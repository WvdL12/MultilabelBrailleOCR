import os, cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import json


from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import functional, Resize

import time
from datetime import datetime
from tqdm import tqdm

from model_utils.braille_utils import num_to_bin, bin_to_num
from model_utils.metrics import bitwise_accuracy
from model_utils.cbr_model import ModelWrapper
from model_utils.torch_utils import myDataLoader
from model_utils.augment import apply_augmentations

from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats.qmc import Sobol

SQR_SIZE = (32, 32, 3)

class IteratedFRace:
    
    def __init__(self, cts_params: dict, discrete_params: dict, bin_params: dict, functions,
                 train_X, train_Y, out_sz, kfold_split, augmentations=None, random_state=None,
                 init_sz=30, iterations=6, n_min=5, alpha=0.05, dev='cuda', log_path=None):
        self.cts_params = cts_params
        self.discrete_params = discrete_params
        self.bin_params = bin_params
        self.functions = functions
        self.train_X = train_X
        self.train_Y = train_Y
        self.out_sz = out_sz
        self.kfold_split = kfold_split
        self.augmentations = augmentations
        self.random_state = random_state
        
        self.init_sz = init_sz
        self.iterations = iterations
        self.n_min = n_min
        self.alpha = alpha
        self.dev = dev
        self.log_path = log_path
        
        self.config_history = {}
        self.score_history = {}
        self.loss_history = {}
        self.acc_history = {}
        
    def add_augmentations(self, trainX, trainY, fold):
        return trainX, trainY
        
    def train_fold(self, model_args, train_loader, val_loader, epochs=10, pat=3, verbose=-1):
        lr, beta_1, beta_2, l2, filters, kernel_sz, dense_sz, activs, padding, grey, sqr = model_args
        in_sz = SQR_SIZE if sqr else train_loader.dataset[0][0].shape
        if grey:
            in_sz = (in_sz[0], in_sz[1], 1)
        
        print(f"lr {round(lr, 4)}, beta_1 {round(beta_1, 4)}, beta_2 {round(beta_2, 4)}, l2 {round(l2, 4)}, filters {filters}, kernel_sz {kernel_sz},",
            f"dense_sz {dense_sz}, activs {self.functions[activs].__name__}, padding {padding},",
            f"in_sz {in_sz}, grey {grey}, sqr {sqr}")
        
        model = ModelWrapper(in_size=in_sz, filts=filters, kerns=kernel_sz, pad=padding, out_size=self.out_sz,
                            activ=self.functions[activs], dense_sz=dense_sz, beta_1=beta_1, beta_2=beta_2, lr=lr, l2=l2,
                            grey_scaled=grey, square_in=sqr, dev=self.dev)
        history = model.train(train_loader, val_loader, epochs=epochs, patience=pat, verbose=verbose)
        
        ce = model.conv_epoch
        t_loss, t_acc = history['train_loss'][ce], history['train_acc'][ce]
        loss, acc = history['val_loss'][ce], history['val_acc'][ce]
        
        return model.trained_epochs, t_loss, loss, t_acc, acc
    
    def iterate(self, model_configs: dict, loss_history: dict, acc_history: dict,
                batch_sz=1024, epochs=10, pat=3, verbose=-1):
        
        accs, losses = {}, {}
        for key in model_configs:
            accs[key] = acc_history[key] if key in acc_history else []
            losses[key] = loss_history[key] if key in loss_history else []
        
        start_t = time.time()
        for  f, (train_ix, val_ix) in enumerate(self.kfold_split):
            print(f"Training fold {f} on list of {len(model_configs)} configurations.")
            ft = time.time()
            trainX, trainY, valX, valY = self.train_X[train_ix], self.train_Y[train_ix], self.train_X[val_ix], self.train_Y[val_ix]
            if self.augmentations is not None:
                trainX, trainY = apply_augmentations(trainX, trainY, self.augmentations, f,
                                                     shuffle=self.random_state is not None, random_state=self.random_state)
            
            train_loader = myDataLoader(trainX, trainY, batch_sz=batch_sz, dev=self.dev)
            val_loader = myDataLoader(valX, valY, batch_sz=batch_sz, dev=self.dev)
                
            for (config_idx, model_args) in model_configs.items():
                if len(losses[config_idx]) > f:
                    print(f"Configuration {config_idx}, evaluated once before -- skipping.")
                    continue
                f_t = time.time()        
                # model_args.append(x_data.shape[1:])
                trained, t_loss, loss, t_acc, acc = self.train_fold(model_args, train_loader, val_loader,
                                                                epochs=epochs, pat=pat, verbose=verbose)
                
                print(f"Configuration {config_idx}, fold {f} trained {trained} epochs in {round(time.time() - f_t, 3)}s. ",
                        f"Loss: tr - {round(t_loss, 3)}, val - {round(loss, 3)}. ",
                        f"Accuracy: tr - {round(t_acc, 3)}, val - {round(acc, 3)}.")
                accs[config_idx].append(acc)
                losses[config_idx].append(loss)
                
            # perform Friedman tests on losses to determine if configs are dropped
            df = pd.DataFrame({key: pd.Series(value) for key, value in losses.items()}).dropna()
            F, pval_friedman = friedmanchisquare(*df.to_numpy().T)
            if pval_friedman < self.alpha:
                print(f"Significant difference in validation losses across group, after {f+1} folds ({pval_friedman})")
                means = df.apply('mean', axis=0).sort_values()
                ranks = means.index
                best_config = df[ranks[0]]
                other_configs = df[ranks[1:]]
            
                alpha_adj = 1-pow(1-self.alpha, 1. / other_configs.shape[1]) # Sidak correction
                for conf in other_configs:
                    W, pval_wilcoxon = wilcoxon(best_config, other_configs[conf], alternative='less')
                    if pval_wilcoxon < self.alpha: # Use adjusted alpha?
                        # Drop configuration from further testing
                        print(f"Configuration {conf} (mean vloss {means[conf]}) dropped with pvalue {pval_wilcoxon} against configuration {ranks[0]} (mean vloss {means[ranks[0]]})")
                        self.score_history[conf] = means[conf]
                        self.loss_history[conf] = losses[conf]
                        self.acc_history[conf] = accs[conf]
                        
                        model_configs.pop(conf)
                        losses.pop(conf)
                        accs.pop(conf)
                print("Paired tests completed")
            else:
                print(f"Not enough evidence to support a difference across group, after {f+1} folds ({pval_friedman})")

            if len(model_configs) <= self.n_min:
                break
                
            del train_loader, val_loader
            torch.cuda.empty_cache()
            
            print(f"Fold {f} training and evaluation complete in {round(time.time() - ft, 3)} seconds. {len(model_configs)} configurations remaining.")
            print('-'*100)
            
        print(f"Iteration completed in {round(time.time() - start_t, 3)}s. Remaining configurations {len(model_configs)}. ",
            f"Mean validation loss {np.round(np.mean([np.mean(v) for v in losses.values()]), 3)} and acc {np.round(np.mean([np.mean(a) for a in accs.values()]), 3)}")
        print('-'*75)
        return model_configs, losses, accs
    
    
    def run(self, batch_sz=1024, epochs=10, pat=3, verbose=-1):
        param_space = self.cts_params.copy()
        param_space.update(self.discrete_params)
        param_space.update(self.bin_params)
        
        param_ranges = {p: ub - lb for p, (lb, ub) in self.cts_params.items()}
        discrete_ranges = {p: ub - lb for p, (lb, ub) in self.discrete_params.items()}
        bin_ranges = {p: ub - lb for p, (lb, ub) in self.bin_params.items()}
        param_ranges.update(discrete_ranges)
        param_ranges.update(bin_ranges)
        
        discrete_dists = {p: np.repeat(1 / pr, pr) for p, pr in discrete_ranges.items()}
        discrete_dists.update({
            p: np.repeat(1 / pr, pr) for p, pr in bin_ranges.items()
        })
        
        D = len(param_space)
        int_idx = len(self.cts_params)
        bin_idx = int_idx + len(self.discrete_params)
 
        # scores_history = {}
        # configs_history = {}
        model_configs = {}
        loss_history = {}
        acc_history = {}
        
        m = int(np.ceil(np.log2(self.init_sz)))
        SAMPLER = Sobol(d=D)
        _ = SAMPLER.fast_forward(2 ** m)
        points = SAMPLER.random_base2(m=m)
        for i in range(self.init_sz):
            sample = points[i]
            params = [p * (ub - lb) + lb for p, (lb, ub) in zip(sample, param_space.values())]
            params = [p if idx < int_idx else round(p) for idx, p in enumerate(params)]
            model_configs[i] = params
            self.config_history[i] = params
        IDX = self.init_sz
        
        start = time.time()
        for l in range(1, self.iterations+1):
            print("#" * 100)
            print(f"Iteration {l} of Iterated F-race.")
            model_configs, loss_history, acc_history = self.iterate(model_configs, loss_history, acc_history,
                                                                        batch_sz=batch_sz, epochs=epochs, pat=pat, verbose=verbose)
            N_e = len(model_configs)

            scores = {idx: np.mean(loss_history[idx]) for idx in model_configs}
            for idx, sc in scores.items():
                self.score_history[idx] = sc
                self.loss_history[idx] = loss_history[idx]
                self.acc_history[idx] = acc_history[idx]
            ranks = list(dict(sorted(scores.items(), key=lambda x: x[1])).keys())
            if N_e > self.n_min:
                print(f"More than desired minimum configurations remaining: {N_e}")
                for i in range(self.n_min, N_e):
                    idx = ranks[i]
                    model_configs.pop(idx)
                    loss_history.pop(idx)
                    acc_history.pop(idx)
                    scores.pop(idx)
                ranks = ranks[:self.n_min]
                N_e = self.n_min
            if self.log_path:
                log_dict = {
                    'model_configs': model_configs,
                    'loss_histories': loss_history,
                    'acc_histories': acc_history,
                    'final_ranks': ranks,
                    'final_scores': {sk: sv if not np.isnan(sv) else 'nan' for sk, sv in scores.items()}
                }
                with open(self.log_path, 'w') as log_file:
                    json.dump(log_dict, log_file)

                log_dict = {
                    'model_configs': self.config_history,
                    'loss_histories': self.loss_history,
                    'acc_histories': self.acc_history,
                    'final_scores': {sk: sv if not np.isnan(sv) else 'nan' for sk, sv in self.score_history.items()}
                }
                with open(self.log_path.replace('elite', 'historic'), 'w') as log_file:
                    json.dump(log_dict, log_file)

            if l == self.iterations:
                break
                
            # Select elite to sample (new one for every new sample? Or once for all new samples?)
            weights = {idx: (N_e - ranks.index(idx)) / (N_e * (N_e + 1) / 2) for idx in model_configs}
            cum_weights = np.cumsum(list(weights.values()))
            r = np.random.rand()
            
            sample_idx = list(weights.keys())[np.searchsorted(cum_weights, r)]
            sample_config = model_configs[sample_idx]
            
            
            # for key in [k for k in model_configs if k != sample_idx and k != ranks[0]]: # Keep sampled elite config and global best config
            for key in [k for k in model_configs if k != sample_idx]: # Keep only sampled elite config
                model_configs.pop(key)
                loss_history.pop(key)
                acc_history.pop(key)
            
            # Recompute sampling parameters
            N_e = len(model_configs)
            sig_mult = pow(1 / self.init_sz, l / D)
            
            for pi, (key, param) in enumerate(zip(param_space, sample_config)):
                if pi < bin_idx:
                    continue
                discrete_dists[key] = np.array([pr * (1 - l / self.iterations) + int(idx + param_space[key][0] == param) * l / self.iterations
                                                for idx, pr in enumerate(discrete_dists[key])])
            
            for i in range(N_e, self.init_sz):
                # Sample around parameters of elite candidate
                new_conf = []
                for pi, (key, param) in enumerate(zip(param_space, sample_config)):
                    bounds = param_space[key]
                    if pi < bin_idx:             
                        sample = np.random.normal(param, param_ranges[key] * sig_mult)
                        sample = bounds[0] if sample < bounds[0] else bounds[1] if sample > bounds[1] else sample
                        if pi >= int_idx:
                            sample = round(sample)
                    else:
                        cum_weights = np.cumsum(discrete_dists[key])
                        r = np.random.rand()

                        sample_idx = np.searchsorted(cum_weights, r)
                        sample = list(range(bounds[0], bounds[1]+1))[sample_idx]
                    new_conf.append(sample)

                # Assign IDX to new candidate, and increment IDX
                model_configs[IDX] = new_conf
                self.config_history[IDX] = new_conf
                IDX += 1

        end = time.time()

        # ranks_records.append(ranks)
        # scores_records.append(scores)
        # configs_records.append(model_configs)
        print('Iterated F-race takes {:.2f} seconds to tune roughly {} configurations'.format(end - start, self.init_sz * self.iterations))
        
        return {
            'model_configs': model_configs,
            'loss_histories': loss_history,
            'acc_histories': acc_history,
            'final_ranks': ranks,
            'final_scores': scores
        }
