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

from model_utils import ModelWrapper, myDataLoader, IteratedFRace
from model_utils import num_to_bin_string, bin_to_num, num_to_bin, ml_to_mc, mc_to_ml
from model_utils import bitwise_accuracy


DATASETS = os.path.join('..', 'Datasets', 'numpy_datasets')
MODEL_PATH = os.path.join('models')
ML_LOGS = os.path.join('opt_logs', 'ml_tuning')
MC_LOGS = os.path.join('opt_logs', 'mc_tuning')

angelina = os.path.join(DATASETS, "angelina.npz")
data = np.load(angelina)

train_X = data["train_x"] / 255.
train_Y = data["train_y"]

# val_X = data["val_x"] / 255.
# val_Y = data["val_y"]

# test_X = data["test_x"] / 255.
# test_Y = data["test_y"]

train_Y_cat = ml_to_mc(data["train_y"])
val_Y_cat = ml_to_mc(data["val_y"])
test_Y_cat = ml_to_mc(data["test_y"])

N, p = train_Y.shape
C = train_Y_cat.shape[1]

kfold = KFold(10, shuffle=True, random_state=220623).split(train_X)
kfold_split = [(t, v) for (t, v) in kfold]

cts_space = {
    'lr': (0.0001, 0.01),
    'beta_1': (0.85, 0.95),
    'beta_2': (0.99, 0.9999),
    'l2': (0.0005, 0.02)
}
disc_space = {
    'filters': (16, 64),
    'kernel_sz': (2,5),
    'dense_sz': (25, 150),
    'activs': (0, 2)
}
bin_space = {
    'padding': (0, 1),
    'grey': (0, 1),
    'sqr': (0, 1)
}

activations = {
    0: F.relu,
    1: F.tanh,
    2: F.leaky_relu
}

log_path_ml = os.path.join(ML_LOGS, f"base_{datetime.now().strftime(format='%Y%m%d_%H%M%S')}_elite_set.json")
log_path_mc = os.path.join(MC_LOGS, f"base_{datetime.now().strftime(format='%Y%m%d_%H%M%S')}_elite_set.json")

SZ = 20
ITER = 5
N_MIN = 5
BATCH = 2048
EPS = 20
PAT = 2
VERB = -1
ALPHA = 0.05

print('$' * 120)
print('Tuning ML Model')
print('$' * 120)

IFR_ML = IteratedFRace(cts_params=cts_space, discrete_params=disc_space, bin_params=bin_space, functions=activations,
                    train_X=train_X, train_Y=train_Y, out_sz=6, kfold_split=kfold_split,
                    init_sz=SZ, iterations=ITER, n_min=N_MIN, alpha=ALPHA, dev='cuda', log_path=log_path_ml)
# results_ML = IFR_ML.run(batch_sz=BATCH, epochs=EPS, pat=PAT, verbose=VERB)

print('$' * 120)
print('Tuning MC Model')
print('$' * 120)

IFR_MC = IteratedFRace(cts_params=cts_space, discrete_params=disc_space, bin_params=bin_space, functions=activations,
                    train_X=train_X, train_Y=train_Y_cat, out_sz=C, kfold_split=kfold_split,
                    init_sz=SZ, iterations=ITER, n_min=N_MIN, alpha=ALPHA, dev='cuda', log_path=log_path_mc)
results_MC = IFR_MC.run(batch_sz=BATCH, epochs=EPS, pat=PAT, verbose=VERB)

# print('$' * 120)
# print('Final ML Results')
# print(results_ML['model_configs'])
# print(results_ML['final_ranks'])
# print('$' * 120)

print('$' * 120)
print('Final MC Results')
print(results_MC['model_configs'])
print(results_MC['final_ranks'])
print('$' * 120)
