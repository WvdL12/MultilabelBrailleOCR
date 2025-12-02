import json
import os

import numpy as np

from .braille_utils import bin_to_num, num_to_bin_string
from .metrics import label_correlations, class_frequency, class_balance
from .braille_utils import ml_to_mc

    
def rotate_sample(x):
    if len(x.shape) < 4:
        return np.rot90(x, k=2)
    return np.rot90(x, k=2, axes=(1,2))


def y_mirror_sample(x):
    if len(x.shape) < 4:
        return np.flip(x, axis=1)
    return np.flip(x, axis=2)


def x_mirror_sample(x):
    if len(x.shape) < 4:
        return np.flip(x, axis=0)
    return np.flip(x, axis=1)


def rotate_label(y):
    return y.T[::-1].T


def y_mirror_label(y):
    return np.hstack([y[:, 3:], y[:,:3]])


def x_mirror_label(y):
    return  y_mirror_label(rotate_label(y))

def rotate(str_label):
    return str_label[::-1]


def y_mirror(str_label):
    return str_label[3:] + str_label[:3]


def x_mirror(str_label):
    return y_mirror(rotate(str_label))


def label_balance_resample(trainY, undersample_candidates, eps=0.1, return_iters=False):
    N = trainY.shape[0]

    undersample = np.zeros(N).astype(bool)
    trainY_rot = rotate_label(trainY)
    resample_rot = np.zeros(N).astype(bool)
    trainY_xmir = x_mirror_label(trainY)
    resample_xmir = np.zeros(N).astype(bool)
    trainY_ymir = y_mirror_label(trainY)
    resample_ymir = np.zeros(N).astype(bool)

    trainY_mc = ml_to_mc(trainY)
    trainY_rot_mc = ml_to_mc(trainY_rot)
    trainY_xmir_mc = ml_to_mc(trainY_xmir)
    trainY_ymir_mc = ml_to_mc(trainY_ymir)
    
    iter = 0
    imbals = []
    correlations = []
    cfreqs_std = []
    lfreqs_std = []
    lfreqs_means = []
    while True:
        iter += 1
        
        new_Y_ml = np.vstack([trainY[~undersample], trainY_rot[resample_rot], trainY_xmir[resample_xmir], trainY_ymir[resample_ymir]])
        new_Y_mc = np.vstack([trainY_mc[~undersample], trainY_rot_mc[resample_rot], trainY_xmir_mc[resample_xmir], trainY_ymir_mc[resample_ymir]])
        class_weights = class_frequency(new_Y_mc)
        class_bal = class_balance(new_Y_mc)
        imbals.append(class_bal)
        cfreqs_std.append(np.std(class_weights))
        
        _, _, label_corrs = label_correlations(new_Y_ml)
        label_weights = class_frequency(new_Y_ml)

        lfreqs_std.append(np.std(label_weights))
        lfreqs_means.append(np.mean(label_weights))
        correlations.append(label_corrs)
        
        abs_corrs = np.abs(label_corrs)
        i, j = np.unravel_index(np.nanargmax(abs_corrs), label_corrs.shape)
        max_cor = abs_corrs[i, j]

        if max_cor < eps:
            print(f'Label imbalance sufficiently reduced after {iter} iterations.')
            break
        else:
            cor = label_corrs[i, j]
            li = int((i > j) or (cor > 0))
            nli = int(not li)
            lj = int((i > j) and (cor < 0))
            
            mask_under = (trainY[:, i] == nli) & (trainY[:, j] == lj) & undersample_candidates & ~undersample
            if sum(mask_under) > 0:
                undersample = undersample | mask_under
                continue
            
            mask_rot = (trainY_rot[:, i] == li) & (trainY_rot[:, j] == lj) & ~resample_rot
            if sum(mask_rot) > 0:
                resample_rot = resample_rot | mask_rot
                continue
            
            mask_xmir = (trainY_xmir[:, i] == li) & (trainY_xmir[:, j] == lj) & ~resample_xmir
            if sum(mask_xmir) > 0:
                resample_xmir = resample_xmir | mask_xmir
                continue
            
            mask_ymir = (trainY_ymir[:, i] == li) & (trainY_ymir[:, j] == lj) & ~resample_ymir
            if sum(mask_ymir) > 0:
                resample_ymir = resample_ymir | mask_ymir
                continue
            
            print(f"No more resampling / undersampling candidates. Ending augmentation after {iter} iterations.")
            break
    augs = {
        'undersample': undersample,
        'rotate': resample_rot,
        'xmir': resample_xmir,
        'ymir': resample_ymir
    }
    
    if return_iters:
        return augs, {
            'class_imbalance': imbals,
            'class_frequency_std': cfreqs_std,
            'label_correlations': correlations,
            'label_frequency_std': lfreqs_std,
            'label_frequency_means': lfreqs_means
        }
    else:
        return augs

def class_balance_resample(trainY, undersample_candidates, eps=0.05, return_iters=False):
    N = trainY.shape[0]

    undersample = np.zeros(N).astype(bool)
    trainY_rot = rotate_label(trainY)
    resample_rot = np.zeros(N).astype(bool)
    trainY_xmir = x_mirror_label(trainY)
    resample_xmir = np.zeros(N).astype(bool)
    trainY_ymir = y_mirror_label(trainY)
    resample_ymir = np.zeros(N).astype(bool)

    trainY_mc = ml_to_mc(trainY)
    trainY_rot_mc = ml_to_mc(trainY_rot)
    trainY_xmir_mc = ml_to_mc(trainY_xmir)
    trainY_ymir_mc = ml_to_mc(trainY_ymir)

    minimised = []
    maximised = [0]

    imbals = []
    correlations = []
    cfreqs_std = []
    lfreqs_std = []
    lfreqs_means = []
    while True:
        new_Y_ml = np.vstack([trainY[~undersample], trainY_rot[resample_rot], trainY_xmir[resample_xmir], trainY_ymir[resample_ymir]])
        new_Y_mc = np.vstack([trainY_mc[~undersample], trainY_rot_mc[resample_rot], trainY_xmir_mc[resample_xmir], trainY_ymir_mc[resample_ymir]])
        class_weights = class_frequency(new_Y_mc)
        class_bal = class_balance(new_Y_mc)
        imbals.append(class_bal)
        cfreqs_std.append(np.std(class_weights))
        
        _, _, label_corrs = label_correlations(new_Y_ml)
        label_weights = class_frequency(new_Y_ml)

        lfreqs_std.append(np.std(label_weights))
        lfreqs_means.append(np.mean(label_weights))
        correlations.append(label_corrs)
        
        class_weights[minimised + maximised] = np.nan
        min_f, max_f = np.nanmin(class_weights), np.nanmax(class_weights)
        min_classes = list(np.where(class_weights == min_f)[0])
        max_classes = list(np.where(class_weights == max_f)[0])
        
        if 1 - class_bal < eps:
            print('Class imbalance sufficiently reduced')
            break
        if len(min_classes) + len(max_classes) == 0:
            print('All classes balanced as much as possible')
            break
        
        for c in max_classes:
            mask_under_c = trainY_mc[:,c].astype(bool) & undersample_candidates & ~undersample
            if sum(mask_under_c) > 0:
                undersample = undersample | mask_under_c
            else:
                minimised.append(c)
                # print(f"No more undersampling candidates for class {c}.")
        
        for c in min_classes:
            mask_rot = trainY_rot_mc[:,c].astype(bool) & ~resample_rot
            if sum(mask_rot) > 0:
                resample_rot = resample_rot | mask_rot
                continue       
            
            mask_xmir = trainY_xmir_mc[:,c].astype(bool) & ~resample_xmir
            if sum(mask_xmir) > 0:
                resample_xmir = resample_xmir | mask_xmir
                continue
            
            mask_ymir = trainY_ymir_mc[:,c].astype(bool) & ~resample_ymir
            if sum(mask_ymir) > 0:
                resample_ymir = resample_ymir | mask_ymir
                continue
            
            maximised.append(c)
            # print(f"No more resampling candidates for class {c}.")
    augs = {
        'undersample': undersample,
        'rotate': resample_rot,
        'xmir': resample_xmir,
        'ymir': resample_ymir
    }
    
    if return_iters:
        return augs, {
            'class_imbalance': imbals,
            'class_frequency_std': cfreqs_std,
            'label_correlations': correlations,
            'label_frequency_std': lfreqs_std,
            'label_frequency_means': lfreqs_means
        }
    else:
        return augs
    
def adaptive_resample(trainY, trainY_mc, sample_diffs,
                      class_eps=0.05, corr_eps=0.1,
                      us_step=250, os_step=100, max_iter=300,
                      return_iters=False):
    N = trainY.shape[0]

    undersample = np.zeros(N).astype(bool)
    trainY_rot = rotate_label(trainY)
    resample_rot = np.zeros(N).astype(bool)
    trainY_xmir = x_mirror_label(trainY)
    resample_xmir = np.zeros(N).astype(bool)
    trainY_ymir = y_mirror_label(trainY)
    resample_ymir = np.zeros(N).astype(bool)
    
    trainY_rot_mc = ml_to_mc(trainY_rot)
    trainY_xmir_mc = ml_to_mc(trainY_xmir)
    trainY_ymir_mc = ml_to_mc(trainY_ymir)
    
    minimised = []
    maximised = [0]
    
    imbals = []
    correlations = []
    cfreqs_std = []
    lfreqs_std = []
    lfreqs_means = []
    iter = 0
    while True:
        # print('-'*20, f'ITERATION {iter}', '-'*20)
        iter += 1
        new_Y_ml = np.vstack([trainY[~undersample], trainY_rot[resample_rot], trainY_xmir[resample_xmir], trainY_ymir[resample_ymir]])
        new_Y_mc = np.vstack([trainY_mc[~undersample], trainY_rot_mc[resample_rot], trainY_xmir_mc[resample_xmir], trainY_ymir_mc[resample_ymir]])
        class_weights = class_frequency(new_Y_mc)
        class_bal = class_balance(new_Y_mc)
        imbals.append(class_bal)
        cfreqs_std.append(np.std(class_weights))
        
        class_weights[minimised + maximised] = np.nan
        min_f, max_f = np.nanmin(class_weights), np.nanmax(class_weights)
        min_classes = list(np.where(class_weights == min_f)[0])
        max_classes = list(np.where(class_weights == max_f)[0])
        
        _, _, label_corrs = label_correlations(new_Y_ml)
        label_weights = class_frequency(new_Y_ml)

        lfreqs_std.append(np.std(label_weights))
        lfreqs_means.append(np.mean(label_weights))
        correlations.append(label_corrs)
        
        abs_corrs = np.abs(label_corrs)
        
        i, j = np.unravel_index(np.nanargmax(abs_corrs), label_corrs.shape)
        cor = label_corrs[i, j]
        li = int((i > j) or (cor > 0))
        nli = int(not li)
        lj = int((i > j) and (cor < 0))
        max_cor = abs_corrs[i, j]

        
        if (
            (1 - class_bal < class_eps) 
            or len(min_classes) + len(max_classes) == 0
            ) and (max_cor < corr_eps):
            print(f'Imbalance sufficiently reduced: class balance {class_bal} and max correlation {max_cor}')
            break
        undersample_candidates = (
                np.any(trainY_mc[:,max_classes].astype(bool), axis=1) |
                ((trainY[:, i] == nli) & (trainY[:, j] == lj))
            ) & ~undersample
        undersample_mask = np.zeros(trainY.shape[0]).astype(bool)
        undersample_mask[
            sample_diffs[undersample_candidates].sort_values(
                    'mean_sample_diff', ascending=True
                ).head(us_step).index
            ] = True
        undersample = undersample | undersample_mask
        
        for c in max_classes:
            if sum(trainY_mc[:,c].astype(bool) & ~undersample) == 0:
                minimised.append(c)        
        
        rot_candidates = (
                np.any(trainY_rot_mc[:,min_classes].astype(bool), axis=1) |
                ((trainY_rot[:, i] == li) & (trainY_rot[:, j] == lj))
            ) & ~resample_rot
        rot_mask = np.zeros(trainY.shape[0]).astype(bool)
        rot_mask[
            sample_diffs[rot_candidates].sort_values(
                    'mean_sample_diff', ascending=False
                ).head(os_step).index
            ] = True
        resample_rot = resample_rot | rot_mask
        
        xmir_candidates = (
                np.any(trainY_xmir_mc[:,min_classes].astype(bool), axis=1) |
                ((trainY_xmir[:, i] == li) & (trainY_xmir[:, j] == lj))
            ) & ~resample_xmir
        xmir_mask = np.zeros(trainY.shape[0]).astype(bool)
        xmir_mask[
            sample_diffs[xmir_candidates].sort_values(
                    'mean_sample_diff', ascending=False
                ).head(os_step).index
            ] = True
        resample_xmir = resample_xmir | xmir_mask
        
        ymir_candidates = (
                np.any(trainY_ymir_mc[:,min_classes].astype(bool), axis=1) |
                ((trainY_ymir[:, i] == li) & (trainY_ymir[:, j] == lj))
            ) & ~resample_ymir
        ymir_mask = np.zeros(trainY.shape[0]).astype(bool)
        ymir_mask[
            sample_diffs[ymir_candidates].sort_values(
                    'mean_sample_diff', ascending=False
                ).head(os_step).index
            ] = True
        resample_ymir = resample_ymir | ymir_mask
        
        for c in min_classes:
            mask_rot = trainY_rot_mc[:,c].astype(bool) & ~resample_rot        
            mask_xmir = trainY_xmir_mc[:,c].astype(bool) & ~resample_xmir        
            mask_ymir = trainY_ymir_mc[:,c].astype(bool) & ~resample_ymir
            
            if sum(mask_rot) + sum(mask_xmir) + sum(mask_ymir) == 0:
                maximised.append(c)
    
        if iter >= max_iter:
            print(f'Max iterations reached: {iter}')
            break
    augs = {
        'undersample': undersample,
        'rotate': resample_rot,
        'xmir': resample_xmir,
        'ymir': resample_ymir
    }
    
    if return_iters:
        return augs, {
            'class_imbalance': imbals,
            'class_frequency_std': cfreqs_std,
            'label_correlations': correlations,
            'label_frequency_std': lfreqs_std,
            'label_frequency_means': lfreqs_means
        }
    else:
        return augs
            
    
def apply_augmentations(trainX, trainY, augments, fold=None, shuffle=False, random_state=None, apply_train=True):
    if fold is not None:
        augments = augments[fold]
    undersample = augments['undersample']
    rotate = augments['rotate']
    xmir = augments['xmir']
    ymir = augments['ymir']
    
    trainY_new = np.vstack([
        trainY[~undersample],
        rotate_label(trainY[rotate]),
        x_mirror_label(trainY[xmir]),
        y_mirror_label(trainY[ymir])
    ])
    arr = np.arange(trainY_new.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(arr)
    
    trainX_new = None
    if apply_train:
        trainX_new = np.vstack([
            trainX[~undersample],
            rotate_sample(trainX[rotate]),
            x_mirror_sample(trainX[xmir]),
            y_mirror_sample(trainX[ymir])
        ])[arr]
    
    return trainX_new , trainY_new[arr]


def main():
    out_file = input("Give an output filename:")
    out_file = os.path.join('../..', 'dicts', out_file)
    in_list = input("Give list of available labels:")
    if in_list == "all":
        in_list = range(1, 64)
    else:
        in_list = in_list.split(" ")
        
    label_dict = {}
    for label in in_list:
        label_bin = num_to_bin_string(label)
        label_dict[label_bin] = {}
        print("Label {}, bin {}".format(label, label_bin))
        label_dict[label_bin]["y_mirror"] = y_mirror(label_bin)
        label_dict[label_bin]["x_mirror"] = x_mirror(label_bin)
        label_dict[label_bin]["rotate"] = rotate(label_bin)
    with open("{}.json".format(out_file), "w") as fp:
        json.dump(label_dict, fp)


if __name__ == "__main__":
    main()
