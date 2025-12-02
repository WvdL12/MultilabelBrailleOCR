import itertools

import numpy as np
import torch
from collections import Counter

from .braille_utils import ml_to_mc

# TODO: Revisit metrics, focus on:
# * Conventional Accuracy for Class and Label Acc
# * Mean Hamming Distance / error to contextualise accuracies -- MCC for multi-class? Simply macro avg? See https://arxiv.org/pdf/1008.2908
# * Class Robust F1 (using rates) - add Precision and Recall for context
# * Label Robust F1, macro average (per label, arithmetic mean) - add Precision and Recall for context

def label_frequency(data_y):
    N = data_y.shape[0]
    label_weights = np.sum(data_y, axis=0) / N
    return label_weights

def class_frequency(data_y):
    N = data_y.shape[0]
    class_weights = np.sum(data_y, axis=0) / N
    return class_weights

def class_balance(data_y, eps=1e-10):
    
    N, K = data_y.shape
    classes = class_frequency(data_y)
    
    H = -sum(np.log(classes+eps) * classes) #shannon entropy
    return H/np.log(K)

def within_label_balance(data_y):
    N = data_y.shape[0]
    label_counts = np.sum(data_y, axis=0)
    return label_counts / (N - label_counts)
    
def between_label_imbalance(data_y):
    N, K = data_y.shape
    label_counts = np.sum(data_y, axis=0)
    imbal_ratio = np.max(label_counts) / label_counts
    imbal_ratio_mean = np.mean(imbal_ratio)
    imbal_ratio_std = np.sqrt(sum((imbal_ratio - imbal_ratio_mean) ** 2 / (K - 1)))
    
    return imbal_ratio_mean, imbal_ratio_std, imbal_ratio

def N_ij(data_y, y1, y2):
    combined = (data_y[:, y1] << 1) | data_y[:, y2]
    counts = np.bincount(combined, minlength=4)
    
    return counts.reshape(2, 2)

def mutual_information(data_y, y1, y2):
    if y1 == y2:
        return np.nan
    N = data_y.shape[0]
    Nij = N_ij(data_y, y1, y2).astype(float)
    Nij[Nij == 0] = np.nan
    
    Nij_over_N = Nij / N
    sum_Nij_i = np.sum(Nij, axis=1)
    sum_Nij_j = np.sum(Nij, axis=0)
    
    mi_matrix = Nij_over_N * (np.log(N) + np.log(Nij) - np.log(sum_Nij_i[:, None]) - np.log(sum_Nij_j[None, :]))
    mi = np.nansum(mi_matrix)
    
    return mi

def mean_mutual_information(data_y):
    k = data_y.shape[1]
    mi = np.nanmean([mutual_information(data_y, y1, y2) for y1 in range(k) for y2 in range(y1+1, k)])
    return mi

def label_correlations(data_y, agg_absolute=False):
    K = data_y.shape[1]
    cov = np.cov(data_y.T)
    D = 1 / np.sqrt(np.diag(cov))
    label_corrs = D * cov * D
    np.fill_diagonal(label_corrs, np.nan)
    
    agg_label_corrs = np.abs(label_corrs) if agg_absolute else label_corrs
    
    corr_mean = np.nanmean(agg_label_corrs)
    corr_std = np.sqrt(np.nansum((agg_label_corrs - corr_mean) ** 2 / (K - 1)))
    
    return corr_mean, corr_std, label_corrs
    
def label_correlations_v2(data_y, agg_absolute=False):
    K = data_y.shape[1]
    cov = np.cov(data_y.T)
    D = 1 / np.sqrt(np.diag(cov))
    label_corrs = D * cov * D.reshape((-1,1))
    np.fill_diagonal(label_corrs, np.nan)
    
    agg_label_corrs = np.abs(label_corrs) if agg_absolute else label_corrs
    
    corr_mean = np.nanmean(agg_label_corrs)
    corr_std = np.sqrt(np.nansum((agg_label_corrs - corr_mean) ** 2 / (K - 1)))
    
    return corr_mean, corr_std, label_corrs
    

def softmax_accuracy(preds, trues):
    """
    Global class accuracy metric for softmax classifier.
    Measured as correct classifications over total.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Global accuracy metric
    """
    if not preds.shape == trues.shape:
        return -1

    count = sum(preds == trues)
    total = preds.shape[0]

    return count / total


def sigmoid_accuracy(preds, trues):
    """
    Global strict / subset accuracy metric for sigmoid classifier.
    Measured as correct sample classifications over total.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Global strict accuracy metric
    """
    if not preds.shape == trues.shape:
        return -1

    count = sum(np.all(preds == trues, axis=1))
    total = preds.shape[0]
    # for i in range(total):
    #     if np.array_equal(preds[i], trues[i]):
    #         count += 1

    return count / total


def bitwise_metrics(preds, trues):
    """
    Calculates the total TP, TN, FP, FN over all classes.

    :param preds: Predicted labels
    :param trues: True labels
    :return: total metrics
    """
    samples, bits = preds.shape
    if type(preds) != np.ndarray:
        TP = torch.sum((preds == 1) & (trues == 1)).item()
        TN = torch.sum((preds == 0) & (trues == 0)).item()
        FP = torch.sum((preds == 1) & (trues == 0)).item()
        FN = torch.sum((preds == 0) & (trues == 1)).item()
    else:
        TP = np.sum(np.logical_and(preds == 1, trues == 1))
        TN = np.sum(np.logical_and(preds == 0, trues == 0))
        FP = np.sum(np.logical_and(preds == 1, trues == 0))
        FN = np.sum(np.logical_and(preds == 0, trues == 1))
    
    return TP, TN, FP, FN


def bitwise_accuracy(preds, trues):
    """
    Global accuracy metric for sigmoid classifier.
    Measured as correct classifications of individual labels over total.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Global accuracy metric
    """
    if not preds.shape == trues.shape:
        return -1
    
    TP, TN, FP, FN = bitwise_metrics(preds, trues)

    return (TP + TN) / (TP + TN + FP + FN)


def bitwise_precision(preds, trues):
    """
    Global precision metric for sigmoid classifier.
    Measured in terms of total TP and FP over all classes.

    Also called micro-average precision.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Micro precision metric
    """
    if not preds.shape == trues.shape:
        return -1

    TP, TN, FP, FN = bitwise_metrics(preds, trues)

    return TP / (TP + FP)


def bitwise_recall(preds, trues):
    """
    Global recall metric for sigmoid classifier.
    Measured in terms of total TP and FN over all classes.

    Also called micro-average recall.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Micro recall metric
    """
    if not preds.shape == trues.shape:
        return -1

    TP, TN, FP, FN = bitwise_metrics(preds, trues)

    return TP / (TP + FN)


def bitwise_F1(preds, trues):
    """
    Global F1 metric for sigmoid classifier.
    Measured in terms of total TP, FP and FN over all classes.
    That is, measured in terms of global precision and recall

    Also called micro-average F1.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Micro F1 metric
    """
    if not preds.shape == trues.shape:
        return -1

    prec = bitwise_precision(preds, trues)
    recall = bitwise_recall(preds, trues)

    return 2 * prec * recall / (prec + recall)


def macro_metrics(preds, trues):
    """
    Calculates the total TP, TN, FP, FN per class.

    :param preds: Predicted labels
    :param trues: True labels
    :return: per class metrics
    """
    samples, bits = preds.shape
    if type(preds) != np.ndarray:
        preds = preds.numpy()
        trues = trues.numpy()

    # Initialize arrays for TP, TN, FP, FN for each class
    TP = np.zeros(bits)
    TN = np.zeros(bits)
    FP = np.zeros(bits)
    FN = np.zeros(bits)
    
    # Calculate TP, TN, FP, FN for each class
    for b in range(bits):
        TP[b] = np.sum(np.logical_and(preds[:, b] == 1, trues[:, b] == 1))
        TN[b] = np.sum(np.logical_and(preds[:, b] == 0, trues[:, b] == 0))
        FP[b] = np.sum(np.logical_and(preds[:, b] == 1, trues[:, b] == 0))
        FN[b] = np.sum(np.logical_and(preds[:, b] == 0, trues[:, b] == 1))
        
    return TP, TN, FP, FN


def per_class_accuracy(preds, trues):
    """
    Global accuracy metric for sigmoid classifier.
    Measured as average of class-wise accuracies.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Macro accuracy metric
    """
    if not preds.shape == trues.shape:
        return -1
    samples, bits = preds.shape

    TP, TN, FP, FN = macro_metrics(preds, trues)
    macro = []
    for i in range(bits):
        macro.append((TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i]))
    return macro

def macro_accuracy(preds, trues):
    # preds = ml_to_mc(preds)
    # trues = ml_to_mc(trues)
    per_class = per_class_accuracy(preds, trues)
    return np.nanmean(per_class)

def per_class_precision(preds, trues):
    """
    Global precision metric for sigmoid classifier.
    Measured as average of class-wise precisions.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Macro precision metric
    """
    if not preds.shape == trues.shape:
        return -1
    samples, bits = preds.shape

    TP, TN, FP, FN = macro_metrics(preds, trues)
    macro = []
    for i in range(bits):
        tpfp = (TP[i] + FP[i])
        if tpfp == 0:
            # No positive predictions - precision 0
            macro.append(0.0)
        else:
            macro.append(TP[i] / tpfp)
    return macro

def macro_precision(preds, trues):
    # preds = ml_to_mc(preds)
    # trues = ml_to_mc(trues)
    per_class = per_class_precision(preds, trues)
    return np.nanmean(per_class)

def per_class_recall(preds, trues):
    """
    Global recall metric for sigmoid classifier.
    Measured as average of class-wise recalls.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Macro recall metric
    """
    if not preds.shape == trues.shape:
        return -1
    samples, bits = preds.shape

    TP, TN, FP, FN = macro_metrics(preds, trues)
    macro = []
    for i in range(bits):
        tpfn = (TP[i] + FN[i])
        if tpfn == 0:
            # No positive samples - recall 1
            macro.append(1.0)
        else:
            macro.append(TP[i] / tpfn)
    return macro

def macro_recall(preds, trues):
    # preds = ml_to_mc(preds)
    # trues = ml_to_mc(trues)
    per_class = per_class_recall(preds, trues)
    return np.nanmean(per_class)

def per_class_F1(preds, trues, return_PR=False):
    """
    Global F1 metric for sigmoid classifier.
    Measured as average of class-wise F1s.
    That is, measured in terms of macro precision and recall.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Macro F1 metric
    """
    if not preds.shape == trues.shape:
        return -1
    samples, bits = preds.shape

    prec = per_class_precision(preds, trues)
    recall = per_class_recall(preds, trues)

    f1 = []
    for i in range(bits):
        pr = prec[i] + recall[i]
        if pr == 0:
            # F1 - 0
            f1.append(0.0)
        else:
            f1.append(2 * prec[i] * recall[i] / pr)
    if return_PR:
        return f1, prec, recall
    return f1

def macro_F1(preds, trues):
    # preds = ml_to_mc(preds)
    # trues = ml_to_mc(trues)
    per_class = per_class_F1(preds, trues)
    return np.nanmean(per_class)

def hamming_distances(preds, trues):
    """
    Calculates the per-sample hamming distances between true and predicted labels.

    :param preds: Predicted labels
    :param trues: True labels
    :return: Hamming distance per sample
    """
    dists = np.sum(np.abs(preds - trues), axis=1)
    return dists

def mean_hamming_distance(preds, trues):
    return np.mean(hamming_distances(preds, trues))

def mean_error_distance(preds, trues):
    errors = np.any(preds != trues, axis=1)
    return mean_hamming_distance(preds[errors], trues[errors])
