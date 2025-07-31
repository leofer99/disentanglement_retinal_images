import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, recall_score)


## 1. Discrimination:
# 1.1 AUROC (recommended)
def calculate_auroc(y_true=None, y_pred_probs=None):
    
    if y_true is None or y_pred_probs is None:
        raise ValueError("y_true and y_pred_probs must be provided.")

    return roc_auc_score(y_true, y_pred_probs)

## 2. Calibration:
# 2.2 O:E Ratio (not essential)
def observed_expected_ratio(y_true=None, y_pred_probs=None):
    """
    Computes the observed expected ratio (O:E).
    O:E compares the total number of actual events (positives) to the total number of expected events (sum of predicted probabilities).
    >1 - model underestimates (more events occurred than predicted)
    <1 - model overestimates (fewer events occurred than predicted)
    """
    if y_true is None or y_pred_probs is None:
        raise ValueError("y_true and y_pred_probs must be provided.")

    observed = np.sum(y_true)
    expected = np.sum(y_pred_probs)
    return observed / expected if expected != 0 else np.nan

# 2.3 ECE (not essential)
def expected_calibration_error(y_true=None, y_pred_probs=None, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    Summarizes the average difference between predicted probabilities and actual outcomes across bins.
    """
    if y_true is None or y_pred_probs is None:
        raise ValueError("y_true and y_pred_probs must be provided.")

    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_pred_probs, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        bin_mask = binids == i
        bin_size = np.sum(bin_mask)
        if bin_size == 0:
            continue
        bin_prob = y_pred_probs[bin_mask]
        bin_true = y_true[bin_mask]
        acc = np.mean(bin_true)
        conf = np.mean(bin_prob)
        ece += (bin_size / len(y_true)) * abs(acc - conf)

    # print("ECE: ", ece)
    return ece

## 4. Classification:
# 4.1 Sensitivity/recall (TPR) + Specificity (FNR) (not essential)
def compute_tpr_fpr_fnr_tnr(y_true=None, y_pred=None):
    
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must be provided.")

    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  # TPs
    FP = cm[0, 1]  # FPs
    TN = cm[0, 0]  # TNs
    FN = cm[1, 0]  # FNs
    
    # Compute TPR and FPR 
    TPR = TP / (FN + TP) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0
    TNR = TN / (FP + TN) if (FP + TN) != 0 else 0
    
    return TPR, FPR, FNR, TNR

def compute_weighted_tpr(self, y_true=None, y_pred=None):

    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must be provided.")
    
    weighted_recall= recall_score(y_true, y_pred, average='weighted')

    return weighted_recall

# 4.2 PPV/precision and NPV
def calculate_ppv_and_npv(y_true=None, y_pred=None):

    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must be provided.")

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ppv = tp / (tp+fp) if (tp+fp) != 0 else 0
    npv = tn / (tn+fn) if (tn+fn) != 0 else 0
    
    return ppv, npv


# 4.3 Classification Threshold Plot
# (showing either sensitivity and specificity or PPV and NPV by threshold)
def plot_sensitivity_specificity(mode='eval', y_true=None, y_pred_probs=None):
    model_name = model_name

    if y_true is None or y_pred_probs is None:
        raise ValueError("y_true and y_pred_probs must be provided.")


    thresholds = np.linspace(0, 1, 101)
    sensitivities = []
    specificities = []

    for th in thresholds:
        y_pred = (y_pred_probs >= th).astype(int)
        tpr, _, fnr, _ = compute_tpr_fpr_fnr_tnr(y_true, y_pred)

        sensitivities.append(tpr)
        specificities.append(fnr)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, label='Sensitivity (Recall)', color='green')
    plt.plot(thresholds, specificities, label='Specificity (TNR)', color='red')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title(f'Sensitivity & Specificity vs. Threshold - {model_name} {mode}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Save image
    save_image(plt, img_name=f'Sensitivity_Specificity_vs_Threshold_{mode}')
    plt.close()

def plot_ppv_npv(mode='eval', y_true=None, y_pred_probs=None):
    model_name=model_name

    if y_true is None or y_pred_probs is None:
        raise ValueError("y_true and y_pred_probs must be provided.")

    thresholds = np.linspace(0, 1, 101)
    ppvs = []
    npvs = []

    for th in thresholds:
        y_pred = (y_pred_probs >= th).astype(int)
        ppv, npv = calculate_ppv_and_npv(y_true, y_pred)

        ppvs.append(ppv)
        npvs.append(npv)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, ppvs, label='PPV (Precision)', color='blue')
    plt.plot(thresholds, npvs, label='NPV', color='orange')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title(f'PPV & NPV vs. Threshold - {model_name} {mode}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Save image
    save_image(plt, img_name=f'PPV_NPV_vs_Threshold_{mode}')
    plt.close()


