from sklearn.metrics import roc_auc_score
from scipy.stats import norm, ttest_ind, mannwhitneyu
from scipy import stats as st
import matplotlib.pyplot as plt
import baycomp
import rpy2.robjects as robjects
import subprocess
import tempfile
import re


import numpy as np
import warnings
import os
import sys
import torch

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))
from src_fairness.bootstrap_analysis import bootstrap_analysis
from src_fairness.delong_analysis import delong_roc_variance

from torchmetrics.functional.classification import (
    accuracy,
    f1_score,
    recall,
    precision,
    auroc
)

from src_fairness.delong_analysis import fastDeLong, calc_pvalue, compute_ground_truth_statistics



def bootstrap_auroc(y_true, y_probs, n_bootstraps=1000, seed=None):
    rng = np.random.default_rng(seed)
    scores = []

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[indices]
        p_sample = y_probs[indices]

        if len(np.unique(y_sample)) < 2:
            continue  # skip if not both classes present

        score = roc_auc_score(y_sample, p_sample)
        scores.append(score)

    return np.array(scores)

def unpaired_whitney_test(subgroup_data):
    if len(subgroup_data) != 2:
        raise ValueError("Expected exactly 2 subgroups for comparison.")

    _, group0_name, group0_true, _, group0_probs, _ = subgroup_data[0]
    _, group1_name, group1_true, _, group1_probs, _ = subgroup_data[1]

    r_result = run_r_roc_test(group0_true, group0_probs, group1_true, group1_probs)
    print(r_result)  # Optional: for debugging

    p_value = parse_whitney_p_value_from_r_output(r_result)
    return p_value

def unpaired_delong_test(subgroup_data):
    if len(subgroup_data) != 2:
        raise ValueError("Expected exactly 2 subgroups for comparison.")

    _, group0_name, group0_true, _, group0_probs, _ = subgroup_data[0]
    _, group1_name, group1_true, _, group1_probs, _ = subgroup_data[1]


    r_result = run_r_roc_test(group0_true, group0_probs, group1_true, group1_probs)
    print(r_result)

    p_value = parse_p_value_from_r_output(r_result)

    return p_value


def get_full_auc_ci(subgroup_data):
    
    _, group0_name, group0_true, _, group0_probs, _ = subgroup_data[0]
    _, group1_name, group1_true, _, group1_probs, _ = subgroup_data[1]

    y = np.concatenate([group0_true, group1_true])
    scores = np.concatenate([group0_probs, group1_probs])


    # Save data to temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        csv_path = f.name
        f.write("truth,score\n")
        for t, s in zip(y, scores):
            f.write(f"{int(t)},{s}\n")

    safe_csv_path = csv_path.replace("\\", "/")
    r_script = f"""
    library(pROC)
    data <- read.csv("{safe_csv_path}")
    roc_obj <- roc(data$truth, data$score, ci=TRUE)
    cat("Full Group AUC CI:\\n")
    print(ci(roc_obj))
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".R") as f:
        r_path = f.name
        f.write(r_script)

    try:
        result = subprocess.check_output(["Rscript", r_path], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        result = f"R error: {e.output}"
    finally:
        os.remove(csv_path)
        os.remove(r_path)

    match = re.search(r"95% CI:\s*([\d.]+)-([\d.]+)", result)
    if match:
        return tuple(map(float, match.groups()))
    
    return None

def parse_p_value_from_r_output(r_output):

    pval_match = re.search(r"p-value\s*=\s*([0-9.eE-]+)", r_output)
    p_value = float(pval_match.group(1)) if pval_match else None

    ci1_match = re.search(r"ROC1 CI:\s*95% CI:\s*([\d\.]+)-([\d\.]+)", r_output)
    ci2_match = re.search(r"ROC2 CI:\s*95% CI:\s*([\d\.]+)-([\d\.]+)", r_output)

    ci1 = tuple(map(float, ci1_match.groups())) if ci1_match else None
    ci2 = tuple(map(float, ci2_match.groups())) if ci2_match else None

    return {
        "p_value": p_value,
        "ci_group0": ci1,
        "ci_group1": ci2,
    }

def parse_whitney_p_value_from_r_output(r_output):
    # Finds Mann-Whitney section and parse the first p-value after that
    mw_section = re.split(r"Mann-Whitney U Test Result:", r_output)
    if len(mw_section) < 2:
        return None
    
    pval_match = re.search(r"p-value\s*=\s*([0-9.eE-]+)", mw_section[1])
    p_value = float(pval_match.group(1)) if pval_match else None

    ci_match = re.search(r"confidence interval:\s*\n\s*\[([\d\.\-eE]+),\s*([\d\.\-eE]+)\]", mw_section[1])
    ci = tuple(map(float, ci_match.groups())) if ci_match else None

    return {
        "p_value": p_value,
        "confidence_interval": ci,
    }


def run_r_roc_test(y1, scores1, y2, scores2):
    # Save data temporarily
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        csv_path = f.name
        f.write("group,truth,score\n")
        for t, s in zip(y1, scores1):
            f.write(f"0,{int(t)},{s}\n")
        for t, s in zip(y2, scores2):
            f.write(f"1,{int(t)},{s}\n")

    # Create R script
    safe_csv_path = csv_path.replace("\\", "/")
    r_script = f"""
    library(pROC)
    data <- read.csv("{safe_csv_path}")
    data0 <- subset(data, group == 0)
    data1 <- subset(data, group == 1)

    roc1 <- roc(data0$truth, data0$score)
    roc2 <- roc(data1$truth, data1$score)

    test_result <- roc.test(roc1, roc2, paired=FALSE, method="delong")
    cat("ROC Test Result:\\n")
    print(test_result)
    cat("\nROC1 CI:\n")
    print(ci(roc1))
    cat("\nROC2 CI:\n")
    print(ci(roc2))

    # Mann-Whitney U test
    mann_whitney_result <- wilcox.test(data0$score, data1$score)

    cat("\\nMann-Whitney U Test Result:\\n")
    print(mann_whitney_result)
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".R") as f:
        r_path = f.name
        f.write(r_script)

    try:
        result = subprocess.check_output(["Rscript", r_path], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        result = f"R error: {e.output}"
    finally:
        os.remove(csv_path)
        os.remove(r_path)

    return result

