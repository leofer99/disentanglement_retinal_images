# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# TorchMetrics Imports
from torchmetrics.functional.classification import (
    accuracy,
    f1_score,
    recall,
    precision,
    auroc
)
from torchmetrics.functional.regression import (
    mean_squared_error,
    pearson_corrcoef,

)

# pingouin-stats
import pingouin as pg

# W&B Imports
import wandb




# Function: Perform bootstrap analysis
def bootstrap_analysis(y_true, y_pred, metric_value, task='binary', metric_name='accuracy', bins=1000, confidence=0.95):
    """
    Author: Dagoberto Pulido-Arias
    Perform bootstrap analysis to calculate confidence intervals for a given metric.

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    metric_fn : callable
        Function to calculate the metric
    metric_name : str, optional
        Name of the metric for progress display, by default ""
    bins : int, optional
        Number of bootstrap samples, by default 1000
    confidence : float, optional
        Confidence level for the interval, by default 0.95

    Returns
    -------
    dict
        Dictionary containing the metric value, lower bound, and upper bound
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    results = []

    # Use tqdm to show progress
    desc = f"Bootstrap analysis for {metric_name}" if metric_name else "Bootstrap analysis"
    for _ in tqdm(range(bins), desc=desc):
        # Generate bootstrap sample indices
        indices = np.random.choice(n, n, replace=True)

        # Calculate metric on bootstrap sample
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]

        if metric_name == "accuracy":
            result = accuracy(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "f1_score":
            result = f1_score(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "precision":
            result = precision(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "recall":
            result = recall(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "auroc":
            bootstrap_true = torch.tensor(bootstrap_true)
            bootstrap_pred = torch.tensor(bootstrap_pred)
            result = auroc(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "mean_squared_error":
            result = mean_squared_error(preds=bootstrap_pred, target=bootstrap_true)
            result = result.item()
        elif metric_name == "pearson_corrcoef":
            result = pearson_corrcoef(preds=bootstrap_pred, target=bootstrap_true)
            result = result.item()
        elif metric_name == "intraclass_corr":
            icc_samples = list()
            icc_judges = list()
            icc_scores = list()

            for i, s in enumerate(list(bootstrap_pred.numpy())):
                icc_samples.append(i)
                icc_judges.append('A')
                icc_scores.append(s)
            
            for j, c in enumerate(list(bootstrap_true.numpy())):
                icc_samples.append(j)
                icc_judges.append('B')
                icc_scores.append(c)
            
            icc_data = {
                'icc_samples':icc_samples,
                'icc_judges':icc_judges,
                'icc_scores':icc_scores
            }
            icc_data_df = pd.DataFrame.from_dict(icc_data)
            icc = pg.intraclass_corr(data=icc_data_df, targets='icc_samples', raters='icc_judges', ratings='icc_scores').round(3)
            result = icc.values[2,2]

        results.append(result)

    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(results, alpha * 100)
    upper_bound = np.percentile(results, (1 - alpha) * 100)

    return {
        'value': metric_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'results':results
    }