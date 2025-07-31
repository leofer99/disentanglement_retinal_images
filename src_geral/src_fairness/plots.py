import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve


# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))


from src_fairness.utils import save_image


def plot_roc_curve(y_true, y_pred_probs, mode='eval', label=None, file_dir=None, subgroup_data=None):
    
    plt.figure(figsize=(8, 6))

    if mode == 'eval':
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

    elif mode == 'subgroup' and subgroup_data:
        # Multiple subgroup ROC curves
        for name_sf, group_name, group_y_true, _, group_y_pred_probs, colour in subgroup_data:
            fpr, tpr, _ = roc_curve(group_y_true, group_y_pred_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name_sf} {group_name} (AUC = {roc_auc:.2f})", color=colour)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    #Save image
    if mode == 'eval':
        save_image(plt, img_name=f'ROC_{mode}', file_dir=file_dir)
    elif mode == 'subgroup':
        save_image(plt, img_name=f'ROC_{mode}_{name_sf}', file_dir=file_dir)

    plt.close()


def plot_calibration_curve(y_true, y_pred_probs, mode='eval', label=None, file_dir=None, subgroup_data=None, n_bins=20):

    plt.figure(figsize=(8, 6))

    if mode == 'eval':
        prob_true, prob_pred_bins = calibration_curve(y_true, y_pred_probs, n_bins=n_bins)
        plt.plot(prob_pred_bins, prob_true, marker="o", label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    

    elif mode == 'subgroup' and subgroup_data:
    # Multiple subgroup ROC curves
        for name_sf, group_name, group_y_true, _, group_y_pred_probs, colour in subgroup_data:
            prob_true, prob_pred_bins = calibration_curve(group_y_true, group_y_pred_probs, n_bins=n_bins)
            plt.plot(prob_pred_bins, prob_true, marker="o", color=colour, label=f"{name_sf} {group_name}")

            # plt.plot(fpr, tpr, label=f"{name_sf} {group_name} (AUC = {roc_auc:.2f})", color=colour)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)

    #Save image
    if mode == 'eval':
        save_image(plt, img_name=f'Calibration plot_{mode}', file_dir=file_dir)
    elif mode == 'subgroup':
        save_image(plt, img_name=f'Calibration plot_{mode}_{name_sf}', file_dir=file_dir)

    plt.close()


def risk_distribution_line_plot(y_true, y_pred_probs, mode='eval', label=None, file_dir=None, subgroup_data=None, n_bins=20):
    """
    Plots a risk distribution density line of predicted probabilities,
    separated by true label.
    """
    plt.figure(figsize=(10, 6))

    if mode == 'eval':
        sns.kdeplot(y_pred_probs[y_true == 0], color="blue", label="No DR (0)", fill=False, alpha=0.6)
        sns.kdeplot(y_pred_probs[y_true == 1], color="red", label="DR (1)", fill=False, alpha=0.6)

    elif mode == 'subgroup' and subgroup_data:
        for name_sf, group_name, group_y_true, _, group_y_pred_probs, colour in subgroup_data:
            sns.kdeplot(group_y_pred_probs[group_y_true == 0], color=colour, linestyle='-', label=f"{group_name} No DR (0)", fill=False, alpha=0.6)
            sns.kdeplot(group_y_pred_probs[group_y_true == 1], color=colour, linestyle='--', label=f"{group_name} DR (1)", fill=False, alpha=0.6)

    plt.xlabel("Predicted Risk (Probability)", fontsize=17)
    plt.ylabel("Density", fontsize=17)
    plt.xticks(fontsize=16)  
    plt.yticks(fontsize=16)
    # plt.legend()
    plt.legend(loc='upper right', fontsize=17, title_fontsize=16)

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if mode == 'eval':
        # plt.title(f"Risk Distribution Line Plot - {mode}")
        save_image(plt, img_name=f'Risk_distribution_line_plot_{mode}', file_dir=file_dir)
    elif mode == 'subgroup':
        # plt.title(f"Risk Distribution Line Plot - {mode} {name_sf}")
        save_image(plt, img_name=f'Risk_distribution_line_plot_{mode}_{name_sf}', file_dir=file_dir)
    
    # plt.show()
    plt.close()

def risk_distribution_histogram_plot(y_true, y_pred_probs, mode='eval', label=None, file_dir=None, subgroup_data=None, n_bins=20):
    """
    Plots a risk distribution histogram of predicted probabilities,
    separated by true label.
    """
    plt.figure(figsize=(10, 6))

    if mode == 'eval':
        sns.histplot(y_pred_probs[y_true == 0], bins=n_bins, color="blue", label="No Event (0)", kde=False, stat="density", alpha=0.6,
                     edgecolor='black', hatch="/")
        sns.histplot(y_pred_probs[y_true == 1], bins=n_bins, color="red", label="Event (1)", kde=False, stat="density", alpha=0.6,
                     edgecolor='black', hatch="\\")

    elif mode == 'subgroup' and subgroup_data:
        colour_map =  [['blue', 'lightsteelblue'], ['green', 'lightgreen']]

        for (name_sf, group_name, group_y_true, _, group_y_pred_probs, _), colour in zip(subgroup_data, colour_map):
            sns.histplot(group_y_pred_probs[group_y_true == 0], bins=n_bins, color=colour[0], label=f"{group_name} No Event (0)", kde=False, stat="density", alpha=0.6)
            sns.histplot(group_y_pred_probs[group_y_true == 1], bins=n_bins, color=colour[1], label=f"{group_name} Event (1)", kde=False, stat="density", alpha=0.6)

    plt.xlabel("Predicted Risk (Probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if mode == 'eval':
        plt.title(f"Risk Distribution Histogram - {mode}")
        save_image(plt, img_name=f'Risk_distribution_histogram_{mode}', file_dir=file_dir)
    elif mode == 'subgroup':
        plt.title(f"Risk Distribution Histogram - {mode} {name_sf}")
        save_image(plt, img_name=f'Risk_distribution_histogram_{mode}_{name_sf}', file_dir=file_dir)

    # plt.show()
    plt.close()



def calculate_net_benefit(thresholds, y_true=None, y_pred_probs=None):
        net_benefits = []

        for th in thresholds:
            tp = ((y_pred_probs >= th) & (y_true == 1)).sum()
            fp = ((y_pred_probs >= th) & (y_true == 0)).sum()
            n = len(y_true)
            th_odds = th / (1 - th)

            net_benefit = (tp / n) - (fp / n) * th_odds
            net_benefits.append(net_benefit)

        return np.array(net_benefits)

def decision_curve_analysis(y_true, y_pred_probs, mode='eval', subgroup_data=None, label=None, file_dir=None, plot=True):
        
        plt.figure(figsize=(9.5, 5.5))
        thresholds = np.linspace(0.01, 0.5, 100)
        # thresholds = np.linspace(0.01, 0.99, 100)

        if mode == 'eval':
            model_nb = calculate_net_benefit(thresholds, y_true, y_pred_probs)
            treat_all = (y_true.sum() / len(y_true)) - ((1 - y_true.mean()) * (thresholds / (1 - thresholds)))
            treat_none = np.zeros_like(thresholds)

            plt.plot(thresholds, model_nb, label=f"Model {label}")


        elif mode == 'subgroup' and subgroup_data:
            for name_sf, group_name, group_y_true, _, group_y_pred_probs, colour in subgroup_data:
                model_nb = calculate_net_benefit(thresholds, group_y_true, group_y_pred_probs)
                treat_all = (group_y_true.sum() / len(group_y_true)) - ((1 - group_y_true.mean()) * (thresholds / (1 - thresholds)))
                treat_none = np.zeros_like(thresholds)

                plt.plot(thresholds, model_nb, label=f"Model {name_sf} {group_name}", color=colour)


        if plot:
            plt.plot(thresholds, treat_all, label="Treat All", linestyle='--', color='chocolate')
            plt.plot(thresholds, treat_none, label="Treat None", linestyle='--', color='red')
            plt.xlabel("Threshold Probability", fontsize=17)
            plt.xticks(fontsize=16)  
            plt.yticks(fontsize=16)
            plt.ylabel("Net Benefit", fontsize=17)
            plt.ylim(-0.1, 0.2)
            plt.legend(loc='lower center', fontsize=17, title_fontsize=16)
            #'lower right' 'upper right'
            plt.grid(True)

             
            #Save image
            if mode == 'eval':
                # plt.title(f"Decision Curve Analysis - {mode}", fontsize=16)
                save_image(plt, img_name=f'Decision_Curve_Analysis_{mode}', file_dir=file_dir)
            elif mode == 'subgroup':
                # plt.title(f"Decision Curve Analysis - {mode} {name_sf}", fontsize=16)
                save_image(plt, img_name=f'Decision_Curve_Analysis_{mode}_{name_sf}', file_dir=file_dir)
            
            # plt.show()
            plt.close()


