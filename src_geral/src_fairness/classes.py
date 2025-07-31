import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from baycomp import two_on_single

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))


from src_fairness.metrics import calculate_auroc, observed_expected_ratio, expected_calibration_error
from src_fairness.plots import plot_roc_curve, plot_calibration_curve, risk_distribution_line_plot, risk_distribution_histogram_plot, decision_curve_analysis
from src_fairness.statistical_tests import unpaired_delong_test, unpaired_whitney_test, get_full_auc_ci


# from utils import update_df_eval, update_df_fairness, update_df_subgroups
from src_fairness.configs import COLUMNS_BRSET, COLUMNS_MBRSET, COLUMNS_MBRSET_DISEN, COLUMNS_BRSET_DISEN

class EvaluationMetrics:

    def __init__(self, df, df_eval, model_name, dataset, file_dir):
        self.df = df
        self.df_eval = df_eval
        self.model_name = model_name
        self.dataset = dataset
        self.file_dir=file_dir

        if dataset == 'BRSET':
            self.y_true = df[COLUMNS_BRSET['y_true']]
            self.y_pred = df[COLUMNS_BRSET['y_pred']]
            self.y_pred_probs = df[COLUMNS_BRSET['y_pred_probs']]

        elif dataset == 'mBRSET':
            self.y_true = df[COLUMNS_MBRSET['y_true']]
            self.y_pred = df[COLUMNS_MBRSET['y_pred']]
            self.y_pred_probs = df[COLUMNS_MBRSET['y_pred_probs']]

        elif dataset == 'mBRSET_disen':
            self.y_true = df[COLUMNS_MBRSET_DISEN['y_true']]
            self.y_pred = df[COLUMNS_MBRSET_DISEN['y_pred']]
            self.y_pred_probs = df[COLUMNS_MBRSET_DISEN['y_pred_probs']]

        elif dataset == 'BRSET_disen':
            self.y_true = df[COLUMNS_BRSET_DISEN['y_true']]
            self.y_pred = df[COLUMNS_BRSET_DISEN['y_pred']]
            self.y_pred_probs = df[COLUMNS_BRSET_DISEN['y_pred_probs']]



    def update_df_eval(self, metric_name, value):
        """
        Helper function to update the df_subgroups with the calculated value.
        """
        self.df_eval.loc[metric_name] = value


    def calculate_and_update_eval_metrics(self):
        """
        Calculate and update all evaluation metrics to df_eval.
        """
        
        ## 1. Discrimination:
        # AUROC (recommended)
        roc_auc_score = calculate_auroc(y_true=self.y_true, y_pred_probs=self.y_pred_probs)*100
        self.update_df_eval(metric_name='AUROC', value=f"{roc_auc_score:.2f}")
        
        # # ROC curve (not essential)
        plot_roc_curve(y_true=self.y_true, y_pred_probs=self.y_pred_probs, mode='eval', label=self.model_name, file_dir=self.file_dir)

        # ## 2. Calibration:
        # # Calibration plot (recommended)
        plot_calibration_curve(y_true=self.y_true, y_pred_probs=self.y_pred_probs, mode='eval', label=self.model_name, file_dir=self.file_dir, n_bins=20)

        # ## 3. Overall:
        # # 3.1 Risk distribution plots (recommended)
        
        risk_distribution_line_plot(y_true=self.y_true, y_pred_probs=self.y_pred_probs, mode='eval', label=self.model_name, file_dir=self.file_dir, n_bins=20)
        
        risk_distribution_histogram_plot(y_true=self.y_true, y_pred_probs=self.y_pred_probs, mode='eval', label=self.model_name, file_dir=self.file_dir, n_bins=20)



        # # 5. Clinical Utility:
        # # 5.1 NB or standardized NB
        decision_curve_analysis(y_true=self.y_true, y_pred_probs=self.y_pred_probs, mode='eval', label=self.model_name, file_dir=self.file_dir, plot=True) # add confidence interval


class SubgroupMetrics:
    def __init__(self, df, df_subgroups, df_fairness, model_name, dataset, file_dir):
        self.df = df
        self.df_subgroups = df_subgroups
        self.df_fairness = df_fairness
        self.model_name = model_name
        self.dataset = dataset
        self.file_dir=file_dir

        if dataset == 'BRSET':
            self.y_true = df[COLUMNS_BRSET['y_true']]
            self.y_pred = df[COLUMNS_BRSET['y_pred']]
            self.y_pred_probs = df[COLUMNS_BRSET['y_pred_probs']]

        elif dataset == 'mBRSET':
            self.y_true = df[COLUMNS_MBRSET['y_true']]
            self.y_pred = df[COLUMNS_MBRSET['y_pred']]
            self.y_pred_probs = df[COLUMNS_MBRSET['y_pred_probs']]

        elif dataset == 'mBRSET_disen':
            self.y_true = df[COLUMNS_MBRSET_DISEN['y_true']]
            self.y_pred = df[COLUMNS_MBRSET_DISEN['y_pred']]
            self.y_pred_probs = df[COLUMNS_MBRSET_DISEN['y_pred_probs']]

        elif dataset == 'BRSET_disen':
            self.y_true = df[COLUMNS_BRSET_DISEN['y_true']]
            self.y_pred = df[COLUMNS_BRSET_DISEN['y_pred']]
            self.y_pred_probs = df[COLUMNS_BRSET_DISEN['y_pred_probs']]


    def update_df_subgroups(self, subgroup_name, metric_name, value):
        """
        Helper function to update the df_subgroups with the calculated value.
        """
        self.df_subgroups.loc[subgroup_name, metric_name] = value

    def update_df_fairness(self, subgroup_name, metric_name, value):
        """
        Helper function to update the df_subgroups with the calculated value.
        """
        self.df_fairness.loc[subgroup_name, metric_name] = value


    def calculate_and_update_subgroup_metrics(self, name_sf):

        sf=self.df[name_sf]
        groups = np.unique(sf)
        subgroup_data = []
        

        for group, group_name, colour in zip(
            groups, ('group0', 'group1'), ('green', 'blue')):

            group_true = self.y_true[sf == group]
            group_pred = self.y_pred[sf == group]
            group_pred_probs = self.y_pred_probs[sf == group]

            # Prepare subgroup data
            subgroup_data.append((name_sf, group_name, group_true, group_pred, group_pred_probs, colour))


            # Group size:
            group_size = len(group_true)
            self.update_df_subgroups(f'{name_sf} {group_name}', 'N', group_size)

            # Group percentage:
            group_percentage = group_size / len(self.df) * 100
            self.update_df_subgroups(f'{name_sf} {group_name}', 'sample size', f"{group_percentage:.2f}")

            # Prevalence:
            group_positive_counts = np.sum(group_true)
            group_prevalence = group_positive_counts / group_size * 100
            self.update_df_subgroups(f'{name_sf} {group_name}', 'prevalence', f"{group_prevalence:.2f}")

            # Model Prevalence:
            group_positive_pred_counts = np.sum(group_pred)
            group_prevalence = group_positive_pred_counts / group_size * 100
            self.update_df_subgroups(f'{name_sf} {group_name}', 'model prevalence', f"{group_prevalence:.2f}")




            # 1. Discrimination:
            # 1.1 AUROC (recommended)
            roc_auc_score = calculate_auroc(y_true=group_true, y_pred_probs=group_pred_probs)*100
            self.update_df_subgroups(f'{name_sf} {group_name}', 'AUROC', f"{roc_auc_score:.2f}")


        # 1.2 ROC curve (not essential)
        plot_roc_curve(y_true=None, y_pred_probs=None, mode='subgroup', subgroup_data=subgroup_data, label=f'{name_sf}_{group_name}', file_dir=self.file_dir)


        ## 2. Calibration:
        # Calibration plot (recommended)
        plot_calibration_curve(y_true=None, y_pred_probs=None, mode='subgroup', subgroup_data=subgroup_data, label=f'{name_sf}_{group_name}', file_dir=self.file_dir, n_bins=20)


        ## 3. Overall:
        # 3.1 Risk distribution plots (recommended)
        risk_distribution_line_plot(y_true=None, y_pred_probs=None, mode='subgroup', subgroup_data=subgroup_data, label=f'{name_sf}_{group_name}', file_dir=self.file_dir, n_bins=20)
        
        risk_distribution_histogram_plot(y_true=None, y_pred_probs=None, mode='subgroup', subgroup_data=subgroup_data, label=f'{name_sf}_{group_name}', file_dir=self.file_dir, n_bins=20)


        # 5. Clinical Utility:
        # 5.1 NB or standardized NB
        decision_curve_analysis(y_true=None, y_pred_probs=None, mode='subgroup', subgroup_data=subgroup_data, label=f'{name_sf}_{group_name}', file_dir=self.file_dir, plot=True)
        
        # Statistical tests:
        # DeLong test:
        g_ci = get_full_auc_ci(subgroup_data)
        ci_str = f"[{g_ci[0]:.4f} - {g_ci[1]:.4f}]" if g_ci else "NA"

        self.update_df_subgroups(f'{name_sf}', 'CI', ci_str)



        delong_results = unpaired_delong_test(subgroup_data)
        p_value = delong_results["p_value"]
        g0_ci = delong_results["ci_group0"]
        g1_ci = delong_results["ci_group1"]

        # Save p-value:
        if delong_results["p_value"] is not None:
            self.update_df_subgroups(f'{name_sf} AUROC', 'p-value (DeLong)', f"{p_value:.4f}")

        #Save CIs:
        for group, group_name, g_ci in zip(
            groups, ('group0', 'group1'), (g0_ci, g1_ci) ):

            ci_str = f"[{g_ci[0]:.4f} - {g_ci[1]:.4f}]" if g_ci else "NA"
            self.update_df_subgroups(f'{name_sf} {group_name}', 'CI', ci_str)

        # Mann-Whitney U test:
        whit_results= unpaired_whitney_test(subgroup_data)
        p_value= whit_results["p_value"]
        g_ci = whit_results["confidence_interval"]
        ci_str = f"[{g_ci[0]:.4f} - {g_ci[1]:.4f}]" if g_ci else "NA"

        #0 g1_ci = whit_results["confidence_interval"]

        self.update_df_subgroups(f'{name_sf} AUROC', 'p-value (Whitney)', f"{p_value:.4f}")
        # self.update_df_subgroups(f'{name_sf} CI', 'CI (Whitney)', ci_str)



        # result = ttest_on_auroc(subgroup_data)
        # g0 = result['group0']
        # g1 = result['group1']


        # stat = result['stat_test']
        # auroc_diff = g1['mean_auroc'] - g0['mean_auroc']
        
        # g0_std = np.std(result['group0']['bootstrap_samples']) if 'bootstrap_samples' in g0 else None
        # g1_std = np.std(result['group1']['bootstrap_samples']) if 'bootstrap_samples' in g1 else None

        # g0_ci = f"[{g0['ci'][0]:.4f}, {g0['ci'][1]:.4f}]"
        # g1_ci = f"[{g1['ci'][0]:.4f}, {g1['ci'][1]:.4f}]"

        # for group, group_name, g, g_ci, g_std in zip(
        #     groups, ('group0', 'group1'), (g0, g1), (g0_ci, g1_ci), (g0_std, g1_std)):

        #     self.update_df_subgroups(f'{name_sf} {group_name}', 'Mean AUROC', f"{g['mean_auroc']:.3f}")
        #     self.update_df_subgroups(f'{name_sf} {group_name}', 'CI', g_ci)

        #     if g0_std is not None and g1_std is not None:
        #         self.update_df_subgroups(f'{name_sf} {group_name}', 'STD', f"{g_std:.4f}")

        # self.update_df_subgroups(f'{name_sf} AUROC', 'AUROC Difference', f"{auroc_diff:.4f}")
        # self.update_df_subgroups(f'{name_sf} AUROC', 'p-value (mannwhitney)', f"{stat['p_value']:.4f}")

        # prob_group0_better, prob_equal, prob_group1_better = two_on_single(
        #     np.array(g0['bootstrap_samples']), np.array(g1['bootstrap_samples']), rope=0.01  
        # )             
        # # ROPE = region of practical equivalence (1% AUROC diff)

        # print(f"Probability Group 0 > Group 1: {prob_group0_better:.4f}")
        # print(f"Probability Group 1 > Group 0: {prob_group1_better:.4f}")
        # print(f"Probability of negligible difference (ROPE ±0.01): {prob_equal:.3f}")

        # self.update_df_subgroups(f'{name_sf} AUROC', 'Prob G0 > G1', f"{prob_group0_better:.4f}")
        # self.update_df_subgroups(f'{name_sf} AUROC', 'Prob G0 < G1', f"{prob_group0_better:.4f}")
        # self.update_df_subgroups(f'{name_sf} AUROC', 'Prob neglig dif', f"{prob_group0_better:.4f}")


