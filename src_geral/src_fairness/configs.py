# config.py

# File paths
MBRSET_DIR1 = r"/nas-ctm01/homes/ljfernandes/Msc-Thesis---Leonor-Fernandes/0_mother_folder"
MBRSET_DIR2 = r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\Código\0_mother_folder"
BRSET_DIR1 = r"/nas-ctm01/homes/ljfernandes/Msc-Thesis---Leonor-Fernandes/0_mother_folder"
BRSET_DIR2 = r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\Código\0_mother_folder"


# Model-specific configurations
MODEL_CONFIG = {
    "DT_less": {"folder_name": "mBRSET_less_features", "filename": "mBRSET_less_DT_feature_model.csv"},
    "XGBoost_less": {"folder_name": "mBRSET_less_features", "filename": "mBRSET_less_XGBoost_feature_model.csv"},
    "LR_less": {"folder_name": "mBRSET_less_features", "filename": "mBRSET_less_LR_feature_model.csv"},

    "DT": {"folder_name": "mBRSET_features", "filename": "mBRSET_DT_feature_model.csv"},
    "XGBoost": {"folder_name": "mBRSET_features", "filename": "mBRSET_XGBoost_feature_model.csv"},
    "LR": {"folder_name": "mBRSET_features", "filename": "mBRSET_LR_feature_model.csv"},
    
    "convnext": {"folder_name": "mBRSET_images", "filename": "predicted_icdr_scores_20890.csv"},
    "dinov": {"folder_name": "mBRSET_images", "filename": "predicted_icdr_scores_20891.csv"},
    "swin": {"folder_name": "mBRSET_images", "filename": "predicted_icdr_scores_20892.csv"},
}

MODEL_CONFIG_DISEN = {
    "convnext_disen": {"folder_name": "mBRSET", "filename": "df_test_disentangled.csv"},
    "dinov_disen": {"folder_name": "mBRSET", "filename": "df_test_disentangled.csv"},
    "swin_disen": {"folder_name": "mBRSET", "filename": "df_test_disentangled.csv"},
    "convnext_no_disen": {"folder_name": "mBRSET", "filename": "df_test_disentangled.csv"},
    "dinov_no_disen": {"folder_name": "mBRSET", "filename": "df_test_disentangled.csv"},
    "swin_no_disen": {"folder_name": "mBRSET", "filename": "df_test_disentangled.csv"},
}

MODEL_CONFIG_BRSET = {
    "DT": {"folder_name": "BRSET_features", "filename": "BRSET_DT_feature_model.csv"},
    "XGBoost": {"folder_name": "BRSET_features", "filename": "BRSET_XGBoost_feature_model.csv"},
    "LR": {"folder_name": "BRSET_features", "filename": "BRSET_LR_feature_model.csv"},
    
    "convnext": {"folder_name": "BRSET_images", "filename": "predicted_icdr_scores_20835.csv"},
    "dinov": {"folder_name": "BRSET_images", "filename": "predicted_icdr_scores_20836.csv"},
    "swin": {"folder_name": "BRSET_images", "filename": "predicted_icdr_scores_20837.csv"},

    "convnext_disen": {"folder_name": "BRSET_disen", "filename": ".csv"},
    "dinov_disen": {"folder_name": "BRSET_disen", "filename": "_.csv"},
    "swin_disen": {"folder_name": "BRSET_disen", "filename": "_.csv"},
}


# Attributes to analyze
ATTRIBUTES_TO_ANALYZE_MBRSET = [
    "age", "sex", "educational_level", "insurance", "obesity"
]
ATTRIBUTES_TO_ANALYZE_MBRSET_LESS = [
    "age", "sex"
]
ATTRIBUTES_TO_ANALYZE_BRSET = [
     "patient_age", "patient_sex" 
]


COLUMNS_MBRSET = {
    "y_true": "final_icdr",
    "y_pred": "predicted_icdr",
    "y_pred_probs": "prob_predicted_icdr",
}
COLUMNS_BRSET = {
    "y_true": "DR_ICDR",
    "y_pred": "predicted_icdr",
    "y_pred_probs": "prob_predicted_icdr",
}
COLUMNS_MBRSET_DISEN = {
    "y_true": "final_icdr",
    "y_pred": "disen_predicted_icdr",
    "y_pred_probs": "disen_prob_predicted_icdr",
}
COLUMNS_BRSET_DISEN = {
    "y_true": "DR_ICDR",
    "y_pred": "disen_predicted_icdr",
    "y_pred_probs": "disen_prob_predicted_icdr",
}



GROUP_NAMES = ["group0", "group1"]
SUBGROUP_METRICS = ['N','sample size','prevalence', 'model prevalence', 'TPR/Recall', 'FPR', 'NPV', 'AUROC', 'Balanced_accuracy', 'f1_score', 'Precision/PPV', 'PR-AUC', 'Clinical benefit', ]
FAIRNESS_METRICS = ['demographic_parity_diff', 'cond_demographic_parity_diff', 'predictive_parity', 'eq_odds', 'eq_opportunity', 'AUC_diff', 'ES-AUC']
EVAL_METRICS = ['AUROC', 'Balanced_accuracy', 'f1_score', 'Precision', 'Recall', 'Precision-Recall']



