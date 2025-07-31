import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import torch

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))

from src_fairness.configs import MBRSET_DIR1, MBRSET_DIR2, MODEL_CONFIG_DISEN, ATTRIBUTES_TO_ANALYZE_MBRSET, SUBGROUP_METRICS, FAIRNESS_METRICS, EVAL_METRICS
from src_fairness.classes import EvaluationMetrics, SubgroupMetrics
from preprocess_features import preprocess_mbrset 


class_number=2 
model_name = 'balaug_swin_disen_insurance'  #run next

#falta:
#model_name = 'balaug_swin_no_disen_educational' 
# model_name = 'balaug_swin_no_disen_insurance' 
# model_name = 'balaug_swin_no_disen_sex' 
# model_name = 'baseline_convnext_sex'
# model_name = 'baseline1_convnext_sex'
# model_name = 'dinov_disen_sex32'

base_model_name = model_name.rsplit('_', 1)[0]
name_sf = model_name.split('_')[-1] 

base_model_name = 'swin_disen'
name_sf='insurance'


# 'DT', 'XGBoost', 'LR'
# 'DT_less', 'XGBoost_less', 'LR_less'
# 'convnext_disen', 'dinov_disen', 'swin_disen'
# 'convnext_no_disen', 'dinov_no_disen', 'swin_no_disen'

# Read data:
if os.path.exists(MBRSET_DIR2):
    mbrset_dir = MBRSET_DIR2
elif os.path.exists(MBRSET_DIR1):
    mbrset_dir = MBRSET_DIR1
else:
    raise FileNotFoundError("mBRSET directory not found.")

model_config = MODEL_CONFIG_DISEN.get(base_model_name)
if not model_config:
        raise ValueError(f"Model '{base_model_name}' is not supported.")
folder_name = model_config["folder_name"]
filename = model_config["filename"]


# Directory to read results:
folder_dir = os.path.join(mbrset_dir, f"Disentanglement Results/{folder_name}/{model_name}")
os.makedirs(folder_dir, exist_ok=True)

csv_file_path = os.path.join(folder_dir, filename)
df = pd.read_csv(csv_file_path)




## Test Medical Prediction from Medical Vector: 
y_true = df['final_icdr']
y_pred = df['disen_predicted_icdr']
y_probs = df['disen_prob_predicted_icdr']




###########
# Prepare excel of the obtained results: 
subgroups=[]
for attribute in ATTRIBUTES_TO_ANALYZE_MBRSET:
    for i in range(2): # Each attribute is divided into group0 and group1
        subgroups.append(f'{attribute} group{i}')

df_subgroups = pd.DataFrame(index=subgroups, columns=SUBGROUP_METRICS)
df_fairness =pd.DataFrame(index=ATTRIBUTES_TO_ANALYZE_MBRSET, columns=FAIRNESS_METRICS)
df_eval =pd.DataFrame(index=ATTRIBUTES_TO_ANALYZE_MBRSET, columns=EVAL_METRICS)

# Directories to save results:
file_dir =  os.path.join(parent_dir, f'Disentanglement Results/{folder_name}/{model_name}')
image_dir = file_dir
file_path = os.path.join(file_dir, f'mBRSET_disen_fairness_{model_name}.xlsx')


#  Evaluate:
# df_ = df.dropna(subset=ATTRIBUTES_TO_ANALYZE_MBRSET)
# y_true = df_['final_icdr']
# y_pred = df_['predicted_icdr']
# y_pred_probs = df_['prob_predicted_icdr']

# icdr:
# y_pred = df_['disen_predicted_icdr']
#este está errado, corrigir!!
# df_['disen_prob_predicted_icdr'] = 1 / (1 + np.exp(-df_['disen_prob_predicted_icdr'].values))

# id:
# pred_id = df['disen_predicted_id']
# prob_pred_id = df['disen_prob_predicted_id'] 




# Evaluate the model as a whole (without subgroups):
Evaluation = EvaluationMetrics(df=df, df_eval=df_eval, model_name=model_name, dataset='mBRSET_disen', file_dir=file_dir)
Evaluation.calculate_and_update_eval_metrics()


# Evaluate the model by subgroups: 
for subgroup_type in ATTRIBUTES_TO_ANALYZE_MBRSET:
# for name_sf in [name_sf]:
    df_ = df.dropna(subset=subgroup_type)
    if subgroup_type != name_sf:
        df_ = preprocess_mbrset(df_, subset=subgroup_type, class_number=class_number, mode='bin')
    
    Subgroup = SubgroupMetrics(df=df_, df_subgroups=df_subgroups, df_fairness=df_fairness, model_name=model_name, dataset='mBRSET_disen', file_dir=file_dir)
    Subgroup.calculate_and_update_subgroup_metrics(subgroup_type)



with pd.ExcelWriter(file_path) as writer:
    df_subgroups.to_excel(writer, sheet_name='Subgroups', index=True)
    df_fairness.to_excel(writer, sheet_name='Fairness', index=True)
    df_eval.to_excel(writer, sheet_name='Evaluation', index=True)

print(f"File saved at: {file_path}")
