import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))

from configs import BRSET_DIR1, BRSET_DIR2, MODEL_CONFIG_BRSET, ATTRIBUTES_TO_ANALYZE_BRSET, SUBGROUP_METRICS, FAIRNESS_METRICS, EVAL_METRICS
from classes import EvaluationMetrics, SubgroupMetrics
from preprocess_features import preprocess_brset 


class_number=2 
model_name = 'convnext' 
# 'DT', 'XGBoost', 'LR'
# 'DT_less', 'XGBoost_less', 'LR_less'
# 'convnext', 'dinov', 'swin'

# Read data:
if os.path.exists(BRSET_DIR2):
    mbrset_dir = BRSET_DIR2
elif os.path.exists(BRSET_DIR1):
    mbrset_dir = BRSET_DIR1
else:
    raise FileNotFoundError("BRSET directory not found.")

model_config = MODEL_CONFIG_BRSET.get(model_name)
if not model_config:
        raise ValueError(f"Model '{model_name}' is not supported.")
folder_name = model_config["folder_name"]
filename = model_config["filename"]

folder_dir = os.path.join(mbrset_dir, f"Results/{folder_name}/{model_name}")
csv_file_path = os.path.join(folder_dir, filename)
df = pd.read_csv(csv_file_path)


# Prepare excel of the obtained results: 
subgroups=[]
for attribute in ATTRIBUTES_TO_ANALYZE_BRSET:
    for i in range(2): # Each attribute is divided into group0 and group1
        subgroups.append(f'{attribute} group{i}')

df_subgroups = pd.DataFrame(index=subgroups, columns=SUBGROUP_METRICS)
df_fairness =pd.DataFrame(index=ATTRIBUTES_TO_ANALYZE_BRSET, columns=FAIRNESS_METRICS)
df_eval =pd.DataFrame(index=ATTRIBUTES_TO_ANALYZE_BRSET, columns=EVAL_METRICS)

# Directories to save results:
file_dir =  os.path.join(parent_dir, f'Fairness Results/{folder_name}/{model_name}')
image_dir = file_dir
file_path = os.path.join(file_dir, f'BRSET_fairness_results_{model_name}.xlsx')

         

if folder_name=='BRSET_images':
    # 0=No retinopathy, 1=Mild non-proliferative, 2=Moderate non-proliferative, 3=Severe non-proliferative, 4=Proliferative diabetic retinopathyand post-laser status
    if class_number==2: # Normal = 0,1; DR = 1, 2, 3, 4;
        
        df['predicted_icdr'] = df['prob_predicted_icdr'].apply(lambda x: 1 if x > 0.5 else 0)

    else: # Normal = 0,1; Non-proliferative = 2, 3; Proliferative = 4
        df['final_icdr']= df['final_icdr'].apply(lambda x: 'Normal' if x in [0,1]  else ('Non-proliferative' if x in [2,3] else 'Proliferative'))
        df['final_icdr']= df['final_icdr'].apply(lambda x: 0 if x == 'Normal' else (1 if x == 'Non-proliferative' else 2)  ).astype(int)

        df['final_icdr']= df['final_icdr'].apply(lambda x: 'Normal' if x in [0,1]  else 'Diabetic Retinopathy')
        df['final_icdr']= df['final_icdr'].apply(lambda x: 0 if x == 'Normal' else 1 ).astype(int)



#  Evaluate:
df_ = df.dropna(subset=ATTRIBUTES_TO_ANALYZE_BRSET)
# y_true = df_['DR_ICDR']
# y_pred = df_['predicted_icdr']
# y_pred_probs = df_['prob_predicted_icdr']

# Evaluate the model as a whole (without subgroups):
Evaluation = EvaluationMetrics(df=df_, df_eval=df_eval, model_name=model_name, dataset='BRSET', file_dir=file_dir)
Evaluation.calculate_and_update_eval_metrics()


# Evaluate the model by subgroups: 
for subgroup_type in ATTRIBUTES_TO_ANALYZE_BRSET:
    df_ = df.dropna(subset=subgroup_type)
    df_, _ = preprocess_brset(df_, subset=subgroup_type, class_number=class_number, mode='bin')
    
    Subgroup = SubgroupMetrics(df=df_, df_subgroups=df_subgroups, df_fairness=df_fairness, model_name=model_name, dataset='BRSET', file_dir=file_dir)
    Subgroup.calculate_and_update_subgroup_metrics(subgroup_type)



with pd.ExcelWriter(file_path) as writer:
    df_subgroups.to_excel(writer, sheet_name='Subgroups', index=True)
    df_fairness.to_excel(writer, sheet_name='Fairness', index=True)
    df_eval.to_excel(writer, sheet_name='Evaluation', index=True)

print(f"File saved at: {file_path}")
