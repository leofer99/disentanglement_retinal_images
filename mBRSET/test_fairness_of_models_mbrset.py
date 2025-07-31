import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, balanced_accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))

from src_fairness.classes import EvaluationMetrics, SubgroupMetrics
from preprocess_features import preprocess_mbrset 




attributes_to_analyse=[
                        'age', 'sex', 
                        'educational_level', 
                        'insurance', 
                        'obesity', 
                        # 'dm_time', 
                    #    'insulin', 
                    #    'insulin_time',
                    #    'oraltreatment_dm', 
                    #    'systemic_hypertension', 
                        # 'alcohol_consumption',
                        # 'smoking', 
                        # 'diabetic_foot', 'acute_myocardial_infarction',
                    #    'nephropathy', 'vascular_disease', 
                    #     'laterality', 'final_artifacts', 'final_quality'
                    #    'final_edema'
                       ]

class_number=2 
model_name = 'convnext_augTrue' #  'XGBoost', 'LR', 'convnext', 'dinov', 'swin'


less=False

mbrset_dir1= r"/nas-ctm01/homes/ljfernandes/Msc-Thesis---Leonor-Fernandes/0_mother_folder"
mbrset_dir2= r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\Código\0_mother_folder"

if os.path.exists(mbrset_dir2):
    mbrset_dir = mbrset_dir2



# Read the excel of the obtained results:
if model_name in ['DT', 'XGBoost', 'LR']:
    if less:
        folder_name='mBRSET_less_features'

        if model_name=='DT':
            filename= "mBRSET_less_DT_feature_model.csv"
        elif model_name=='XGBoost':
            filename= "mBRSET_less_XGBoost_feature_model.csv"
        elif model_name=='LR':
            filename= "mBRSET_less_LR_feature_model.csv"     

        if os.path.exists(mbrset_dir):
            folder_dir = os.path.join(mbrset_dir, f"Results/{folder_name}/{model_name}") #model_type, model)


    else:
        folder_name='mBRSET_features'

        if model_name=='DT':
            filename= "mBRSET_DT_feature_model.csv"
        elif model_name=='XGBoost':
            filename= "mBRSET_XGBoost_feature_model.csv"
        elif model_name=='LR':
            filename= "mBRSET_LR_feature_model.csv"

        if os.path.exists(mbrset_dir):
            folder_dir = os.path.join(mbrset_dir, f"Results/{folder_name}/{model_name}") #model_type, model)


else:  #convnext, dinov, swin
    folder_name='mBRSET_images'

    if model_name=='convnext':
        filename= "predicted_icdr_scores_20890.csv"

    elif model_name=='dinov':
        filename= "predicted_icdr_scores_20891.csv"
    elif model_name in ['convnext_augTrue', 'convnext_augFalse', 'dinov_augTrue', 'dinov_augFalse','swin_augTrue', 'swin_augFalse']:
        filename= "model_scores_9842.csv"

    elif model_name=='swin':
        filename= "predicted_icdr_scores_20892.csv"

    if os.path.exists(mbrset_dir):
        folder_dir = os.path.join(mbrset_dir, f"Fairness Results/{folder_name}/{model_name}")
         #folder_name, model_name)





csv_file_path= os.path.join(folder_dir, filename)
df = pd.read_csv(csv_file_path)


# Directories to save results:
file_dir =  os.path.join(parent_dir, f'Fairness Results/{folder_name}/{model_name}')
image_dir = file_dir
file_path = os.path.join(file_dir, f'mBRSET_fairness_results_{model_name}.xlsx')



if folder_name=='mBRSET_images':
    # 0=No retinopathy, 1=Mild non-proliferative, 2=Moderate non-proliferative, 3=Severe non-proliferative, 4=Proliferative diabetic retinopathyand post-laser status
    if class_number==2: # Normal = 0,1; DR = 1, 2, 3, 4;
        
        df['predicted_icdr'] = df['prob_predicted_icdr'].apply(lambda x: 1 if x >= 0.5 else 0)

    else: # Normal = 0,1; Non-proliferative = 2, 3; Proliferative = 4
        df['final_icdr']= df['final_icdr'].apply(lambda x: 'Normal' if x in [0,1]  else ('Non-proliferative' if x in [2,3] else 'Proliferative'))
        df['final_icdr']= df['final_icdr'].apply(lambda x: 0 if x == 'Normal' else (1 if x == 'Non-proliferative' else 2)  ).astype(int)

        df['final_icdr']= df['final_icdr'].apply(lambda x: 'Normal' if x in [0,1]  else 'Diabetic Retinopathy')
        df['final_icdr']= df['final_icdr'].apply(lambda x: 0 if x == 'Normal' else 1 ).astype(int)



# Prepare df with results: 
subgroups=[]
for attribute in attributes_to_analyse:
    for i in range(2): # Each attribute is divided into group0 and group1
        subgroups.append(f'{attribute} group{i}')

subgroup_metrics= ['N','sample_size','prevalence', 'model_prevalence/PPR', 'TPR/Recall', 'FPR', 'NPV', 'AUROC', 'Balanced_accuracy', 'f1_score', 'Precision/PPV', 'PR-AUC', 'Clinical benefit', ]
df_subgroups = pd.DataFrame(index=subgroups, columns=subgroup_metrics)

fairness_metrics=['demographic_parity_diff', 'cond_demographic_parity_diff', 'predictive_parity', 'eq_odds', 'eq_opportunity', 'AUC_diff', 'ES-AUC']
df_fairness =pd.DataFrame(index=attributes_to_analyse, columns=fairness_metrics)

eval_metrics=['AUROC', 'Balanced_accuracy', 'f1_score', 'Precision', 'Recall', 'Precision-Recall']
df_eval =pd.DataFrame(index=attributes_to_analyse, columns=eval_metrics)


# 1. Evaluate the model as a whole:
# df_ = df.dropna(subset=attributes_to_analyse)
# y_true = df['final_icdr']
# y_pred = df['predicted_icdr']
# y_pred_probs = df['prob_predicted_icdr']

# bal_acc = balanced_accuracy_score(y_true, y_pred)
# print(f"Balanced Accuracy: {bal_acc:.4f}")


# Obtains results for the whole dataset (without subgroups))
# Evaluation = EvaluationMetrics(df_eval, model_name, dataset='mBRSET', file_dir=file_dir, y_true=y_true, y_pred=y_pred, y_pred_probs=y_pred_probs)

Evaluation = EvaluationMetrics(df=df, df_eval=df_eval, model_name=model_name, dataset='mBRSET', file_dir=file_dir)
Evaluation.calculate_and_update_eval_metrics()


# 2. Evaluate the model by subgroups:
for subgroup_type in attributes_to_analyse:
    df_ = df.dropna(subset=subgroup_type)
    df_= preprocess_mbrset(df_, subset=subgroup_type, class_number=class_number, mode='bin')
    
    y_true = df_['final_icdr']
    y_pred = df_['predicted_icdr']
    y_pred_probs = df_['prob_predicted_icdr']

    Subgroup = SubgroupMetrics(df=df_, df_subgroups=df_subgroups, df_fairness=df_fairness, model_name=model_name, dataset='mBRSET', file_dir=file_dir)
    Subgroup.calculate_and_update_subgroup_metrics(subgroup_type)


with pd.ExcelWriter(file_path) as writer:
    df_subgroups.to_excel(writer, sheet_name='Subgroups', index=True)
    df_fairness.to_excel(writer, sheet_name='Fairness', index=True)
    df_eval.to_excel(writer, sheet_name='Evaluation', index=True)

print(f"File saved at: {file_path}")
