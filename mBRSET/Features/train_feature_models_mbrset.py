import xgboost as xgb
# print(xgb.__version__)

import matplotlib.pyplot as plt
import os
import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from scipy.stats import uniform

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))

from preprocess_features import preprocess_mbrset
from fairness_metrics import FairnessMetrics
from src_image_luis.get_dataset import get_dataset, split_data_mbrset_features, plot_labels_distribution


def save_image(plt, img_name, model_name='DT'):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    image_dir = os.path.join(parent_dir, f'Results/mBRSET_features/{model_name}')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, img_name))
    print(f"Image saved to '{image_dir}'")

# Save Results to .csv file:
def save_results(df_test, y_pred, y_pred_prob, model_name):

    df_test['predicted_icdr'] = y_pred
    df_test['prob_predicted_icdr'] = y_pred_prob
    name=f'mBRSET_{model_name}_feature_model.csv'

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(parent_dir, f'Results/mBRSET_features/{model_name}')
    os.makedirs(results_dir, exist_ok=True)

    df_test.to_csv(os.path.join(results_dir, name), index=False)
    print(f"Predictions saved to '{name}.csv'")

def plot_ROC(y_test, y_pred, model_name, save=False, type=None):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')  # Random classifier (diagonal line)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve - {model_name} ({type})')
    plt.legend(loc='lower right')
    plt.grid(True)

    if save:
        save_image(plt, img_name=f'{model_name}_ROC_{type}', model_name=model_name)
    
    plt.show()
    plt.close()

def plot_cm(y_test, y_pred, model_name, save=False, type=None):
    cm = confusion_matrix(y_test, y_pred)
    # print("Decision Tree - Confusion Matrix (Train):", cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name} ({type}) - mBRSET')

    if save:
        save_image(plt, model_name=model_name, img_name=f'{model_name}_cm_{type}')
    
    # plt.show()
    plt.close()



def apply_tree_classifier(X_train, y_train, X_test, y_test, X_val, y_val, save=False):
    
    tree_model = DecisionTreeClassifier(
        ccp_alpha=0.001,
        class_weight=None, 
        criterion='gini',
        max_depth=20,
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=4,
        min_samples_split=9,
        splitter='best',

        random_state=50
)

# Best Parameters: {'ccp_alpha': 0.001, 'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 4, 'min_samples_split': 9, 'splitter': 'best'}

#     #Grid search
#     param_grid = {
#        'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
#         'max_depth': [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, None],  # Maximum depth of the tree
#         'min_samples_split': [2, 3, 5, 7, 9, 10, 15, 20],  # Minimum samples required to split an internal node
#         'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],  # Minimum samples required to be at a leaf node
#         'max_features': [None, 'sqrt', 'log2', 0.5, 0.75],
#         'splitter': ['best', 'random'],  # Strategy to split at each node
#         'class_weight': [None, 'balanced'], 
#         'max_leaf_nodes': [None, 10, 50, 100],  # Maximum number of leaf nodes in the tree
#         'min_impurity_decrease': [0.0, 0.01, 0.1],  # Minimum impurity decrease required to split a node
#         'ccp_alpha': [0.0, 0.001, 0.01]  # Complexity parameter used for pruning
# }
    
#     grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     print(f"Best Parameters: {best_params}")
#     best_model = grid_search.best_estimator_
#     tree_model = best_model

    tree_model.fit(X_train, y_train)


    # Train:
    y_pred = tree_model.predict(X_train)
    y_pred_prob = tree_model.predict_proba(X_train)[:, 1] #probabil for class 1
    plot_cm(y_train, y_pred, model_name='DT', type='Train', save=save)
    plot_ROC(y_train, y_pred_prob, model_name='DT', type='Train', save=save)


    # Test:
    y_pred = tree_model.predict(X_test)
    y_pred_prob = tree_model.predict_proba(X_test)[:, 1] #probabil for class 1
    plot_cm(y_test, y_pred, model_name='DT', type='Test', save=save)
    plot_ROC(y_test, y_pred_prob, model_name='DT', type='Test', save=save)

    
    # Show feature importance:
    feature_importance= tree_model.feature_importances_  
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # print(importance_df)

    # Plot feature importance:
    plt.figure(figsize=(10, 6))
    ax= sns.barplot(x='Feature', y='Importance', data=importance_df, hue=None, palette='viridis')
    plt.title('Feature Importance of Decision Tree - mBRSET')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(True)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}',  # Format the value with 4 decimal places
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=10, color='black', 
                    xytext=(0, 5), textcoords='offset points')

    if save:
        save_image(plt, model_name="DT", img_name='DT_feature_importance')

    plt.show()
    plt.close()


    # Plot the decision tree:
    # plt.figure(figsize=(20,10))
    # plot_tree(clf, filled=True, feature_names=X_test.columns, class_names=[str(i) for i in clf.classes_])
    # plt.show()

    if save:
        save_results(df_test, y_pred, y_pred_prob, model_name='DT')


    return tree_model



def apply_xg_boost(X_train, y_train, X_test, y_test, X_val, y_val, save=False):

    scale_pos_weight= sum(y_train==0) / sum(y_train==1) #weights for class imbalancement


# Best Parameters: {'alpha': 0.5, 'colsample_bytree': 1.0, 'gamma': 0.2, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 70, 'subsample': 0.7}

    xgb_1 = xgb.XGBClassifier(
        alpha=0.5,
        colsample_bytree=1.0, #max value is 1
        gamma=0.20,
        lambda_=1,
        learning_rate =0.05,
        max_depth=6, #between 3-10
        min_child_weight=1,
        n_estimators=70,
        subsample=0.70, #typically between 0.5-0.9

        objective= 'binary:logistic',
        nthread=4,
        # scale_pos_weight=scale_pos_weight,
        seed=27,
        early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
        )
    
    xgb_model= xgb_1


# Best Parameters: {'alpha': 0.5, 'colsample_bytree': 1.0, 'gamma': 0.2, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.7}

    # # Hyperparameter tuning
    # param_grid = {
    # 'learning_rate': [0.01, 0.05, 0.1],
    # 'n_estimators': [100, 200, 500],
    # 'max_depth': [3, 5, 6, 8],
    # 'min_child_weight': [1, 3, 5],
    # 'subsample': [0.7, 0.8, 1.0],
    # 'colsample_bytree': [0.5, 0.7, 1.0],
    # 'gamma': [0, 0.1, 0.2],
    # 'lambda': [0, 0.1, 1],
    # 'alpha': [0, 0.1, 0.5],
    # }

# Best Parameters: {'alpha': 0.5, 'colsample_bytree': 1.0, 'gamma': 0.2, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 70, 'subsample': 0.7}

    # # Hyperparameter tuning
    # param_grid = {
    # 'learning_rate': [0.03, 0.05, 0.07],
    # 'n_estimators': [70, 100, 130],
    # 'max_depth': [5, 6, 7],
    # 'min_child_weight': [1, 2, 3],
    # 'subsample': [0.65, 0.70, 0.75],
    # 'colsample_bytree': [0.8, 1.0], 
    # 'gamma': [0.15, 0.2, 0.25],
    # 'lambda': [0.9, 1, 1.1],
    # 'alpha': [0.4, 0.5, 0.6],
    # }
    

    # grid_search = GridSearchCV(estimator=xgb_1, param_grid=param_grid, 
    #                        scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
    # grid_search.fit(X_train, y_train,
    # eval_set=[(X_val, y_val)],  # Use validation data here
    # verbose=False)
    # best_model= grid_search.best_estimator_
    # print(f"Best Parameters: {grid_search.best_params_}")
    # xgb_model=best_model


    xgb_model.fit( X_train, y_train,
    eval_set=[(X_val, y_val)],  # Use validation data here
    verbose=False)


    # Train:
    y_pred = xgb_model.predict(X_train)
    y_pred_prob = xgb_model.predict_proba(X_train)[:, 1]
    plot_cm(y_train, y_pred, model_name='XGBoost', type='Train', save=save)
    plot_ROC(y_train, y_pred_prob, model_name='XGBoost', type='Train', save=save)


    # Validation:
    y_pred = xgb_model.predict(X_val)
    y_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
    plot_cm(y_val, y_pred, model_name='XGBoost', type='Validation', save=save)
    plot_ROC(y_val, y_pred_prob, model_name='XGBoost', type='Validation', save=save)


    # Test:
    y_pred = xgb_model.predict(X_test)
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    plot_cm(y_test, y_pred, model_name='XGBoost', type='Test', save=save)
    plot_ROC(y_test, y_pred_prob, model_name='XGBoost', type='Test', save=save)


    ax = xgb.plot_importance(xgb_model, height=0.5, importance_type='weight')
    fig = plt.gcf()
    fig.tight_layout()
    fig.subplots_adjust(left=0.32)  # Adjust left margin for long feature names
    save_image(plt, 'XGBoost_importance_plot.png', model_name='XGBoost')
    plt.close()

    if save:
        save_results(df_test, y_pred, y_pred_prob, model_name='XGBoost')

    return xgb_model


def apply_lr(X_train, y_train, X_test, y_test, X_val, y_val, class_number, save=False):

    # y_train=y_train.values.ravel()
    lr_model = LogisticRegression(
        C= 0.3913043478260869, 
        class_weight= None, 
        fit_intercept= True, 
        max_iter= 50, 
        penalty= 'l2', 
        solver= 'lbfgs', 
        tol= 0.001
        )
    
    # Best Parameters: {'C': 0.3913043478260869, 'class_weight': None, 'fit_intercept': True, 'max_iter': 50, 'penalty': 'l2', 'solver': 'lbfgs', 'tol': 0.001}
    # param_grid = [{
    #     'penalty':['l1','l2','elasticnet'],
    #     'C' : np.linspace(0.2, 0.6, 300), 
    #     'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    #     'max_iter': [50, 100, 200, 500, 1000],
    #     'tol': [1e-4, 1e-3, 1e-2],  # Tolerance values for more control over optimization precision
    #     'class_weight': ['balanced', None],  # Include both balanced and unbalanced class weights
    #     'fit_intercept': [True, False],
    # }]

    # grid_search = GridSearchCV(lr_model, param_grid = param_grid, cv = 5, verbose=True,n_jobs=-1)
    # grid_search.fit(X_train, y_train) 
    # best_params = grid_search.best_params_
    # print(f"Best Parameters: {best_params}")
    # clf_best = grid_search.best_estimator_
    # lr_model=clf_best



    lr_model.fit(X_train, y_train.squeeze())  #doesn't use validation data


    # Train:
    y_pred = lr_model.predict(X_train)
    y_pred_prob = lr_model.predict_proba(X_train)[:, 1]
    plot_cm(y_train, y_pred, model_name='LR', type='Train', save=save)
    plot_ROC(y_train, y_pred_prob, model_name='LR', type='Train', save=save)


    # Validation:
    y_pred = lr_model.predict(X_val)
    y_pred_prob = lr_model.predict_proba(X_val)[:, 1]
    plot_cm(y_val, y_pred, model_name='LR', type='Validation', save=save)
    plot_ROC(y_val, y_pred_prob, model_name='LR', type='Validation', save=save)


    # Test:
    y_pred = lr_model.predict(X_test)
    y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
    plot_cm(y_test, y_pred, model_name='LR', type='Test', save=save)
    plot_ROC(y_test, y_pred_prob, model_name='LR', type='Test', save=save)


    # Show feature importance:
    if class_number==2:
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': lr_model.coef_.flatten()  # Flatten to 1D array
        })
        # Sort by absolute importance (higher magnitude means higher importance)
        feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
        feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

        # Display results
        print(tabulate(feature_importance.drop(columns=['Abs_Coefficient']), headers='keys', tablefmt='pretty'))

    else: 
        feature_importance = pd.DataFrame(
            lr_model.coef_,  # Each row corresponds to one class
            columns=X_train.columns,  # Feature names
            index=[f"Class {i}" for i in range(lr_model.coef_.shape[0])] 
         ) # Class labels
        # print(feature_importance.drop(columns=['Abs_Coefficient']))


    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='Feature',
        y='Coefficient',
        data=feature_importance,
        palette='coolwarm'
    )
    plt.title('Feature Importance (Logistic Regression - Binary Classification)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()

    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=9, color='black',
                    xytext=(0, 5), textcoords='offset points')

    save_image(plt, 'LR_importance_plot.png', model_name='LR')
    plt.show()
    plt.close()


    if save:
        save_results(df_test, y_pred, y_pred_prob, model_name='LR')




    return lr_model



# 1. Load dataset
mbrset_dir1= r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\mBRSET"
mbrset_dir2= r"/nas-ctm01/datasets/public/MEDICAL/mbrset/physionet.org/files/mbrset/1.0"

if os.path.exists(mbrset_dir2):
    mbrset_dir = mbrset_dir2
else:
    mbrset_dir = mbrset_dir1

image_folder_path= os.path.join(mbrset_dir, "images")
csv_file_path= os.path.join(mbrset_dir, "labels_mbrset.csv")

patients_info = pd.read_csv(csv_file_path)
image_filenames = set(os.listdir(image_folder_path))
patients_info = patients_info[patients_info['file'] != '985.1.jpg']

#selects only patients that have a corresponding image
filtered_patients_info = patients_info[patients_info['file'].isin(image_filenames)]
# patients_info['final_artifacts'].isna().sum() #had 0 NA values


attributes_to_analyse=['age', 'sex', 'dm_time', 
                       'insulin', 
                    #    'insulin_time',
                       'oraltreatment_dm', 
                       'systemic_hypertension', 'insurance', 'educational_level', 'alcohol_consumption',
                       'smoking', 'obesity', 'acute_myocardial_infarction', 'diabetic_foot', 
                    #    'final_icdr',
                    #    'nephropathy', 'vascular_disease', 
                        # 'laterality', 'final_artifacts', 'final_quality'
                    #    'final_edema'
                       ]

class_number=2

# 2.0 # Diabetic retinopathy:
# 0=No retinopathy, 1=Mild non-proliferative, 2=Moderate non-proliferative, 3=Severe non-proliferative, 4=Proliferative diabetic retinopathyand post-laser status
if class_number==2:
# Normal = 0,1; DR = 1, 2, 3, 4;
    filtered_patients_info['final_icdr']= filtered_patients_info['final_icdr'].apply(lambda x: 'Normal' if x in [0,1]  else 'Diabetic Retinopathy')
    filtered_patients_info['final_icdr']= filtered_patients_info['final_icdr'].apply(lambda x: 0 if x == 'Normal' else 1 ).astype(int)

else: 
# Normal = 0,1; Non-proliferative = 2, 3; Proliferative = 4
    filtered_patients_info['final_icdr']= filtered_patients_info['final_icdr'].apply(lambda x: 'Normal' if x in [0,1]  else ('Non-proliferative' if x in [2,3] else 'Proliferative'))
    filtered_patients_info['final_icdr']= filtered_patients_info['final_icdr'].apply(lambda x: 0 if x == 'Normal' else (1 if x == 'Non-proliferative' else 2)  ).astype(int)


# 3. Split data into training and test
df = pd.DataFrame(filtered_patients_info)
# print("1- NaNs in df before cut!!:", pd.isna(df[attributes_to_analyse]).sum().sum())
df = df.dropna(subset=attributes_to_analyse)
# print("NaNs in df after cut!!:", pd.isna(df[attributes_to_analyse]).sum().sum())

df_train, df_test = split_data_mbrset_features(df, column="final_icdr", split=0.2, undersample=False)
df_train, df_val = split_data_mbrset_features(df_train, column="final_icdr", split=1/8, undersample=False)

df_train_copy= df_train.copy()
df_test_copy= df_test.copy()
df_val_copy= df_val.copy()
df_train_ = preprocess_mbrset(df=df_train_copy, subset=attributes_to_analyse, class_number=class_number)
df_test_ = preprocess_mbrset(df=df_test_copy, subset=attributes_to_analyse, class_number=class_number)
df_val_ = preprocess_mbrset(df=df_val_copy, subset=attributes_to_analyse, class_number=class_number)

X_train=df_train_[attributes_to_analyse]
X_test=df_test_[attributes_to_analyse]
X_val=df_val_[attributes_to_analyse]

y_train=df_train_[["final_icdr"]].to_numpy().flatten()
y_test=df_test_[["final_icdr"]].to_numpy().flatten()
y_val=df_val_[["final_icdr"]].to_numpy().flatten()




model_tree= None
model_xgb= None
model_lr= None

# Tree Classifier:
model_tree= apply_tree_classifier(X_train, y_train, X_test, y_test, X_val, y_val, save=True)

# XGBoost Classifier:
model_xgb= apply_xg_boost(X_train, y_train, X_test, y_test, X_val, y_val, save=True)

# # Logistic Regression:
df_train_lr = preprocess_mbrset(df=df_train_copy, subset=attributes_to_analyse, class_number=class_number, mode='lr')
df_test_lr = preprocess_mbrset(df=df_test_copy, subset=attributes_to_analyse, class_number=class_number, mode='lr')
df_val_lr = preprocess_mbrset(df=df_val_copy, subset=attributes_to_analyse, class_number=class_number, mode='lr')

X_train=df_train_lr[attributes_to_analyse]
X_test=df_test_lr[attributes_to_analyse]
X_val=df_val_lr[attributes_to_analyse]

y_train=df_train_lr[["final_icdr"]].to_numpy().flatten()
y_test=df_test_lr[["final_icdr"]].to_numpy().flatten()
y_val=df_val_lr[["final_icdr"]].to_numpy().flatten()

# model_lr= apply_lr(X_train, y_train, X_test, y_test, X_val, y_val, class_number, save=True)


for model, model_name in zip(
    [model_tree, model_xgb, model_lr],
    ['DT', 'XGBoost', 'LR']):

    if model is not None:

        # Save trained model:
        save_directory= os.path.join(parent_dir, 'Models')
        os.makedirs(save_directory, exist_ok=True)
        model_path=  os.path.join(save_directory, f'mbrset_full_2classicdr_{model_name}.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved at: {model_path}")




# # Feature Selection:
# y=y.values.ravel()

# # K-Best
# kbest = SelectKBest(f_classif, k=5)
# X_KBest =kbest.fit_transform(X, y)
# X_KBest.shape #selects 10
# selected_features_mask = kbest.get_support()
# selected_features_KBest = X.columns[selected_features_mask]
# print("Selected K-Best Features:", selected_features_KBest)

# # RFE
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X_RFE = model.transform(X)
# X_RFE.shape #selects 9
# selected_features_mask = model.get_support()
# selected_features_RFE = X.columns[selected_features_mask]
# print("Selected RFFeatures:", selected_features_RFE)

# #New X:
# X=df[selected_features_KBest]
