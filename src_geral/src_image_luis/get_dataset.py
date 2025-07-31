import subprocess
import os
import getpass
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np   

def check_columns(row, columns):
    for column in columns:
        if row[column] != 0:
            return 'abnormal'
    return 'normal'

def get_dataset(labels_path, info=False):
    """
    Reads the dataset CSV file and provides information about the DataFrame.

    Parameters:
    data_dir (str): The directory where the dataset is stored.
    download (bool, optional): Whether to download the dataset if it's not already available. Defaults to False.

    Returns:
    pd.DataFrame: The loaded DataFrame containing dataset information.

    Example:
    df = get_dataset("data/", download=True)

    This example would download the dataset if not already available, then load the 'labels.csv' file from the specified directory.
    The resulting DataFrame will contain information about the dataset.

    Note: Make sure to have the 'labels.csv' file in the specified directory.
    """
    
    print(f'loading csv file in {labels_path}')
    df = pd.read_csv(labels_path)
        

    # Provide information about the DataFrame
    if info:
        print(f"Number of Rows: {df.shape[0]}")
        print(f"Number of Columns: {df.shape[1]}")
        print(f"Column Names: {', '.join(df.columns)}")
        print("\nInfo:")
        print(df.info())

        #print("\nDescription:")
        #print(df.describe())
    
    #
    '''
    columns = ['diabetic_retinopathy', 'macular_edema', 'scar', 'nevus',
               'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 
               'drusens', 'hemorrhage', 'retinal_detachment',
               'myopic_fundus', 'increased_cup_disc', 'other'
              ]
    '''
    columns = ['patient', 'age', 'sex', 'dm_time', 'insulin', 'insulin_time',
       'oraltreatment_dm', 'systemic_hypertension', 'insurance',
       'educational_level', 'alcohol_consumption', 'smoking', 'obesity',
       'vascular_disease', 'acute_myocardial_infarction', 'nephropathy',
       'neuropathy', 'diabetic_foot', 'file', 'laterality', 'final_artifacts',
       'final_quality', 'final_icdr', 'final_edema']
    
    df['normality'] = df.apply(check_columns, args=(columns,),  axis=1)

    return df


def get_dataset_brset(labels_path, info=False):
    """
    Reads the dataset CSV file and provides information about the DataFrame.

    Parameters:
    data_dir (str): The directory where the dataset is stored.
    download (bool, optional): Whether to download the dataset if it's not already available. Defaults to False.

    Returns:
    pd.DataFrame: The loaded DataFrame containing dataset information.

    Example:
    df = get_dataset("data/", download=True)

    This example would download the dataset if not already available, then load the 'labels.csv' file from the specified directory.
    The resulting DataFrame will contain information about the dataset.

    Note: Make sure to have the 'labels.csv' file in the specified directory.
    """
    
        
    print(f'loading csv file in {labels_path}')
    df = pd.read_csv(labels_path)
    
    # Provide information about the DataFrame
    if info:
        print(f"Number of Rows: {df.shape[0]}")
        print(f"Number of Columns: {df.shape[1]}")
        print(f"Column Names: {', '.join(df.columns)}")
        print("\nInfo:")
        print(df.info())

        #print("\nDescription:")
        #print(df.describe())
    
    #
    columns = ['patient_age', 'patient_sex',
               'nationality', 
               'diabetes_time_y', 'insuline', 
               'diabetes', 
               'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus',
               'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 
               'drusens', 'hemorrhage', 'retinal_detachment',
               'myopic_fundus', 'increased_cup_disc', 'other', 
               'exam_eye', 
               'DR_ICDR'
              ]
    
    df['normality'] = df.apply(check_columns, args=(columns,),  axis=1)

    return df


# for mBRSET (image models)- uses macula image of eye with stronger icdr to stratify the dataset
# uses all images, but groups images of the same patient together (both in train or both in test)
def split_data_mbrset_images(df, column, split, undersample=False, undersample_ratio=1.0, random_state=42):
    """
    Split a DataFrame into training and testing sets while stratifying by a specified column.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be split.
    - column: str
        The column used for stratification. (ensure similar distribution of training/test set values)
    - split: float
        The proportion of the dataset to include in the test split (0.0 to 1.0).
    - undersample: bool, default=False
        Whether to undersample the majority class in the training set.
    - undersample_ratio: float, default=1.0
        The percentage of undersampling relative to the minority class (1.0 means undersample to the size of the minority class).


    Returns:
    - train_data: pandas DataFrame
        Training set.
    - test_data: pandas DataFrame
        Testing set.
    """

    #See evolution with icdr:
    df_macula_right= df[df['file'].str.endswith('.1.jpg')]
    df_macula_left= df[df['file'].str.endswith('.3.jpg')]
    # df_disc_right= df[df['file'].str.endswith('.2.jpg')]
    # df_disc_left= df[df['file'].str.endswith('.4.jpg')]

    #Prioritizes left_eye information:
    # common_patients = df_macula_left[ df_macula_left['patient'].isin(df_macula_right['patient']) ]

    # Prioritizes right_eye information:
    # common_patients = df_macula_right[ df_macula_right['patient'].isin(df_macula_left['patient']) ]

    #Prioritizes information of eye with stronger icdr:
    common_patients_prep = df_macula_left['patient'].isin(df_macula_right['patient']) #lists of patients with both eyes
    common_patients= pd.DataFrame()

    for patient_id in df_macula_left[common_patients_prep]['patient']:
      
        left_row = df_macula_left[df_macula_left['patient'] == patient_id]
        right_row = df_macula_right[df_macula_right['patient'] == patient_id]
        
        # Select highest label 'final_icdr'
        if left_row[column].values[0] > right_row[column].values[0]: #if left eye has a higher label than right eye, use that label
            common_patients = pd.concat([common_patients, left_row], ignore_index=True)
        else:
            common_patients = pd.concat([common_patients, right_row], ignore_index=True)

    right_only_patients = df_macula_right[~df_macula_right['patient'].isin(df_macula_left['patient'])] #add right eye only
    left_only_patients = df_macula_left[~df_macula_left['patient'].isin(df_macula_right['patient'])] #add left eye only

    all_icdr = []
    all_patients = []

    # Merge the 3 types of patient dfs:
    if not common_patients.empty:
        all_icdr += common_patients[column].tolist()
        all_patients += common_patients['patient'].tolist()

    if not right_only_patients.empty:
        all_icdr += right_only_patients[column].tolist()
        all_patients += right_only_patients['patient'].tolist()

    if not left_only_patients.empty:
        all_icdr += left_only_patients[column].tolist()
        all_patients += left_only_patients['patient'].tolist()

    # # Convert to pandas Series for use in train_test_split
    # all_icdr = pd.Series(all_icdr)
    # all_patients = pd.Series(all_patients)

    # Divides the patients by type:   
    train_patients, test_patients = train_test_split(all_patients, 
                                                    test_size=split, 
                                                    stratify=all_icdr, 
                                                    random_state=42)
    
    # Having divided, gets the patients, for each eye:
    train_data_left = df_macula_left[df_macula_left['patient'].isin(train_patients)]
    test_data_left = df_macula_left[df_macula_left['patient'].isin(test_patients)]

    train_data_right = df_macula_right[df_macula_right['patient'].isin(train_patients)]
    test_data_right = df_macula_right[df_macula_right['patient'].isin(test_patients)]

    # Adds
    train_data = pd.concat([train_data_left, train_data_right], ignore_index=True)
    test_data = pd.concat([test_data_left, test_data_right], ignore_index=True)

    train_data_n = len( train_data[train_data[column]=='Normal'] )
    train_data_dr = len( train_data[train_data[column]=='DR'] )
    test_data_n = len( test_data[test_data[column]=='Normal'] )
    test_data_dr = len( test_data[test_data[column]=='DR'] )

    # print(f"Train: {train_data_n}|{train_data_dr} Test size: {test_data_n}|{test_data_dr}")

    if undersample:
        # Determine the class with the fewest samples
        min_class_count = train_data[column].value_counts().min()
        # Determine the class with the most samples ang get the name
        max_class_label = train_data[column].value_counts().idxmax()
        # Downsample the majority classes to balance the dataset
        
        downsampled_classes = []
        for class_label in train_data[column].unique():
            if class_label == max_class_label:
                class_data = train_data[train_data[column] == class_label]
                downsampled_class = resample(class_data, replace=False, n_samples=int(min_class_count*undersample_ratio), random_state=random_state)
                downsampled_classes.append(downsampled_class)
            else:
                downsampled_class = train_data[train_data[column] == class_label]
                downsampled_classes.append(downsampled_class)
        
        # Combine the downsampled classes
        train_data = pd.concat(downsampled_classes)


    # print(f"Train data shape: {train_data.shape}")
    # print(f"Test data shape: {test_data.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plot_labels_distribution(train_data, column, title='Train Label Distribution')
    plt.subplot(1, 2, 2)
    plot_labels_distribution(test_data, column, title='Test Label Distribution')
    plt.show()

    return train_data, test_data

# for mBRSET (feature models)- chooses macula image of eye with stronger icdr to stratify
# uses each patient only once! 
def split_data_mbrset_features(df, column, split, undersample=False, undersample_ratio=1.0, random_state=42):
    """
    Split a DataFrame into training and testing sets while stratifying by a specified column.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be split.
    - column: str
        The column used for stratification. (ensure similar distribution of training/test set values)
    - split: float
        The proportion of the dataset to include in the test split (0.0 to 1.0).
    - undersample: bool, default=False
        Whether to undersample the majority class in the training set.
    - undersample_ratio: float, default=1.0
        The percentage of undersampling relative to the minority class (1.0 means undersample to the size of the minority class).


    Returns:
    - train_data: pandas DataFrame
        Training set.
    - test_data: pandas DataFrame
        Testing set.
    """

    #See evolution with icdr:
    df_macula_right= df[df['file'].str.endswith('.1.jpg')]
    df_macula_left= df[df['file'].str.endswith('.3.jpg')]
    # df_disc_right= df[df['file'].str.endswith('.2.jpg')]
    # df_disc_left= df[df['file'].str.endswith('.4.jpg')]

    #Prioritizes left_eye information:
    # common_patients = df_macula_left[ df_macula_left['patient'].isin(df_macula_right['patient']) ]

    # Prioritizes right_eye information:
    # common_patients = df_macula_right[ df_macula_right['patient'].isin(df_macula_left['patient']) ]

    #Prioritizes information of eye with stronger icdr:
    common_patients_prep = df_macula_left['patient'].isin(df_macula_right['patient']) #lists of patients with both eyes
    df_1_patient= pd.DataFrame()

    for patient_id in df_macula_left[common_patients_prep]['patient']:
      
        left_row = df_macula_left[df_macula_left['patient'] == patient_id]
        right_row = df_macula_right[df_macula_right['patient'] == patient_id]
        
        # Select highest label 'final_icdr'
        if left_row[column].values[0] > right_row[column].values[0]: #if left eye has a higher label than right eye, use that label
            df_1_patient= pd.concat([df_1_patient, left_row], ignore_index=True)
        else:
            df_1_patient= pd.concat([df_1_patient, right_row], ignore_index=True)

    right_only_patients = df_macula_right[~df_macula_right['patient'].isin(df_macula_left['patient'])] #add right eye only
    left_only_patients = df_macula_left[~df_macula_left['patient'].isin(df_macula_right['patient'])] #add left eye only

    df_1_patient = pd.concat([df_1_patient, right_only_patients, left_only_patients])

    # Divides the patients by type:   
    train_patient, test_patient = train_test_split(df_1_patient, 
                                                    test_size=split, 
                                                    stratify=df_1_patient[column], 
                                                    random_state=42)
    
    # Adds
    train_data_n = len( train_patient[train_patient[column]==0] )
    train_data_dr = len( train_patient[train_patient[column]==1] )
    test_data_n = len( test_patient[test_patient[column]==0] )
    test_data_dr = len( test_patient[test_patient[column]==1] )

    # print(f"Train: {train_data_n}|{train_data_dr} Test size: {test_data_n}|{test_data_dr}")

    if undersample:
        # Determine the class with the fewest samples
        min_class_count = train_patient[column].value_counts().min()
        # Determine the class with the most samples ang get the name
        max_class_label = train_patient[column].value_counts().idxmax()
        # Downsample the majority classes to balance the dataset
        
        downsampled_classes = []
        for class_label in train_patient[column].unique():
            if class_label == max_class_label:
                class_data = train_patient[train_patient[column] == class_label]
                downsampled_class = resample(class_data, replace=False, n_samples=int(min_class_count*undersample_ratio), random_state=random_state)
                downsampled_classes.append(downsampled_class)
            else:
                downsampled_class = train_patient[train_patient[column] == class_label]
                downsampled_classes.append(downsampled_class)
        
        # Combine the downsampled classes
        train_patient = pd.concat(downsampled_classes)


    # print(f"Train data shape: {train_patient.shape}")
    # print(f"Test data shape: {test_patient.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plot_labels_distribution(train_patient, column, title='Train Label Distribution')
    plt.subplot(1, 2, 2)
    plot_labels_distribution(test_patient, column, title='Test Label Distribution')
    plt.show()

    return train_patient, test_patient


# TBD
# for BRSET (image models)- uses macula image of eye with stronger icdr to stratify the dataset
# uses all images, but groups images of the same patient together (both in train or both in test)
def split_data_brset_images(df, column, split, undersample=False, undersample_ratio=1.0, random_state=42):
    """
    Split a DataFrame into training and testing sets while stratifying by a specified column.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be split.
    - column: str
        The column used for stratification. (ensure similar distribution of training/test set values)
    - split: float
        The proportion of the dataset to include in the test split (0.0 to 1.0).
    - undersample: bool, default=False
        Whether to undersample the majority class in the training set.
    - undersample_ratio: float, default=1.0
        The percentage of undersampling relative to the minority class (1.0 means undersample to the size of the minority class).


    Returns:
    - train_data: pandas DataFrame
        Training set.
    - test_data: pandas DataFrame
        Testing set.
    """

    # Separate by eye: (1- right, 2-left)
    df_right= df[df['exam_eye']==1]
    df_left= df[df['exam_eye']==2]


    #Prioritizes information of eye with stronger icdr:
    common_patients_prep = df_left['patient_id'].isin(df_right['patient_id']) #lists of patients with both eyes
    common_patients= pd.DataFrame()

    for patient_id in df_left[common_patients_prep]['patient_id']:
      
        left_row = df_left[df_left['patient_id'] == patient_id]
        right_row = df_right[df_right['patient_id'] == patient_id]
        
        # Select highest label 'final_icdr'
        if left_row[column].values[0] > right_row[column].values[0]: #if left eye has a higher label than right eye, use that label
            common_patients = pd.concat([common_patients, left_row], ignore_index=True)
        else:
            common_patients = pd.concat([common_patients, right_row], ignore_index=True)

    right_only_patients = df_right[~df_right['patient_id'].isin(df_left['patient_id'])] #add right eye only
    left_only_patients = df_left[~df_left['patient_id'].isin(df_right['patient_id'])] #add left eye only

    all_icdr = []
    all_patients = []


 # Merge the 3 types of patient dfs:
    if not common_patients.empty:
        all_icdr += common_patients[column].tolist()
        all_patients += common_patients['patient_id'].tolist()

    if not right_only_patients.empty:
        all_icdr += right_only_patients[column].tolist()
        all_patients += right_only_patients['patient_id'].tolist()

    if not left_only_patients.empty:
        all_icdr += left_only_patients[column].tolist()
        all_patients += left_only_patients['patient_id'].tolist()


    # Divides the patients by type:   
    train_patients, test_patients = train_test_split(all_patients, 
                                                    test_size=split, 
                                                    stratify=all_icdr, 
                                                    random_state=42)
    
    # Having divided, gets the patients, for each eye:
    train_data_left = df_left[df_left['patient_id'].isin(train_patients)]
    test_data_left = df_left[df_left['patient_id'].isin(test_patients)]

    train_data_right = df_right[df_right['patient_id'].isin(train_patients)]
    test_data_right = df_right[df_right['patient_id'].isin(test_patients)]

    # Adds
    train_data = pd.concat([train_data_left, train_data_right], ignore_index=True)
    test_data = pd.concat([test_data_left, test_data_right], ignore_index=True)

    train_data_n = len( train_data[train_data[column]=='Normal'] )
    train_data_dr = len( train_data[train_data[column]=='DR'] )
    test_data_n = len( test_data[test_data[column]=='Normal'] )
    test_data_dr = len( test_data[test_data[column]=='DR'] )

    # print(f"Train: {train_data_n}|{train_data_dr} Test size: {test_data_n}|{test_data_dr}")



    if undersample:
        # Determine the class with the fewest samples
        min_class_count = train_data[column].value_counts().min()
        # Determine the class with the most samples ang get the name
        max_class_label = train_data[column].value_counts().idxmax()
        # Downsample the majority classes to balance the dataset
        
        downsampled_classes = []
        for class_label in train_data[column].unique():
            if class_label == max_class_label:
                class_data = train_data[train_data[column] == class_label]
                downsampled_class = resample(class_data, replace=False, n_samples=int(min_class_count*undersample_ratio), random_state=random_state)
                downsampled_classes.append(downsampled_class)
            else:
                downsampled_class = train_data[train_data[column] == class_label]
                downsampled_classes.append(downsampled_class)
        
        # Combine the downsampled classes
        train_data = pd.concat(downsampled_classes)


    # print(f"Train data shape: {train_data.shape}")
    # print(f"Test data shape: {test_data.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plot_labels_distribution(train_data, column, title='Train Label Distribution')
    plt.subplot(1, 2, 2)
    plot_labels_distribution(test_data, column, title='Test Label Distribution')
    plt.show()

    return train_data, test_data

# TBD
# for BRSET (feature models)- chooses macula image of eye with stronger icdr to stratify
# uses each patient only once! 
def split_data_brset_features(df, column, split, undersample=False, undersample_ratio=1.0, random_state=42):
    """
    Split a DataFrame into training and testing sets while stratifying by a specified column.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be split.
    - column: str
        The column used for stratification. (ensure similar distribution of training/test set values)
    - split: float
        The proportion of the dataset to include in the test split (0.0 to 1.0).
    - undersample: bool, default=False
        Whether to undersample the majority class in the training set.
    - undersample_ratio: float, default=1.0
        The percentage of undersampling relative to the minority class (1.0 means undersample to the size of the minority class).


    Returns:
    - train_data: pandas DataFrame
        Training set.
    - test_data: pandas DataFrame
        Testing set.
    """
    # Separate by eye: (1- right, 2-left)
    df_right= df[df['exam_eye']==1]
    df_left= df[df['exam_eye']==2]

    # Prioritizes information of eye with stronger icdr (for patients with both eyes):
    common_patients_prep = df_left['patient_id'].isin(df_right['patient_id']) #list of patients with both eyes
    df_1_patient= pd.DataFrame()

    for patient_id in df_left[common_patients_prep]['patient_id']:
          
        left_row = df_left[df_left['patient_id'] == patient_id]
        right_row = df_right[df_right['patient_id'] == patient_id]
        
        # Select highest label 'final_icdr'
        if left_row[column].values[0] > right_row[column].values[0]: 
            #if left eye has a higher label than right eye, use that label

            df_1_patient= pd.concat([df_1_patient, left_row], ignore_index=True)
        else: #right eye has equal/higher label
            df_1_patient= pd.concat([df_1_patient, right_row], ignore_index=True)

    right_only_patients = df_right[~df_right['patient_id'].isin(df_left['patient_id'])] #add patients with right eye only
    left_only_patients = df_left[~df_left['patient_id'].isin(df_right['patient_id'])] #add left eye only

    df_1_patient = pd.concat([df_1_patient, right_only_patients, left_only_patients])

                       
    # Divides the patients by type:   
    train_patient, test_patient = train_test_split(df_1_patient, test_size=split, 
                                                    stratify=df_1_patient[column], 
                                                    random_state=42)
    
    # Adds
    train_data_n = len( train_patient[train_patient[column]==0] )
    train_data_dr = len( train_patient[train_patient[column]==1] )
    test_data_n = len( test_patient[test_patient[column]==0] )
    test_data_dr = len( test_patient[test_patient[column]==1] )

    # print(f"Train: {train_data_n}|{train_data_dr} Test size: {test_data_n}|{test_data_dr}")


    if undersample:
        # Determine the class with the fewest samples
        min_class_count = train_data[column].value_counts().min()
        # Determine the class with the most samples ang get the name
        max_class_label = train_data[column].value_counts().idxmax()
        # Downsample the majority classes to balance the dataset
        
        downsampled_classes = []
        for class_label in train_data[column].unique():
            if class_label == max_class_label:
                class_data = train_data[train_data[column] == class_label]
                downsampled_class = resample(class_data, replace=False, n_samples=int(min_class_count*undersample_ratio), random_state=random_state)
                downsampled_classes.append(downsampled_class)
            else:
                downsampled_class = train_data[train_data[column] == class_label]
                downsampled_classes.append(downsampled_class)
        
        # Combine the downsampled classes
        train_data = pd.concat(downsampled_classes)


    # print(f"Train data shape: {train_patient.shape}")
    # print(f"Test data shape: {test_patient.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plot_labels_distribution(train_patient, column, title='Train Label Distribution')
    plt.subplot(1, 2, 2)
    plot_labels_distribution(test_patient, column, title='Test Label Distribution')
    plt.show()

    return train_patient, test_patient


def plot_labels_distribution(df, column, title=None):
    # Get value counts for the new categories
    category_counts = df[column].value_counts()
    
    # Plotting the value counts
    #category_counts.plot(kind='bar', rot=0, color='skyblue')
    plt.bar(category_counts.index, category_counts.values, color='skyblue')
    plt.xlabel(f'{column} Categories')
    plt.ylabel('Count')
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of {column} Categories')