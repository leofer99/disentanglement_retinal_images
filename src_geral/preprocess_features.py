import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_brset(df, subset, class_number, mode=None):

    unaltered_df = df.copy()

    if 'patient_age' in subset and len(df['patient_age'].unique()) != 2: #Int

        df['patient_age'] = df['patient_age'].replace({'>= 90': 91})
        df['patient_age']= df['patient_age'].astype(int)
        
        if mode=='bin': # Binarize age
            # th_age= np.median(df['patient_age'])
            th_age=50
            df['patient_age']=df['patient_age'].apply(lambda x: 0 if x <= th_age else 1 ).astype(int)

        else: #scale to values in range [0,100]
            scaler = MinMaxScaler(feature_range=(0, 100))
            df['patient_age'] = scaler.fit_transform(df['patient_age'].values.reshape(-1, 1))

            if mode=='lr': #scale to [0, 1]
                df['patient_age']= df['patient_age']/100

            # age_lb=0
            # age_ub=100
            # df['age']= (df['age']  - age_lb)/(age_ub - age_lb) 


    if 'patient_sex' in subset: # 0=Female, 1=Male
        df['patient_sex']= df['patient_sex'].apply(lambda x: 0 if x in [2, 0] else 1).astype(int)


    if 'nationality' in subset: #...
        df['nationality']= df['nationality']
        df= pd.get_dummies(df, columns=['nationality'], drop_first=True)

    if 'diabetes_time_y' in subset:
        df['diabetes_time_y'] = pd.to_numeric(df['diabetes_time_y'], errors='coerce')

        if mode=='bin':
            th_dmtime= np.median(df['diabetes_time_y'])
            df.loc[:, 'diabetes_time_y']=df['diabetes_time_y'].apply(lambda x: 0 if x <= th_dmtime else 1 ).astype(int)

        elif mode=='lr':
            scaler = MinMaxScaler()
            df['diabetes_time_y'] = scaler.fit_transform(df['diabetes_time_y'].values.reshape(-1, 1))

        print("7- NaNs in df before cut!!:", pd.isna(df).sum().sum())
        df = df.dropna(subset=['diabetes_time_y'])
        print("NaNs in df after cut!!:", pd.isna(df).sum().sum())
        unaltered_df = unaltered_df.loc[df.index]

    if 'camera' in subset: #Camera Prediction:
        #0: 'Canon CR', 1:'NIKON NF5050'        
        df['camera']= df['camera'].apply(lambda x: 1 if x in ['NIKON NF5050', 1] else 0 )


    if 'insuline' in subset: # 0=No, 1=Yes
        df['insuline']= df['insuline'].apply(lambda x: 1 if x == 'yes' else 0 )

    if 'diabetes' in subset:
        df['diabetes']= df['diabetes'].apply(lambda x: 1 if x == 'yes' else 0 )
        # df['oraltreatment_dm']= df['oraltreatment_dm'].apply(lambda x: '1' if x == 1 else '0' )

    if 'macular_edema' in subset:
        df['macular_edema']= df['macular_edema'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'scar' in subset:
        df['scar']= df['scar'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'nevus' in subset:
        df['nevus']= df['nevus'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'amd' in subset:
        df['amd']= df['amd'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'vascular_occlusion' in subset:
        df['vascular_occlusion']= df['vascular_occlusion'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'hypertensive_retinopathy' in subset:
        df['hypertensive_retinopathy']= df['hypertensive_retinopathy'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'drusens' in subset:
        df['drusens']= df['drusens'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'hemorrhage' in subset:
        df['hemorrhage']= df['hemorrhage'] #.apply(lambda x: 1 if x == 'yes' else 0 )

    if 'retinal_detachment' in subset:
        df['retinal_detachment']= df['retinal_detachment'] #.apply(lambda x: 1 if x == 'yes' else 0 )
    
    if 'myopic_fundus' in subset:
        df['myopic_fundus']= df['myopic_fundus'] #.apply(lambda x: 1 if x == 'yes' else 0 )
    
    if 'increased_cup_disc' in subset:
        df['increased_cup_disc']= df['increased_cup_disc'] #.apply(lambda x: 1 if x == 'yes' else 0 )
    
    if 'other' in subset:
        df['other']= df['other'] #.apply(lambda x: 1 if x == 'yes' else 0 )
    
    if 'exam_eye' in subset: # 1=right eye, 2=left_eye 
        df['exam_eye']= df['exam_eye'].apply(lambda x: 1 if x == 1 else 2 )



    return df, unaltered_df



def preprocess_mbrset(df, subset, class_number, mode=None):

    if 'age' in subset and len(df['age'].unique()) != 2:
        df['age'] = df['age'].replace({'>= 90': 91})
        df['age']= df['age'].astype(int)

        if mode=='bin': # Binarize age
            # th_age= np.median(df['age'])
            th_age=50
            df['age']=df['age'].apply(lambda x: 0 if x <= th_age else 1 ).astype(int)


        else:  #scale to values in range [0,100]
            scaler = MinMaxScaler(feature_range=(0, 100))
            df['age'] = scaler.fit_transform(df['age'].values.reshape(-1, 1))
            
            if mode=='lr': #normalize to 0-1
                df['age']= df['age']/100

            # age_lb=0
            # age_ub=100
            # df['age']= (df['age']  - age_lb)/(age_ub - age_lb) 


    if 'sex' in subset: # 0=Female, 1=Male
        # df['sex']= df['sex'].astype(int)
        df['sex']= df['sex'].apply(lambda x: 0 if x == 0 else 1)


    if 'dm_time' in subset: # Int (Duration of diabetes diagnosis in years)
        if mode=='bin':
            th_dmtime= np.median(df['dm_time'])
            df.loc[:, 'dm_time']=df['dm_time'].apply(lambda x: 0 if x <= th_dmtime else 1 ).astype(int)

        elif mode=='lr':
            scaler = MinMaxScaler()
            df['dm_time']= scaler.fit_transform(df['dm_time'].values.reshape(-1, 1))


    if 'insulin' in subset: # 0=No, 1=Yes (use of insulin)
        df['insulin']= df['insulin'].astype(int)

    if 'insulin_time' in subset: # Int (Duration of insulin use in years)
    #has a lot of NA values
        if mode=='bin':
            th_dmtime= np.median(df['insulin_time'])
            df.loc[:, 'insulin_time']=df['insulin_time'].apply(lambda x: 0 if x <= th_dmtime else 1 ).astype(int)

        elif mode=='lr':
            scaler = MinMaxScaler()
            df['insulin_time']= scaler.fit_transform(df['insulin_time'].values.reshape(-1, 1))

    if 'oraltreatment_dm' in subset: # 0=No, 1=Yes (Use of oral drugs for diabetes) 
        df['oraltreatment_dm']= df['oraltreatment_dm'].astype(int)

    if 'systemic_hypertension' in subset: #0=No, 1=Yes (Diagnosis of systemic arterial hypertension)
        df['systemic_hypertension']= df['systemic_hypertension'].astype(int)

    if 'insurance' in subset: #0=No, 1=Yes (Health insurance status)
        df['insurance']= df['insurance'].astype(int)

    if 'educational_level' in subset: #Highest educational level achieved: 
        #(Separate illiterate from literate)
    # 1=Illiterate, 2=Incomplete Primary, 3=Complete Primary, 4=Incomplete Secondary
    # 5=Complete Secondary, 6=Incomplete Rertiary, 7=Complete Tertiary
        df['educational_level']= df['educational_level'].astype(int)
        df['educational_level']= df['educational_level'].apply(lambda x: 1 if x in [1,2] else 0 )

    if 'alcohol_consumption' in subset: # 0=No, 1=Yes (Regular alcohol consumption)
        df['alcohol_consumption']= df['alcohol_consumption'].astype(int)

    if 'smoking' in subset: # 0=No, 1=Yes (Active smoking status)
        df['smoking']= df['smoking'].astype(int)

    if 'obesity' in subset:  # 0=No, 1=Yes (Diagnosis of obesity)
        df['obesity']= df['obesity'].astype(int)

    if 'acute_myocardial_infarction' in subset: # 0=No, 1=Yes (History of acute myocardial infarction) 
        df['acute_myocardial_infarction']= df['acute_myocardial_infarction'].astype(int)

    if 'diabetic_foot' in subset: #  0=No, 1=Yes (Presence of diabetic foot)
        df['diabetic_foot']= df['diabetic_foot'].astype(int)

    if 'vascular_disease' in subset: #  0=No, 1=Yes (Diabetes-related vascular disease)
        df['vascular_disease']= df['vascular_disease'].astype(int)

    if 'nephropathy' in subset: # 0=No, 1=Yes (Nephropathy)
        df['nephropathy']= df['nephropathy'].astype(int)

    if 'laterality' in subset: # 0=Left, 1=Right (Laterality of retinal fundus photo)
        df['laterality']= df['laterality'].apply(lambda x: '1' if x == 'Left' else '0' )

    if 'final_artifacts' in subset: # 0=No, 1=Yes (Presence of artifacts in the retinal image)
        df['final_artifacts']= df['final_artifacts'].apply(lambda x: '1' if x == 'Yes' else '0' )

    if 'final_quality' in subset: # 0=No, 1=Yes (Final quality assessment of the retinal image)
        df['final_quality']= df['final_quality'].apply(lambda x: '1' if x == 'Yes' else '0' )

    if 'final_edema' in subset: # 0=No, 1=Yes (Presence of macular edema)
        df['final_edema']= df['final_edema'].apply(lambda x: '1' if x == 'Yes' else '0' )


    return df
