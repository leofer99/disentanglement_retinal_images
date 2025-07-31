import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('src_geral')
from preprocess_features import preprocess_mbrset



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




attributes_to_analyse=['final_icdr', 'age', 'sex', 'dm_time', 
                    #    'insulin', 
                       'insulin_time',
                       'oraltreatment_dm', 
                       'systemic_hypertension', 'insurance', 'educational_level', 'alcohol_consumption',
                       'smoking', 'obesity', 
                    #    'vascular_disease', 'acute_myocardial_infarction',
                    #    'nephropathy', 'diabetic_foot', 
                        # 'laterality', 'final_artifacts', 'final_quality',
                    #    'final_edema'
                       ]

filtered_patients_info = filtered_patients_info.dropna(subset=attributes_to_analyse)


# filtered_patients_info = filtered_patients_info.dropna(subset=attributes_to_analyse)

#Pre-processing
filtered_patients_info['final_icdr'] = filtered_patients_info['final_icdr'].apply(lambda x: '1' if x in [0, 1] else '0')
filtered_patients_info = preprocess_mbrset(df=filtered_patients_info, subset=attributes_to_analyse, class_number=2)



# filtered_patients_info['age'] = filtered_patients_info['age'].replace({'>= 90': 91})
# age_column = filtered_patients_info['age'].values.reshape(-1, 1)
# discretizer = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
# discretizer.fit(age_column)
# filtered_patients_info['age'] = discretizer.transform(age_column)

# dm_time_column = filtered_patients_info['dm_time'].values.reshape(-1, 1)
# discretizer = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
# discretizer.fit(dm_time_column)
# filtered_patients_info['dm_time'] = discretizer.transform(dm_time_column)

# insulin_time_column = filtered_patients_info['insulin_time'].values.reshape(-1, 1)
# discretizer = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
# discretizer.fit(insulin_time_column)
# filtered_patients_info['insulin_time'] = discretizer.transform(insulin_time_column)


# filtered_patients_info['laterality'] = filtered_patients_info['laterality'].apply(lambda x: '1' if x == 'Left' else '0' )
# filtered_patients_info['final_artifacts'] = filtered_patients_info['final_artifacts'].apply(lambda x: '1' if x == 'Yes' else '0' )
# filtered_patients_info['final_quality'] = filtered_patients_info['final_quality'].apply(lambda x: '1' if x == 'Yes' else '0' )
# filtered_patients_info['final_edema'] = filtered_patients_info['final_edema'].apply(lambda x: '1' if x == 'yes' else '0' )

# #Separate by Iliterate (1) or not
# filtered_patients_info['educational_level'] = filtered_patients_info['educational_level'].apply(lambda x: '1' if x in [1, 2] else '0')

# Compute correlation matrix
corr = filtered_patients_info[attributes_to_analyse].astype(float).corr()


# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))


plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": .8}
)

# Crop the axes to remove empty space
n = len(corr)
ax.set_xlim(0, n)
ax.set_ylim(n, 0)

plt.title("Correlation matrix (lower triangle only)")
plt.tight_layout()
plt.show()


# Print correlation matrix
#print("Correlation Matrix:")
#print(correlation_matrix)

# # Plot heatmap for better visualization
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation matrix of mBRSET attributes")
# plt.tight_layout()

# plt.show()



