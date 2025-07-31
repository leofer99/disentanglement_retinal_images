import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src_disentanglement.image_utils import rescale_image_to_minus1_1


# Function to process text labels and one-hot encode them
def process_labels(df, col='answer', mlb=None, train_columns=None):
    """
    Process text labels and perform one-hot encoding using MultiLabelBinarizer.

    Args:
    - df (pd.DataFrame): The DataFrame containing the labels.
    - col (str): The column name containing the labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): The MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Returns:
    pd.DataFrame: One-hot encoded labels.
    sklearn.preprocessing.MultiLabelBinarizer: MultiLabelBinarizer object.
    list: List of columns from the training set.

    Example:
    one_hot_labels, mlb, train_columns = process_labels(df, col='answer')
    """
    if mlb is None:
        mlb = MultiLabelBinarizer()
        
        if df[col].dtype == int: #int classes
            labels= df[col].values.reshape(-1, 1)

            # train_columns = col
            # one_hot_labels = pd.get_dummies(df[col], prefix=None)
            # train_columns = one_hot_labels.columns

            # return one_hot_labels, mlb #, train_columns
            
        else: #classes as string
            labels = df[col].apply(lambda x: set(x.split(', ')))

        one_hot_labels = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_)
        # Save the columns from the training set
        train_columns = one_hot_labels.columns
        
        return one_hot_labels, mlb, train_columns



        # if df[col].dtype == int  and (len(df[col].unique()) == 2):
        #     # label = df[col]
        #     labels = pd.get_dummies(df[col], prefix=col) #creates [class0, class1] tensor 

        # else:
        #     labels = df[col].apply(lambda x: set(x.split(', ')))

        # if df[col].dtype == int and (len(df[col].unique()) == 2): #binary classification
        #     train_columns = col
        #     one_hot_labels = label
        # else:
        #     one_hot_labels = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_)
        #     # Save the columns from the training set
        #     train_columns = one_hot_labels.columns

        # return one_hot_labels, mlb, train_columns

    else:
        if df[col].dtype == int:
            labels = df[col]
        else:
            labels = df[col].apply(lambda x: set(x.split(', ')))

        if df[col].dtype == int and (len(df[col].unique()) == 2):
            one_hot_labels = labels
        else:
            one_hot_labels = pd.DataFrame(mlb.transform(labels), columns=train_columns)

        return one_hot_labels


class mBRSETDataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - image_cols (str): Column name containing the path to the images.
    - label_col (str): Column name containing labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Attributes:
    - image_data (np.ndarray): Array of image data.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.
    - labels (np.ndarray): Array of one-hot encoded labels.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(idx): Returns a dictionary with 'text', 'image', and 'labels'.

    Example:
    dataset = BRSETDataset(df, image_cols='image1', label_col='answer', mlb=mlb, train_columns=train_columns)
    """
    def __init__(self, df, image_cols, images_dir, label_col, mlb, train_columns, shape=(224, 224), transform=None, disentangled=False, augmentation=False, id_info='age'):

        # Images
        self.df=df
        self.image_data = df[image_cols].values
        self.images_dir = images_dir
        self.shape = shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.disentangled=disentangled
        self.augmentation = augmentation
        self.transform_augmentation = transforms.Compose([
                transforms.Resize((224,224)),

                transforms.RandomRotation(degrees=20), 
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),


                # transforms.RandomCrop(SHAPE),
                transforms.ToTensor(),  # Converts image to [0, 1]
                transforms.Lambda(rescale_image_to_minus1_1),  # Rescale [0, 1] to [-1, 1]

                # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                # transforms.RandomRotation(50),  # Randomly rotate the image by up to 10 degrees
            ])

        # Label Disease:
        self.mlb = mlb
        self.train_columns = train_columns
        self.labels = process_labels(df, col=label_col, mlb=mlb, train_columns=train_columns).values
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)

        if disentangled == True:
            # Label ID: 
            self.label_id = df[id_info]
        
        # print(f"Image shape: {self.image_data.shape}, Label ID shape: {self.label_id.shape}, Label Dis shape: {self.labels.shape}")



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Images:
        img_path = self.image_data[idx]
        img = Image.open( os.path.join(self.images_dir, img_path) ).convert("RGB")
        img = self.transform(img)

        label_dis = torch.tensor(self.labels[idx], dtype=torch.float32)

        # augmentation = getattr(self, "augmentation", False)
        # if augmentation: #and label_dis.item()==1:
        #     img = self.transform_augmentation(img)

        # else:
        #     img = self.transform(img)
            
        # print(f"Image shape: {img.shape}, Label ID shape: {label_id.shape}, Label Dis shape: {label_dis.shape}")


        if self.disentangled:
            label_id = torch.tensor(self.label_id[idx], dtype=torch.float32)

            return {
                'image': img,
                'label_id': label_id,
                'label_dis': label_dis }
        
        else:
            return {
                'image': img,
                'labels': label_dis }


  
    # def __init__(self, base_dataset):
    #     mbrset_dir= r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\mBRSET"
    #     images_path= os.path.join(mbrset_dir, "images")
    #     csv_file_path= os.path.join(mbrset_dir, "labels_mbrset.csv")

    #     #Assign class variables
    #     self.mbrset_dir = mbrset_dir
    #     self.csv_file_path = csv_file_path
    #     self.images_path = images_path

    #     #Open CSV file
    #     filtered_patients_info = self.create_database(csv_file_path,images_path)

    # def create_database(self, csv_file_path,image_folder_path):
    #     patients_info = pd.read_csv(csv_file_path)
    #     image_filenames = set(os.listdir(image_folder_path))
    #     patients_info = patients_info[ patients_info['file_names'].isin(image_filenames) ]
        
    #     return patients_info



class BRSETDataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - image_cols (str): Column name containing the path to the images.
    - label_col (str): Column name containing labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Attributes:
    - image_data (np.ndarray): Array of image data.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.
    - labels (np.ndarray): Array of one-hot encoded labels.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(idx): Returns a dictionary with 'text', 'image', and 'labels'.

    Example:
    dataset = BRSETDataset(df, image_cols='image1', label_col='answer', mlb=mlb, train_columns=train_columns)
    """
    def __init__(self, df, image_cols, images_dir, label_col, mlb, train_columns, shape=(224, 224), transform=None, disentangled=False, augmentation=False, id_info='patient_age'):

        # Images
        self.df=df
        self.image_data = df[image_cols].values
        self.images_dir = images_dir
        self.shape = shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.disentangled = disentangled
        self.augmentation = augmentation


        # Label Disease:
        self.mlb = mlb
        self.train_columns = train_columns
        self.labels = process_labels(df, col=label_col, mlb=mlb, train_columns=train_columns).values
        # print(self.labels.shape)
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)
        
        if disentangled == True:
            # Label ID:
            self.label_id = df[id_info]

        # print(f"Image shape: {self.image_data.shape}, Label ID shape: {self.label_id.shape}, Label Dis shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Images:
        img_name= self.df['image_id'].iloc[idx] + ('.jpg')
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label_dis = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # print(f"Image shape: {img.shape}, Label ID shape: {label_id.shape}, Label Dis shape: {label_dis.shape}")

        if self.disentangled:
            label_id = torch.tensor(self.label_id[idx], dtype=torch.float32)

            return {
                'image': img,
                'label_id': label_id,
                'label_dis': label_dis }
        
        else:
            return {
                'image': img,
                'labels': label_dis }


#################################################