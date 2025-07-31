import os
import sys
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))


from src_image_luis.get_dataset import get_dataset, split_data_mbrset_images
from src_image_luis.data_loader import mBRSETDataset, process_labels
# from src.RetFound import get_retfound
from src_image_luis.FocalLoss import FocalLoss, BinaryFocalLoss
from src_image_luis.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from src_image_luis.train import train
from src_image_luis.test import test


if __name__ == '__main__':

    # Constants:
    mbrset_dir1= r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\mBRSET"
    mbrset_dir2= r"/nas-ctm01/datasets/public/MEDICAL/mbrset/physionet.org/files/mbrset/1.0"

    if os.path.exists(mbrset_dir2):
        mbrset_dir = mbrset_dir2
    elif os.path.exists(mbrset_dir1):
        mbrset_dir = mbrset_dir1
    else: 
        print("Neither path works!")

    image_folder_path= os.path.join(mbrset_dir, "images")
    IMAGES= image_folder_path
    csv_file_path= os.path.join(mbrset_dir, "labels_mbrset.csv")

    DOWNLOAD = False
    IMAGE_COL = 'file'

    """
    Dataset Mean and Std:
    NORM_MEAN = [0.5896205017400412, 0.29888971649817453, 0.1107679405196557]
    NORM_STD = [0.28544273712830986, 0.15905456049750208, 0.07012281660980953]

    ImageNet Mean and Std:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    """

    NORM_MEAN =  [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    BACKBONE = 'convnextv2_large'
    MODE = 'fine_tune'
    backbone_mode = 'fine_tune'

    SHAPE = (224,224)
    TEST_SIZE = 0.2
    UNDERSAMPLE = False

    HIDDEN = [128]
    BATCH_SIZE = 4 #changed from 16
    # BATCH_SIZE = 8
    NUM_WORKERS = 4
    LOSS = 'focal_loss'
    OPTIMIZER = 'adam'


    # Define your hyperparameters
    parser = argparse.ArgumentParser(description='conv_next_v2_large.py')
    parser.add_argument('--num_classes', type=int, default=2, required=False, help="Number of classes.")
    parser.add_argument('--epochs', type=int, default=1, required=False, help="Number of training epochs.")
    # parser.add_argument('--model_names', nargs='+', type=str, required=False, default=['vgg16'], help="Model to train.")
    parser.add_argument('--classification', type=str, required=False, default='2class_DR', help="The type of classification performed.")
    parser.add_argument('--augmentation', type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help="If data augmentation is performed.")

    args = parser.parse_args()

    # Hyperparameters and training settings
    num_classes = args.num_classes  #2
    num_epochs = args.epochs        #50
    # model_names = args.model_names  
    TASK= args.classification       #'2class_DR', '3class_DR', 'gender'
    augmentation = args.augmentation  #False
    learning_rate = 1e-5


    print('Model:', BACKBONE,'| Task:', TASK, '| epochs:', num_epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read csv file:
    df = get_dataset(csv_file_path, info=False)
    df = df[df[IMAGE_COL] != '985.1.jpg'] # this is missing
    df.head()

    # List all files in the image folder - splits .jpg extension
    image_files = os.listdir(image_folder_path)
    filtered_df = df[df[IMAGE_COL].isin(image_files)]
    # print(f' image_id without corresponding img: {len(df)-len(filtered_df)}')
    df= filtered_df

    # value = df.loc[df['file'] == '15.1.jpg', 'final_icdr'].values[0]
    # print(value)    
    # value = df.loc[df['final_icdr'] == 4.0, 'file'].values[0]
    # print(value)

    # Select Task Prediction:
    if TASK=='2class_DR': #Divides into 2 DR classes:
        LABEL = 'final_icdr'
        num_classes = 2
        df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x in [0,1] else ('DR'))
        df[LABEL] = df[LABEL].apply(lambda x: 0 if x in 'Normal' else (1))

    elif TASK=='3class_DR': #Divides into 3 DR classes:
        LABEL = 'final_icdr'
        num_classes=3
        df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
        df[LABEL] = df[LABEL].apply(lambda x: 0 if x == 'Normal' else (1 if x in 'Non-proliferative' else 2))

    elif TASK=='sex': #Gender Prediction:
        LABEL='sex'
        num_classes = 2
        df[LABEL]= df[LABEL].astype(int)
        # df[LABEL]= df[LABEL].apply(lambda x: '1' if x == 1 else '0' )


    elif TASK=='age':
        LABEL = 'age'
        #has a '>= 90' value
        df[LABEL] = df[LABEL].replace({'>= 90': 91})
        df = df.dropna(subset=[LABEL])
        df[LABEL] = df[LABEL].astype(int)


        # Bin age groups into two classes (e.g., 0: young, 1: old)
        bins = [0, 50, 100]  # Split point (here, 0-49 and 50+)
        bin_labels = [0, 1]
        df[LABEL] = pd.cut(df[LABEL], bins=bins, labels=bin_labels, right=False)
        df[LABEL] = df[LABEL].astype(int)

        print(df[LABEL].isna().sum())
        print(df[LABEL].unique())

 
    elif TASK=='educational_level':
        LABEL = 'educational_level'
        # 7 educational levels
        df = df.dropna(subset=[LABEL])
        df[LABEL] = df[LABEL].astype(int)      

        print(df[LABEL].isna().sum())
        print(df[LABEL].unique())

        # Differentiate only between literate (0) and illiterate (1)
        df[LABEL] = df[LABEL].apply(lambda x: 1 if x == 1 else 0 ).astype(int)

    elif TASK=='insurance':
        LABEL = 'insurance'

        df = df.dropna(subset=[LABEL])
        df[LABEL] = df[LABEL].astype(int)

        print(df[LABEL].isna().sum())
        print(df[LABEL].unique())

    elif TASK=='obesity':
        LABEL = 'obesity'

        df = df.dropna(subset=[LABEL])
        df[LABEL] = df[LABEL].astype(int)

        print(df[LABEL].isna().sum())
        print(df[LABEL].unique())


    label_counts = df[LABEL].value_counts()

#4932 images with labels:
#0: 3834
#1: 1038

#4660 images with good quality labels:
#0: 3812
#1: 848


    # Select only images of good quality:
    # Good quality images
    adequate_df = df[df['final_quality'] == 'yes']
    df = adequate_df


    # Split dataset into train, test and validation: 
    #Splitting with stratification and ensuring same patient images are on the same side!
    df_train, df_test = split_data_mbrset_images(df, LABEL, TEST_SIZE, undersample=False)
    df_train, df_val = split_data_mbrset_images(df_train, column=LABEL, split=1/8, undersample=False)
    # df_val = df_test


    # Process labels: Train the one hot encoder on the train set and get the labels for the test and validation sets:
    train_labels, mlb, train_columns = process_labels(df_train, col=LABEL)
 
    if augmentation:
    # Dataloaders
    # Define the target image shape
        train_transforms = transforms.Compose([
            transforms.Resize((224,224)),

            transforms.RandomRotation(degrees=20), 
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),

            transforms.ToTensor(),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((224,224)),

            # transforms.RandomCrop(SHAPE),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            # transforms.RandomRotation(50),  # Randomly rotate the image by up to 10 degrees
        ])


    if NORM_MEAN is not None and NORM_STD is not None:
        train_transforms.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

    test_transform = transforms.Compose([
        transforms.Resize(SHAPE),
        transforms.ToTensor(),
    ])

    if NORM_MEAN is not None and NORM_STD is not None:
        test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))


    # Create the custom dataset
    train_dataset = mBRSETDataset(
        df_train, 
        IMAGE_COL, 
        IMAGES, 
        LABEL, 
        mlb, 
        train_columns, 
        transform=train_transforms
    )

    test_dataset = mBRSETDataset(
        df_test, 
        IMAGE_COL, 
        IMAGES, 
        LABEL, 
        mlb, 
        train_columns, 
        transform=test_transform
    )

    val_dataset = mBRSETDataset(
        df_val, 
        IMAGE_COL, 
        IMAGES, 
        LABEL, 
        mlb, 
        train_columns, 
        transform=test_transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


    # Print 6 samples with their labels
    # Iterate through the DataLoader and plot the images with labels
    for batch in train_dataloader:

        images, labels = batch['image'], batch['labels']

        for i in range(len(images)):
            if i == 6:
                break
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))  # Permute to (H, W, C) from (C, H, W)
            plt.title(f"Label: {np.argmax(labels[i])}")
            plt.axis('off')
        plt.show()
        break


    # Model
    # Create a DataLoader to generate embeddings
    #model = get_retfound(weights='/scratch/liyues_root/liyues/chenweiw/retina_datasets/retfound_weigths/RETFound_cfp_weights.pth', num_classes=3)
    backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=MODE)
    model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
    model.to(device)

    # Use DataParallel to parallelize the model across multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, [0,1])
    # Training:
    if LOSS == 'focal_loss':
        # class_distribution = train_dataloader.dataset.labels.sum(axis=0)
        class_distribution= np.bincount(train_dataloader.dataset.labels.flatten())
        print(f'Class distribution: {class_distribution}')
        class_dis = np.array(class_distribution)
        class_weights =1-class_dis/np.sum(class_dis)

        # class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)

        weights = torch.tensor(class_weights).to(device)
        #criterion = FocalLoss()  # Focal Loss
        if num_classes==2:
            criterion = BinaryFocalLoss(gamma=2, alpha=weights)
        else:
            criterion = FocalLoss(gamma=2, alpha=weights)    
    else:
        # Assuming train_loader.dataset.labels is a one-hot representation
        class_indices = np.argmax(train_dataloader.dataset.labels, axis=1)

        # Compute class weights using class indices
        class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        #criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy Loss

    if OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    

    #Train
    model, train_losses, val_losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=num_epochs, save=True, num_classes=num_classes, device=device, backbone=f'{BACKBONE}_{num_classes}class_{LABEL}')


    # Save trained model:
    save_directory= os.path.join(parent_dir, 'mBRSET','Images',f'convnext_aug{augmentation}')
    os.makedirs(save_directory, exist_ok=True)
    model_path=  os.path.join(save_directory, f'mbrset_{BACKBONE}_{TASK}_{num_epochs}epochs_model.pkl')

    with open(model_path, 'wb') as f:
        # pickle.dump(model, f)
        torch.save(model.state_dict(), model_path)

    print(f"Model saved at: {model_path}")

    losses_path = os.path.join(save_directory, 'training_validation_losses.pkl')
    with open(losses_path, 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)
    print(f"Losses saved at: {losses_path}")

    plt.close()
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_directory, 'training_validation_losses.png'))  # Save the plot
    plt.show()

    # Test
    print("Complete Dataloader Results: ")
    test(model, test_dataloader, saliency=True, device=device, save=True, df_test=df_test, num_classes=num_classes, save_dir=save_directory, label=LABEL)

