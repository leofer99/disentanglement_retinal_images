import sys
import os
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
from transformers import ConvNextV2Model, ConvNextV2ForImageClassification,AutoModelForImageClassification

from src.get_dataset import get_dataset, split_data, plot_labels_distribution
from src.data_loader import mBRSETDataset, process_labels
# from src.RetFound import get_retfound
from src.FocalLoss import FocalLoss
from src.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from src.train import train
from src.test import test
from src_l.models import DenseNet121, VGG16



if __name__ == '__main__':

    # Constants:
    mbrset_dir1= r"C:\Users\HP\OneDrive - Universidade do Porto\Uni\Tese\mBRSET"
    mbrset_dir2= r"/nas-ctm01/datasets/public/MEDICAL/mbrset/physionet.org/files/mbrset/1.0"

    if os.path.exists(mbrset_dir2):
        DATASET = mbrset_dir2
    else:
        DATASET = mbrset_dir1

    DOWNLOAD = False
    IMAGES = os.path.join(DATASET, 'images')
    LABELS_PATH = os.path.join(DATASET, 'labels_mbrset.csv')
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

    BACKBONE = 'swinv2'
    MODE = 'fine_tune'
    backbone_mode = 'fine_tune'

    LABEL = 'final_icdr'
    num_classes = 2
    SHAPE = (224,224)
    TEST_SIZE = 0.2
    UNDERSAMPLE = False
    
    HIDDEN = [128]
    BATCH_SIZE = 2 #changed from 16
    NUM_WORKERS = 4

    LOSS = 'focal_loss'
    OPTIMIZER = 'adam'


    # Define your hyperparameters
    parser = argparse.ArgumentParser(description='conv_next_v2_large.py')
    parser.add_argument('--num_classes', type=int, default=2, required=False, help="Number of classes.")
    parser.add_argument('--epochs', type=int, default=50, required=False, help="Number of training epochs.")
    # parser.add_argument('--model_names', nargs='+', type=str, required=False, default=['vgg16'], help="Model to train.")
    parser.add_argument('--classification', type=str, required=False, default='2class_DR', help="The type of classification performed.")
    args = parser.parse_args()

    # Hyperparameters and training settings
    num_classes = args.num_classes  #2
    num_epochs = args.epochs        #50
    # model_names = args.model_names  
    TASK= args.classification       #'2class_DR', '3class_DR', 'gender'
    learning_rate = 1e-5


    print('Model:', BACKBONE,'| Task:', TASK, '| epochs:', num_epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read csv file:
    df = get_dataset(LABELS_PATH, info=False)
    df = df[df['file'] != '985.1.jpg'] # this is missing
    df.head()


    # Select Task Prediction:
    if TASK=='2class_DR': #Divides into 2 DR classes:
        LABEL = 'final_icdr'
        num_classes = 2
        df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x in [0,1] else ('DR'))
    
    elif TASK=='3class_DR': #Divides into 3 DR classes:
        LABEL = 'final_icdr'
        num_classes=3
        df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))

    elif TASK=='gender': #Gender Prediction:
        LABEL='sex'
        num_classes = 2
        df[LABEL]= df[LABEL].astype(int)
        df[LABEL]= df[LABEL].apply(lambda x: '1' if x == 1 else '0' )

    label_counts = df[LABEL].value_counts()


    #Cut Dataset - original size is (5163, 25)
    df=df[:64]


    # Select only images of good quality:
    # Good quality images
    adequate_df = df[df['final_quality'] == 'yes']
    df = adequate_df


    #Remove nephropathy and vascular disease:
    df = df.drop('nephropathy', axis=1)
    df = df.drop('vascular_disease', axis=1)


    # Split dataset into train, test and validation:
    df_train, df_test = split_data(df, LABEL, TEST_SIZE, undersample=False)
    print('Getting validation set...')
    df_val = df_test


    # Process labels: Train the one hot encoder on the train set and get the labels for the test and validation sets:
    train_labels, mlb, train_columns = process_labels(df_train, col=LABEL)


    # Dataloaders
    # Define the target image shape
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
    # Load the pre-trained ConvNeXt V2 model
    model_name = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Modify the classifier head according to the new number of classes
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)

    # Use DataParallel to parallelize the model across multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, [0,1])

    # Training:
    if LOSS == 'focal_loss':
        class_distribution = train_dataloader.dataset.labels.sum(axis=0)
        print(f'Class distribution: {class_distribution}')
        # class_dis = np.array(class_distribution)
        # class_weights =1-class_dis/np.sum(class_dis)
        # weights = torch.tensor(class_weights).to(device)
        # #criterion = FocalLoss()  # Focal Loss
        # criterion = FocalLoss(gamma=2, alpha=weights)

        #one hot encoder: each row in y_train represents the class membership. [0 1] belongs to class1
        #transform from one-hot encoded to 1D array
        y_train = np.argmax(train_dataset.labels, axis=1)

        class_counts, _ = np.unique(y_train, return_counts=True)
        weights = compute_class_weight(class_weight="balanced", classes=class_counts, y=y_train)
        class_weights = torch.from_numpy(weights)
        positive_weight = weights[1]/weights[0]
        weights=torch.tensor([1.0, positive_weight], dtype=torch.float64)
        weights.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(weight=weights[1])
        criterion.to(device)

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
    model= VGG16(
            channels=3, 
            height=224, 
            width=224, 
            nr_classes=num_classes
        )  
    # class_counts, _ = np.unique(y_train, return_counts=True)
    # weights = compute_class_weight(class_weight="balanced", classes=class_counts, y=y_train)
    # class_weights = torch.from_numpy(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,weight_decay=1e-8)
    # metrics_per_subset,mean_confusion_matrix= train_and_evaluate(model, train_loader, val_loader, test_loader, class_weights, optimizer, epochs, num_classes, model_name, analysis,fold,metrics_per_subset,mean_confusion_matrix)

    model = train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=num_epochs, save=True, device=device, backbone=f'{BACKBONE}_{num_classes}class_{LABEL}')


    # Test
    print("Complete Dataloader Results: ")
    test(model, test_dataloader, saliency=True, device=device, df_test=df_test)




