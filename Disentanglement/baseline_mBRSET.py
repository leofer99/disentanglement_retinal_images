import os
import sys
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TMF
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# get 0_folder
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))


from src_image_luis.get_dataset import get_dataset, split_data_mbrset_images
from src_image_luis.data_loader import mBRSETDataset, process_labels

from src_disentanglement.model import FeatureExtractor, Generator, DiseaseClassifier, IdentityClassifier
from src_disentanglement.loss import compute_custom_loss
from src_disentanglement.init_train_eval import initialize_models, setup_optimizers, train_one_epoch, train_one_epoch_baseline, evaluate_model, evaluate_model_val
from src_disentanglement.image_utils import plot_generated_images, rescale_image_to_minus1_1, rescale_image_to_0_1


# 0. Data Loading
def mbrset_data_loading(id_info, batch_size, num_epochs, num_classes_id, save_dir, df_size='all', augmentation='False'):

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

    BACKBONE='disentanglement'
    TASK= '2class_DR'
    num_epochs=num_epochs
    IMAGE_COL='file'
    TEST_SIZE= 0.2

    
    # NORM_MEAN =  [0.485, 0.456, 0.406]
    # NORM_STD = [0.229, 0.224, 0.225]
    NORM_MEAN = None
    NORM_STD = None

    SHAPE = (224,224)
    BATCH_SIZE = batch_size
    NUM_WORKERS = 4

    print('mBRSET - Model:', BACKBONE,'| Task:', TASK, '| ID info:', id_info, '| Epochs:', num_epochs, '| df_size:', df_size)

    # Read csv file:
    df = get_dataset(csv_file_path, info=False)
    df = df[df[IMAGE_COL] != '985.1.jpg'] # this is missing

    # List all files in the image folder - splits .jpg extension
    image_files = os.listdir(image_folder_path)
    filtered_df = df[df[IMAGE_COL].isin(image_files)]
    # print(f' image_id without corresponding img: {len(df)-len(filtered_df)}')
    df= filtered_df

    if df_size != 'all':
        df=df[:df_size]


    # Select Task Prediction:
    if TASK=='2class_DR': #Divides into 2 DR classes:
        LABEL = 'final_icdr'
        # num_classes = 2
        df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x in [0,1] else ('DR'))
        df[LABEL] = df[LABEL].apply(lambda x: 0 if x in 'Normal' else (1)).astype(int)

    elif TASK=='3class_DR': #Divides into 3 DR classes:
        LABEL = 'final_icdr'
        # num_classes=3
        df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
        df[LABEL] = df[LABEL].apply(lambda x: 0 if x == 'Normal' else (1 if x in 'Non-proliferative' else 2)).astype(int)


    # Preprocess id_info:
    df[id_info] = df[id_info].dropna()

    
    if id_info=='age':
        #has a '>= 90' value
        df[id_info] = df[id_info].replace({'>= 90': 91})
        df[id_info]= df[id_info].astype(int)

        # # Bin age groups
        # bins= [0, 20, 40, 60, 80, 100]
        # bin_labels= [0, 1, 2, 3, 4] #0-20 labeled 0, and so on
        # df[id_info] = pd.cut(df[id_info], bins=bins, labels=bin_labels, right=False)
        # df[id_info] = df[id_info].astype(int)

        # Bin age groups into two classes (e.g., 0: young, 1: old)
        bins = [0, 50, 100]  # Split point (here, 0-49 and 50+)
        bin_labels = [0, 1]
        df[id_info] = pd.cut(df[id_info], bins=bins, labels=bin_labels, right=False)
        df[id_info] = df[id_info].astype(int)

        print(df[id_info].isna().sum())
        print(df[id_info].unique())

        # # Convert to one-hot encoding
        # one_hot = pd.get_dummies(df[id_info]).astype(int)
        # print(one_hot)
        # df[id_info] = one_hot 

    elif id_info=='sex': # 0=Female, 1=Male
        df[id_info]= df[id_info].astype(int)

        print(df[id_info].isna().sum())
        print(df[id_info].unique())
 
    elif id_info=='educational_level':
        # 7 educational levels
        df = df.dropna(subset=[id_info])
        df[id_info] = df[id_info].astype(int)      

        print(df[id_info].isna().sum())
        print(df[id_info].unique())

        # Differentiate only between literate (0) and illiterate (1)
        df[id_info] = df[id_info].apply(lambda x: 1 if x == 1 else 0 ).astype(int)

    elif id_info=='insurance':
        # df[id_info] = df[id_info].dropna().astype(int)

        
        df = df.dropna(subset=[id_info])
        df[id_info] = df[id_info].astype(int)

        print(df[id_info].isna().sum())
        print(df[id_info].unique())


    # Insurance


    label_counts = df[LABEL].value_counts()

    # Select only images of good quality:
    adequate_df = df[df['final_quality'] == 'yes']
    df = adequate_df

    # Split dataset into train, test and validation: 
    #Splitting with stratification and ensuring same patient images are on the same side!
    df_train, df_test = split_data_mbrset_images(df, LABEL, TEST_SIZE, undersample=False)
    df_train, df_val = split_data_mbrset_images(df_train, column=LABEL, split=1/8, undersample=False)

    # Process labels: Train the one hot encoder on the train set and get the labels for the test and validation sets:
    train_labels, mlb_dis, train_columns_dis = process_labels(df_train, col=LABEL)
    # train_labels_id, mlb_id, train_columns_id = process_labels(df_train, col=id_info)

    if augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize((224,224)),

            transforms.RandomRotation(degrees=20), 
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
         
            # transforms.RandomCrop(SHAPE),
            transforms.ToTensor(),  # Converts image to [0, 1]
            transforms.Lambda(rescale_image_to_minus1_1),  # Rescale [0, 1] to [-1, 1]

        ])
    else:
            train_transforms = transforms.Compose([
            transforms.Resize((224,224)),

            transforms.ToTensor(),  # Converts image to [0, 1]
            transforms.Lambda(rescale_image_to_minus1_1),  # Rescale [0, 1] to [-1, 1]

            # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            # transforms.RandomRotation(50),  # Randomly rotate the image by up to 10 degrees
        ])


    if NORM_MEAN is not None and NORM_STD is not None:
        train_transforms.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

    test_transform = transforms.Compose([
        transforms.Resize(SHAPE),
        transforms.ToTensor(),  # Converts image to [0, 1]
        transforms.Lambda(rescale_image_to_minus1_1),  # Rescale [0, 1] to [-1, 1]
    ])

    if NORM_MEAN is not None and NORM_STD is not None:
        test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))


    # Create the custom dataset
    train_dataset = mBRSETDataset(
        df_train, 
        IMAGE_COL, 
        IMAGES, 
        LABEL, 
        mlb=mlb_dis, 
        train_columns=train_columns_dis, 
        transform=train_transforms, 
        disentangled=True,
        id_info=id_info,
        augmentation=augmentation
    )

    test_dataset = mBRSETDataset(
        df_test, 
        IMAGE_COL, 
        IMAGES, 
        LABEL, 
        mlb=mlb_dis, 
        train_columns=train_columns_dis, 
        transform=test_transform,
        disentangled=True,
        id_info=id_info
        # no augmentation
    )

    val_dataset = mBRSETDataset(
        df_val, 
        IMAGE_COL, 
        IMAGES, 
        LABEL, 
        mlb=mlb_dis, 
        train_columns=train_columns_dis, 
        transform=test_transform,
        disentangled=True,
        id_info=id_info
        # no augmentation
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


    # Print 6 samples with their labels
    # Iterate through the DataLoader and plot the images with labels
    for batch in train_dataloader:

        images, labels = batch['image'], batch['label_dis']

        for i in range(len(images)):
            if i == 6:
                break
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))  # Permute to (H, W, C) from (C, H, W)
            plt.title(f"Label: {np.argmax(labels[i])}")
            plt.axis('off')
        plt.show()
        break

    #Save df_train, df_val, df_test
    df_train.to_csv(os.path.join(save_dir, 'df_train.csv'), index=False)
    df_val.to_csv(os.path.join(save_dir, 'df_val.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir, 'df_test.csv'), index=False)

    return train_dataloader, val_dataloader, test_dataloader




if __name__ == '__main__':

    torch.cuda.empty_cache()
        
    # ============ Setup ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Define your hyperparameters
    parser = argparse.ArgumentParser(description='baseline_mBRSET.py')
    parser.add_argument('--num_classes', type=int, default=2, required=False, help="Number of classes.")
    parser.add_argument('--latent_dim', type=int, default=256, required=False, help="Latent Dimensions.")
    parser.add_argument('--batch_size', type=int, default=32, required=False, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=2, required=False, help="Number of training epochs.")
    parser.add_argument('--model', type=str, required=False, default='dinov', help="Model to train.")
    parser.add_argument('--id_info', type=str, required=False, default='sex', help="Attribute to be disentangled.")
    parser.add_argument('--df_size', type=str, default='250', required=False, help="Size of the dataset.")
    parser.add_argument('--learning_rate_gen', type=float, default=5e-5, required=False, help="Generator learning rate.")
    # parser.add_argument('--learning_rate_disc', type=float, default=5e-7, required=False, help="Discriminator learning rate.")
    parser.add_argument('--disentanglement', type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help="If disentanglement is performed.")
    parser.add_argument('--augmentation', type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help="If data augmentation is performed.")
    args = parser.parse_args()

    # Hyperparameters and training settings 
    latent_dim = args.latent_dim  #256
    batch_size = args.batch_size  #32
    num_epochs = args.epochs        #100

    if args.df_size.lower() == 'all':
        df_size = 'all'
    else:
        df_size = int(args.df_size)

    # df_size= args.df_size  # 'all' or int
    learning_rate_gen = args.learning_rate_gen  #5e-5
    # learning_rate_disc = args.learning_rate_disc  #5e-7

    decay = 1e-6
    id_info= args.id_info   #'age', 'sex', 'educational_level', 'insurance'
    num_classes = args.num_classes  #2
    model = args.model  # 'normal', 'convnext', 'swin', 'dinov'

    disentanglement = args.disentanglement  #True
    augmentation = args.augmentation  #True


    # Path to be created
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'no_job_id')

    #BALANCED LOSSES
    run_name = f"baseline_aug{augmentation}_{model}_id_{str(id_info)}_Disen{disentanglement}_ld{latent_dim}_bs{batch_size}_lrG{learning_rate_gen}_DF{df_size}"
    save_dir= os.path.join(parent_dir, 'Disentanglement', 'Models (mBRSET)', run_name)
    os.makedirs(save_dir, exist_ok=True)

    print('LR_generator:', learning_rate_gen, '| model: ', model, '| disentanglement:', disentanglement)

    # ============ Data Loading ============ #
    train_dataloader, val_dataloader, test_dataloader = mbrset_data_loading(id_info=id_info, batch_size=batch_size, num_epochs=num_epochs, num_classes_id=num_classes, save_dir=save_dir, df_size=df_size, augmentation=augmentation)


    # ---- BCEWithLogitsLoss for disease (binary) ----
    # Get all disease labels from the training set
    disease_labels = []
    for batch in train_dataloader:
        disease_labels.extend(batch['label_dis'].cpu().numpy())
    disease_labels = np.array(disease_labels).flatten()
    num_0 = np.sum(disease_labels == 0)
    num_1 = np.sum(disease_labels == 1)
    pos_weight = torch.tensor([num_0 / num_1], dtype=torch.float32).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- CrossEntropyLoss for identity (multi-class) ----
    identity_labels = []
    for batch in train_dataloader:
        identity_labels.extend(batch['label_id'].cpu().numpy())
    identity_labels = np.array(identity_labels).flatten()
    class_weights = compute_class_weight('balanced', classes=np.unique(identity_labels), y=identity_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    cce = nn.CrossEntropyLoss(weight=class_weights)



    # Save dataloaders
    dataloader_save_path = os.path.join(save_dir, "dataloaders.pkl")
    with open(dataloader_save_path, "wb") as f:
        pickle.dump({
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader
        }, f)
    print(f"Dataloaders saved to {dataloader_save_path}")


    # Model Initialization
    models = initialize_models(latent_dim, num_classes, device, model)

    # Optimizer Setup
    opt = setup_optimizers(models, learning_rate_gen, decay)

    # Unpack models
    feature_extractor, generator, disease_classifier, identity_classifier = models

    # Initialize lists to store training losses
    epoch_total_losses = []
    # disc_real_losses = []
    # disc_fake_losses = []
    epoch_loss_recon = []
    epoch_loss_dis = []
    epoch_loss_id = []
    epoch_loss_disen = []

    # Initialize lists to store validation losses
    val_total_losses = []
    val_loss_recon = []
    val_loss_dis = []
    val_loss_id = []
    val_loss_disen = []


    # Training Loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # total_loss, loss_recon, loss_disen, loss_dis, loss_id = train_one_epoch(models=models, dataloader=train_dataloader, opt=opt, device=device, disentanglement=disentanglement)        # epoch_loss, disc_real_loss, disc_fake_loss, loss_dis, loss_id, loss_disen = train_one_epoch(models, train_dataloader, opt, disc_opt, device)
        total_loss, loss_recon, loss_disen, loss_dis, loss_id = train_one_epoch_baseline(models=models, dataloader=train_dataloader, opt=opt, device=device, disentanglement=disentanglement)        # epoch_loss, disc_real_loss, disc_fake_loss, loss_dis, loss_id, loss_disen = train_one_epoch(models, train_dataloader, opt, disc_opt, device)

        # Store training losses
        epoch_total_losses.append(np.mean(total_loss))
        # disc_real_losses.append(np.mean(disc_real_loss))
        # disc_fake_losses.append(np.mean(disc_fake_loss))
        epoch_loss_recon.append(np.mean(loss_recon))
        epoch_loss_dis.append(np.mean(loss_dis))
        epoch_loss_id.append(np.mean(loss_id))
        epoch_loss_disen.append(np.mean(loss_disen))

        # Compute validation losses
        val_loss, val_recon, val_dis, val_id, val_disen = evaluate_model_val(models=models, dataloader=val_dataloader, device=device, disentanglement=disentanglement)        
        val_total_losses.append(val_loss)
        # val_loss_recon.append(val_recon)
        val_loss_dis.append(val_dis)
        val_loss_id.append(val_id)
        # val_loss_disen.append(val_disen)


        # if epoch in [1, 5, 10, 20, 30, 40, 50, 100, 150, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            # plot_generated_images(epoch, train_dataloader, models, device)

        # ---- Save/Validate every N num_epochs ---- #
        if epoch in [num_epochs, 80, 100, 120]:
        # if epoch % 20 == 0:  #epoch%50
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                # 'generator': generator.state_dict(),
                # 'discriminator': discriminator.state_dict(),
                'disease_classifier': disease_classifier.state_dict(),
                'identity_classifier': identity_classifier.state_dict(),
            }, os.path.join(save_dir, f"model_epoch_{epoch}.pt"))

            # Save the loss variables
            losses = {
                'epoch_losses': epoch_total_losses,
                # 'disc_real_losses': disc_real_losses,
                # 'disc_fake_losses': disc_fake_losses,
                # 'epoch_loss_recon': epoch_loss_recon,
                'epoch_loss_dis': epoch_loss_dis,
                'epoch_loss_id': epoch_loss_id,
                # 'epoch_loss_disen': epoch_loss_disen,

                'val_losses': val_total_losses,
                # 'val_loss_recon': epoch_loss_recon,
                'val_loss_dis': val_loss_dis,
                'val_loss_id': val_loss_id,
                # 'val_loss_disen': val_loss_disen,
            }
            
            losses_save_path = os.path.join(save_dir, 'losses.pkl')
            with open(losses_save_path, 'wb') as f:
                pickle.dump(losses, f)
            print(f"Loss variables saved to {losses_save_path}")





    # Save the loss variables
    losses = {
        'epoch_losses': epoch_total_losses,
        # 'disc_real_losses': disc_real_losses,
        # 'disc_fake_losses': disc_fake_losses,
        # 'epoch_loss_recon': epoch_loss_recon,
        'epoch_loss_dis': epoch_loss_dis,
        'epoch_loss_id': epoch_loss_id,
        # 'epoch_loss_disen': epoch_loss_disen,
        
        'val_losses': val_total_losses,
        # 'val_loss_recon': val_loss_recon,
        'val_loss_dis': val_loss_dis,
        'val_loss_id': val_loss_id,
        # 'val_loss_disen': val_loss_disen,
    }

    losses_save_path = os.path.join(save_dir, 'losses.pkl')
    with open(losses_save_path, 'wb') as f:
        pickle.dump(losses, f)
    print(f"Loss variables saved to {losses_save_path}")

    # Plot the losses after training
    plt.figure(figsize=(12, 8))
    # plt.plot(epoch_total_losses, label='Generator Total Loss')
    # plt.plot(disc_real_losses, label='Discriminator Real Loss')
    # plt.plot(disc_fake_losses, label='Discriminator Fake Loss')
    # plt.plot(epoch_loss_recon, label='Reconstruction Loss')
    plt.plot(epoch_loss_dis, label='Disease Loss')
    plt.plot(epoch_loss_id, label='Identity Loss')
    # plt.plot(epoch_loss_disen, label='Disentanglement Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))  # Save the plot
    plt.show()

   # Plot the validation losses 
    plt.figure(figsize=(12, 8))
    # plt.plot(val_total_losses, label='Validation Generator Total Loss')
    # plt.plot(disc_real_losses, label='Discriminator Real Loss')
    # plt.plot(disc_fake_losses, label='Discriminator Fake Loss')
    # plt.plot(val_loss_recon, label='Validation Reconstruction Loss')
    plt.plot(val_loss_dis, label='Validation Disease Loss')
    plt.plot(val_loss_id, label='Validation Identity Loss')
    # plt.plot(val_loss_disen, label='Validation Disentanglement Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Losses Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'validation_losses.png'))  # Save the plot
    plt.show()

    # Plot the training vs validation losses
    plt.figure(figsize=(12, 8))
    # plt.plot(epoch_total_losses, label='Training Generator Total Loss')
    # plt.plot(val_total_losses, label='Validation Generator Total Loss')

    # plt.plot(epoch_loss_recon, label='Training Reconstruction Loss')
    # plt.plot(val_loss_recon, label='Validation Reconstruction Loss')
    plt.plot(epoch_loss_dis, label='Training Disease Loss')
    plt.plot(val_loss_dis, label='Validation Disease Loss')
    plt.plot(epoch_loss_id, label='Training Identity Loss')
    plt.plot(val_loss_id, label='Validation Identity Loss')
    # plt.plot(epoch_loss_disen, label='Training Disentanglement Loss')
    # plt.plot(val_loss_disen, label='Validation Disentanglement Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'training_validation_losses.png'))  # Save the plot
    plt.show()

    # Evaluation
    evaluate_model(models, test_dataloader, device)
