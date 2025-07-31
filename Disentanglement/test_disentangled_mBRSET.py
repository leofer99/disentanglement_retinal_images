import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import pandas as pd
import argparse

# from captum.attr import LayerGradCam, visualize_image_attr

# from torchcam.methods import GradCAM
# from torchcam.utils import overlay_mask
# from torchvision.transforms.functional import to_pil_image

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))

from disentanglement_network_mBRSET import mbrset_data_loading
from src_disentanglement.init_train_eval import initialize_models, load_model, evaluate_model, evaluate_identity_model
from src_disentanglement.image_utils import plot_generated_images, rescale_image_to_0_1, rescale_image_to_minus1_1


def compute_saliency_map(image, label, disease_classifier):
    # Set the model to evaluation mode
    disease_classifier.eval()

    # Ensure the image requires gradients

    # Forward pass through the disease classifier
    m_feat, i_feat, o_feat = feature_extractor(image)
    m_feat.requires_grad_()

    output = disease_classifier(m_feat)

    # Compute the loss for the target label
    target_score = output  # Assume label is the index of the target class
    target_score.backward()  # Compute gradients

    # Get the gradient of the input image
    saliency = image.grad.data.abs().squeeze().cpu().numpy()

    # Normalize the saliency map
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency

# def plot_generated_images_with_gradcam(epoch):
#     n = 2  # Number of images to view

#     # Initialize Grad-CAM
#     # model.0: first Linear layers as target
#     # model.2: final Linear layer as target

#     cam_extractor = GradCAM(disease_classifier, target_layer="model.0")  

#     for batch in train_dataloader:
#         images, labels = batch['image'], batch['label_dis']

#         # Select one image and its corresponding label
#         image = images[0:1].to(device)  # Select the first image and move it to the device
#         label = int(labels[0].item())  # Get the corresponding label as an integer

#         # Forward pass through the disease classifier
#         m_feat, i_feat, o_feat = feature_extractor(image)
#         output = disease_classifier(m_feat)
#         output.requires_grad_(True)


#         output = output.squeeze(1)  # Shape: (1,)
#         # output = output.unsqueeze(0)  # Shape: (1, num_classes), num_classes = 1 for binary classification

#         # Generate Grad-CAM for the target class
#         activation_map = cam_extractor(class_idx=label, scores=output, retain_graph=True)

#         # Overlay the Grad-CAM heatmap on the original image
#         result = overlay_mask(
#             to_pil_image(image.squeeze().cpu()),
#             to_pil_image(activation_map[0].squeeze(0), mode="F"),
#             alpha=0.5
#         )

#         # Visualize the Grad-CAM heatmap
#         plt.figure(figsize=(6, 6))
#         plt.imshow(result)
#         plt.title(f"Grad-CAM for Epoch {epoch}")
#         plt.axis("off")
#         plt.show()

#         break  # Process only the first batch





if __name__ == '__main__':

    # ============ Configuration ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Define your hyperparameters
    parser = argparse.ArgumentParser(description='disentanglement_network_mBRSET.py')
    parser.add_argument('--latent_dim', type=int, default=256, required=False, help="Latent Dimensions.")
    parser.add_argument('--batch_size', type=int, default=32, required=False, help="Batch size.")
    parser.add_argument('--num_classes', type=int, default=2, required=False, help="Number of classes.")
    parser.add_argument('--folder_name', type=str, default='00', required=False, help="Folder name.")
    parser.add_argument('--model_epoch', type=int, default=100, required=False, help="Number of epochs.")
    parser.add_argument('--model', type=str, required=False, default='dinov', help="Model to train.")
    args = parser.parse_args()

    # latent_dim = 256  # Hyperparameter
    # batch_size = 32
    # num_classes = 2  # Number of identity classes
    # folder_name = 'id_insurance_dinov_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse'
    # model_epoch = 80  # Specify the epoch of the model to load
    # model= 'dinov'  # 'normal', 'convnext', 'swin', 'dinov'

    latent_dim = args.latent_dim
    batch_size = args.batch_size
    num_classes = args.num_classes
    folder_name = args.folder_name
    model_epoch = args.model_epoch
    model = args.model


    # Paths
    save_dir = os.path.join(parent_dir, 'Disentanglement', 'Models (mBRSET)', folder_name)
    model_path = os.path.join(save_dir, f"model_epoch_{model_epoch}.pt")


    # ============ Load Data ============ #
    dataloader_save_path = os.path.join(save_dir, "dataloaders.pkl")
    df_train_save_path = os.path.join(save_dir, "df_train.csv")
    # df_val_save_path = os.path.join(save_dir, "df_val.csv")
    df_test_save_path = os.path.join(save_dir, "df_test.csv")


    # Load dataloaders
    with open(dataloader_save_path, "rb") as f:
        dataloaders = pickle.load(f)

    train_dataloader = dataloaders["train_dataloader"]
    # val_dataloader = dataloaders["val_dataloader"]
    test_dataloader = dataloaders["test_dataloader"]

    print("Dataloaders loaded successfully.")

    if os.path.exists(df_train_save_path):
        df_train = pd.read_csv(df_train_save_path)
        # df_val = pd.read_csv(df_val_save_path)
        df_test = pd.read_csv(df_test_save_path)
    else:
        df_train = None
        # df_val = None
        df_test = None

    # ============ Initialize Models ============ #
    models = initialize_models(latent_dim, num_classes, device, model)

    # ============ Load Saved Model ============ #
    load_model(model_path, device, models)

    # ============ Plot Generated Images ============ #
    # print("Plotting generated images...")
    # plot_generated_images(epoch=model_epoch, train_dataloader=train_dataloader, models=models, device=device)
    

    # Evaluation
    print("Train - Model: ")
    evaluate_model(models=models, dataloader=train_dataloader, device=device, df_train=df_train, save_path=save_dir)
    
    # print("Val Dataloader Results: ")
    # evaluate_model(models=models, dataloader=val_dataloader, device=device, df_val=df_val, save_path=save_dir)
    
    print("Test - Model: ")
    evaluate_model(models=models, dataloader=test_dataloader, device=device, df_test=df_test, save_path=save_dir)


    # Gender Evaluation
    print("Train - Switch vectors Eval: ")
    evaluate_identity_model(models=models, dataloader=train_dataloader, device=device, df_train=df_train, save_path=save_dir)
    
    print("Test - Switch vectors Eval: ")
    evaluate_identity_model(models=models, dataloader=test_dataloader, device=device, df_test=df_test, save_path=save_dir)


    # plot_generated_images_with_gradcam(epoch=model_epoch)


    # image, label = next(iter(train_dataloader))  # Get a batch of images and labels
    # image = image[0:1].to(device)  # Select one image and move to the device
    # label = label[0].item()  # Get the corresponding label
    # saliency_map = compute_saliency_map(image, label, disease_classifier)

    # # Visualize the saliency map
    # plt.imshow(saliency_map, cmap='hot')
    # plt.colorbar()
    # plt.title("Saliency Map")
    # plt.show()
