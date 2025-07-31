import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def rescale_image_to_0_1(img):
    return (img + 1) / 2

def rescale_image_to_minus1_1(img):
    return img * 2 - 1

def plot_generated_images(epoch, train_dataloader, models, device):

    # Unpack models
    feature_extractor, generator, _, _ = models

    cols= 5
    rows = 2 # number of images to view


    for batch in train_dataloader:

        images, labels = batch['image'], batch['label_dis']

        # Select one image and its corresponding label
        image = images[0:1].to(device)  # Select the first image and move it to the device
        label = labels[0].item()  # Get the corresponding label as an integer

        # saliency_map = compute_saliency_map(image, label, disease_classifier)

        # # Visualize the saliency map
        # plt.imshow(saliency_map, cmap='hot')
        # plt.colorbar()
        # plt.title("Saliency Map")
        # plt.show()

        # Pass images through the feature extractor
        m_feat, i_feat = feature_extractor(images.to(device))

        # Generate blank vector X (same shape as features, filled with zeros)
        X = torch.zeros_like(m_feat)

        generated_images_rec_X = generator(m_feat, i_feat)  # Only med and id info
        generated_images_med_X = generator(m_feat, X)
        generated_images_id_X = generator(X, i_feat)  # Only id and other info
        # generated_images_oth_X = generator(X, X, o_feat)  # Only med and other info


        titles = ["Original", "Reconstructed", "Medical", "ID"]
        image_sets = [images, generated_images_rec_X, generated_images_med_X, generated_images_id_X]
        plt.figure(figsize=(cols*4, rows*5))  # Adjust the figure size for a 1x5 grid
        
        for i in range(rows):
            for j, (title, image_set) in enumerate(zip(titles, image_sets)):
                
                # Calculate subplot index
                ax = plt.subplot(rows, cols, i * cols + j + 1)

                # Extract and process the image
                img = image_set[i].detach().cpu().permute(1, 2, 0).numpy()
                min_value = np.min(img)
                max_value = np.max(img)
                img = (img - min_value) / (max_value - min_value) * 255
                img = np.clip(img, 0, 255).astype(np.uint8)

                # Display the image
                plt.imshow(img)
                ax.set_title(title)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


    # Path to be created
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'no_job_id')

    img_dir= os.path.join(parent_dir, 'Disentanglement', 'Images mBRSET', slurm_job_id)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"generated_img_epoch_{epoch}.png")
    plt.savefig(img_path)
    plt.close()

