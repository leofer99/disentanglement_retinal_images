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

    
# NORM_MEAN =  [0.485, 0.456, 0.406]
# NORM_STD = [0.229, 0.224, 0.225]

def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean

def rescale_image_to_0_1(img): #from [-1, 1] to [0, 1]
    return (img + 1) / 2

# Loss Implementation
def compute_custom_loss_no_disen(inputs, outputs, targets, weights, cce=None, bce=None):
    ''' 
    loss_dis - disease prediction loss (medical features) - binary (0 or 1)
    loss_id - id recognition - multiclass (1 out of 115)
    loss_disc - fooling discriminator (GAN training)
    recon_loss - how close fake img is to original visually
    loss_feat_consistency - feature disentanglement - REMOVED
    '''

    # Unpack inputs
    img_real = inputs['real'] #real img
    img_fake = outputs['reconstructed'] #generated img
    img_fake_resized = F.interpolate(img_fake, size=(224, 224), mode='bilinear', align_corners=False)
    img_fake=img_fake_resized

    # Features from encoder
    feat_med, feat_id = outputs['features']

    # Classifier outputs
    dis_out = outputs['disease_out']
    id_out = outputs['identity_out']
    # disc_out = outputs['discriminator_out']

    target_dis = targets['disease'] # true disease label (0 or 1)
    target_id = targets['identity'] # pred label (class index)   (one-hot id_vector [0 0 1 0 ...])

    # Loss functions
    if bce is None:
        bce = nn.BCEWithLogitsLoss() #disease classification loss
    if cce is None:
        cce = nn.CrossEntropyLoss() #receives logits (id classification loss)
    mse = nn.MSELoss() #disentanglement loss 

    # disease classification loss (binary)
    loss_dis = bce(dis_out, target_dis)

    # id classification loss (multi-class)
    # print("target_id min:", target_id.min().item(), "max:", target_id.max().item(), "unique:", target_id.unique())
    # print("num_classes:", id_out.shape[1])
    
    
    loss_id = cce(id_out, target_id.squeeze().long()) # one-hot → class index


    # discriminator loss
    # loss_disc = bce(disc_out, torch.ones_like(disc_out))

    # # # Normalize images to compute pixel-level losses
    # NORM_MEAN =  [0.485, 0.456, 0.406]
    # NORM_STD = [0.229, 0.224, 0.225]
    # img_real = denormalize_image(img_real, mean=NORM_MEAN, std=NORM_STD)

    # print('max img_real', img_real.max())
    # print('min img_real', img_real.min())
    # print('max img_fake', img_fake.max())
    # print('min img_fake', img_fake.min())

    img_fake = rescale_image_to_0_1(img_fake)
    img_real = rescale_image_to_0_1(img_real)


    # Pixel-level losses (image reconstruction): need to be in range [0, 1]
    loss_psnr = (48 - TMF.peak_signal_noise_ratio(img_real, img_fake)) / 24
    loss_ssim = 1 - TMF.structural_similarity_index_measure(img_real, img_fake)
    recon_loss = loss_psnr + loss_ssim


    noisy_med, noisy_id = outputs['features_noisy']
    feat_med_rec, feat_id_rec = outputs['features_rec']
    feat_med_altered_med, feat_id_altered_med = outputs['feat_altered_med']
    feat_med_altered_id, feat_id_altered_id = outputs['feat_altered_id']
    # feat_med_altered_oth, feat_id_altered_oth, feat_oth_altered_oth = outputs['feat_altered_oth']

    # Disentanglement loss (Feature consistency)
    loss_disentanglement = (
        # Altered med
        mse(feat_med_rec, feat_med_altered_med) +
        mse(noisy_id, feat_id_altered_med) +
        # mse(noisy_oth, feat_oth_altered_med) + 
        # Altered id
        mse(noisy_med, feat_med_altered_id) +
        mse(feat_id_rec, feat_id_altered_id) 
        # mse(noisy_oth, feat_oth_altered_id) + 
        # Altered other
        # mse(noisy_med, feat_med_altered_oth) +
        # mse(noisy_id, feat_id_altered_oth) +
        # mse(feat_oth_rec, feat_oth_altered_oth)  
    )


# # Originally:
    # total_loss = (
    #     loss_dis +
    #     loss_id +
    #     weights['disc_weight'] * (0.1 * loss_disc + recon_loss) +
    #     5 * loss_feat_consistency
    # )

    total_loss = (
        loss_dis +
        loss_id +
        weights['disc_weight'] * (recon_loss) 
    )

    loss= {'loss_dis': loss_dis,    
                'loss_id': loss_id, 
                # 'loss_disc': loss_disc, 
                'recon_loss': recon_loss, 
                'loss_disentanglement': torch.tensor(0.0)}


    return total_loss, loss
