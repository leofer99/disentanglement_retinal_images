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
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification, AutoModelForImageClassification

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from src_image_luis.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from src_disentanglement.image_utils import rescale_image_to_0_1

# 1. ResBlock
class ResBlock(nn.Module):
    ''' 
Input image goes through layers and learns adjustments (residual path)
Input image is copied and bypasses all layers (skip connection path)
The og image and the learned adjustments are combined together
    '''
    def __init__(self, in_channels, out_channels, resample=None):
        ''' 
        in_channels: 
        out_channels: 
        resample: upsample, downsample or do nothing
        '''
        super().__init__()
        self.resample = resample
        # stride = 1 if resample in [None, 'upsample'] else 2 # if downsampling

        # # Shortcut path 
        # if resample is None and in_channels == out_channels:
        #     self.shortcut = nn.Sequential() # empty seq is identity function

        # #use conv if it has diff size (upsampling/downsampling) or in_channels != out_channels
        # else: 
        #     self.shortcut = nn.Sequential(
        #         nn.Upsample(scale_factor=2) if resample == 'upsample' else nn.Identity(),
        #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #match output channels
        #         nn.BatchNorm2d(out_channels) # stabilize
        #     )

        # # Main path
        # layers = []
        # if resample == 'upsample':
        #     layers.append(nn.Upsample(scale_factor=2))

        # layers += [
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        # ]
        # self.main = nn.Sequential(*layers)



        if resample == 'upsample':
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )

        elif resample == 'downsample':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )

        else:  # no resampling
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )

            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        # add adjustments + og image, then activates
        return F.leaky_relu(self.main(x) + self.shortcut(x), 0.2)



# 2. Feature Extractor
class FeatureExtractor(nn.Module):
    ''' From input image, extracts 3 distinct ft vectors:
    Medical features, identity features, other features
    '''
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()

        ####
        self.conv1 = nn.Conv2d(input_channels, latent_dim // 4, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # res blocks - downsampling to reduce spatial dims and increase feature depth 
        self.res_blocks = nn.Sequential(
            ResBlock(latent_dim // 4, latent_dim // 2, 'downsample'),
            ResBlock(latent_dim // 2, latent_dim // 2),
            ResBlock(latent_dim // 2, latent_dim, 'downsample'),
            ResBlock(latent_dim, latent_dim),
            ResBlock(latent_dim, latent_dim * 2, 'downsample'),
            ResBlock(latent_dim * 2, latent_dim * 2)
        )
        ####


        #reduces each feature map to a single value (1d ft vector: (batch_size, channels, 1, 1)) 
        self.global_pool = nn.AdaptiveAvgPool2d(1) 
        #uses linear layers to produce the three distinct feature vectors 
        self.medical_fc = nn.Linear(latent_dim * 2, latent_dim) 
        self.identity_fc = nn.Linear(latent_dim * 2, latent_dim)
        # self.other_fc = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.res_blocks(x)
        x = self.global_pool(x).view(x.size(0), -1) #flattens to (batch_size, channels) from (batch_size, channels, 1, 1) 
        
        return self.medical_fc(x), self.identity_fc(x) #, self.other_fc(x)


class FeatureExtractor_convnext(nn.Module):
    ''' From input image, extracts 3 distinct ft vectors:
    Medical features, identity features, other features
    '''
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()

        ####
        BACKBONE = 'convnextv2_large'
        MODE = 'fine_tune'
        backbone_mode = 'fine_tune'
        HIDDEN = [128]
        num_classes=2
        backbone_name="facebook/convnextv2-large-22k-224"

        self.preprocessor = AutoImageProcessor.from_pretrained(backbone_name, do_rescale=False)
        self.backbone = ConvNextV2ForImageClassification.from_pretrained(backbone_name, output_hidden_states=True)
        in_features = self.backbone.classifier.in_features
        # self.backbone = nn.Sequential(*list(backbone_full.children())[:-1]) # Removes the classifier layer

        # self.backbone = FoundationalCVModel(backbone=BACKBONE, mode='feature_extractor')
        # model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
    
        
        ####
        #reduces each feature map to a single value (1d ft vector: (batch_size, channels, 1, 1)) 
        self.global_pool = nn.AdaptiveAvgPool2d(1) 

        #uses linear layers to produce the three distinct feature vectors 
        self.medical_fc = nn.Linear(in_features, latent_dim)
        self.identity_fc = nn.Linear(in_features, latent_dim)
        # self.other_fc = nn.Linear(in_features, latent_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x):


        # Rescale images from [-1, 1] to [0, 1]
        x = rescale_image_to_0_1(x)

        # Preprocess the input image
        inputs = self.preprocessor(images=x, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(next(self.backbone.parameters()).device)


        # Extract features using the ConvNeXt backbone
        with torch.no_grad():
            outputs = self.backbone(pixel_values)  # Pass the entire dictionary of inputs
            features = outputs.hidden_states[-1]  # Use the last hidden state (without the classifier layer)

        # Global average pooling to reduce spatial dimensions
        x = self.global_pool(features).view(features.size(0), -1) 

        # Generate the three distinct feature vectors
        medical_features = self.leaky_relu(self.medical_fc(x))
        identity_features = self.leaky_relu(self.identity_fc(x))
        # other_features = self.leaky_relu(self.other_fc(x))


        return medical_features, identity_features #, other_features


class FeatureExtractor_swin(nn.Module):
    ''' From input image, extracts 3 distinct ft vectors:
    Medical features, identity features, other features
    '''
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()

        ####
        BACKBONE = 'swinv2'
        MODE = 'fine_tune'
        backbone_mode = 'fine_tune'
        HIDDEN = [128]
        num_classes=2
        backbone_name="microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"

        self.preprocessor = AutoImageProcessor.from_pretrained(backbone_name, do_rescale=False)
        self.backbone =  AutoModelForImageClassification.from_pretrained(backbone_name, output_hidden_states=True)
        in_features = self.backbone.classifier.in_features  
        # self.backbone = nn.Sequential(*list(backbone_full.children())[:-1]) # Removes the classifier layer

        # self.backbone = FoundationalCVModel(backbone=BACKBONE, mode='feature_extractor')
        # model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
    

        #reduces each feature map to a single value (1d ft vector: (batch_size, channels, 1, 1)) 
        self.global_pool = nn.AdaptiveAvgPool2d(1) 

        #uses linear layers to produce the three distinct feature vectors 
        self.medical_fc = nn.Linear(in_features, latent_dim)
        self.identity_fc = nn.Linear(in_features, latent_dim)
        # self.other_fc = nn.Linear(in_features, latent_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        # Rescale images from [-1, 1] to [0, 1]
        x = rescale_image_to_0_1(x)

        # Preprocess the input image
        inputs = self.preprocessor(images=x, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(next(self.backbone.parameters()).device)


        # Extract features using the Swin backbone
        with torch.no_grad():
            outputs = self.backbone(pixel_values)  # Pass the entire dictionary of inputs
            features = outputs.hidden_states[-1]  # Use the last hidden state

        # Global average pooling to reduce spatial dimensions
        # print("features shape:", features.shape)
        # x = features
        x = features.mean(dim=1) 

        # Generate the three distinct feature vectors
        medical_features = self.leaky_relu(self.medical_fc(x))
        identity_features = self.leaky_relu(self.identity_fc(x))
        # other_features = self.leaky_relu(self.other_fc(x))

        # print("medical_features shape:", medical_features.shape)
        # print("identity_features shape:", identity_features.shape)


        return medical_features, identity_features #, other_features


class FeatureExtractor_dinov(nn.Module):
    ''' From input image, extracts 3 distinct ft vectors:
    Medical features, identity features, other features
    '''
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()

        ####
        BACKBONE = 'dinov2_large'
        MODE = 'fine_tune'
        backbone_mode = 'fine_tune'
        HIDDEN = [128]
        num_classes=2
        backbone_name="facebook/dinov2-large"

        self.preprocessor = AutoImageProcessor.from_pretrained(backbone_name, do_rescale=False)
        self.backbone =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        in_features = self.backbone.embed_dim

        #reduces each feature map to a single value (1d ft vector: (batch_size, channels, 1, 1)) 
        self.global_pool = nn.AdaptiveAvgPool2d(1) 

        #uses linear layers to produce the three distinct feature vectors 
        self.medical_fc = nn.Linear(in_features, latent_dim)
        self.identity_fc = nn.Linear(in_features, latent_dim)
        # self.other_fc = nn.Linear(in_features, latent_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        # Rescale images from [-1, 1] to [0, 1]
        x = rescale_image_to_0_1(x)

        # Preprocess the input image
        inputs = self.preprocessor(images=x, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(next(self.backbone.parameters()).device)


        # Extract features using the Swin backbone
        with torch.no_grad():
            outputs = self.backbone(pixel_values)  # Pass the entire dictionary of inputs
            features = outputs  # For torch.hub DINOv2, outputs is the features tensor

        # Global average pooling to reduce spatial dimensions
        # x = self.global_pool(features).view(features.size(0), -1) 
        x = features

        # Generate the three distinct feature vectors
        medical_features = self.leaky_relu(self.medical_fc(x))
        identity_features = self.leaky_relu(self.identity_fc(x))
        # other_features = self.leaky_relu(self.other_fc(x))


        return medical_features, identity_features #, other_features



# 3. Classifiers
class DiseaseClassifier(nn.Module):
    def __init__(self, latent_dim=256): # input is feature vector (e.g. with 256 dims)
        super(DiseaseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), # transforms into 128 ft vector 
            nn.Dropout(0.5), # turns off 50% of neurons (avoid overfitting)
            nn.Linear(latent_dim // 2, 1), #output layer: reduces from 128 dims to 1 dim
            # nn.Sigmoid() # activation function (maps between 0 and 1)
        )

    def forward(self, x):
        return self.model(x)


class IdentityClassifier(nn.Module): # multi-class classification, with 115 classes
    def __init__(self, latent_dim=256, num_classes=5): # input is feature vector (e.g. with 256 dims) and num of classes
        super(IdentityClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), # 256 → 128
            nn.Dropout(0.5), #turn off 50% neurons (reduce overfitting)
            nn.Linear(latent_dim // 2, num_classes) # 128 → 115
            
            # outputs logits (no need for softmax)
        )

    def forward(self, x):
        return self.model(x)


# # 4. Discriminator
# # Predicts if image is real/fake
# class Discriminator(nn.Module):
#     def __init__(self, input_channels=3, latent_dim=256):
#         super().__init__()
#         self.augment = nn.Sequential(
#             nn.Dropout2d(0.1)  # Basic augmentation (can be improved)
#         )

#         self.conv_start = nn.Sequential(
#             nn.Conv2d(input_channels, latent_dim // 4, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2)
#         )

#         self.res_blocks = nn.Sequential(
#             ResBlock(latent_dim // 4, latent_dim, 'downsample'),
#             ResBlock(latent_dim, latent_dim)
#         )

#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(latent_dim, latent_dim),
#             nn.Dropout(0.5),
#             nn.Linear(latent_dim, latent_dim // 4),
#             nn.Dropout(0.5),
#             nn.Linear(latent_dim // 4, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.augment(x)
#         x = self.conv_start(x)
#         x = self.res_blocks(x)
#         return self.classifier(x)



# 5. Generator
# Produces an image from three latent vectors
class Generator(nn.Module):
    def __init__(self, latent_dim=256, output_channels=3):
        super().__init__()
        self.units = 128
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 4 * 4 * self.units),
            nn.ReLU()
        )
        self.reshape_dim = (self.units, 4, 4)

        self.res_blocks = nn.Sequential(
            ResBlock(self.units, self.units // 2, 'upsample'),
            ResBlock(self.units // 2, self.units // 2),

            ResBlock(self.units // 2, self.units // 4, 'upsample'),
            ResBlock(self.units // 4, self.units // 4),

            ResBlock(self.units // 4, self.units // 8, 'upsample'),
            ResBlock(self.units // 8, self.units // 8),

            ResBlock(self.units // 8, self.units // 16, 'upsample'),
            ResBlock(self.units // 16, self.units // 16),
        )

        self.final_conv = nn.Conv2d(self.units // 16, output_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, medical, identity): #, other):
        x = torch.cat([medical, identity], dim=1) #, other], dim=1)
        x = self.fc(x)
        x = x.view(-1, *self.reshape_dim)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x=self.tanh(x) # Output in range [-1, 1]
        return x

