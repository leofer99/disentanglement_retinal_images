from torchvision import models
from transformers import ConvNextV2ForImageClassification
from transformers import ViTModel
from transformers import CLIPModel
import torch
import torch.nn as nn
import subprocess
import os
from .RetFound import get_retfound
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision import models
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")


class CLIPImageEmbeddings(nn.Module):
    """
    A PyTorch module for generating image embeddings using the CLIP vision model.

    This module takes an image as input and produces embeddings that can be used in various downstream tasks.

    Parameters:
    - vision_model (torch.nn.Module): The CLIP vision model used for image feature extraction.
    - visual_projection (torch.nn.Module): The visual projection head.

    Methods:
    - forward(images): Forward pass to generate image embeddings from input images.

    Example Usage:
    ```python
    vision_model = CLIPImageEmbeddings(vision_model, visual_projection)
    image_features = vision_model(images)
    ```

    Note:
    - The CLIPImageEmbeddings class is designed to work with CLIP vision models.
    - It takes an image as input and produces embeddings for downstream tasks.

    Dependencies:
    - PyTorch

    For more information on CLIP, see:
    https://openai.com/research/clip
    """
    def __init__(self, vision_model, visual_projection):
        """
        Initialize the CLIPImageEmbeddings module.

        Args:
        - vision_model (torch.nn.Module): The CLIP vision model used for image feature extraction.
        - visual_projection (torch.nn.Module): The visual projection head.
        """
        super(CLIPImageEmbeddings, self).__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, images):
        # Pass the images through the vision model
        vision_output = self.vision_model(images)['pooler_output']

        # Apply the visual projection
        image_embeddings = self.visual_projection(vision_output)

        return image_embeddings

class FoundationalCVModel(torch.nn.Module):
    """
    A PyTorch module for loading and using foundational computer vision models.

    This module allows you to load and use various foundational computer vision models for tasks like image classification.

    Parameters:
    - backbone (str): The name of the foundational CV model to load.
    - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    - forward(x): Forward pass to obtain features from input data.

    Example Usage:
    ```python
    cv_model = FoundationalCVModel(backbone='vit_base', mode='eval')
    features = cv_model(input_data)
    ```

    Note:
    - This module provides access to various foundational CV models such as ViT, CLIP, ConvNets, and more.
    - It allows for both evaluation and fine-tuning modes.

    Dependencies:
    - PyTorch
    - Hugging Face Transformers (for ViT and CLIP models)
    - Facebook Research DINOv2 (for DINOv2 models)

    For more information on specific models, refer to the respective model's documentation.
    """
    
    def __init__(self, backbone, mode='eval', weights=None):
        """
        Initialize the FoundationalCVModel module.

        Args:
        - backbone (str): The name of the foundational CV model to load.
        - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.
        - if model is retfound, weights is the path to the weights file
        """
        super(FoundationalCVModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Select the backbone from the possible foundational models
        if backbone in ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']:
            # Repo: https://github.com/facebookresearch/dinov2
            # Paper: https://arxiv.org/abs/2304.07193
            backbone_path = {
                'dinov2_small': 'dinov2_vits14',
                'dinov2_base': 'dinov2_vitb14',
                'dinov2_large': 'dinov2_vitl14',
                'dinov2_giant': 'dinov2_vitg14',
            }
            self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_path[backbone])

            
        elif backbone in ['convnextv2_tiny', 'convnextv2_base', 'convnextv2_large','convnextv2_384']:
            # Repo: https://huggingface.co/facebook/convnextv2-base-22k-224
            # Paper: https://arxiv.org/abs/2301.00808
            backbone_path = {
                'convnextv2_tiny': 'facebook/convnextv2-tiny-22k-224',
                'convnextv2_base': 'facebook/convnextv2-base-22k-224',
                'convnextv2_large': 'facebook/convnextv2-large-22k-224',
                'convnextv2_384':'facebook/convnextv2-large-22k-384'
            }
            
            self.backbone = ConvNextV2ForImageClassification.from_pretrained(backbone_path[backbone])
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone in ['swinv2']:
            from transformers import Swinv2ForImageClassification
            self.backbone = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'convnext_tiny':
            # Get the backbone
            self.backbone = models.convnext.convnext_tiny(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1], nn.Flatten())
        elif backbone == 'convnext_small':
            # Get the backbone
            self.backbone = models.convnext.convnext_small(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1], nn.Flatten())
        elif backbone == 'convnext_base':
            # Get the backbone
            self.backbone = models.convnext.convnext_base(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1], nn.Flatten())
        elif backbone == 'convnext_large':
            # Get the backbone
            self.backbone = models.convnext.convnext_large(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1], nn.Flatten())
             
        # https://pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html#torchvision.models.swin_t
        elif backbone == 'swin_tiny':
            # Get the backbone
            self.backbone = models.swin_transformer.swin_t(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'swin_small':
            # Get the backbone
            self.backbone = models.swin_transformer.swin_s(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'swin_base':
            # Get the backbone
            self.backbone = models.swin_transformer.swin_b(pretrained=True)
            # Remove the final classifier layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        elif backbone in ['vit_base', 'vit_large']:
            # https://huggingface.co/docs/transformers/model_doc/vit
            # paper: https://arxiv.org/abs/2010.11929
            backbone_path = {
                'vit_base': "google/vit-base-patch16-224-in21k",
                'vit_large': 'google/vit-large-patch16-224-in21k',
            }
            # Get the backbone
            self.backbone = ViTModel.from_pretrained(backbone_path[backbone])

        elif backbone in ['clip_base', 'clip_large']:
            # https://huggingface.co/openai/clip-vit-base-patch16
            # paper: https://arxiv.org/abs/2103.00020
            backbone_path = {
                'clip_base': "openai/clip-vit-large-patch14",
                'clip_large': 'openai/clip-vit-base-patch16',
            }
            clip_model = CLIPModel.from_pretrained(backbone_path[backbone])
            # Get image part of CLIP model
            self.backbone = CLIPImageEmbeddings(clip_model.vision_model, clip_model.visual_projection)

        elif backbone == 'retfound':
            self.backbone = get_retfound(weights=weights, backbone=True)
            
        elif backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            # Select the ResNet model based on the 'backbone' parameter
            resnet_models = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152,
            }
            
            # Load the pretrained ResNet model
            self.backbone = resnet_models[backbone](pretrained=True)
            #backbone = resnet_models[backbone](pretrained=True)
            
            # Remove the fully connected layer to use the model as a feature extractor
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1], nn.Flatten())
            #backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

        else:
            raise ValueError(f"Unsupported backbone model: {backbone} \n Supported models: 'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'swin_tiny', 'swin_small', 'swin_base', 'vit_base', 'vit_large', 'clip_base', 'clip_large', 'retfound'")
            
        # Set the model to evaluation or fine-tuning mode
        self.mode = mode
        if mode == 'eval':
            self.eval()
        elif mode == 'fine_tune':
            self.train()
            
            
    def download_and_rename(self, url, filename):
        """Downloads a file from the given URL and renames it to the given new file name.

        Args:
            url: The URL of the file to download.
            new_file_name: The new file name for the downloaded file.
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        print(f'Downloading the weights of the model: {url} ...')
        subprocess.run(["wget", "-q", "-O", filename, url])
        print(f'Done!')

    def forward(self, x):
        """
        Forward pass to obtain features from input data.

        Args:
        - x (torch.Tensor): Input data to obtain features from.

        Returns:
        torch.Tensor: Features extracted from the input data using the selected foundational CV model.
        """

        # Pass the input image to the model
        features = self.backbone(x)
        
        if self.backbone_name in ['vit_base', 'vit_large', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large']:
            features = features['pooler_output']

        # Return the features
        return features
    
    

class FoundationalCVModelWithClassifier(torch.nn.Module):
    """
    A PyTorch module that combines a foundational computer vision model with a classifier for image classification.

    This module allows you to create a complete image classification model by combining a foundational CV model
    with a classifier on top of it. It supports both evaluation and fine-tuning modes.

    Parameters:
    - backbone (torch.nn.Module): The foundational CV model used for feature extraction.
    - num_classes (int): The number of output classes for classification.
    - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    - forward(x): Forward pass to obtain class predictions from input data.

    Example Usage:
    ```python
    backbone_model = FoundationalCVModel(backbone='vit_base', mode='eval')
    image_classifier = FoundationalCVModelWithClassifier(backbone_model, num_classes=10, mode='eval')
    predictions = image_classifier(input_data)
    ```

    Note:
    - This module combines a foundational CV model with a classifier to create an image classification model.
    - It is suitable for tasks where you need to classify images into multiple classes.
    - The `num_classes` parameter defines the number of output classes.

    Dependencies:
    - PyTorch

    For more information on specific foundational CV models, refer to their respective documentation.
    """
    def __init__(self, backbone, hidden, num_classes, mode='eval', backbone_mode='eval'):
        """
        Initialize the FoundationalCVModelWithClassifier module.

        Args:
        - backbone (torch.nn.Module): The foundational CV model used for feature extraction.
        - num_classes (int): The number of output classes for classification.
        - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine-tune' for fine-tuning. Default is 'eval'.
        """
        super(FoundationalCVModelWithClassifier, self).__init__()
        assert num_classes >= 2

        # Define the backbone
        self.backbone = backbone
        self.hidden = hidden
        
        output_dim = self.calculate_backbone_out()
        
        # Initialize layers as an empty list
        layers = []
        
        # Add the linear layer and ReLU activation if 'hidden' is an integer
        if isinstance(hidden, int): # (1 single fully connected layer)
            layers.append(nn.Linear(output_dim, hidden)) 
            # layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.2))
            # layers.append(nn.BatchNorm1d(hidden))
            output_dim = hidden
            
        # Add the linear layer and ReLU activation for each element in 'hidden' if it's a list
        elif isinstance(hidden, list): # (multiple hidden layers)
            for h in hidden:
                layers.append(nn.Linear(output_dim, h))
                # layers.append(nn.ReLU())
                # layers.append(nn.Dropout(p=0.2))
                # layers.append(nn.BatchNorm1d(h))
                output_dim = h
        
        if hidden: # hidden layers are grouped into a sequential container
            self.hidden_layers = nn.Sequential(*layers)
        else: # if there are no hidden layers, batch normalization is applied
            self.norm = nn.BatchNorm1d(output_dim)

        if num_classes==2: 
            self.classifier = nn.Linear(output_dim, 1) # binary classification: 1 output
        else:
            self.classifier = nn.Linear(output_dim, num_classes) # multi-class classification

            
        # Set the mode
        self.mode = mode
        self.backbone_mode = backbone_mode
        
        if backbone_mode == 'eval':
            self.backbone.eval()
        elif backbone_mode == 'fine_tune':
            self.backbone.train()
            
        if mode == 'eval':
            self.eval()
        elif mode == 'fine_tune':
            self.train()
            
    def calculate_backbone_out(self):
        # calculates output dim of backbone
        sample_input = torch.randn(1, 3, 224, 224)
        
        self.backbone.eval()
        # Forward pass 1 sample input through the model
        with torch.no_grad():
            output = self.backbone(sample_input)
        try:
            a= output.shape[1]
        except:
            a= output.logits.shape[1]
        #    a=output.last_hidden_state.shape[1]
        
        return a
        # return output.logits.shape[1]
        

    def forward(self, x):
        """
        Forward pass to obtain class predictions from input data.

        Args:
        - x (torch.Tensor): Input data to obtain class predictions for.

        Returns:
        torch.Tensor: Class predictions generated by the model for the input data.
        """
        # Pass the input through the backbone (feature extractor)
        features = self.backbone(x) 
        
        if self.hidden: #hidden layers
            features = self.hidden_layers(features)
        else: #normalization
            features = self.norm(features)

        # Apply the classifier to obtain class predictions
        logits = self.classifier(features)
        
        # Get the probabilities
        # probabilities = self.activation_f(logits)

        return logits
    
    