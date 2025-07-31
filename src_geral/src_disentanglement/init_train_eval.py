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
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# get 0_folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..'))

# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))


from src_disentanglement.model import FeatureExtractor, FeatureExtractor_convnext, FeatureExtractor_swin, FeatureExtractor_dinov, Generator, DiseaseClassifier, IdentityClassifier
from src_disentanglement.loss import compute_custom_loss
from src_disentanglement.loss_no_disen import compute_custom_loss_no_disen


# ============ Load Model ============ #
def load_model(model_path, device, models):
    feature_extractor, generator, disease_classifier, identity_classifier = models
    # feature_extractor, generator, discriminator, disease_classifier, identity_classifier = models

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        generator.load_state_dict(checkpoint['generator'])
        # discriminator.load_state_dict(checkpoint['discriminator'])
        disease_classifier.load_state_dict(checkpoint['disease_classifier'])
        identity_classifier.load_state_dict(checkpoint['identity_classifier'])

    else:
        print(f"Model file not found: {model_path}")
        exit()

def load_model_baseline(model_path, device, models):
    feature_extractor, generator, disease_classifier, identity_classifier = models
    # feature_extractor, generator, discriminator, disease_classifier, identity_classifier = models

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        # generator.load_state_dict(checkpoint['generator'])
        # discriminator.load_state_dict(checkpoint['discriminator'])
        disease_classifier.load_state_dict(checkpoint['disease_classifier'])
        identity_classifier.load_state_dict(checkpoint['identity_classifier'])

    else:
        print(f"Model file not found: {model_path}")
        exit()


# ============ Model Initialization ============ #
def initialize_models(latent_dim, num_classes, device, model='normal'):

    if model=='normal':
        feature_extractor = FeatureExtractor(latent_dim=latent_dim).to(device)

    elif model=='convnext':
        feature_extractor = FeatureExtractor_convnext(latent_dim=latent_dim).to(device)

    elif model=='swin':
        feature_extractor = FeatureExtractor_swin(latent_dim=latent_dim).to(device)

    elif model=='dinov':
        feature_extractor = FeatureExtractor_dinov(latent_dim=latent_dim).to(device)



    generator = Generator(latent_dim=latent_dim).to(device)
    # discriminator = Discriminator(latent_dim=latent_dim).to(device)
    disease_classifier = DiseaseClassifier(latent_dim=latent_dim).to(device)
    identity_classifier = IdentityClassifier(latent_dim=latent_dim, num_classes=num_classes).to(device)
    
    return feature_extractor, generator, disease_classifier, identity_classifier
    # return feature_extractor, generator, discriminator, disease_classifier, identity_classifier

# ============ Optimizer Setup ============ #
def setup_optimizers(models, learning_rate_gen, decay):
# def setup_optimizers(models, learning_rate_gen, learning_rate_disc, decay):
    feature_extractor, generator, disease_classifier, identity_classifier = models
    # feature_extractor, generator, discriminator, disease_classifier, identity_classifier = models
    
    params = list(feature_extractor.parameters()) + \
             list(generator.parameters()) + \
             list(disease_classifier.parameters()) + \
             list(identity_classifier.parameters())
    # generator:
    opt = optim.Adam(params, lr=learning_rate_gen, weight_decay=decay)
 
    #discriminator:
    # disc_opt = optim.Adam(discriminator.parameters(), lr=learning_rate_disc, weight_decay=decay)

    return opt
    # return opt, disc_opt




# ============ Training Loop ============ #
def train_one_epoch_baseline(models, dataloader, opt, device, disentanglement=True, cce=None, bce=None):
# def train_one_epoch(models, dataloader, opt, disc_opt, device):

    feature_extractor, _, disease_classifier, identity_classifier = models
    feature_extractor.train()
    # generator.train()
    # discriminator.train()
    disease_classifier.train()
    identity_classifier.train()

    epoch_total_loss = []
    # epoch_loss_adv = []
    epoch_loss_recon = []
    epoch_loss_disen=[]
    epoch_loss_dis=[]
    epoch_loss_id=[]

    # disc_real_loss = []
    # disc_fake_loss = []


    for batch in dataloader:
        images = batch['image'].to(device)
        labels_id = batch['label_id'].to(device)
        labels_dis = batch['label_dis'].to(device)

        # # ---- Discriminator Update ---- #
        # disc_opt.zero_grad()
        # with torch.no_grad():
        #     med_feat, id_feat, oth_feat = feature_extractor(images)
        #     fake_images = generator(med_feat, id_feat, oth_feat)

        # real_pred = discriminator(images)
        # fake_pred = discriminator(fake_images)

        # bce = nn.BCELoss()
        # loss_real = bce(real_pred, torch.ones_like(real_pred))
        # loss_fake = bce(fake_pred, torch.zeros_like(fake_pred))
        # loss_disc = loss_real + loss_fake

        # loss_disc.backward()
        # disc_opt.step()
        # # Track discriminator loss
        # disc_real_loss.append(loss_real.item())
        # disc_fake_loss.append(loss_fake.item())

        # ---- Generator & Classifiers Update ---- #
        opt.zero_grad()
        med_feat, id_feat = feature_extractor(images)
        noisy_med = med_feat + torch.randn_like(med_feat) * 0.1
        noisy_id = id_feat + torch.randn_like(id_feat) * 0.1
        # noisy_oth = oth_feat + torch.randn_like(oth_feat) * 0.1

        # fake_images = generator(noisy_med, noisy_id)
        disease_out = disease_classifier(noisy_med)
        identity_out = identity_classifier(noisy_id)
        # discriminator_out = discriminator(fake_images)

        # Disentanglement: use altered_med/id/oth 
        # feat_med_rec, feat_id_rec = feature_extractor(fake_images)
        
        # # Generate new images, using the altered vectors
        # fake_images_med = generator(feat_med_rec, noisy_id)
        # fake_images_id = generator(noisy_med, feat_id_rec)

        # # extrair
        # feat_med_altered_med, feat_id_altered_med = feature_extractor(fake_images_med)
        # feat_med_altered_id, feat_id_altered_id = feature_extractor(fake_images_id)
        # feat_med_altered_oth, feat_id_altered_oth = feature_extractor(fake_images_oth)

        zero_feat = torch.zeros_like(med_feat)
        zero_img = torch.zeros_like(images)
        zero_id = torch.zeros_like(id_feat)
        zero_tuple = (zero_feat, zero_id)

        outputs = {
            'reconstructed': zero_img,
            'features': (zero_feat, zero_id),
            'features_noisy': (zero_feat, zero_id),
            'features_rec': (zero_feat, zero_id),

            'disease_out': disease_out,
            'identity_out': identity_out,
            #torch.zeros((disease_out.shape[0], 2), device=device), #because id is only 2 classes
            # 'discriminator_out': discriminator_out,

            'feat_altered_med': (zero_feat, zero_id),
            'feat_altered_id': (zero_feat, zero_id),
            # 'feat_altered_oth': (feat_med_altered_oth, feat_id_altered_oth, feat_oth_altered_oth)
        }

        inputs = {'real': images}
        targets = {'disease': labels_dis, 'identity': labels_id}
        weights = {'disc_weight': torch.tensor(1.0, device=device)}

        if disentanglement:
            total_loss, loss = compute_custom_loss(inputs, outputs, targets, weights, cce, bce)
        else:
            total_loss, loss = compute_custom_loss_no_disen(inputs, outputs, targets, weights, cce, bce)
        
        total_loss.backward()
        opt.step()

        # Total generator loss
        epoch_total_loss.append(total_loss.item())

        # Track losses:
        # epoch_loss_adv.append(loss['loss_disc'].item())  # Adversarial loss (how well generator fools discriminator)
        epoch_loss_recon.append(loss['recon_loss'].item())  # Reconstruction loss (Generator)
        epoch_loss_dis.append(loss['loss_dis'].item())
        epoch_loss_id.append(loss['loss_id'].item())
        epoch_loss_disen.append(loss['loss_disentanglement'].item())


    # Print loss values:
    print(f"Epoch - Total Loss: {np.mean(epoch_total_loss):.4f}" )   
    print(f"Recon Loss: {np.mean(epoch_loss_recon):.4f} |  Disentang Loss: {np.mean(epoch_loss_disen):.4f}")
    print(f"Dis Loss: {np.mean(epoch_loss_dis):.4f} | Id Loss: {np.mean(epoch_loss_id):.4f}")

    return epoch_total_loss, epoch_loss_recon, epoch_loss_disen, epoch_loss_dis, epoch_loss_id
    # return epoch_loss, disc_real_loss, disc_fake_loss, epoch_loss_dis, epoch_loss_id, epoch_loss_disen

def train_one_epoch(models, dataloader, opt, device, disentanglement=True, cce=None, bce=None):
# def train_one_epoch(models, dataloader, opt, disc_opt, device):

    feature_extractor, generator, disease_classifier, identity_classifier = models
    feature_extractor.train()
    generator.train()
    # discriminator.train()
    disease_classifier.train()
    identity_classifier.train()

    epoch_total_loss = []
    # epoch_loss_adv = []
    epoch_loss_recon = []
    epoch_loss_disen=[]
    epoch_loss_dis=[]
    epoch_loss_id=[]

    # disc_real_loss = []
    # disc_fake_loss = []


    for batch in dataloader:
        images = batch['image'].to(device)
        labels_id = batch['label_id'].to(device)
        labels_dis = batch['label_dis'].to(device)

        # # ---- Discriminator Update ---- #
        # disc_opt.zero_grad()
        # with torch.no_grad():
        #     med_feat, id_feat, oth_feat = feature_extractor(images)
        #     fake_images = generator(med_feat, id_feat, oth_feat)

        # real_pred = discriminator(images)
        # fake_pred = discriminator(fake_images)

        # bce = nn.BCELoss()
        # loss_real = bce(real_pred, torch.ones_like(real_pred))
        # loss_fake = bce(fake_pred, torch.zeros_like(fake_pred))
        # loss_disc = loss_real + loss_fake

        # loss_disc.backward()
        # disc_opt.step()
        # # Track discriminator loss
        # disc_real_loss.append(loss_real.item())
        # disc_fake_loss.append(loss_fake.item())

        # ---- Generator & Classifiers Update ---- #
        opt.zero_grad()
        med_feat, id_feat = feature_extractor(images)
        noisy_med = med_feat + torch.randn_like(med_feat) * 0.1
        noisy_id = id_feat + torch.randn_like(id_feat) * 0.1
        # noisy_oth = oth_feat + torch.randn_like(oth_feat) * 0.1

        fake_images = generator(noisy_med, noisy_id)
        disease_out = disease_classifier(noisy_med)
        identity_out = identity_classifier(noisy_id)
        # discriminator_out = discriminator(fake_images)

        # Disentanglement: use altered_med/id/oth 
        feat_med_rec, feat_id_rec = feature_extractor(fake_images)
        
        # Generate new images, using the altered vectors
        fake_images_med = generator(feat_med_rec, noisy_id)
        fake_images_id = generator(noisy_med, feat_id_rec)

        # extrair
        feat_med_altered_med, feat_id_altered_med = feature_extractor(fake_images_med)
        feat_med_altered_id, feat_id_altered_id = feature_extractor(fake_images_id)
        # feat_med_altered_oth, feat_id_altered_oth = feature_extractor(fake_images_oth)

        outputs = {
            'reconstructed': fake_images,
            'features': (med_feat, id_feat),
            'features_noisy': (noisy_med, noisy_id),
            'features_rec': (feat_med_rec, feat_id_rec),

            'disease_out': disease_out,
            'identity_out': identity_out,
            # 'discriminator_out': discriminator_out,

            'feat_altered_med': (feat_med_altered_med, feat_id_altered_med),
            'feat_altered_id': (feat_med_altered_id, feat_id_altered_id),
            # 'feat_altered_oth': (feat_med_altered_oth, feat_id_altered_oth, feat_oth_altered_oth)
        }

        inputs = {'real': images}
        targets = {'disease': labels_dis, 'identity': labels_id}
        weights = {'disc_weight': torch.tensor(1.0, device=device)}

        if disentanglement:
            total_loss, loss = compute_custom_loss(inputs, outputs, targets, weights, cce, bce)
        else:
            total_loss, loss = compute_custom_loss_no_disen(inputs, outputs, targets, weights, cce, bce)
        
        total_loss.backward()
        opt.step()

        # Total generator loss
        epoch_total_loss.append(total_loss.item())

        # Track losses:
        # epoch_loss_adv.append(loss['loss_disc'].item())  # Adversarial loss (how well generator fools discriminator)
        epoch_loss_recon.append(loss['recon_loss'].item())  # Reconstruction loss (Generator)
        epoch_loss_dis.append(loss['loss_dis'].item())
        epoch_loss_id.append(loss['loss_id'].item())
        epoch_loss_disen.append(loss['loss_disentanglement'].item())


    # Print loss values:
    print(f"Epoch - Total Loss: {np.mean(epoch_total_loss):.4f}" )   
    print(f"Recon Loss: {np.mean(epoch_loss_recon):.4f} |  Disentang Loss: {np.mean(epoch_loss_disen):.4f}")
    print(f"Dis Loss: {np.mean(epoch_loss_dis):.4f} | Id Loss: {np.mean(epoch_loss_id):.4f}")

    return epoch_total_loss, epoch_loss_recon, epoch_loss_disen, epoch_loss_dis, epoch_loss_id
    # return epoch_loss, disc_real_loss, disc_fake_loss, epoch_loss_dis, epoch_loss_id, epoch_loss_disen

# ============ Evaluation ============ #
def evaluate_model(models, dataloader, device, df_train=None, df_val=None, df_test=None, save_path=None):
    
    generator=None

    # Set models to evaluation mode
    feature_extractor, generator, disease_classifier, identity_classifier = models
    # feature_extractor, _, _, disease_classifier, identity_classifier = models

    feature_extractor.eval()
    generator.eval()
    # discriminator.eval()
    disease_classifier.eval()
    identity_classifier.eval()

    val_loss = []
    val_loss_dis = []
    val_loss_id = []
    val_loss_disen = []


    # Extract features from the test set
    all_images, all_med_feat, all_id_feat = [], [], []
    all_y_id_test, all_y_dis_test = [], []
    image_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels_id = batch['label_id'].to(device)
            labels_dis = batch['label_dis'].to(device)

            # Get original image names
            batch_image_names = dataloader.dataset.image_data[batch_idx * dataloader.batch_size:
                                                              (batch_idx + 1) * dataloader.batch_size]
            image_names.extend(batch_image_names)

            # Forward pass
            med_feat, id_feat = feature_extractor(images)

            all_images.append(images)
            all_med_feat.append(med_feat)
            all_id_feat.append(id_feat)
            # all_oth_feat.append(oth_feat)
            all_y_id_test.append(labels_id)
            all_y_dis_test.append(labels_dis)


        images = torch.cat(all_images)
        med_feat = torch.cat(all_med_feat)
        id_feat = torch.cat(all_id_feat)
        # oth_feat = torch.cat(all_oth_feat)
        y_id_test = torch.cat(all_y_id_test)
        y_dis_test = torch.cat(all_y_dis_test)


        disease_outputs = disease_classifier(med_feat)
        identity_outputs = identity_classifier(id_feat) # Logits for both classes


    # Disease Classifier Evaluation
    disease_probs = torch.sigmoid(disease_outputs)
    disease_preds = (disease_probs >= 0.5).int()

    disease_accuracy = (disease_preds == y_dis_test).float().mean().item()
    # print(f"Medical Accuracy with Med vector: {disease_accuracy:.4f}")
    
    roc_auc = roc_auc_score(y_dis_test.cpu().numpy(), disease_probs.cpu().numpy())
    print(f"Medical ROC with Med vector: {roc_auc:.4f}")
    print(classification_report(y_dis_test.cpu(), disease_preds.cpu(), target_names=['not DR', 'DR']))
    print(confusion_matrix(y_dis_test.cpu(), disease_preds.cpu()))


    # Identity Classifier Evaluation
    identity_probs = torch.softmax(identity_outputs, dim=1) #Convert logits to probabilities
    identity_preds = torch.argmax(identity_outputs, dim=1) # Predicted class (0 or 1)
    identity_accuracy = (identity_preds == y_id_test).float().mean().item()
    # print(f"SA Accuracy with SA vector: {identity_accuracy:.4f}")
    
    roc_auc = roc_auc_score(y_id_test.cpu().numpy(), identity_probs[:, 1].cpu().numpy())
    print(f"SA ROC with SA vector: {roc_auc:.4f}")
    print(classification_report(y_id_test.cpu(), identity_preds.cpu()))
    print(confusion_matrix(y_id_test.cpu(), identity_preds.cpu()))

    # print("True (unique): ", torch.unique(y_id_test.cpu()))
    # print("Pred (unique): ", torch.unique(identity_preds.cpu()))


    # # Generate images
    # generated_images = generator(med_feat, id_feat, oth_feat)
    # generated_images = F.interpolate(generated_images, size=(224, 224), mode='bilinear', align_corners=False)

    # # Reconstruction metrics
    # mse_loss = nn.MSELoss()
    # mse = mse_loss(generated_images, images).item()
    # ssim = TMF.structural_similarity_index_measure(generated_images, images).item()
    # psnr = TMF.peak_signal_noise_ratio(generated_images, images).item()
    # print(f"MSE: {mse:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}")

    # # Feature reconstruction
    # rec_med_feat, rec_id_feat, rec_oth_feat = feature_extractor(generated_images)
    # mse_med = mse_loss(med_feat, rec_med_feat).item()
    # mse_id = mse_loss(id_feat, rec_id_feat).item()
    # mse_oth = mse_loss(oth_feat, rec_oth_feat).item()
    # print(f"MSE med: {mse_med:.4f}, MSE id: {mse_id:.4f}, MSE other: {mse_oth:.4f}")

    # # Evaluate reconstructed features
    # rec_disease_outputs = disease_classifier(rec_med_feat)
    # rec_disease_preds = (rec_disease_outputs >= 0.5).int()
    # rec_disease_accuracy = (rec_disease_preds == y_dis_test).float().mean().item()
    # print(f"Reconstructed Medical Results Accuracy: {rec_disease_accuracy:.4f}")
    # print(confusion_matrix(y_dis_test.cpu(), rec_disease_preds.cpu()))

    # rec_identity_outputs = identity_classifier(rec_id_feat)
    # rec_identity_preds = torch.argmax(rec_identity_outputs, dim=1)
    # rec_identity_accuracy = (rec_identity_preds == y_id_test).float().mean().item()
    # print(f"Reconstructed Identity Results Accuracy: {rec_identity_accuracy:.4f}")


    if df_train is not None:
        # Extract patient IDs from image names
        patient_ids = [int(name.split('.')[0]) for name in image_names]  # Extract patient ID from image name
        # print( f"Patient IDs: {patient_ids}")
        
        # df_train['patient_ID_disen'] = patient_ids
        df_train['true_dis'] = y_dis_test.cpu().numpy() 

        df_train['disen_predicted_icdr'] = disease_preds.cpu().numpy() 
        df_train['disen_prob_predicted_icdr'] = disease_probs.detach().cpu().numpy() 
        df_train['disen_predicted_id'] = identity_preds.cpu().numpy()
        df_train['disen_prob_predicted_id'] = identity_probs[:, 1].detach().cpu().numpy() # Probability of being in class 1
        
        # Save the DataFrame to a CSV file
        df_train.to_csv(os.path.join(save_path, 'df_train_disentangled.csv'), index=False)
        print("Successfully saved df_train!")


    if df_val is not None:
        df_val['disen_predicted_icdr'] = disease_preds.cpu().numpy() 
        df_val['disen_prob_predicted_icdr'] = disease_probs.detach().cpu().numpy() 
        df_val['disen_predicted_id'] = identity_preds.cpu().numpy()
        df_val['disen_prob_predicted_id'] = identity_probs[:, 1].detach().cpu().numpy() # Probability of being in class 1
        
        # Save the DataFrame to a CSV file
        df_val.to_csv(os.path.join(save_path, 'df_val_disentangled.csv'), index=False)
        print("Successfully saved df_val!")


    if df_test is not None:
        df_test['disen_predicted_icdr'] = disease_preds.cpu().numpy() 
        df_test['disen_prob_predicted_icdr'] = disease_probs.detach().cpu().numpy() 
        df_test['disen_predicted_id'] = identity_preds.cpu().numpy()
        df_test['disen_prob_predicted_id'] = identity_probs[:, 1].detach().cpu().numpy() # Probability of being in class 1
        
        # Save the DataFrame to a CSV file
        df_test.to_csv(os.path.join(save_path, 'df_test_disentangled.csv'), index=False)
        print("Successfully saved df_test!")


    # if save_path is not None and generator is not None:
    #     # Save the reconstructed images
    #     save_dir = os.path.join(save_path, 'generated_images')
    #     os.makedirs(save_dir, exist_ok=True)

    #     #USE THE PATIENT ID TO SAVE THE IMAGES
    #     # Convert tensors to images and save them
    #     to_pil = ToPILImage()
    #     for img_tensor, original_name in zip(generated_images, image_names):
    #         img = to_pil(img_tensor.cpu().clamp(0, 1))  # Convert tensor to PIL image
    #         img.save(os.path.join(save_dir, f"rec_{original_name}.jpg"))  # Save as .jpg
    #     print(f"Reconstructed images saved to {save_dir}")

    #     save_dir = os.path.join(save_path, 'disentangled_features')
    #     os.makedirs(save_dir, exist_ok=True)

        # Save the med, id and other features (that are disentangled)


# ============ Evaluation ============ #
def evaluate_model_baseline(models, dataloader, device, df_train=None, df_val=None, df_test=None, save_path=None):
    
    generator=None

    # Set models to evaluation mode
    feature_extractor, _, disease_classifier, identity_classifier = models
    # feature_extractor, _, _, disease_classifier, identity_classifier = models

    feature_extractor.eval()
    # generator.eval()
    # discriminator.eval()
    disease_classifier.eval()
    identity_classifier.eval()

    val_loss = []
    val_loss_dis = []
    val_loss_id = []
    val_loss_disen = []

    # Extract features from the test set
    all_images, all_med_feat, all_id_feat = [], [], []
    all_y_id_test, all_y_dis_test = [], []
    image_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels_id = batch['label_id'].to(device)
            labels_dis = batch['label_dis'].to(device)

            # Get original image names
            batch_image_names = dataloader.dataset.image_data[batch_idx * dataloader.batch_size:
                                                              (batch_idx + 1) * dataloader.batch_size]
            image_names.extend(batch_image_names)

            # Forward pass
            med_feat, id_feat = feature_extractor(images)

            all_images.append(images)
            all_med_feat.append(med_feat)
            all_id_feat.append(id_feat)
            all_y_id_test.append(labels_id)
            all_y_dis_test.append(labels_dis)

        images = torch.cat(all_images)
        med_feat = torch.cat(all_med_feat)
        id_feat = torch.cat(all_id_feat)
        y_id_test = torch.cat(all_y_id_test)
        y_dis_test = torch.cat(all_y_dis_test)

        disease_outputs = disease_classifier(med_feat)
        identity_outputs = identity_classifier(id_feat) # Logits for both classes


    # Disease Classifier Evaluation
    disease_probs = torch.sigmoid(disease_outputs)
    disease_preds = (disease_probs >= 0.5).int()

    disease_accuracy = (disease_preds == y_dis_test).float().mean().item()
    # print(f"Medical Accuracy with Med vector: {disease_accuracy:.4f}")
    
    roc_auc = roc_auc_score(y_dis_test.cpu().numpy(), disease_probs.cpu().numpy())
    print(f"Medical ROC with Med vector: {roc_auc:.4f}")
    print(classification_report(y_dis_test.cpu(), disease_preds.cpu(), target_names=['not DR', 'DR']))
    print(confusion_matrix(y_dis_test.cpu(), disease_preds.cpu()))


    # Identity Classifier Evaluation
    identity_probs = torch.softmax(identity_outputs, dim=1) #Convert logits to probabilities
    identity_preds = torch.argmax(identity_outputs, dim=1) # Predicted class (0 or 1)
    identity_accuracy = (identity_preds == y_id_test).float().mean().item()
    # print(f"SA Accuracy with SA vector: {identity_accuracy:.4f}")
    
    roc_auc = roc_auc_score(y_id_test.cpu().numpy(), identity_probs[:, 1].cpu().numpy())
    print(f"SA ROC with SA vector: {roc_auc:.4f}")
    print(classification_report(y_id_test.cpu(), identity_preds.cpu()))
    print(confusion_matrix(y_id_test.cpu(), identity_preds.cpu()))



    if df_train is not None:
        # Extract patient IDs from image names
        patient_ids = [int(name.split('.')[0]) for name in image_names]  # Extract patient ID from image name
        # print( f"Patient IDs: {patient_ids}")
        
        # df_train['patient_ID_disen'] = patient_ids
        df_train['true_dis'] = y_dis_test.cpu().numpy() 

        df_train['disen_predicted_icdr'] = disease_preds.cpu().numpy() 
        df_train['disen_prob_predicted_icdr'] = disease_probs.detach().cpu().numpy() 
        df_train['disen_predicted_id'] = identity_preds.cpu().numpy()
        df_train['disen_prob_predicted_id'] = identity_probs[:, 1].detach().cpu().numpy() # Probability of being in class 1
        
        # Save the DataFrame to a CSV file
        df_train.to_csv(os.path.join(save_path, 'df_train_disentangled.csv'), index=False)
        print("Successfully saved df_train!")


    if df_val is not None:
        df_val['disen_predicted_icdr'] = disease_preds.cpu().numpy() 
        df_val['disen_prob_predicted_icdr'] = disease_probs.detach().cpu().numpy() 
        df_val['disen_predicted_id'] = identity_preds.cpu().numpy()
        df_val['disen_prob_predicted_id'] = identity_probs[:, 1].detach().cpu().numpy() # Probability of being in class 1
        
        # Save the DataFrame to a CSV file
        df_val.to_csv(os.path.join(save_path, 'df_val_disentangled.csv'), index=False)
        print("Successfully saved df_val!")


    if df_test is not None:
        df_test['disen_predicted_icdr'] = disease_preds.cpu().numpy() 
        df_test['disen_prob_predicted_icdr'] = disease_probs.detach().cpu().numpy() 
        df_test['disen_predicted_id'] = identity_preds.cpu().numpy()
        df_test['disen_prob_predicted_id'] = identity_probs[:, 1].detach().cpu().numpy() # Probability of being in class 1
        
        # Save the DataFrame to a CSV file
        df_test.to_csv(os.path.join(save_path, 'df_test_disentangled.csv'), index=False)
        print("Successfully saved df_test!")


# ============ Evaluation ============ #
def evaluate_model_val(models, dataloader, device, df_train=None, df_val=None, df_test=None, save_path=None, disentanglement=True):    

    # Set models to evaluation mode
    feature_extractor, generator, disease_classifier, identity_classifier = models
    # feature_extractor, generator, discriminator, disease_classifier, identity_classifier = models

    feature_extractor.eval()
    generator.eval()
    # discriminator.eval()
    disease_classifier.eval()
    identity_classifier.eval()

    val_loss = []
    val_loss_recon = []
    val_loss_dis = []
    val_loss_id = []
    val_loss_disen = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels_id = batch['label_id'].to(device)
            labels_dis = batch['label_dis'].to(device)

            # Forward pass
            med_feat, id_feat = feature_extractor(images)
            
            noisy_med = med_feat + torch.randn_like(med_feat) * 0.1
            noisy_id = id_feat + torch.randn_like(id_feat) * 0.1
            # noisy_oth = oth_feat + torch.randn_like(oth_feat) * 0.1

            fake_images = generator(med_feat, id_feat)
            disease_out = disease_classifier(med_feat)
            identity_out = identity_classifier(id_feat)
            # discriminator_out = discriminator(fake_images)


            # Disentanglement: use altered_med/id/oth 
            feat_med_rec, feat_id_rec = feature_extractor(fake_images)
            
            # Generate new images, using the altered vectors
            fake_images_med = generator(feat_med_rec, noisy_id)
            fake_images_id = generator(noisy_med, feat_id_rec)
            # fake_images_oth = generator(noisy_med, noisy_id, feat_oth_rec)

            # extrair
            feat_med_altered_med, feat_id_altered_med = feature_extractor(fake_images_med)
            feat_med_altered_id, feat_id_altered_id  = feature_extractor(fake_images_id)
            # feat_med_altered_oth, feat_id_altered_oth = feature_extractor(fake_images_oth)

            outputs = {
                'reconstructed': fake_images,
                'features': (med_feat, id_feat),
                'features_noisy': (noisy_med, noisy_id),
                'features_rec': (feat_med_rec, feat_id_rec),

                'disease_out': disease_out,
                'identity_out': identity_out,
                # 'discriminator_out': discriminator_out,

                'feat_altered_med': (feat_med_altered_med, feat_id_altered_med),
                'feat_altered_id': (feat_med_altered_id, feat_id_altered_id),
                # 'feat_altered_oth': (feat_med_altered_oth, feat_id_altered_oth, feat_oth_altered_oth)
            }

            inputs = {'real': images}
            targets = {'disease': labels_dis, 'identity': labels_id}
            weights = {'disc_weight': torch.tensor(1.0, device=device)}
            
            if disentanglement:
                total_loss, loss = compute_custom_loss(inputs, outputs, targets, weights)
            else:
                total_loss, loss = compute_custom_loss_no_disen(inputs, outputs, targets, weights)

            # Store losses
            val_loss.append(total_loss.item())
            val_loss_recon.append(loss['recon_loss'].item())  # Reconstruction loss (Generator)
            val_loss_dis.append(loss['loss_dis'].item())
            val_loss_id.append(loss['loss_id'].item())
            val_loss_disen.append(loss['loss_disentanglement'].item())


    return (
        np.mean(val_loss),
        np.mean(val_loss_recon),
        np.mean(val_loss_dis),
        np.mean(val_loss_id),
        np.mean(val_loss_disen),
    )


# ============ Gender Evaluation ============ #
def evaluate_identity_model(models, dataloader, device, df_train=None, df_val=None, df_test=None, save_path=None):
    
    generator=None

    # Set models to evaluation mode
    feature_extractor, generator, disease_classifier, identity_classifier = models
    # feature_extractor, _, _, disease_classifier, identity_classifier = models

    feature_extractor.eval()
    generator.eval()
    # discriminator.eval()
    disease_classifier.eval()
    identity_classifier.eval()

    val_loss = []
    val_loss_dis = []
    val_loss_id = []
    val_loss_disen = []


    # Extract features from the test set
    all_images, all_med_feat, all_id_feat = [], [], []
    all_y_id_test, all_y_dis_test = [], []
    image_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels_id = batch['label_id'].to(device)
            labels_dis = batch['label_dis'].to(device)

            # Get original image names
            batch_image_names = dataloader.dataset.image_data[batch_idx * dataloader.batch_size:
                                                              (batch_idx + 1) * dataloader.batch_size]
            image_names.extend(batch_image_names)

            # Forward pass
            med_feat, id_feat = feature_extractor(images)

            all_images.append(images)
            all_med_feat.append(med_feat)
            all_id_feat.append(id_feat)
            # all_oth_feat.append(oth_feat)
            all_y_id_test.append(labels_id)
            all_y_dis_test.append(labels_dis)


        images = torch.cat(all_images)
        med_feat = torch.cat(all_med_feat)
        id_feat = torch.cat(all_id_feat)
        # oth_feat = torch.cat(all_oth_feat)
        y_id_test = torch.cat(all_y_id_test)
        y_dis_test = torch.cat(all_y_dis_test)


        #disease_outputs = disease_classifier(med_feat)
        disease_outputs = disease_classifier(id_feat)
        identity_outputs = identity_classifier(med_feat) # Logits for both classes

        # Save outputs
        np.save(os.path.join(save_path, "disease_features.npy"), med_feat.cpu().numpy())
        np.save(os.path.join(save_path, "identity_features.npy"), id_feat.cpu().numpy())



    # Disease Classifier Evaluation
    disease_probs = torch.sigmoid(disease_outputs)
    disease_preds = (disease_probs >= 0.5).int()

    disease_accuracy = (disease_preds == y_dis_test).float().mean().item()
    # print(f"Medical Accuracy with SA vector: {disease_accuracy:.4f}")

    roc_auc = roc_auc_score(y_dis_test.cpu().numpy(), disease_probs.cpu().numpy())
    print(f"Medical ROC with SA vector: {roc_auc:.4f}")
    print(classification_report(y_dis_test.cpu(), disease_preds.cpu(), target_names=['not DR', 'DR']))
    print(confusion_matrix(y_dis_test.cpu(), disease_preds.cpu()))


    # Identity Classifier Evaluation
    identity_probs = torch.softmax(identity_outputs, dim=1) #Convert logits to probabilities
    identity_preds = torch.argmax(identity_outputs, dim=1) # Predicted class (0 or 1)
    identity_accuracy = (identity_preds == y_id_test).float().mean().item()
    
    # print(f"SA Accuracy with med vector: {identity_accuracy:.4f}")
    roc_auc = roc_auc_score(y_id_test.cpu().numpy(), identity_probs[:, 1].cpu().numpy())
    print(f"SA ROC with med vector: {roc_auc:.4f}")
    print(classification_report(y_id_test.cpu(), identity_preds.cpu()))
    print(confusion_matrix(y_id_test.cpu(), identity_preds.cpu()))


    if df_train is not None:
        df_train['pred_id_with_med'] = identity_preds.cpu().numpy() 
        df_train['prob_id_with_med'] = identity_probs[:, 1].detach().cpu().numpy() 
        df_train['pred_dr_with_id'] = disease_preds.cpu().numpy() 
        df_train['prob_dr_with_id'] = disease_probs.detach().cpu().numpy() 

        # Save the DataFrame to a CSV file
        df_train.to_csv(os.path.join(save_path, 'df_train_disentangled.csv'), index=False)
        print("Successfully saved df_train!")


    if df_test is not None:
        df_test['pred_id_with_med'] = identity_preds.cpu().numpy() 
        df_test['prob_id_with_med'] = identity_probs[:, 1].detach().cpu().numpy() 
        df_train['pred_dr_with_id'] = disease_preds.cpu().numpy() 
        df_train['prob_dr_with_id'] = disease_probs.detach().cpu().numpy() 

        # Save the DataFrame to a CSV file
        df_test.to_csv(os.path.join(save_path, 'df_test_disentangled.csv'), index=False)
        print("Successfully saved df_test!")
