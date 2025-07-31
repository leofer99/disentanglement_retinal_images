import os
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np


# get 0_folder
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

# add scr_geral
sys.path.append(os.path.join(parent_dir, 'src_geral'))

from src_disentanglement.init_train_eval import (
    initialize_models, setup_optimizers, train_one_epoch, load_model, evaluate_model_val
)


# --- CONFIGURATION ---
folder_name = 'bal_augTrue_dinov_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall'
save_dir= os.path.join(parent_dir, 'Disentanglement', 'Models (mBRSET)', folder_name)

start_epoch = 150                    # The epoch you are resuming from
resume_epochs = 600                   # How many more epochs to train
latent_dim = 256
num_classes = 2
model_name = 'dinov'  
learning_rate_gen = 5e-5
decay = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD DATA ---
with open(os.path.join(save_dir, "dataloaders.pkl"), "rb") as f:
    dataloaders = pickle.load(f)
train_dataloader = dataloaders["train_dataloader"]
val_dataloader = dataloaders["val_dataloader"]

# --- LOAD LOSSES ---
losses_path = os.path.join(save_dir, "losses.pkl")
with open(losses_path, "rb") as f:
    losses = pickle.load(f)

# --- LOAD MODEL ---
model_path = os.path.join(save_dir, f"model_epoch_{start_epoch}.pt")
models = initialize_models(latent_dim, num_classes, device, model=model_name)
load_model(model_path, device, models)
feature_extractor, generator, disease_classifier, identity_classifier = models

# --- OPTIMIZER ---
opt = setup_optimizers(models, learning_rate_gen=learning_rate_gen, decay=decay)

# --- RESUME TRAINING ---
for epoch in range(start_epoch + 1, start_epoch + resume_epochs + 1):
    print(f"Epoch {epoch}/{start_epoch+resume_epochs}")

    total_loss, loss_recon, loss_disen, loss_dis, loss_id = train_one_epoch(
        models=models, dataloader=train_dataloader, opt=opt, device=device, disentanglement=True
    )
    # Store training losses
    losses['epoch_losses'].append(np.mean(total_loss))
    losses['epoch_loss_recon'].append(np.mean(loss_recon))
    losses['epoch_loss_dis'].append(np.mean(loss_dis))
    losses['epoch_loss_id'].append(np.mean(loss_id))
    losses['epoch_loss_disen'].append(np.mean(loss_disen))

    # Store validation losses
    val_loss, val_recon, val_dis, val_id, val_disen = evaluate_model_val(
        models=models, dataloader=val_dataloader, device=device, disentanglement=True
    )
    losses['val_losses'].append(val_loss)
    losses['val_loss_recon'].append(val_recon)
    losses['val_loss_dis'].append(val_dis)
    losses['val_loss_id'].append(val_id)
    losses['val_loss_disen'].append(val_disen)

    # Save model at specific epochs
    if epoch in [200, 300] or epoch == start_epoch + resume_epochs:
        torch.save({
            'feature_extractor': feature_extractor.state_dict(),
            'generator': generator.state_dict(),
            'disease_classifier': disease_classifier.state_dict(),
            'identity_classifier': identity_classifier.state_dict(),
        }, os.path.join(save_dir, f"model_epoch_{epoch}.pt"))
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)
        print(f"Checkpoint and losses saved at epoch {epoch}")

# --- SAVE LOSSES ---
losses_save_path = os.path.join(save_dir, 'losses.pkl')
with open(losses_save_path, 'wb') as f:
    pickle.dump(losses, f)
print(f"Loss variables saved to {losses_save_path}")


# --- PLOT LOSSES AFTER TRAINING---
epoch_total_losses = losses['epoch_losses']
epoch_loss_recon = losses['epoch_loss_recon']
epoch_loss_dis = losses['epoch_loss_dis']
epoch_loss_id = losses['epoch_loss_id']
epoch_loss_disen = losses['epoch_loss_disen']

val_total_losses = losses['val_losses']
val_loss_recon = losses['val_loss_recon']
val_loss_dis = losses['val_loss_dis']
val_loss_id = losses['val_loss_id']
val_loss_disen = losses['val_loss_disen']

plt.figure(figsize=(12, 8))
plt.plot(epoch_total_losses, label='Generator Total Loss')
# plt.plot(disc_real_losses, label='Discriminator Real Loss')
# plt.plot(disc_fake_losses, label='Discriminator Fake Loss')
plt.plot(epoch_loss_recon, label='Reconstruction Loss')
plt.plot(epoch_loss_dis, label='Disease Loss')
plt.plot(epoch_loss_id, label='Identity Loss')
plt.plot(epoch_loss_disen, label='Disentanglement Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses Over Epochs')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, 'training_losses.png'))  # Save the plot
plt.show()

# Plot the validation losses 
plt.figure(figsize=(12, 8))
plt.plot(val_total_losses, label='Validation Generator Total Loss')
# plt.plot(disc_real_losses, label='Discriminator Real Loss')
# plt.plot(disc_fake_losses, label='Discriminator Fake Loss')
plt.plot(val_loss_recon, label='Validation Reconstruction Loss')
plt.plot(val_loss_dis, label='Validation Disease Loss')
plt.plot(val_loss_id, label='Validation Identity Loss')
plt.plot(val_loss_disen, label='Validation Disentanglement Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Losses Over Epochs')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, 'validation_losses.png'))  # Save the plot
plt.show()

# Plot the training vs validation losses
plt.figure(figsize=(12, 8))
plt.plot(epoch_total_losses, label='Training Generator Total Loss')
plt.plot(val_total_losses, label='Validation Generator Total Loss')

plt.plot(epoch_loss_recon, label='Training Reconstruction Loss')
plt.plot(val_loss_recon, label='Validation Reconstruction Loss')
plt.plot(epoch_loss_dis, label='Training Disease Loss')
plt.plot(val_loss_dis, label='Validation Disease Loss')
plt.plot(epoch_loss_id, label='Training Identity Loss')
plt.plot(val_loss_id, label='Validation Identity Loss')
plt.plot(epoch_loss_disen, label='Training Disentanglement Loss')
plt.plot(val_loss_disen, label='Validation Disentanglement Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, 'training_validation_losses.png'))  # Save the plot
plt.show()
