import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np

def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=50, backbone='Retina', save=False, num_classes=2, device='cpu', patience=7):
    model.to(device)

    if num_classes==2:
        binary=True
    else: 
        binary=False

    # print(f"Binary: {binary}")
    train_losses = []
    val_losses = []
    f1_scores = []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,
    }
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)

        for batch in tqdm(train_dataloader, total=num_train_batches):
            inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # print("Inputs: ", inputs)
            outputs = model(inputs)

            if hasattr(outputs, 'logits'): 
                outputs = outputs.logits

            if not binary: #yes
                labels= torch.argmax(labels, dim=1)  #return the index of the class with the maximum value for each sample.
            
            # print("Binary: ", outputs)
            # print("labels: ", labels)

            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_train_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():

            for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                val_inputs = val_batch['image'].to(device)
                val_labels = val_batch['labels'].to(device)

                val_outputs = model(val_inputs) #.logits

                # print(f"Preds_: {val_outputs}")
                # print(f"Labels_: {val_labels}")

                if hasattr(val_outputs, 'logits'): 
                    val_outputs = val_outputs.logits
                
                if not binary: 
                    val_labels= torch.argmax(val_labels, dim=1)
                    # print(f"Labels_1: {val_labels}")

                val_loss += criterion(val_outputs, val_labels).item()

                if binary:
                    probs = torch.sigmoid(val_outputs)
                    preds = (probs > 0.5).float()
                    # print(f"Probs_: {probs}")

                else:
                    preds = torch.argmax(val_outputs, dim=1)  
                    # labels= torch.argmax(val_labels, dim=1)
                
                labels = val_labels
                # print(f"Preds: {preds}")
                # print(f"Labels: {labels}")

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        # print("y_true:", type(all_labels))
        # print("y_pred:", type(all_preds))
        # print("y_true shape:", len(all_labels))
        # print("y_pred shape:", len(all_preds))
        # print("y_true:", all_labels)
        # print("y_pred:", all_preds)


        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        f1_scores.append(f1)
        confusion_matrix_sc = confusion_matrix(all_labels, all_preds)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, F1 Score: {f1}, acc{acc}')
        print(f"cm{confusion_matrix_sc}")
        

        if f1 > best_model_info['f1_score']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break



    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

    if save:
        os.makedirs('Models', exist_ok=True)
        model.load_state_dict(best_model_info['state_dict'])
        torch.save(model.state_dict(), f'Models/fine_tuned_{backbone}_best.pth')

    return model, train_losses, val_losses