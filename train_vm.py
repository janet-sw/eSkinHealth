import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision import models

import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image


import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import wandb

import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score

from Model.prepare_model import *
from Data.skindataset import SkinDataset
from utils import *
    


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='skin condition classfication')
    args.add_argument('--model', type=str, default='vitb16', help='model name')
    args.add_argument('--batch_size', type=int, default=128, help='batch size')
    args.add_argument('--num_classes', type=int, default=7, help='number of classes')
    args.add_argument('--epochs', type=int, default=30, help='number of epochs')
    args.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args.add_argument('--data_dir', type=str, default='data/', help='data directory')
    args.add_argument('--save_dir', type=str, default='results/', help='directory to save checkpoints')
    args.add_argument('--train_csv_path', type=str, default='data/train.csv', help='path to csv file')
    args.add_argument('--test_csv_path', type=str, default='data/test.csv', help='path to csv file')
    args.add_argument('--seed', type=int, default=42, help='random seed')
    args.add_argument('--mode', default='linear', type=str, help='linear or full')
    args.add_argument('--reweight', action='store_true', help='reweight the classes')

    args = args.parse_args()
    print(args)
    print()
    device = torch.device("cuda")
    
    model, optimizer, hook_layer = prepare_vm(args.model, args.num_classes, lr=args.lr, mode=args.mode)
    model = model.to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    # Define data transformations
    train_dataset = SkinDataset(data_dir=args.data_dir,
                               csv_path=args.train_csv_path,
                               transform=train_transforms)
    test_dataset = SkinDataset(data_dir=args.data_dir,
                              csv_path=args.test_csv_path,
                              transform=test_transforms)
    label_count = [train_dataset.label_count, test_dataset.label_count] 
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    exp_name = f"{args.model}_{args.mode}_{args.lr}_{len(train_dataset)}_{len(test_dataset)}_{args.num_classes}"
    wandb.init(project="SkinBench", name=exp_name)
    wandb.config.update(args)
    wandb_log = {}
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
        
    # else:
    #     raise ValueError(f"Mode {args.mode} not supported.")
    file_name = f"{len(train_dataset)}_{len(test_dataset)}_{args.num_classes}_{args.lr}"
    save_dir = os.path.join(args.save_dir, args.model, f'{args.mode}_{args.seed}', file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if args.mode == 'linear':
        train_feature, train_labels, train_preds = extract_feature(train_loader, model, hook_layer, device)
        test_feature, test_labels, test_preds = extract_feature(test_loader, model, hook_layer, device)
        
        print(f"Training features shape: {train_feature.shape}")
        print(f"Test features shape: {test_feature.shape}")
        
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_feature), torch.from_numpy(train_labels))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_feature), torch.from_numpy(test_labels))
        
        if args.reweight:
            print("Reweighting the classes")
            unique_classes, counts = np.unique(train_labels, return_counts=True)
            print(f"Unique classes: {unique_classes}")
            print(f"Counts: {counts}")
            weight = 1.0 / counts
            print(f"Weight: {weight}")
            sample_weights = np.array([weight[label] for label in train_labels])
            sample_weights = torch.from_numpy(sample_weights).float()
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = nn.Linear(train_feature.shape[1], args.num_classes).to(device)
    
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        if epoch % (args.epochs//5) == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        wandb_log['epoch'] = epoch
        wandb_log['train loss'] = epoch_loss
        wandb_log['train acc'] = epoch_acc
        
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pred_list.extend(predicted.cpu().numpy())
                label_list.extend(labels.cpu().numpy())
        val_loss = running_loss / len(test_loader.dataset)
        val_acc = correct / total
        f1, accuracy, precision, recall, balanced_acc = plot_confusion_matrix(label_list, pred_list, args.num_classes, file_name, label_count)
        if epoch % (args.epochs//5) == 0:
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        wandb_log['val loss'] = val_loss
        wandb_log['val acc'] = val_acc
        wandb_log['f1'] = f1
        wandb_log['precision'] = precision
        wandb_log['recall'] = recall
        wandb_log['balanced_acc'] = balanced_acc
        wandb.log(wandb_log)
        
        # Save the model if the accuracy is improved
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'pred_list': pred_list,
                'label_list': label_list,
            }
            
            torch.save(ckpt, os.path.join(save_dir, f"best.pth"))
            f1, accuracy, precision, recall, balanced_acc = plot_confusion_matrix(label_list, pred_list, args.num_classes, file_name, label_count)
        
        
    # Save the final model
    final_ckpt = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        'acc': val_acc,
        'f1': f1,
        'precision': precision,
        'balanced_acc': balanced_acc,
        'recall': recall,
        'pred_list': pred_list,
        'label_list': label_list,
    }
    torch.save(final_ckpt, os.path.join(save_dir, f"final.pth"))
    
    wandb.finish()
    # print(f"Best Validation Accuracy: {best_acc:.4f}")
    # torch.save({'y_true': label_list, 'y_pred': pred_list}, f"predictions_{len(label_list)}.pth")
    print(f"FINAL RESULTS:\n")
    print(f"{best_acc*100:.3f}  {best_f1*100:.3f}  {best_precision*100:.3f}  {best_recall*100:.3f}  {balanced_acc*100:.3f}")
    # print(f"{val_acc*100:.3f}  {f1*100:.3f}  {precision*100:.3f}  {recall*100:.3f}  {balanced_acc*100:.3f}")
    print()



