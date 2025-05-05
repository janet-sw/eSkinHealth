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
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, SiglipProcessor


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
    args.add_argument('--mode', default='zero_shot', type=str, help='linear or full')
    args.add_argument('--reweight', action='store_true', help='reweight the classes')
    args.add_argument('--template', type=str, default='default', help='template for text embedding')

    args = args.parse_args()
    print(args)
    device = torch.device("cuda")
    
    # model, preprocess = prepare_vlm(args.model, mode=args.mode)
    if 'clip_' in args.model:
        model, preprocess = clip.load(clip_model_dict[args.model], device='cpu')
        model.eval()
    elif 'siglip' in args.model:
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        img_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        def preprocess(image):
            processed = img_processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)  # remove batch dimension
            return pixel_values

    model = model.to(device)
    # Define data transformations
    train_dataset = SkinDataset(data_dir=args.data_dir,
                               csv_path=args.train_csv_path,
                               transform=preprocess)
    test_dataset = SkinDataset(data_dir=args.data_dir,
                              csv_path=args.test_csv_path,
                              transform=preprocess)
    label_count = [train_dataset.label_count, test_dataset.label_count] 
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    print()
    
    
    if args.template == 'default':
        template = DEFAULT_TEMPLATE
    elif args.template == 'custom':
        template = CUSTOM_TEMPLATE
        import json

        # Load JSON file
        with open('./Model/visual_feature_summaries.json', 'r') as f:
            description_dict = json.load(f)
    class_names = train_dataset.class_names
    if 'clip_' in args.model:
        templates = [template]
        # txt_emb = get_text_ensemble_embedding(class_names, templates, model)
        with torch.no_grad():
            zeroshot_weights = []
            for classname in class_names:
                if args.template == 'default':
                    texts = [template.format(classname) for template in templates]
                elif args.template == 'custom':
                    texts = [template.format(classname, description_dict[classname]) for template in templates]
                texts = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
            
        txt_emb = zeroshot_weights
        print(len(templates), "templates")
        print(f"{len(class_names)} Class Names: {class_names}")
        print(f"Text Embedding Shape: {txt_emb.shape}")
        print()
        
        emb_names = np.array([f"T{i // len(class_names)} {class_names[i % len(class_names)]}" for i in range(txt_emb.size(0))])
        
        def network(x):
            x_emb = model.encode_image(x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            # logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
            logits = model.logit_scale.exp() * x_emb @ txt_emb
            return logits
        
    elif 'siglip' in args.model:
        with torch.no_grad():
            if args.template == 'default':
                texts = [template.format(classname) for classname in class_names]
                text_inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
            elif args.template == 'custom':
                texts = [template.format(classname, description_dict[classname]) for classname in class_names]
                text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
                
            text_embeddings = model.get_text_features(**text_inputs)
            text_embeds = model.get_text_features(**text_inputs)
            # normalize embeddings
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            txt_emb = text_embeds
        
        def network(x):
            # x is a preprocessed tensor (already passed through transform/preprocessor)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)  # add batch dimension if needed

            x_emb = model.get_image_features(pixel_values=x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)

            logits = model.logit_scale.exp() * (x_emb @ txt_emb.T)  # note the transpose
            return logits
        

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    exp_name = f"{args.model}_{args.mode}_{args.lr}_{len(train_dataset)}_{len(test_dataset)}_{args.num_classes}"
    # wandb.init(project="SkinBench", name=exp_name)
    # wandb.config.update(args)
    # wandb_log = {}
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
        
    # else:
    #     raise ValueError(f"Mode {args.mode} not supported.")
    file_name = f"{len(train_dataset)}_{len(test_dataset)}_{args.num_classes}_{args.lr}"
    save_dir = os.path.join(args.save_dir, args.model, f'{args.mode}_{args.seed}', file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    
    best_acc = 0.0
    for epoch in range(1):
        # model.train()
        # running_loss = 0.0
        # correct = 0
        # total = 0
        
        # for images, labels in train_loader:
        #     images, labels = images.to(device), labels.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
            
        #     running_loss += loss.item() * images.size(0)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
        
        # scheduler.step()
        # epoch_loss = running_loss / len(train_loader.dataset)
        # epoch_acc = correct / total
        # if epoch % (args.epochs//5) == 0:
        #     print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        # wandb_log['epoch'] = epoch
        # wandb_log['train loss'] = epoch_loss
        # wandb_log['train acc'] = epoch_acc
        
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
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
        # wandb_log['val loss'] = val_loss
        # wandb_log['val acc'] = val_acc
        # wandb_log['f1'] = f1
        # wandb_log['precision'] = precision
        # wandb_log['recall'] = recall
        # wandb_log['balanced_acc'] = balanced_acc
        # wandb.log(wandb_log)
        
        # Save the model if the accuracy is improved
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
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
        # 'optimizer_state_dict': optimizer.state_dict(),
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
    
    # wandb.finish()
    # print(f"Best Validation Accuracy: {best_acc:.4f}")
    # torch.save({'y_true': label_list, 'y_pred': pred_list}, f"predictions_{len(label_list)}.pth")
    print(f"FINAL RESULTS:\n")
    print(f"{best_acc*100:.3f}  {best_f1*100:.3f}  {best_precision*100:.3f}  {best_recall*100:.3f}  {balanced_acc*100:.3f}")
    # print(f"{val_acc*100:.3f}  {f1*100:.3f}  {precision*100:.3f}  {recall*100:.3f}  {balanced_acc*100:.3f}")
    print()



