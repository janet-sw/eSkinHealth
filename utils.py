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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from utils import *

def extract_feature(data_loader, model, layer, device):
    features = []
    outputs = []
    targets = []

    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    forward_hook = layer.register_forward_hook(hook_fn_forward)

    model.eval()
    model.to(device)
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            targets.append(target)
            data = data.to(device)
            # print(data.shape, _)
            _ = model(data)

    forward_hook.remove()

    features = torch.cat([x for x in features]).numpy()
    outputs = torch.cat([x for x in outputs])
    predictions = F.softmax(outputs, dim=-1).numpy()
    targets = torch.cat([x for x in targets]).numpy()

    return features, targets, predictions

def plot_confusion_matrix(y_true, y_pred, classes, title, label_count, save_path=None, plot=False): 

    conf_mat = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    

    if plot:
        # Plotting the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(5)
        plt.xticks(tick_marks, range(5), rotation=45)
        plt.yticks(tick_marks, range(5))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{title} || Acc {accuracy:.4f} || F1 {f1:.4f}')

        # Annotate each cell with the numeric value
        thresh = conf_mat.max() / 2.
        for i, j in np.ndindex(conf_mat.shape):
            plt.text(j, i, conf_mat[i, j],
                    horizontalalignment="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
            
        plt.text(0.5, -0.15, f'Train: {label_count[0]}', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.5, -0.2, f'Test: {label_count[1]}', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(f'{save_path}')
        plt.close()
    return f1, accuracy, precision, recall, balanced_acc