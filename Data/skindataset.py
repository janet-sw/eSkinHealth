import pandas as pd
import numpy as np
from PIL import Image
import torch
import os

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.load_data()

    def load_data(self):
        # Load image paths and labels from CSV file
        df = pd.read_csv(self.csv_path)
        condition_list = df.label.unique().tolist()
        condition_list.sort()
        label_mapping = {condition: i for i, condition in enumerate(condition_list)}
        self.label_mapping = label_mapping
        label_count = [0] * len(condition_list)
        print(f'{len(condition_list)} conditions found: {label_mapping}')
        for index, row in df.iterrows():
            self.image_paths.append(os.path.join(self.data_dir, f"{row['image_id']}.jpg"))
            self.labels.append(label_mapping[row['label']])  
            label_count[label_mapping[row['label']]] += 1
        
        print(f'Label count: {label_count}')        
        print()
        self.label_count = label_count
        self.class_names = list(label_mapping.keys())
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label