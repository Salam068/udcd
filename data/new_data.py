import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import random

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, fold, mode='train', transform=None, nce_p=1, mode_type='exact'):
        self.root_dir = os.path.join(root_dir, mode.capitalize(), f'fold_{fold}')
        self.mode = mode
        self.nce_p = nce_p
        self.mode_type = mode_type
        
        # Get class folders
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        
        # Create class index mapping
        self.class_index = [
            [i for i, path in enumerate(self.get_image_paths()) if os.path.dirname(path).endswith(cls)]
            for cls in self.classes
        ]
        
        # Get all image paths
        self.image_paths = self.get_image_paths()
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def get_image_paths(self):
        image_paths = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            image_paths.extend([os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and transform main image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Get class label
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        label_one_hot = torch.zeros(self.n_classes)
        label_one_hot[label] = 1
        
        # For NCE sampling
        if self.mode_type == 'exact':
            # Exact positive sample (same class)
            same_class_paths = [
                p for p in self.image_paths if os.path.dirname(p) == os.path.dirname(img_path) and p != img_path
            ]
            if same_class_paths:
                pos_img_path = random.choice(same_class_paths)
                pos_image = Image.open(pos_img_path).convert('RGB')
                pos_image = self.transform(pos_image)
            else:
                pos_image = image
        elif self.mode_type == 'relax':
            # Randomly sample a mix of positive and diverse samples
            pos_img_path = random.choice(self.image_paths)
            pos_image = Image.open(pos_img_path).convert('RGB')
            pos_image = self.transform(pos_image)
        else:
            # Multiple positive samples
            pos_image = image
        
        return (image, pos_image), label_one_hot, idx, idx

def load_dataset(args, p=1, mode='exact'):
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Assume data is in a folder structure like: 
    # Data/Train/fold_1/, Data/Test/fold_1/, Data/Val/fold_1/
    root_dir = os.path.join(args.root_path, '..')
    
    # Load train and test datasets
    train_ds = MedicalImageDataset(root_dir, fold=int(args.split[-1]), mode='train', 
                                   transform=transform, nce_p=p, mode_type=mode)
    test_ds = MedicalImageDataset(root_dir, fold=int(args.split[-1]), mode='test', 
                                  transform=transform, nce_p=p, mode_type=mode)
    
    return train_ds, test_ds