import torch
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class CityscapesDataset(Dataset):

    def __init__(self, root, transforms=None):
        super(CityscapesDataset, self).__init__()
        self.transforms = transforms
        self.dataset = glob.glob(root+"/train/*.jpg")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        name = self.dataset[idx]
        img = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
        img = img[:, :img.shape[1]//2]
        img = cv2.resize(img, (128,128))
        if self.transforms:
            img = self.transforms(img)
        return img

class tvtLaneDataset(Dataset):

    def __init__(self, root, transforms=None):
        super(tvtLaneDataset, self).__init__()
        self.transforms = transforms
        self.dataset = []
        with open(root, 'r') as image_name:
            while True:
                lines = image_name.readline()
                if not lines:
                    break
                img_dir = lines.strip().split()[:-1]
                for i in range(len(img_dir)):
                    img_dir[i] = img_dir[i][3:]
                self.dataset.append(img_dir[:-1])
        image_name.close()
        self.dataset = [img for nest in self.dataset for img in nest]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        name = self.dataset[idx]
        img = cv2.imread(name)
        if self.transforms:
            img = self.transforms(img)
        return img
        
class CULanesDataset(Dataset):

    def __init__(self, root, transforms=None):
        super(CULanesDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.dataset = glob.glob(root+"/*/*.jpg")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        name = self.dataset[idx]
        img = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img