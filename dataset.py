# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/2/18
"""
import os
import torch
import cv2
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from utils import LABEL, random_noise, gasuss_noise


class YZMDataset(Dataset):
    def __init__(self, root, file_list, train=True, transform=None):
        split_f = os.path.join(root, file_list)
        self.images = []
        self.labels = []
        with open(split_f, "r") as f:
            for x in f.readlines():
                if x.strip():
                    line = x.split()
                    self.images.append(os.path.join(root, line[0]))
                    self.labels.append(line[1])
        assert (len(self.images) == len(self.labels))
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = cv2.imread(self.images[index])
        if self.train and random.random() < 0.5:
            img = random_noise(img)
        if self.train and random.random() < 0.5:
            img = gasuss_noise(img)
        if self.train and random.random() < 0.5:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.stack([gray, gray, gray], axis=-1)
        target = []
        for i, s in enumerate(self.labels[index]):
            cls = LABEL.index(s)
            target.append(cls)
        target = torch.LongTensor(target)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform():
    transforms = T.Compose([
        T.Resize((40, 100)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms


def get_dataset(root, is_train, file_list):
    dataset = YZMDataset(root=root, file_list=file_list, train=is_train, transform=get_transform())
    return dataset


