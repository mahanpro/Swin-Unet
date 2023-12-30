import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from PIL import Image
import logging
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class Synapse_dataset(Dataset):
    def __init__(self, indices, base_dir, label_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.indices = indices
        self.sample_list = open(os.path.join(list_dir +'/train.txt')).readlines()
        # print(self.sample_list)
        self.data_dir = base_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # idx = self.indices[idx]
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path  = os.path.join(self.data_dir, slice_name)
            label_path = os.path.join(self.label_dir, slice_name)
            image = np.load(data_path)
            label = np.load(label_path)
            image = image['arr_0']
            label = label['arr_0']
        elif self.split == "validation":
            slice_name = self.sample_list[idx].strip('\n')
            data_path  = os.path.join(self.data_dir, slice_name)
            label_path = os.path.join(self.label_dir, slice_name)
            image = np.load(data_path)
            label = np.load(label_path)
            image = image['arr_0']
            label = label['arr_0']
        
        if self.split == "train" and self.transform:
            image = self.transform(image)
            label = self.transform(label)
        # if self.transform:
        # image = self.transform(image)
        # label = self.transform(label)

        return image, label, slice_name
