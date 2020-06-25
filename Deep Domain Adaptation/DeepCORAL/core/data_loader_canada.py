from os.path import join as pjoin
import collections
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler

from torch.utils import data


class PatchLoader(data.Dataset):
    """
        Data loader for the patch-based model
    """
    def __init__(self, split='train', is_transform=True, augmentations=None):
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 3 

        if 'train' in self.split: 
            # Normal train/val mode
            self.seismic = np.load(pjoin('data_canada', 'train', 'images.npy'))
            self.labels = np.load(pjoin('data_canada', 'train', 'labels.npy'))
            self.mean = np.round(np.mean(self.seismic), 6) # average of the training data
        elif 'valid' in self.split:
            self.seismic = np.load(pjoin('data_canada', 'valid', 'images.npy'))
            self.labels = np.load(pjoin('data_canada', 'valid', 'labels.npy'))
            self.mean = np.round(np.mean(self.seismic), 6) # average of the training data
        elif 'test' in self.split:
            self.seismic = np.load(pjoin('data_canada', 'test', 'images.npy'))
            self.labels = np.load(pjoin('data_canada', 'test', 'labels.npy'))
            self.mean = np.round(np.mean(self.seismic), 6) # average of the training data
        else:
            raise ValueError('Unknown split.')

    def __len__(self):
        return len(self.seismic)

    def __getitem__(self, index):

        im = self.seismic[index]
        lbl = self.labels[index]

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
            
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        img = img -  self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()
                
        return img, lbl
        
    def get_sampler(self):

        class_count = np.unique(self.labels, return_counts=True)[1]
        weight = 1. / class_count
        samples_weight = weight[self.labels]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        return sampler