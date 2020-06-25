from os.path import join as pjoin
import collections
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data


class PatchLoader(data.Dataset):
    """
        Data loader for the patch-based model
    """
    def __init__(self, split='train', is_transform=True, augmentations=None):
        self.root = 'data_old/'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 7 
        self.mean = 0.000941 # average of the training data  

        if 'train' in self.split: 
            # Normal train/val mode
            self.seismic = np.load(pjoin('data_old', 'train', 'images.npy'))
            self.labels = np.load(pjoin('data_old', 'train', 'labels.npy'))
        elif 'valid' in self.split:
            self.seismic = np.load(pjoin('data_old', 'valid', 'images.npy'))
            self.labels = np.load(pjoin('data_old', 'valid', 'labels.npy'))
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
        #img -= self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()
                
        return img, lbl

    def get_seismic_labels(self):
        return np.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39], [255, 255, 255]])

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb