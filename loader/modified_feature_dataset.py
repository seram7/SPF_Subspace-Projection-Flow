import os
import re
import sys
import numpy as np
from scipy.io import loadmat
import pickle
from PIL import Image
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.lsun import LSUN
from torchvision.datasets.utils import verify_str_arg, check_integrity
from torchvision.transforms import ToTensor, ToPILImage
from utils import get_shuffled_idx

class Feature_dataset(TensorDataset):
    def __init__(self, root):
        self.root = root
        self.feat_tensor = torch.from_numpy(np.load(self.root+'features.npy'))
        self.label_tensor = torch.from_numpy(np.load(self.root+'labels.npy'))
        super().__init__(self.feat_tensor)
    
    def __getitem__(self, index):
        return self.feat_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.feat_tensor.size(0)

        