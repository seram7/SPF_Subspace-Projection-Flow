"""
modified_dataset.py
===================
Inherited and modified pytorch datasets
"""
import os
import re
import sys
import numpy as np
from scipy.io import loadmat
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.lsun import LSUN
from torchvision.datasets.utils import verify_str_arg, check_integrity
from torchvision.transforms import ToTensor, ToPILImage
from utils import get_shuffled_idx


class Gray2RGB:
    """change grayscale PIL image to RGB format.
    channel values are copied"""
    def __call__(self, x):
        return x.convert('RGB')


# class MNIST_OOD(MNIST):
    # """
    # See also the original MNIST class: 
    #     https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    # """
    # def __init__(self, root, split='training', transform=None, target_transform=None,
    #              download=False, seed=1):
    #     super(MNIST_OOD, self).__init__(root, transform=transform,
    #                                     target_transform=target_transform, download=download)
    #     assert split in ('training', 'validation', 'evaluation')
    #     num_data = len(self.data) #60000
    #     num_train_data = int(0.9*num_data)
    #     if split == 'training' or split == 'validation':
    #         self.train = True
    #         shuffle_idx = get_shuffled_idx(num_data, seed)
    #     else:
    #         self.train = False
    #     self.split = split

    #     if download:
    #         self.download()

    #     if not self._check_exists():
    #         raise RuntimeError('Dataset not found.' +
    #                            ' You can use download=True to download it')

    #     if self.train:
    #         data_file = self.training_file
    #     else:
    #         data_file = self.test_file

    #     data, targets = torch.load(os.path.join(self.processed_folder, data_file))

    #     if split == 'training':
    #         self.data = data[shuffle_idx][:num_train_data]
    #         self.targets = targets[shuffle_idx][:num_train_data]
    #     elif split == 'validation':
    #         self.data = data[shuffle_idx][num_train_data:]
    #         self.targets = targets[shuffle_idx][num_train_data:]
    #     elif split == 'evaluation':
    #         self.data = data
    #         self.targets = targets

    # @property
    # def raw_folder(self):
    #     return os.path.join(self.root, 'MNIST', 'raw')

    # @property
    # def processed_folder(self):
    #     return os.path.join(self.root, 'MNIST', 'processed')

class MNIST_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(MNIST_OOD, self).__init__()
        self.root_dir = root #./datasets
        print(os.path.join(root, 'mnist'))
        self.split = split
        self.transform = transform

        self.list_dir = os.path.join(root, 'datalists/test_mnist.txt')
        self.listfile = open(self.list_dir, 'r')
        self.train_list = self.listfile.readlines()
        self.l_img_file = []
        for line in self.train_list:
            tokens = line.split(' ', 1)
            img_name = tokens[0]
            real_dir = os.path.join(self.root_dir, img_name)
            self.l_img_file.append(real_dir)

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)\


class FashionMNIST_OOD(FashionMNIST):
    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=False, seed=1):
        super(FashionMNIST_OOD, self).__init__(root, transform=transform,
                                        target_transform=target_transform, download=download)
        assert split in ('training', 'validation', 'evaluation')
        num_data = len(self.data) #50000
        num_train_data = int(0.9*num_data)
        if split == 'training' or split == 'validation':
            self.train = True
            shuffle_idx = get_shuffled_idx(60000, seed)
        else:
            self.train = False
        self.split = split

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data, targets = torch.load(os.path.join(self.processed_folder, data_file))

        if split == 'training':
            self.data = data[shuffle_idx][:54000]
            self.targets = targets[shuffle_idx][:54000]
        elif split == 'validation':
            self.data = data[shuffle_idx][54000:]
            self.targets = targets[shuffle_idx][54000:]
        elif split == 'evaluation':
            self.data = data
            self.targets = targets

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'FashionMNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'FashionMNIST', 'processed')

class CIFAR10_OOD(CIFAR10):
    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=True, seed=1):

        super(CIFAR10_OOD, self).__init__(root, transform=transform,
                                          target_transform=target_transform, download=True, train=(split in ['training', 'validation', 'training_full']))
        assert split in ('training', 'validation', 'evaluation', 'training_full')
        self.split = split

        num_data = len(self.data) #50000
        num_train_data = int(0.9*num_data)
        if self.train==True:
            shuffle_idx = get_shuffled_idx(num_data, seed)

        self.targets = torch.tensor(self.targets)
        if split == 'training':
            self.data = self.data[shuffle_idx][:num_train_data]
            self.targets = self.targets[shuffle_idx][:num_train_data]
        elif split == 'validation':
            self.data = self.data[shuffle_idx][num_train_data:]
            self.targets = self.targets[shuffle_idx][num_train_data:]
        elif split == 'training_full':
            pass
        self._load_meta()

class CIFAR100_OOD(CIFAR10_OOD):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class CIFAR10_OOD_noise(CIFAR10): 
    ### noise ratio should be added instead of using '500', '10000', '30000'...
    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=True, seed=1, noise_type= 'uniform', noise_value=0, classn = 10):
        self.split = split

        self.noise_type = noise_type
        self.noise_value = noise_value
        self.classn = classn
        self.rng = np.random.default_rng(seed)
        
        self.check_cnt = [0]*self.classn
        self.targetscomp = torch.tensor([101]*50000)

        super(CIFAR10_OOD_noise, self).__init__(root, transform=transform,
                                          target_transform=target_transform, download=True, train=(split in ['training', 'validation', 'training_full']))
        assert split in ('training', 'validation', 'evaluation', 'training_full')

        num_data = len(self.data) #50000
        num_train_data = int(0.9*num_data)
        # if self.train==True:
        #     shuffle_idx = get_shuffled_idx(num_data, seed)

        self.targets = torch.tensor(self.targets)

        if self.noise_type == 'uniform':
            self.triggerone = False
            shuffle_idx_noise = get_shuffled_idx(num_data, seed+1)
            self.data_cnt = [0] * self.classn
            noise_index = self.rng.choice(self.classn-1, int(num_data/self.classn *self.noise_value), replace = True)
            for i in shuffle_idx_noise:
                for j in range(self.classn):
                    if self.targets[i] == j:
                        if self.data_cnt[j] < int(500*self.noise_value/self.classn):
                            if self.targetscomp[i]<101:
                                raise NotImplementedError
                            if self.targets[i] > noise_index[self.data_cnt[j]+int(500*self.noise_value*j/self.classn)]:
                                self.targetscomp[i] = noise_index[self.data_cnt[j]+int(500*self.noise_value*j/self.classn)]
                            else:
                                self.targetscomp[i] = noise_index[self.data_cnt[j]+int(500*self.noise_value*j/self.classn)]+1
                            self.data_cnt[j] += 1
            for i in range(50000):
                if self.targetscomp[i] != 101:
                    self.targets[i]=self.targetscomp[i]
            self.targetscomp = self.targetscomp[shuffle_idx]

        elif self.noise_type == 'imbalanced':
            shuffle_idx_noise = get_shuffled_idx(num_data, seed+1)
            self.data_number = np.array([int(num_data/self.classn)] * self.classn)
            self.data_cnt = np.array([0] * self.classn)
            imbalanced_index = self.rng.choice(self.classn, self.classn/2, replace = False)
            self.data_number[imbalanced_index] = 10000/self.classn
            for i in shuffle_idx_noise:
                for j in range(self.classn):
                    if self.targets[i]==j:
                        if self.data_cnt[j] == self.data_number[j]:
                            self.targets[i] = 101
                        else:
                            self.data_cnt[j] += 1
            self.data = self.data[self.targets!=101]
            self.targets = self.targets[self.targets!=101]

        elif self.noise == 'imbalanced_comp':
            shuffle_idx_noise = get_shuffled_idx(num_data, seed+1)
            self.data_cnt = np.array([0]* self.classn)
            self.data_number = np.array([int(30000/self.classn)]*self.classn)
            for i in shuffle_idx_noise:
                for j in range(self.classn):
                    if self.targets[i]==j:
                        if self.data_cnt[j] == self.data_number[j]:
                            self.targets[i] = 101
                        else:
                            self.data_cnt[j] += 1
            self.data = self.data[self.targets!=101]
            self.targets = self.targets[self.targets!=101]



        """
        #elif self.noise == 'aggregation_superclass':
        """     
        if split == 'training':
            if self.noise_type == 'imbalanced' or self.noise_type=='imbalanced_comp':
                shuffle_idx = get_shuffled_idx(30000, seed)
                self.data = self.data[shuffle_idx][:27000]
                self.targets = self.targets[shuffle_idx][:27000]
            else:
                self.data = self.data[shuffle_idx][:45000]
                self.targets = self.targets[shuffle_idx][:45000]
                
        elif split == 'validation':
            if self.noise_type == 'imbalanced' or self.noise_type == 'imbalanced_comp':
                shuffle_idx = get_shuffled_idx(30000, seed)
                self.data = self.data[shuffle_idx][27000:]
                self.targets = self.targets[shuffle_idx][27000:]
            else:
                self.data = self.data[shuffle_idx][45000:]
                self.targets = self.targets[shuffle_idx][45000:]
        elif split == 'training_full':
            pass

        self._load_meta()
"""
    def classtosuperclass(self, x, mode):
        coarselabel = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        samegroup = []
        semigroup = []
        for i in range(20):
            for j in range(100):
                if coarselabel[j]==i:
                    semigroup.append(j)
            samegroup.append(semigroup)
            semigroup=[]
"""    

class CIFAR100_OOD_noise(CIFAR10_OOD_noise):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class SVHN_OOD(SVHN):

    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=True, seed=1):
        if split == 'training' or split == 'validation':
            svhn_split = 'train'
        else:
            svhn_split = 'test'
        assert split in ('training', 'validation', 'evaluation')

        super().__init__(root, transform=transform,
                         target_transform=target_transform, split=svhn_split, download=download)
        
        num_data = len(self.data) #73257
        num_train_data = int(0.9*num_data)
        if self.split=='train':
            shuffle_idx = get_shuffled_idx(num_data, seed)

        if split == 'training':
            self.data = self.data[shuffle_idx][:num_train_data]
            self.labels = self.labels[shuffle_idx][:num_train_data]
        elif split == 'validation':
            self.data = self.data[shuffle_idx][num_train_data:]
            self.labels = self.labels[shuffle_idx][num_train_data:]


class Constant_OOD(Dataset):
    def __init__(self, root, split='training', size=(32, 32), transform=None, channel=3, seed=1):
        super(Constant_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.split = split
        self.transform = transform
        self.root = root
        self.img_size = size
        self.channel = channel
        rng = np.random.default_rng(seed)

        if split == 'training':
            self.vals = rng.random((32000,3)) 
        elif split == 'validation':
            self.vals = rng.random((4000, 3)) 
        elif split == 'evaluation':
            self.vals = rng.random((4000, 3)) 

    def __getitem__(self, index):
        img = np.ones(self.img_size + (self.channel,), dtype=np.float32) * self.vals[index]  # (H, W, C)
        # convert into PIL image
        img = Image.fromarray((img * 255).astype(np.uint8))
        if self.channel == 1:
            img = img[:, :, 0:1]

        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.vals)


class ConstantGray_OOD(Dataset):
    def __init__(self, root, split='training', size=(32, 32), transform=None, channel=3, seed=1):
        super(ConstantGray_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.split = split
        self.transform = transform
        self.root = root
        self.img_size = size
        self.channel = channel
        rng = np.random.default_rng(seed)

        if split == 'training':
            self.vals = rng.random((32000, 3)) 
        elif split == 'validation':
            self.vals = rng.random((4000, 3)) 
        elif split == 'evaluation':
            self.vals = rng.random((4000, 3)) 

    def __getitem__(self, index):
        img = np.ones(self.img_size + (self.channel,), dtype=np.float32) * self.vals[index]  # (H, W, C)
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.vals)


class Noise_OOD(Dataset):
    def __init__(self, root, split='training', transform=None, channel=3, size=(32,32), seed=1):
        super(Noise_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.split = split
        self.transform = transform
        self.root = root
        rng = np.random.default_rng(seed)
        self.channel = channel
        self.size = size

        if split == 'training':
            self.vals = rng.random((32000, 32, 32, 3)) 
        elif split == 'validation':
            self.vals = rng.random((4000, 32, 32, 3)) 
        elif split == 'evaluation':
            self.vals = rng.random((4000, 32, 32, 3)) 

    def __getitem__(self, index):
        img = self.vals[index]
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.channel == 1:
            img = img[:, :, 0:1]
        if self.size != (32, 32):
            img = img[:self.size[0], :self.size[1], :]
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.vals)
    

class LSUN_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        """
        Only for split = 'evaluation'
        The number of images : 10000
        """
        super().__init__()
        self.root = os.path.join(root, 'LSUN/test')
        print(self.root)
        self.split = split
        self.transform = transform
        self.shuffle_idx = get_shuffled_idx(10000, seed)

        self.imgdir = self.root
        self.l_img_file = sorted(os.listdir(self.imgdir))
        

    def __getitem__(self, index):
        imgpath = os.path.join(self.imgdir, self.l_img_file[index])
        im = Image.open(imgpath)
        if self.transform is not None:
            im = self.transform(im)
        return im, 0 

    def __len__(self):
        return len(self.l_img_file)

class ISUN_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        """
        Only for split = 'evaluation'
        The number of images : 8925
        """
        super().__init__()
        self.root = os.path.join(root, 'iSUN/iSUN_patches')
        print(self.root)
        self.split = split
        self.transform = transform
        self.shuffle_idx = get_shuffled_idx(8925, seed)

        self.imgdir = self.root
        self.l_img_file = sorted(os.listdir(self.imgdir))  

    def __getitem__(self, index):
        imgpath = os.path.join(self.imgdir, self.l_img_file[index])
        im = Image.open(imgpath)
        if self.transform is not None:
            im = self.transform(im)
        return im, 0 

    def __len__(self):
        return len(self.l_img_file)

class Places365_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(Places365_OOD, self).__init__()
        self.root_dir = root #./datasets
        print(os.path.join(root, 'places365'))
        self.split = split
        self.transform = transform

        self.list_dir = os.path.join(root, 'datalists/test_places365.txt')
        self.listfile = open(self.list_dir, 'r')
        self.train_list = self.listfile.readlines()
        self.l_img_file = []
        for line in self.train_list:
            tokens = line.split(' ', 1)
            img_name = tokens[0]
            real_dir = os.path.join(self.root_dir, img_name)
            self.l_img_file.append(real_dir)

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)

class TinyImageNet_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        assert split in ('validation', 'evaluation')

        super(TinyImageNet_OOD, self).__init__()
        self.root_dir = root #./datasets
        print(os.path.join(root, 'tin'))
        self.split = split
        self.transform = transform

        if split == 'validation': # validation split in OpenOOD V1.5
            self.list_dir = os.path.join(root, 'datalists/val_tin.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                real_dir = os.path.join(self.root_dir, img_name)
                self.l_img_file.append(real_dir)

        elif split == 'evaluation': 
            self.list_dir = os.path.join(root, 'datalists/test_tin.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                real_dir = os.path.join(self.root_dir, img_name)
                self.l_img_file.append(real_dir)
        else:
            raise ValueError(f'{split}')

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)

class ImageNet1K_OOD(Dataset):
    def __init__(self, root, split='training', transform=None, seed=1):

        super(ImageNet1K_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.root_dir = root #./datasets
        print(os.path.join(root, 'imagenet_1k'))
        self.split = split
        self.transform = transform

        if split == 'training':  # whole train split
            self.list_dir = os.path.join(root, 'datalists/train_imagenet.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            self.targets = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                label = int(tokens[1].split('\n',1)[0])
                real_dir = os.path.join(self.root_dir, img_name)
                if(os.path.exists(real_dir)):
                    self.l_img_file.append(real_dir)
                    self.targets.append(label)

        elif split == 'evaluation': 
            '''
                ImageNet_1K/test : 100,000 images
                ImageNet_1K/val : 50,000 images
                OpenOOD v1.5 : split 'ImageNet_1K/val' into 45,000 test images and 5,000 validation images
                                i.e. 'ImageNet_1K/test' is not used
            '''
            self.list_dir = os.path.join(root, 'datalists/test_imagenet.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            self.targets = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                label = int(tokens[1].split('\n',1)[0])
                real_dir = os.path.join(self.root_dir, img_name)
                if(os.path.exists(real_dir)):
                    self.l_img_file.append(real_dir)
                    self.targets.append(label)


        elif split == 'validation':
            '''
                ImageNet_1K/test : 100,000 images
                ImageNet_1K/val : 50,000 images
                OpenOOD v1.5 : split 'ImageNet_1K/val' into 45,000 test images and 5,000 validation images
                                i.e. 'ImageNet_1K/test' is not used
            '''
            self.list_dir = os.path.join(root, 'datalists/val_imagenet.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            self.targets = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                label = int(tokens[1].split('\n',1)[0])
                real_dir = os.path.join(self.root_dir, img_name)
                if(os.path.exists(real_dir)):
                    self.l_img_file.append(real_dir)
                    self.targets.append(label)
        else:
            raise ValueError(f'{split}')

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        label = self.targets[index]
        #label = torch.tensor(label)
        return im, label

    def __len__(self):
        return len(self.l_img_file)
    

class ImageNet1K_ViM_OOD(Dataset):
    def __init__(self, root, split='training', transform=None, seed=1):
        
        super(ImageNet1K_ViM_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')

        self.root_dir = os.path.join(root, 'Imagenet_1K')
        print(self.root_dir)
        self.split = split
        self.transform = transform

        self.imgdir = self.root_dir

        if split == 'training':  # whole train split
            self.list_dir = os.path.join(root, 'datalists/imagenet2012_train_random_200k.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                real_dir = os.path.join(self.root_dir, line)
                dir_split = real_dir.split()
                real_dir = dir_split[0]
                self.l_img_file.append(real_dir)

        elif split == 'validation':  # whole train split
            self.list_dir = os.path.join(root, 'datalists/imagenet2012_train_random_200k.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                real_dir = os.path.join(self.root_dir, line)
                dir_split = real_dir.split()
                real_dir = dir_split[0]
                self.l_img_file.append(real_dir)

        elif split == 'evaluation':  # whole val split
            self.list_dir = os.path.join(root, 'datalists/imagenet2012_val_list.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                real_dir = os.path.join(self.root_dir, line)
                dir_split = real_dir.split()
                real_dir = dir_split[0]
                self.l_img_file.append(real_dir)
        else:
            raise ValueError(f'{split}')

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)


class ImageNet_O_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(ImageNet_O_OOD, self).__init__()
        self.root_dir = os.path.join(root, 'imagenet-o')
        print(self.root_dir)
        self.split = split
        self.transform = transform

        self.imgdir = self.root_dir
        

        self.img_folder_list = os.listdir(self.imgdir)
        i=0
        self.l_img_file = []
        self.labels=[]
        for folders in self.img_folder_list:
            self.sp_imgdir = os.path.join(self.imgdir, folders)
            if os.path.isdir(self.sp_imgdir):
                self.sp_img = os.listdir(self.sp_imgdir)
                for imgs in self.sp_img:
                    self.l_img_file.append(os.path.join(self.sp_imgdir, imgs))
                    self.labels.append(i)
                i +=1

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.labels[index] 

    def __len__(self):
        return len(self.l_img_file)


class Texture_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        """
            In OpenOOD1.5, they use
            texture images for cifar100 : 5640 images
            texture images for imagenet-1k : 5160 images
            We have used all 5640 images
        """
        super(Texture_OOD, self).__init__()
        self.root_dir = os.path.join(root, 'texture')
        print(self.root_dir)
        self.split = split
        self.transform = transform

        self.imgdir = os.path.join(self.root_dir, 'images')
        self.img_folder_list = os.listdir(self.imgdir)
        i=0
        self.l_img_file = []
        self.labels=[]
        for folders in self.img_folder_list:
            self.sp_imgdir = os.path.join(self.imgdir, folders)
            if os.path.isdir(self.sp_imgdir):
                self.sp_img = os.listdir(self.sp_imgdir)
                for imgs in self.sp_img:
                    self.l_img_file.append(os.path.join(self.sp_imgdir, imgs))
                    self.labels.append(i)
                i +=1

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.labels[index] 

    def __len__(self):
        return len(self.l_img_file)
    

class Texture_ViM_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(Texture_ViM_OOD, self).__init__()
        self.root_dir = os.path.join(root, 'Texture')
        print(self.root_dir)
        self.split = split
        self.transform = transform

        self.imgdir = self.root_dir
        self.list_dir = os.path.join(root, 'datalists/texture.txt')
        self.listfile = open(self.list_dir, 'r')
        self.train_list = self.listfile.readlines()
        self.l_img_file = []
        for line in self.train_list:
            real_dir = os.path.join(self.root_dir, line)
            self.l_img_file.append(real_dir)

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)

        
class Openimages_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(Openimages_OOD, self).__init__()
        self.root = os.path.join(root, 'Openimages')
        print(self.root)
        self.split = split
        self.transform = transform

        self.imgdir = self.root
        self.l_img_file = sorted(os.listdir(self.imgdir))
        self.sorted_l_img_file = []
        for imgs in self.l_img_file:
            if not imgs.endswith('.tar'):
                self.sorted_l_img_file.append(os.path.join(self.imgdir, imgs))
        self.l_img_file = self.sorted_l_img_file

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)
    
class Openimages_O_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        assert split in ('validation', 'evaluation')

        super(Openimages_O_OOD, self).__init__()
        self.root_dir = root #./datasets
        print(os.path.join(root, 'openimage_o/images'))
        self.split = split
        self.transform = transform

        if split == 'validation': # validation split in OpenOOD V1.5
                                # 1763 images
            self.list_dir = os.path.join(root, 'datalists/val_openimage_o.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                real_dir = os.path.join(self.root_dir, img_name)
                self.l_img_file.append(real_dir)

        elif split == 'evaluation': # 15869 images
            self.list_dir = os.path.join(root, 'datalists/test_openimage_o.txt')
            self.listfile = open(self.list_dir, 'r')
            self.train_list = self.listfile.readlines()
            self.l_img_file = []
            for line in self.train_list:
                tokens = line.split(' ', 1)
                img_name = tokens[0]
                real_dir = os.path.join(self.root_dir, img_name)
                self.l_img_file.append(real_dir)
        else:
            raise ValueError(f'{split}')

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)


class iNaturalist_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        # 10000 images
        super(iNaturalist_OOD, self).__init__()
        self.root = os.path.join(root, 'inaturalist/images')
        print(self.root)
        self.split = split
        self.transform = transform

        self.imgdir = self.root
        self.l_img_file = sorted(os.listdir(self.imgdir))
        self.sorted_l_img_file = []
        for imgs in self.l_img_file:
            if not imgs.endswith('.tar'):
                self.sorted_l_img_file.append(os.path.join(self.imgdir, imgs))
        self.l_img_file = self.sorted_l_img_file

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)
   

class SSB_hard_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(SSB_hard_OOD, self).__init__()
        self.root_dir = root
        self.split = split
        self.transform = transform
        self.imgdir = os.path.join(root, 'ssb_hard')
        print(os.path.join(self.root_dir, 'ssb_hard'))

        self.img_folder_list = os.listdir(self.imgdir)
        i=0
        self.l_img_file = []
        self.labels=[]
        for folders in self.img_folder_list:
            self.sp_imgdir = os.path.join(self.imgdir, folders)
            if os.path.isdir(self.sp_imgdir):
                self.sp_img = os.listdir(self.sp_imgdir)
                for imgs in self.sp_img:
                    self.l_img_file.append(os.path.join(self.sp_imgdir, imgs))
                    self.labels.append(i)
                i +=1

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.labels[index] 

    def __len__(self):
        return len(self.l_img_file)
    
    
class NINCO_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #only evaluation set
        super(NINCO_OOD, self).__init__()
        self.root_dir = root
        self.split = split
        self.transform = transform
        self.imgdir = os.path.join(root, 'ninco')
        print(os.path.join(self.root_dir, 'ninco'))

        self.img_folder_list = os.listdir(self.imgdir)
        i=0
        self.l_img_file = []
        self.labels=[]
        for folders in self.img_folder_list:
            self.sp_imgdir = os.path.join(self.imgdir, folders)
            if os.path.isdir(self.sp_imgdir):
                self.sp_img = os.listdir(self.sp_imgdir)
                for imgs in self.sp_img:
                    self.l_img_file.append(os.path.join(self.sp_imgdir, imgs))
                    self.labels.append(i)
                i +=1

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.labels[index] 

    def __len__(self):
        return len(self.l_img_file)

class Places_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #ood for imagenet 1k; NPOS
        #only evaluation set
        super(Places_OOD, self).__init__()
        self.root = os.path.join(root, 'Places/images')
        print(self.root)
        self.split = split
        self.transform = transform

        self.imgdir = self.root
        self.l_img_file = sorted(os.listdir(self.imgdir))
        self.sorted_l_img_file = []
        for imgs in self.l_img_file:
            if not imgs.endswith('.tar'):
                self.sorted_l_img_file.append(os.path.join(self.imgdir, imgs))
        self.l_img_file = self.sorted_l_img_file

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)
    
class SUN_OOD(Dataset):
    def __init__(self, root, split='evaluation', transform=None, seed=1):
        #ood for imagenet 1k; NPOS
        #only evaluation set
        super(SUN_OOD, self).__init__()
        self.root = os.path.join(root, 'SUN/images')
        print(self.root)
        self.split = split
        self.transform = transform

        self.imgdir = self.root
        self.l_img_file = sorted(os.listdir(self.imgdir))
        self.sorted_l_img_file = []
        for imgs in self.l_img_file:
            if not imgs.endswith('.tar'):
                self.sorted_l_img_file.append(os.path.join(self.imgdir, imgs))
        self.l_img_file = self.sorted_l_img_file

    def __getitem__(self, index):
        imgpath = self.l_img_file[index]
        im = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, 0

    def __len__(self):
        return len(self.l_img_file)



#
# CelebA
#
IMAGE_EXTENSTOINS = [".png", ".jpg", ".jpeg", ".bmp"]
ATTR_ANNO = "list_attr_celeba.txt"


def _is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext.lower() in IMAGE_EXTENSTOINS


def _find_images_and_annotation(root_dir):
    images = {}
    attr = None
    assert os.path.exists(root_dir), "{} not exists".format(root_dir)
    img_dir = os.path.join(root_dir, 'Img/img_align_celeba')
    for fname in os.listdir(img_dir):
        if _is_image(fname):
            path = os.path.join(img_dir, fname)
            images[os.path.splitext(fname)[0]] = path

    attr = os.path.join(root_dir, 'Anno', ATTR_ANNO)
    assert attr is not None, "Failed to find `list_attr_celeba.txt`"

    # begin to parse all image
    final = []
    with open(attr, "r") as fin:
        image_total = 0
        attrs = []
        for i_line, line in enumerate(fin):
            line = line.strip()
            if i_line == 0:
                image_total = int(line)
            elif i_line == 1:
                attrs = line.split(" ")
            else:
                line = re.sub("[ ]+", " ", line)
                line = line.split(" ")
                fname = os.path.splitext(line[0])[0]
                onehot = [int(int(d) > 0) for d in line[1:]]
                assert len(onehot) == len(attrs), "{} only has {} attrs < {}".format(
                    fname, len(onehot), len(attrs))
                final.append({
                    "path": images[fname],
                    "attr": onehot
                })
    print("Find {} images, with {} attrs".format(len(final), len(attrs)))
    return final, attrs


class CelebA_OOD(Dataset):
    def __init__(self, root_dir, split='training', transform=None, seed=1):
        """attributes are not implemented"""
        super().__init__()
        assert split in ('training', 'validation', 'evaluation')
        if split == 'training':
            setnum = 0
        elif split == 'validation':
            setnum = 1
        elif split == 'evaluation':
            setnum = 2
        else:
            raise ValueError(f'Unexpected split {split}')

        d_split = self.read_split_file(root_dir)
        self.data = d_split[setnum]
        self.transform = transform
        self.split = split
        self.root_dir = os.path.join(root_dir, 'CelebA', 'Img', 'img_align_celeba')


    def __getitem__(self, index):
        filename = self.data[index]
        path = os.path.join(self.root_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0. 

    def __len__(self):
        return len(self.data)

    def read_split_file(self, root_dir):
        split_path = os.path.join(root_dir, 'CelebA', 'Eval', 'list_eval_partition.txt')
        d_split = {0:[], 1:[], 2:[]}
        with open(split_path) as f:
            for line in f:
                fname, setnum = line.strip().split()
                d_split[int(setnum)].append(fname)
        return d_split


class NotMNIST(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        super().__init__()
        self.transform = transform
        shuffle_idx = np.load(os.path.join(root_dir, 'notmnist_trainval_idx.npy'))
        datadict = loadmat(os.path.join(root_dir, 'NotMNIST/notMNIST_small.mat'))
        data = datadict['images'].transpose(2, 0, 1).astype('float32')
        data = data[shuffle_idx]
        targets = datadict['labels'].astype('float32')
        targets = targets[shuffle_idx]
        if split == 'training':
            self.data = data[:14979]
            self.targets = targets[:14979]
        elif split == 'validation':
            self.data = data[14979:16851]
            self.targets = targets[14979:16851]
        elif split == 'evaluation':
            self.data = data[16851:]
            self.targets = targets[16851:]
        else:
            raise ValueError

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.data)


class ImageNet32(Dataset):
    def __init__(self, root_dir, split='training', transform=None, seed=1, train_split_ratio=0.8):
        """
        split: 'training' - the whole train split (1281149)
               'evaluation' - the whole val split (49999)
               'train_train' - (train_split_ratio) portion of train split
               'train_val' - (1 - train_split_ratio) portion of train split
        """
        super().__init__()
        self.root_dir = os.path.join(root_dir, 'ImageNet32')
        self.split = split
        self.transform = transform
        self.shuffle_idx = get_shuffled_idx(1281149, seed)
        n_train = int(len(self.shuffle_idx) * train_split_ratio)

        if split == 'training':  # whole train split
            self.imgdir = os.path.join(self.root_dir, 'train_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
        elif split == 'evaluation':  # whole val split
            self.imgdir = os.path.join(self.root_dir, 'valid_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
        elif split == 'train_train':  # 80 % of train split
            self.imgdir = os.path.join(self.root_dir, 'train_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
            self.l_img_file = [self.l_img_file[i] for i in self.shuffle_idx[:n_train]]
        elif split == 'train_val':  # 20 % of train split
            self.imgdir = os.path.join(self.root_dir, 'train_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
            self.l_img_file = [self.l_img_file[i] for i in self.shuffle_idx[n_train:]]
        else:
            raise ValueError(f'{split}')

    def __getitem__(self, index):
        imgpath = os.path.join(self.imgdir, self.l_img_file[index])
        im = Image.open(imgpath)
        if self.transform is not None:
            im = self.transform(im)
        return im, 0 

    def __len__(self):
        return len(self.l_img_file)

from torch.utils.data import TensorDataset
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