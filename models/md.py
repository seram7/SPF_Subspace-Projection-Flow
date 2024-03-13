import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    InterpolationMode,
    ToTensor,
)
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from models.latent import LatentOOD
import os
import numpy as np

class Mahalanobis(LatentOOD):
    def __init__(
        self,
        backbone_name="vit_base_patch16_224",
        spherical=True,
        centercrop=False,
        n_class=None,
        pretrained=True,
        name=None,
        feat_dir = None,
        noise_type='',
        noise_value=0
    ):
        self.noise_type = noise_type
        self.noise_value = noise_value
        if self.noise_type == 'aggregation':
            self.n_class = n_class - self.noise_value
        else:
            self.n_class = n_class
        super().__init__(backbone_name, spherical, centercrop, self.n_class, pretrained, name)
        """
        self.backbone_name = backbone_name
        assert backbone_name in {"vit_base_patch16_224", "resnetv2_50x1_bitm"}
        self.spherical = spherical
        self.centercrop = centercrop
        self.n_class = n_class
        self.name = name
        """
        self.feat_dir = feat_dir
        self.stat_dir = os.path.join(self.feat_dir, 'features_stat.pkl')
        
        
        

        if(os.path.exists(self.stat_dir)):
            feat_stat = torch.load(self.stat_dir)
            self.all_means = feat_stat['all_means']
            self.invcov = feat_stat['invcov']
            self.whole_invcov = feat_stat['whole_invcov']
            self.whole_mean = feat_stat['whole_mean']
            self._initialized = True
        else:
            self._initialized = False
        
        

        
    def forward(self, x):
        if not self._initialized:
            raise ValueError('Model not initialized properly; Check Mahalanobis stat file.')

        z = self.backbone(x)
        device = z.get_device()
        if(self.name == 'md'):
            return self.md_score(z, self.all_means, self.invcov, self.whole_mean, self.whole_invcov, device)
        elif(self.name == 'rmd'):
            return self.rmd_score(z, self.all_means, self.invcov, self.whole_mean, self.whole_invcov, device)
        
    def predict(self, x):
        return self(x)

    def md_score(self, z, all_means, invcov, whole_mean, whole_invcov, device):
        if (type(device) == int) and (device < 0):
            device = 'cpu'
        
        all_means = all_means.to(device)
        invcov = invcov.to(device)
        whole_mean = whole_mean.to(device)
        whole_invcov = whole_invcov.to(device)
        
        z = z.unsqueeze(-1)  # .double()
        z = z - all_means.double()
        op1 = torch.einsum("ijk,jl->ilk", z, invcov.double())
        op2 = torch.einsum("ijk,ijk->ik", op1, z)
        ##for test op2 = torch.einsum("ijk,ijk->ik", z, z)

        return torch.min(op2, dim=1).values.float()
    
    def rmd_score(self, z, all_means, invcov, whole_mean, whole_invcov, device):
        if (type(device) == int) and (device < 0):
            device = 'cpu'
        all_means = all_means.to(device)
        invcov = invcov.to(device)
        whole_mean = whole_mean.to(device)
        whole_invcov = whole_invcov.to(device)
        
        md_score = self.md_score(z, all_means, invcov, whole_mean, whole_invcov, device)
        z = z - whole_mean.double()
        op1 = torch.mm(z, whole_invcov.double())
        op2 = torch.mm(op1, z.t())
        background_maha = op2.diag().float()
        return md_score-background_maha
    
    def extract_feature(self, d_dataloaders, device):
        train_features = []
        train_labels = []
        with torch.no_grad():
            for x, y in d_dataloaders:
                x = x.to(device)
                feat_batch = self.backbone(x).cpu().numpy()
                y = y.numpy()
                train_features.append(feat_batch)
                train_labels.append(y)
    
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis = 0)

        key = self.name
        d_result = {}
        d_result[key] = train_features
        d_result[key+'_labels'] = train_labels
        return d_result

    def extract_stat(self, features):
        x = features[self.name]
        y = features[self.name+'_labels']
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        l_mean = []
        l_outer = []
        for k in range(self.n_class):
            subset_x = x[y == k]
            subset_mean = torch.mean(subset_x, dim=0, keepdim=True)

            v = subset_x - subset_mean
            # subset_outer = v.T.mm(v)
            subset_outer = v.T.mm(v) / len(subset_x)
            l_mean.append(subset_mean)
            l_outer.append(subset_outer)
        # pooled_cov = torch.sum(torch.stack(l_outer), dim=0) / len(x)
        pooled_cov = torch.mean(torch.stack(l_outer), dim=0)
        all_means = torch.stack(l_mean, dim=-1)
        invcov = torch.linalg.inv(pooled_cov.double())
        # invcov = torch.linalg.inv(pooled_cov)
        
        '''relative mahalanobis statistics'''
        whole_mean = torch.mean(x, dim=0, keepdim=True)
        v = x - whole_mean
        whole_cov = v.T.mm(v) / len(x)
        whole_invcov = torch.linalg.inv(whole_cov.double())
        # whole_invcov = torch.linalg.inv(whole_cov)
        maha_stat = {}
        maha_stat['all_means'] = all_means
        maha_stat['invcov'] = invcov
        maha_stat['whole_mean'] = whole_mean
        maha_stat['whole_invcov'] = whole_invcov
        return maha_stat