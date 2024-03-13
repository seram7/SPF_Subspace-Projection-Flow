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
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
import numpy as np


class ViM(LatentOOD):
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
        noise_value=0,
        ft_model_dir = '',
    ):
        self.noise_type = noise_type
        self.noise_value = noise_value
        if self.noise_type == 'aggregation':
            self.n_class = n_class - self.noise_value
        else:
            self.n_class = n_class
        super().__init__(backbone_name, spherical, centercrop, n_class, pretrained, name)
        """
        self.backbone_name = backbone_name
        assert backbone_name in {"vit_base_patch16_224", "resnetv2_50x1_bitm"}
        self.spherical = spherical
        self.centercrop = centercrop
        self.n_class = n_class
        self.name = name
        """
        self.feat_dir = feat_dir
        # self.stat_dir = os.path.join(self.feat_dir, 'features_stat.pkl')
        self.stat_dir = os.path.join(ft_model_dir, 'features_stat.pkl')


        if(os.path.exists(self.stat_dir)):
            feat_stat = torch.load(self.stat_dir)
            self.w = feat_stat['w']
            self.b = feat_stat['b']
            self.u = feat_stat['u']
            self.alpha = feat_stat['alpha']
            self.NS = feat_stat['NS']
            self._initialized = True
        else:
            self._initialized = False
    
    def forward(self, x):
        if not self._initialized:
            raise ValueError('Model not initialized properly; Check ViM stat file.')
        z = self.backbone(x)
        device = z.get_device()
        return self.vim_score(z, self.w, self.b, self.u, self.alpha, self.NS, device)
        
    def predict(self, x):
        return self(x)

    def vim_score(self, z, w, b, u, alpha, NS, device):
        if (type(device) == int) and (device < 0):
            device = 'cpu'
        w = w.to(device)
        b = b.to(device)
        u = u.to(device)
        alpha = alpha.to(device)
        NS = NS.to(device)
        with torch.no_grad():    
            logit_z = torch.matmul(z, w.T+b)
            vlogit_z = torch.linalg.norm(torch.matmul(z- u, NS), axis=-1) * alpha
            energy_z = logsumexp(logit_z.cpu().numpy(), axis=-1)
            score = vlogit_z.cpu().numpy() - energy_z
            score = torch.Tensor(score).to(device)
        return score
    
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
    
    def fcweight_bias(self):
        w = self.classifier.weight
        b = self.classifier.bias
        return w, b

    def extract_stat(self, features):
        w, b = self.fcweight_bias()
        w = w.cpu()
        b = b.cpu()
        DIM=512
        u = -torch.matmul(torch.linalg.pinv(w), b)

        x = features[self.name]
        x = torch.Tensor(x)

        logit_id_train = torch.matmul(x, w.T+b)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(x.detach().numpy() - u.detach().numpy())
        ec_cov = torch.Tensor(ec.covariance_)
        eig_vals, eigen_vectors = torch.linalg.eig(ec_cov)

        NS = np.ascontiguousarray((eigen_vectors.detach().numpy().T[np.argsort(eig_vals.detach().numpy() * -1)[DIM:]]).T)
        NS = torch.Tensor(NS)
        vlogit_id_train = torch.linalg.norm(torch.matmul(x - u, NS), axis=-1)
        alpha = logit_id_train.detach().numpy().max(axis=-1).mean() / vlogit_id_train.detach().numpy().mean()
        vim_stat = {}
        vim_stat['w'] = w
        vim_stat['b'] = b
        vim_stat['u'] = u
        vim_stat['alpha'] = torch.tensor(alpha)
        vim_stat['NS'] = NS
        return vim_stat
