import torch
import torch.nn as nn
from torch.autograd import grad
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
import faiss
from models.modules import Flow_Module
from tqdm import tqdm    

class Backbone_Euclidean(LatentOOD):
    def __init__(
        self,
        backbone_name="vit_base_patch16_224",
        spherical=False,
        feat_normalize=False,
        centercrop=False,
        n_class=None,
        pretrained=True,
        name=None,
        feat_dir = None,
        ft_model_dir = ''
    ):
        super().__init__(backbone_name, spherical, centercrop, n_class, pretrained, name)

        ###Although prototypes aren't used, it is initialize for extract stat method
        self.register_buffer("prototypes", torch.zeros(self.n_class,self.backbone.num_features))
                
        self.feat_dir = feat_dir
        self.stat_dir = os.path.join(self.feat_dir, 'features_stat.pkl')
        self.ft_model_dir = ft_model_dir
        self.feat_normalize= feat_normalize
                
    def train_step(self, x, y, opt, **kwargs):
        """
        fine-tuning with cross-entropy classification loss
        x: data
        y: label
        opt: optimizer
        """
        self.train()
        #print("being trained with comp_loss")
        logit = self.class_logit(x)
        loss = nn.CrossEntropyLoss()(logit, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        d_train = {"loss": loss.item()}
        return d_train
        
    def extract_feature(self, d_dataloaders, device, show_tqdm=False):
        train_features = []
        train_labels = []
        d_dataloaders = tqdm(d_dataloaders) if show_tqdm else d_dataloaders 
        with torch.no_grad():
            for x, y in d_dataloaders:
                x = x.to(device)
                feat_batch = self.encode(x)
                ##if feature normalization is required
                if(self.feat_normalize):
                    feat_batch = feat_batch/feat_batch.norm(dim=1, keepdim=True)
                
                feat_batch = feat_batch.cpu().numpy()
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
        # u = -torch.matmul(torch.linalg.pinv(w), b)

        x = features[self.name]
        x = torch.Tensor(x)
        u = x.mean(axis=0)
        #if self.proj_error_type=='Sphere':
        #    u = torch.zeros_like(u)
        prototypes = self.prototypes.cpu()

        logit_id_train = torch.matmul(x, w.T+b)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(x.detach().numpy() - u.detach().numpy())
        ec_cov = torch.Tensor(ec.covariance_)
        eig_vals, eigen_vectors = torch.linalg.eigh(ec_cov)

        # NS = np.ascontiguousarray((eigen_vectors.detach().numpy().T[np.argsort(eig_vals.detach().numpy() * -1)[self.dim:]]).T)
        NS = np.ascontiguousarray((eigen_vectors.detach().numpy().T[np.argsort(eig_vals.detach().numpy() * -1)[:]]).T)
        NS = torch.Tensor(NS)
        vlogit_id_train = torch.linalg.norm(torch.matmul(x - u, NS), axis=-1)
        vim_stat = {}
        vim_stat['w'] = w
        vim_stat['b'] = b
        vim_stat['u'] = u
        vim_stat['eig_vals'] = eig_vals
        vim_stat['NS'] = NS
        vim_stat['prototypes'] = prototypes
        return vim_stat

    def validation_step(self, x, y,**kwargs):
        """
        validation based on kl
        """
        self.eval()
        #print('being tuned with crossentropy')
        logit = self.class_logit(x)
        loss = nn.CrossEntropyLoss()(logit, y)
        acc = (logit.argmax(dim=1) == y).float().mean().item()
        d_val = {"loss": loss.item(), "acc_": acc}
        return d_val