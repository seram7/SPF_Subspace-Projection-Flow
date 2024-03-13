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
from hydra.utils import instantiate

class KNN(nn.Module):
    def __init__(
        self,
        k,
        in_dl
    ):
        super().__init__()
        self.in_dl = in_dl
        self.k = k 

    def forward(self, z):
        device = z.get_device()

        z = z / z.norm(dim=1, keepdim=True)

        total_batch_wise_dist = []
        for batch, _ in self.in_dl:
            batch = batch.to(device)
            batch = batch/batch.norm(dim=1, keepdim=True)
            rep_batch = batch.repeat(z.shape[0], 1, 1)
            rep_z = z.unsqueeze(1).repeat(1, batch.shape[0], 1)
            batch_wise_dist = (rep_batch - rep_z).norm(dim=2)
            total_batch_wise_dist.append(batch_wise_dist)
        total_batch_wise_dist = torch.cat(total_batch_wise_dist, dim=1)

        sorted, _ = torch.sort(total_batch_wise_dist)

        return sorted[:, self.k]


    # def forward(self, z):
    #     if not self._initialized:
    #         raise ValueError('Model not initialized properly; Check ViM stat file.')
    #     device = z.get_device()
    #     return self.vim_score(z, self.w, self.b, self.u, self.alpha, self.NS, device)
        
    def predict(self, x):
        return self(x)

    # def vim_score(self, z, w, b, u, alpha, NS, device):
    #     if (type(device) == int) and (device < 0):
    #         device = 'cpu'
    #     w = w.to(device)
    #     b = b.to(device)
    #     u = u.to(device)
    #     alpha = alpha.to(device)
    #     NS = NS.to(device)
    #     with torch.no_grad():    
    #         logit_z = torch.matmul(z, w.T+b)
    #         vlogit_z = torch.linalg.norm(torch.matmul(z- u, NS), axis=-1) * alpha
    #         energy_z = logsumexp(logit_z.cpu().numpy(), axis=-1)
    #         score = vlogit_z.cpu().numpy() - energy_z
    #         score = torch.Tensor(score).to(device)
    #     return score
        
    # def extract_feature(self, d_dataloaders, device, show_tqdm=False):
    #     train_features = []
    #     train_labels = []
    #     d_dataloaders = tqdm(d_dataloaders) if show_tqdm else d_dataloaders 
    #     with torch.no_grad():
    #         for x, y in d_dataloaders:
    #             x = x.to(device)
    #             feat_batch = self.encode(x).cpu().numpy()
    #             y = y.numpy()
    #             train_features.append(feat_batch)
    #             train_labels.append(y)
    #     train_features = np.concatenate(train_features, axis=0)
    #     train_labels = np.concatenate(train_labels, axis = 0)
    #     key = self.name
    #     d_result = {}
    #     d_result[key] = train_features
    #     d_result[key+'_labels'] = train_labels
    #     return d_result
    
    # def fcweight_bias(self):
    #     w = self.classifier.weight
    #     b = self.classifier.bias
    #     return w, b

    # def extract_stat(self, features):
    #     w, b = self.fcweight_bias()
    #     w = w.cpu()
    #     b = b.cpu()
    #     DIM=512
    #     u = -torch.matmul(torch.linalg.pinv(w), b)

    #     x = features[self.name]
    #     x = torch.Tensor(x)

    #     logit_id_train = torch.matmul(x, w.T+b)

    #     ec = EmpiricalCovariance(assume_centered=True)
    #     ec.fit(x.detach().numpy() - u.detach().numpy())
    #     ec_cov = torch.Tensor(ec.covariance_)
    #     eig_vals, eigen_vectors = torch.linalg.eig(ec_cov)

    #     NS = np.ascontiguousarray((eigen_vectors.detach().numpy().T[np.argsort(eig_vals.detach().numpy() * -1)[DIM:]]).T)
    #     NS = torch.Tensor(NS)
    #     vlogit_id_train = torch.linalg.norm(torch.matmul(x - u, NS), axis=-1)
    #     alpha = logit_id_train.detach().numpy().max(axis=-1).mean() / vlogit_id_train.detach().numpy().mean()
    #     vim_stat = {}
    #     vim_stat['w'] = w
    #     vim_stat['b'] = b
    #     vim_stat['u'] = u
    #     vim_stat['alpha'] = torch.tensor(alpha)
    #     vim_stat['NS'] = NS
    #     return vim_stat

    # def validation_step(self, x,y):
    #     """
    #     validation based on kl
    #     """
    #     self.eval()
    #     #print('being tuned with crossentropy')
    #     logit = self.class_logit(x)
    #     loss = nn.CrossEntropyLoss()(logit, y)
    #     acc = (logit.argmax(dim=1) == y).float().mean().item()
    #     d_val = {"loss": loss.item(), "acc_": acc}
    #     return d_val
    
    # def virtual_forward(self, x, custom_feat_num = False):
    #     """
    #     Returns Class Logit AND feature space representation
    #     z necessary to sample virtual outlier in training time scheme
    #     """
    #     z = self.encode(x)
    #     feature = None
    #     if custom_feat_num == True:
    #         feature = nn.Sequential()(z)
    #     return feature, z