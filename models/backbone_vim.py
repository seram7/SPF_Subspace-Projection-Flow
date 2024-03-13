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

class Backbone_ViM(LatentOOD):
    def __init__(
        self,
        backbone_name="vit_base_patch16_224",
        spherical=False,
        centercrop=False,
        n_class=None,
        pretrained=True,
        name=None,
        feat_dir = None,
        evr_threshold = 0.7,
        ft_model_dir = '',
        custom_dim = False
    ):
        super().__init__(backbone_name, spherical, centercrop, n_class, pretrained, name)

                
        self.feat_dir = feat_dir
        # self.stat_dir = os.path.join(self.feat_dir, 'features_stat.pkl')
        self.stat_dir = os.path.join(ft_model_dir, 'features_stat.pkl')
        self.evr_threshold = evr_threshold

        if(os.path.exists(self.stat_dir)):
            feat_stat = torch.load(self.stat_dir)
            self.w = feat_stat['w']
            self.b = feat_stat['b']
            self.u = feat_stat['u']
            x = torch.tensor(np.load(os.path.join(ft_model_dir, 'features.npy')), dtype=torch.float32)
            
            if 'eig_vals' in feat_stat:
                self.register_buffer('eig_vals', feat_stat['eig_vals'])
            else:
                self.register_buffer('eig_vals', None)
            
            if(custom_dim):
                self.evr_dim = custom_dim
            else:
                if self.eig_vals is not None:
                    self.evr_dim = self.compute_evr_dimension(feat_stat['NS'])
                else:
                    print("Warning : Key 'eig_vals' Not Found in feat_stat.pkl, Cannot Compute EVR Dimension. Use custom_dim or extract ViM stat again.")
                    raise KeyError

            print(f"Computed EVR Dimension : {self.evr_dim}")
            self.register_buffer('NS', feat_stat['NS'][:,self.evr_dim:])
            
            logit_id_train = torch.matmul(x, self.w.T+self.b)
            vlogit_id_train = torch.linalg.norm(torch.matmul(x - self.u, self.NS), axis=-1)
            self.alpha = torch.tensor(logit_id_train.detach().numpy().max(axis=-1).mean() / vlogit_id_train.detach().numpy().mean())
            
            print('loading model statistics from {}'.format(self.stat_dir))
            print('initializing projection matrix')
            self.register_buffer('proj_mat', torch.matmul(self.NS,self.NS.transpose(1,0)))
            print('Initialized')
            print(self.proj_mat)
            
            self._initialized = True
        else:
            self._initialized = False
            
    def compute_evr_dimension(self,feat_space):
        eig_val_sum = torch.sum(self.eig_vals)
        for i in range(feat_space.size(1)):
            curr_eig_val_sum = torch.sum(self.eig_vals[feat_space.size(1)-i-1:])
            if curr_eig_val_sum > self.evr_threshold*eig_val_sum:
                return i+1
            
    def forward(self, z):
        if not self._initialized:
            raise ValueError('Model not initialized properly; Check ViM stat file.')
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
        
    def extract_feature(self, d_dataloaders, device, show_tqdm=False):
        train_features = []
        train_labels = []
        d_dataloaders = tqdm(d_dataloaders) if show_tqdm else d_dataloaders 
        with torch.no_grad():
            for x, y in d_dataloaders:
                x = x.to(device)
                feat_batch = self.encode(x).cpu().numpy()
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

        x = features[self.name]
        x = torch.Tensor(x)
        u = x.mean(axis=0)

        logit_id_train = torch.matmul(x, w.T+b)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(x.detach().numpy() - u.detach().numpy())
        ec_cov = torch.Tensor(ec.covariance_)
        eig_vals, eigen_vectors = torch.linalg.eigh(ec_cov)

        # NS = np.ascontiguousarray((eigen_vectors.detach().numpy().T[np.argsort(eig_vals.detach().numpy() * -1)[self.dim:]]).T)
        NS = np.ascontiguousarray((eigen_vectors.detach().numpy().T[np.argsort(eig_vals.detach().numpy() * -1)[:]]).T)
        NS = torch.Tensor(NS)
        vim_stat = {}
        vim_stat['w'] = w
        vim_stat['b'] = b
        vim_stat['u'] = u
        vim_stat['eig_vals'] = eig_vals
        vim_stat['NS'] = NS
        return vim_stat

    def validation_step(self, x,y):
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
    
    def virtual_forward(self, x, custom_feat_num = False):
        """
        Returns Class Logit AND feature space representation
        z necessary to sample virtual outlier in training time scheme
        """
        z = self.encode(x)
        feature = None
        if custom_feat_num == True:
            feature = nn.Sequential()(z)
        return feature, z