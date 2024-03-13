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
from models.modules import MLP_onedimoutput as MLP
from utils import weight_norm


class Flow_Sphere_EBM(LatentOOD):
    def __init__(
        self,
        backbone_name="vit_base_patch16_224",
        spherical=True,
        centercrop=False,
        n_class=None,
        pretrained=True,
        name=None,
        feat_dir = None, # Absolutely not used
        proj_error_type = 'Euclidean',
        coef_dsm = 1,
        coef_entropy = 0, 
        coef_classify= 0, 
        coef_inlier_projection_error = 1,
        dsm_sigma = 0.5,
        id_reg = True,
        dsm_reg= False,
        coef_reg =0,
        coef_l2reg=None,
        tau = 0.1,
        evr_threshold = 0.9,
        custom_dim = False,
        sampler = None,
        neg_initial_mode = 'cd',
        z_flow = True,
        z_flow_type = None,
        z_multi_flow_num = 2,
        dsm_scale = True,
        ft_model_dir = '',
        random_vec = False,
        n_hidden_mlp = None,
        intermediate_dim = 1000,
        input_noise=None,
    ):
        """
        coef_l2reg: weight norm regularization
        input_noise: If not None, add noise to input during training.
                     Intended to use when spherical=False.
                     Recommended value is 0.01.
        """
        super().__init__(backbone_name, spherical, centercrop, n_class, pretrained, name)

        self.tau = tau
        self.proj_error_type = proj_error_type
        self.coef_dsm = coef_dsm
        self.coef_inlier_projection_error = coef_inlier_projection_error
        self.dsm_sigma = dsm_sigma
        self.dsm_scale = dsm_scale
        self.coef_reg = coef_reg
        self.id_reg = id_reg
        self.dsm_reg = dsm_reg
        self.coef_l2reg = coef_l2reg
        self.z_flow = z_flow
        self.z_multi_flow_num = z_multi_flow_num
        self.evr_threshold = evr_threshold
        self.custom_dim = custom_dim
        self.sampler = sampler
        self.neg_initial_mode = neg_initial_mode
        assert neg_initial_mode in {'cd', 'projected'}

        self.register_buffer("prototypes", torch.zeros(self.n_class,self.backbone.num_features))
        self.temperature = 0.1
        self.input_noise = input_noise
        self.ft_model_dir = ft_model_dir

        stat_dir = os.path.join(ft_model_dir, 'features_stat.pkl')
        print(stat_dir)

        if(os.path.exists(stat_dir)):
            feat_stat = torch.load(stat_dir, map_location='cpu')

            if "prototypes" in feat_stat:
                self.register_buffer('prototypes', feat_stat['prototypes'])
                self.class_logit = self.class_logit_cider
            else:
                self.class_logit = self.class_logit_softmax

            self.register_buffer('w', feat_stat['w'])
            self.register_buffer('b', feat_stat['b'])
            self.register_buffer('u', feat_stat['u'])
            if self.proj_error_type=='Sphere':
                u = torch.zeros_like(u)
            if 'eig_vals' in feat_stat:
                self.register_buffer('eig_vals', feat_stat['eig_vals'])
            else:
                self.register_buffer('eig_vals', None)
            ###compute_evr_norm
            if(self.custom_dim):
                self.evr_dim = self.custom_dim
            else:
                if self.eig_vals is not None:
                    self.evr_dim = self.compute_evr_dimension(feat_stat['NS'])
                else:
                    print("Warning : Key 'eig_vals' Not Found in feat_stat.pkl, Cannot Compute EVR Dimension. Use custom_dim or extract ViM stat again.")
                    raise KeyError
            
            print(f"Computed EVR Dimension : {self.evr_dim}")
            
            if(random_vec):
                rand_idx = torch.randperm(self.backbone.num_features)[:self.backbone.num_features-self.evr_dim]
                self.register_buffer('NS', feat_stat['NS'][:,rand_idx])
            else:
                self.register_buffer('NS', feat_stat['NS'][:,self.evr_dim:])
            self._initialized = True
            print('loading model statistics from {}'.format(stat_dir))
            print('initializing projection matrix')
            self.register_buffer('proj_mat', torch.matmul(self.NS,self.NS.transpose(1,0)))
            print('Initialized')
            print(self.proj_mat)
        else:
            self._initialized = False

        if(self.z_flow == False):
            self.z1_to_z2 = nn.Sequential()

        if(self.z_flow):
            ###Modify as desired g(z)
            assert z_flow_type in {None,'complex','multi-flow','mlp'}
            if z_flow_type is None:
                self.z1_to_z2 = Flow_Module(intermediate_dim,feature_num=self.backbone.num_features,
                                            flow_num=1,complex=True, spherical=self.spherical)
            if z_flow_type == 'complex':
                self.z1_to_z2 = Flow_Module(intermediate_dim,feature_num=self.backbone.num_features,
                                            flow_num=self.z_multi_flow_num,complex=True, spherical=self.spherical)
            if z_flow_type == 'multi-flow':
                self.z1_to_z2 = Flow_Module(intermediate_dim,feature_num=self.backbone.num_features,
                                            flow_num=self.z_multi_flow_num,complex=False, spherical=self.spherical)
            if z_flow_type == 'mlp':
                self.z1_to_z2 = MLP(intermediate_dim,feature_num=self.backbone.num_features,num_hidden=n_hidden_mlp, spherical=self.spherical)
            if(self.dsm_scale):
                self.scale = torch.nn.Linear(1,1)
                
    def get_transform(self):
        return None
            
    def compute_evr_dimension(self,feat_space):
        eig_val_sum = torch.sum(self.eig_vals)
        for i in range(feat_space.size(1)):
            curr_eig_val_sum = torch.sum(self.eig_vals[feat_space.size(1)-i-1:])
            if curr_eig_val_sum > self.evr_threshold*eig_val_sum:
                return i+1
            
    def forward(self, x):
        if not self._initialized:
            raise ValueError('Model not initialized properly; Check ViM stat file.')
        # z = self.backbone(x)
        z=x
        if(self.z_flow):
            z,_ = self.z1_to_z2(x)
        device = z.get_device()
        if device < 0:
            device = 'cpu'
        u = self.u.to(device)
        proj_mat = self.proj_mat.to(device)
        return self.compute_projection_error(z, u, proj_mat) # same as vim score
        # return self.vim_score(z, self.w, self.b, self.u, self.NS, device)
    
    def langevin_forward(self, z):
        if(self.z_flow):
            z,_ = self.z1_to_z2(z)
        device = z.get_device()
        if device < 0:
            device = 'cpu'
        u = self.u.to(device)
        proj_mat = self.proj_mat.to(device)
        
        with torch.no_grad():
            res = self.compute_projection_error(z, u, proj_mat)
        return res
    
    def intermediate_forward(self,x,n_th=0):
        if not self._initialized:
            raise ValueError('Model not initialized properly; Check stat file.')

        z_before = x

        for i,flow in enumerate(self.z1_to_z2.flows):
            if i >= int(n_th):
                break;      
            res = flow(z_before)
            z_before = res+z_before
            z_before = z_before / z_before.norm(dim=1, p=2, keepdim=True)
        
        device = z_before.get_device()
        if device < 0:
            device = 'cpu'
        u = self.u.to(device)
        proj_mat = self.proj_mat.to(device)
        return self.compute_projection_error(z_before, u, proj_mat)
        
    def predict(self, x):
        return self(x)
    
    def vim_predict(self, x):
        logit = self.class_logit(x)
        logits_max, _ = torch.max(logit, dim=1, keepdim=True)
        new_logits = logit - logits_max.detach()
        logitmax = new_logits.max(dim=1)[0]
        exp_logits = torch.exp(new_logits) 
        log_prob = new_logits - torch.log(exp_logits.sum(1, keepdim=True))
        prob = torch.exp(log_prob)
        probmax = prob.max(dim=1)[0]
        if(self.z_flow):
            x,_ = self.z1_to_z2(x)
            # w, _ = self.z1_to_z2(self.w)
        # print(logit)
        vlogit = torch.linalg.norm(torch.matmul(x- self.u, self.NS), axis=-1) * 1.2
        energy = probmax.detach().cpu().numpy()
        score = vlogit.detach().cpu().numpy() - energy
        score = torch.Tensor(score).to(x.device)
        return score
    
    def class_logit_softmax(self, x):
        return torch.matmul(x, self.w.T) + self.b

    def class_logit_cider(self, x):
        z = x
        # z_p, _ = self.z1_to_z2(z)
        anchor_feature = z

        if(self.prototypes.device != anchor_feature.device):
            self.prototypes = self.prototypes.to(anchor_feature.device)

        #print(f"proto : {self.prototypes}")
        #compute prototype after flow
        # prototype_z2 = self.prototypes
        # prototype_z2,_ = self.z1_to_z2(self.prototypes)

        contrast_feature = self.prototypes
        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        
        return anchor_dot_contrast
    
    def compute_projection_error(self,z_p,u,proj_mat):
        if(self.proj_error_type=='Euclidean'):
            pos_projection_vectors = torch.matmul(z_p - u, proj_mat)
            return torch.norm(pos_projection_vectors, dim=1, p=2)
        elif self.proj_error_type == 'EuclideanSq':
            pos_projection_vectors = torch.matmul(z_p - u, proj_mat)
            return pos_projection_vectors.pow(2).sum(dim=1)
        elif(self.proj_error_type=='Sphere'):
            ##Note: Vaild only when u is zero vector
            assert torch.norm(u,p=2)<1e-9
            pos_projection_vectors = torch.matmul(z_p, proj_mat)
            pos_projection_vectors = z_p - pos_projection_vectors
            pos_projection_vectors = pos_projection_vectors / pos_projection_vectors.norm(dim=1, p=2, keepdim=True)
            return torch.norm(z_p-pos_projection_vectors, dim=1, p=2)
        else:
            raise NotImplementedError

    def get_dsm_loss(self, z, single_MLP = False):
        ## DSM from here
        z_clone = z.clone()
        z_clone.requires_grad = True
        sigma = self.dsm_sigma
        noise = torch.randn_like(z_clone)

        # do we need z_noise projections?
        z_noise = z_clone + sigma * noise
        z_noise = self._project(z_noise)
        new_noise = z_noise - z_clone
        z_p_dsm, dsm_reg_loss = self.z1_to_z2(z_noise)
        
        if single_MLP:
            # if z1_to_z2 is MLP
            energy = z_p_dsm
        else:
            energy = self.compute_projection_error(z_p_dsm,self.u,self.proj_mat)
            if(self.dsm_scale):
                energy = self.scale(energy.unsqueeze(dim=1))
        score = - grad(energy.sum(), z_noise, create_graph=True)[0]
        # z_clone.requires_grad = False

        dsm_loss = torch.norm(new_noise/sigma+score, dim=-1)**2
        dsm_loss = 1/2 * dsm_loss.mean()
        ## DSM End here
        
        return dsm_loss, dsm_reg_loss

    def train_step(self, x, y, opt, **kwargs):
        """
        fine-tuning dsm loss
        x: data
        y: label
        opt: optimizer
        NS: null-space
        """
        proj_mat = self.proj_mat
        u = self.u
        # coef of loss terms; value : float or None
        coef_inlier_projection_error = self.coef_inlier_projection_error
        coef_dsm = self.coef_dsm
        coef_l2reg = self.coef_l2reg
        coef_reg = self.coef_reg
        
        # reg_loss components; value : bool
        id_reg = self.id_reg
        dsm_reg = self.dsm_reg
        
        reg_loss = torch.tensor(0, dtype=torch.float32).to(x.device)
        self.train()
        opt.zero_grad()
        
        if self.input_noise is not None:
            x = x + self.input_noise * torch.randn_like(x)
        z=x
        z_p,ID_reg_loss = self.z1_to_z2(z)
        
        ## Calculate Inlier projection error
        if z_p.shape[1] == 1:
            # if z1_to_z2 is MLP
            inlier_projection_error = torch.tensor(0, dtype=torch.float32).to(x.device)
        else:
            inlier_projection_error = self.compute_projection_error(z_p, u, proj_mat).mean()
                    
        loss = coef_inlier_projection_error * inlier_projection_error
        
        if id_reg:
            reg_loss += ID_reg_loss

        if coef_dsm is not None:
            dsm_loss, dsm_reg_loss = self.get_dsm_loss(z, z_p.shape[1] == 1)
            loss += coef_dsm * dsm_loss
            if dsm_reg:
                reg_loss += dsm_reg_loss
            loss += coef_reg * reg_loss
        else:
            # skip dsm computation
            dsm_loss = torch.tensor(0, dtype=torch.float32)
            
        flow_weight_norm = torch.tensor(0, dtype=torch.float32)
        if coef_l2reg is not None:
            flow_weight_norm = weight_norm(self.z1_to_z2)
            loss += coef_l2reg * flow_weight_norm

        loss.backward()
        opt.step()
        d_train = {"loss": loss.item(), 
                   "inlier_energy_":inlier_projection_error.item(),
                   "dsm_loss_": dsm_loss.item(),
                   "flow_weight_norm_":flow_weight_norm.item(),
                   "reg_loss_":reg_loss.item(),
                   "ID_reg_loss_":ID_reg_loss.item()}
        
        return d_train

    def validation_step(self, x, y,**kwargs):
        """
        validation based on kl
        """
        self.eval()
        
        proj_mat = self.proj_mat
        u = self.u
        tau = self.tau

        z = x
        z_p,_ = self.z1_to_z2(z)

        inlier_projection_error = self.compute_projection_error(z_p, u, proj_mat).mean()
        loss = self.coef_inlier_projection_error * inlier_projection_error

        if self.coef_dsm is not None:
            dsm_loss, _ = self.get_dsm_loss(z, z_p.shape[1] == 1)
            loss += self.coef_dsm * dsm_loss

        d_val = {"loss": loss.item()}
        return d_val
        
