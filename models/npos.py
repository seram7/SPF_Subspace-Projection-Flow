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

from torch.distributions import MultivariateNormal
from npos_utils import generate_outliers #CompLoss, DispLoss
import faiss

class NPOS(nn.Module):
    """OOD detection algorithm with pre-trained model
    MSP.

    fine-tuning (classification)
    classification
    outlier detection (ex. Mahalanobis) mean covariance computation
    MSP: maximum softmax probability
    MD: Mahalanobis distance
    """

    def __init__(
        self,
        backbone_name= "vit_base_patch16_224",
        spherical=True,
        centercrop=False,
        n_class=None,
        pretrained=True,
        sample_number = 1000,
        name=None,
        noise_type='',
        noise_value=0
    ):
        """
        spherical: project the representation to the unit sphere
        centercrop: use centercrop in preprocessing
        n_class: number of classes. None if no classification.
        """
        super().__init__()
        self.backbone_name = backbone_name
        assert backbone_name in {"vit_base_patch16_224", "resnetv2_50x1_bitm"}
        self.spherical = spherical
        self.centercrop = centercrop
        self.n_class = n_class
        self.sample_number = sample_number
        self.name = name
        self.noise_type = noise_type
        self.noise_value = noise_value

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )  # num_classes=0 means feature extraction
        if n_class is not None:
            self.mlp = nn.Sequential(
                nn.Linear(self.backbone.num_features, self.backbone.num_features),
                nn.ReLU(inplace=True),
                nn.Linear(self.backbone.num_features,1)
            )
        else:
            self.mlp = None

        self.register_buffer("data_dict", torch.zeros(self.n_class, self.sample_number, self.backbone.num_features))
        self.number_dict = {}
        for i in range(self.n_class):
            self.number_dict[i] = 0

        self.register_buffer("prototypes", torch.zeros(self.n_class,self.backbone.num_features))
        self.temperature = 0.1

    def get_transform(self):
        config = resolve_data_config({}, model=self.backbone)
        transform = create_transform(**config)
        if self.centercrop:
            return transform
        else:
            if self.backbone_name == "vit_base_patch16_224":
                return Compose(
                    [
                        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                        ToTensor(),
                        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )
            else:
                raise NotImplementedError

    def virtual_forward(self, x, custom_feat_num = False):
        """
        Returns Class Logit AND feature space representation
        z necessary to sample virtual outlier in training time scheme
        """
        z = self.backbone(x)
        z = self._project(z)
        feature = None
        if custom_feat_num == True:
            feature = nn.Sequential()(z)
        return feature, z

    def forward(self, x):
        """anomaly score for npos"""
        class_logit = self.class_logit(x)
        logits_max, _ = torch.max(class_logit, dim=1, keepdim=True)
        new_logits = class_logit - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(new_logits) 
        log_prob = new_logits - torch.log(exp_logits.sum(1, keepdim=True))
        prob = torch.exp(log_prob)
        return -prob.max(dim=1)[0]  # minus because we assign high score for anomaly

    def predict(self, x):
        """anomaly score"""
        return self(x)

    def class_logit(self, x):
        z = self.backbone(x)
        z = self._project(z)
        # compute logits
        anchor_feature = z
        if(self.prototypes.device != anchor_feature.device):
            self.prototypes = self.prototypes.to(anchor_feature.device)
        contrast_feature = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        
        return anchor_dot_contrast

    def encode(self, x):
        z = self.backbone(x)
        return self._project(z)

    def _project(self, x):
        if self.spherical:
            return x / x.norm(dim=1, keepdim=True)
        return x
    

    def train_step(self, x, y, opt, sample_from, epoch, start_epoch, 
                   ID_points_num,K,select,cov_mat,sampling_ratio,pick_nums,criterion_disp,criterion_comp,
                   w_disp, w_comp, ur_loss_weight, device):
        """
        fine-tuning with cross-entropy classification loss
        x: data
        y: label
        opt: optimizer
        """
        if device != 'cpu':
            res = faiss.StandardGpuResources()
            KNN_index = faiss.GpuIndexFlatL2(res, self.backbone.num_features)
        else:
            KNN_index = faiss.IndexFlatL2(self.backbone.num_features)     
                   
        self.data_dict.to(device)
        self.train()

        _, z = self.virtual_forward(x)

        sum_temp = 0
        for cls_idx in range(self.n_class):
            sum_temp += self.number_dict[cls_idx]

        ur_reg_loss = torch.zeros(1).to(device)[0]

        if sum_temp >= self.n_class * self.sample_number and epoch < start_epoch:
            y_numpy = y.cpu().data.numpy()

            for idx in range(len(y_numpy)):
                dict_key = y_numpy[idx]
                self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:].to(device),
                                                      z[idx].detach().view(1, -1)), 0)


        elif sum_temp >= self.n_class * self.sample_number and epoch >= start_epoch:
            print("Regularization")
            y_numpy = y.cpu().data.numpy()
            for idx in range(len(y_numpy)):
                dict_key = y_numpy[idx]
                self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:].to(device),
                                                      z[idx].detach().view(1, -1)), 0)
            
            # Standard Gaussian distribution
            new_dis = MultivariateNormal(torch.zeros(self.backbone.num_features).to(device), torch.eye(self.backbone.num_features).to(device))
            negative_samples = new_dis.rsample((sample_from,))
            for index in range(self.n_class):
                ID = self.data_dict[index]
                sample_point = generate_outliers(ID, input_index=KNN_index,
                                                 negative_samples=negative_samples, ID_points_num=ID_points_num, K=K, select=select,
                                                 cov_mat=cov_mat, sampling_ratio=sampling_ratio, pic_nums=pick_nums, depth=self.backbone.num_features,device=device)
                if index == 0:
                    ood_samples = sample_point
                else:
                    ood_samples = torch.cat((ood_samples, sample_point), 0)
                
            if len(ood_samples) != 0:
                energy_score_for_fg = self.mlp(z)
                energy_score_for_bg = self.mlp(ood_samples)
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), 0).squeeze()
                labels_for_lr = torch.cat((torch.ones(len(energy_score_for_fg)).to(device),
                                           torch.zeros(len(energy_score_for_bg)).to(device)), -1)
                criterion_BCE = torch.nn.BCEWithLogitsLoss()
                ur_reg_loss = criterion_BCE(input_for_lr.view(-1), labels_for_lr)

        else:
            y_numpy = y.cpu().data.numpy()
            for cls_idx in range(len(y_numpy)):
                dict_key = y_numpy[cls_idx]
                if self.number_dict[dict_key] < self.sample_number:
                    self.data_dict[dict_key][self.number_dict[dict_key]] = z[cls_idx].detach()
                    self.number_dict[dict_key] += 1

        #normed_features = F.normalize(logit, dim=1)
        disp_loss = criterion_disp(z, y)
        comp_loss = criterion_comp(z, criterion_disp.prototypes, y)

        loss = w_disp * disp_loss + w_comp * comp_loss
        loss = ur_loss_weight * ur_reg_loss + loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        self.prototypes = criterion_disp.prototypes.detach().cpu().data
        self.temperature = criterion_comp.temperature

        d_train = {"loss": loss.item(), "disp_loss":disp_loss.item(), "comp_loss":comp_loss.item()}
        return d_train
    
    def validation_step(self, x, y, w_disp, w_comp, **kwargs):
        """
        return classification accuracy and loss
        """
        self.eval()
        logit = self.class_logit(x)
        loss = nn.CrossEntropyLoss()(logit, y)
        acc = (logit.argmax(dim=1) == y).float().mean().item()
        d_val = {"loss": loss.item(), "acc_": acc}
        return d_val
       