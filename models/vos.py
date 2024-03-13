import torch
import torch.nn as nn
import torch.nn.functional as F
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

class VOS(nn.Module):
    """Virtual Outlier Synthesis algorithm with pre-trained models"
    default : ViT / Options : ResNET, DenseNet
    """
    def __init__(
        self,
        backbone_name= "vit_base_patch16_224",
        spherical= False,
        centercrop=False,
        n_class=None,
        sample_number = 1000,
        pretrained=True,
        name=None,
        noise_type='',
        noise_value=0     
    ):
        """
        spherical: project the representation to the unit sphere
        centercrop: use centercrop in preprocessing
        n_class: number of classes. None if no classification.
        weight_energy : learnable weight for energy score
        logistic regression : logistic regression on energy value. "to learn flexible energy surface" 
        """
        super().__init__()
        self.backbone_name = backbone_name
        assert backbone_name in {"vit_base_patch16_224"} ##Free to add
        self.spherical = spherical
        self.centercrop = centercrop
        self.n_class = n_class
        self.name = name
        self.noise_type = noise_type
        self.noise_value = noise_value

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )  # num_classes=0 means feature extraction
        if n_class is not None:
            self.classifier = nn.Linear(
                self.backbone.num_features, n_class
            )  # num class is set here
        else:
            self.classifier = None

        self.weight_energy = torch.nn.Linear(n_class, 1)
        torch.nn.init.uniform_(self.weight_energy.weight)

        self.logistic_regression = nn.Sequential(torch.nn.Linear(1, 500),nn.LeakyReLU(),
                                                 torch.nn.Linear(500, 100),nn.LeakyReLU(),
                                                 torch.nn.Linear(100, 2))

        self.data_dict = torch.zeros(self.n_class, sample_number, self.backbone.num_features)
        self.number_dict = {}
        for i in range(self.n_class):
            self.number_dict[i] = 0

    def get_transform(self):
        config = resolve_data_config({}, model=self.backbone)
        transform = create_transform(**config)
        if self.centercrop:
            return transform
        else:
            if self.backbone_name == "vit_base_patch16_224":
                return Compose(
                    [
                        Resize(224, interpolation=InterpolationMode.BICUBIC),
                        ToTensor(),
                        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )
            else:
                raise NotImplementedError

    def virtual_forward(self, x):
        """
        Returns Class Logit AND feature space representation
        z necessary to sample virtual outlier in training time scheme
        """
        z = self.backbone(x)
        z = self._project(z)
        logit = self.classifier(z) 
        return logit, z
    
    def forward(self, x):
        """anomaly score: maximum softmax probability"""

        """
        logit = self.class_logit(x)
        prob = torch.softmax(logit, dim=1)
        return -prob.max(dim=1)[0]  # minus because we assign high score for anomaly
        """

        """VOS Energy Score """
        logit = self.class_logit(x)
        energy_score = -self.num_stable_logsumexp(logit,dim=1)
        binary_ood_logit = self.logistic_regression(energy_score.view(-1, 1))
        
        #print(torch.softmax(binary_ood_logit, dim=1))
        
        return torch.softmax(binary_ood_logit, dim=1)[:,0] ###probability of being anomaly

    def predict(self, x):
        """anomaly score"""
        return self(x)

 
    def class_logit(self, x):
        z = self.backbone(x)
        z = self._project(z)
        logit = self.classifier(z)
        return logit

    def encode(self, x):
        z = self.backbone(x)
        return self._project(z)

    def _project(self, x):
        if self.spherical:
            return x / x.norm(dim=1, keepdim=True)
        return x
    
    
    def num_stable_logsumexp(self,value, dim=None, keepdim=False):
        """
        numerically stable logsumexp, heavily based on VOS
        """
        import math
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(self.weight_energy.weight * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

    def train_step(self, x, y, opt, sample_number,sample_from,select,epoch,start_epoch,ur_loss_weight, device, **kwargs):
        """
        fine-tuning with cross-entropy classification loss + uncertainty regularization loss
        x: data
        y: label
        opt: optimizer
        sample_number : number of samples in each class to estimate means and tied varinace
        sample_from : number of candidate samples to generate from conditional gaussians
        select : number of outliers to select from each inliner class
        epoch : current_epoch
        start_epoch : epoch where uncertainty regularization initialize
        ur_loss_weight : weight for uncertainty regularization 
        """
        self.data_dict.to(device)
        self.train()
        logit, z = self.virtual_forward(x)

        ### Energy Regularization
        sum_temp = 0
        for cls_idx in range(self.n_class):
            sum_temp += self.number_dict[cls_idx]

        ur_reg_loss = torch.zeros(1)[0] #.cuda()[0]
        if sum_temp >= self.n_class * sample_number and epoch < start_epoch:
            """
            maintains ID data queue for each class : regularization after start_epoch
            """
            y_numpy = y.cpu().data.numpy()
            for index in range(len(y)):
                dict_key = y_numpy[index]
                self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:].to(device),
                                                      z[index].detach().view(1, -1)), 0)

        elif sum_temp >= self.n_class * sample_number and epoch >= start_epoch:
            print("Energy_regularation_ON")
            y_numpy = y.cpu().data.numpy()
            for index in range(len(y)):
                dict_key = y_numpy[index]
                self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:].to(device), 
                                                      z[index].detach().view(1, -1)), 0)

            ###Compute Estimate of Covariance Mtx
            
            #1: Data Centering
            for cls_idx in range(self.n_class):
                if cls_idx == 0:
                    X = self.data_dict[cls_idx].to(device)
                    X_mean = X.mean(0)
                    X -= X_mean
                    mean_embed_id = X_mean.view(1,-1)
                else:
                    next_X = self.data_dict[cls_idx].to(device)
                    next_X_mean = next_X.mean(0)
                    X = torch.cat((X, next_X - next_X_mean), 0)
                    mean_embed_id = torch.cat((mean_embed_id, next_X_mean.view(1, -1)), 0)

            #2 Covariance Computation
            cov_mat = torch.mm(X.t(),X) / len(X)
            cov_mat += 0.0001 * torch.eye(self.backbone.num_features).to(device)

            #Sample 
            for index in range(self.n_class):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(mean_embed_id[index], 
                                                                                     covariance_matrix=cov_mat)
                negative_samples = new_dis.rsample((sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                top_samples, top_index = torch.topk(-prob_density, select)
                if index == 0:
                    ood_samples = negative_samples[top_index]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[top_index]), 0)  
            #Energy Computation
            if len(ood_samples) != 0:
                energy_id = -self.num_stable_logsumexp(logit,dim=1)
                predictions_ood = self.classifier(ood_samples.to(device))
                energy_ood = -self.num_stable_logsumexp(predictions_ood,dim=1)

                input_ur = torch.cat((energy_id, energy_ood), 0)
                labels_ur = torch.cat((torch.ones(len(logit)).to(device),
                                           torch.zeros(len(ood_samples)).to(device)), 0)
                #In vos setting, in_data is labeled as 1
                criterion = torch.nn.CrossEntropyLoss()
                output_ur= self.logistic_regression(input_ur.view(-1, 1))
                ur_reg_loss = criterion(output_ur, labels_ur.long())

        ###if queue is not full, add data to the in_distribution_queue
        else:
            y_numpy = y.cpu().data.numpy()
            for cls_idx in range(len(y)):
                dict_key = y_numpy[cls_idx]
                if self.number_dict[dict_key] < sample_number:
                    self.data_dict[dict_key][self.number_dict[dict_key]] = z[cls_idx].detach().view(1,-1)
                    self.number_dict[dict_key] += 1
        
        #computation of loss
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(logit, y).to(device)
        loss += ur_loss_weight * ur_reg_loss
        loss.backward()
        opt.step()

        d_train = {"loss": loss.item()}
        """
        for key, value in self.number_dict.items():
            print(key, ":", value, end = " ")
        
        print(self.data_dict[0])
        print("\n")
        """
        return d_train

    def validation_step(self, x, y, **kwargs):
        """
        return classification accuracy and loss
        """
        self.eval()
        logit = self.class_logit(x)
        loss = nn.CrossEntropyLoss()(logit, y)
        acc = (logit.argmax(dim=1) == y).float().mean().item()
        d_val = {"loss": loss.item(), "acc_": acc}
        return d_val