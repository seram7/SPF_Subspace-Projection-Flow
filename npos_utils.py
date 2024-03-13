import numpy as np
import torch
import faiss
import time
#import matplotlib.pyplot as plt
import faiss.contrib.torch_utils
from sklearn import manifold, datasets
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from tqdm import tqdm


def Approx_KNN_dis_search(target, index, K=50, select=1, normalize = True):
    ##여기서 주어지는 index는 target과 비교될 vector의 faiss.index -> 졍렬되어있다면 0,1,2,3,4... / Costumize되어있다면 ID로 주어짐
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features
    if normalize:
        target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
        normed_target = target / target_norm
    else:
        normed_target =  target

    #index에 embedding 중, 거리가 가장 가까운 K개 추출
    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    #k_th_output_index = output_index[:, -1]

    #target 중, 가장 먼 select개의 instance를 추출
    k_th_distance, minD_idx = torch.topk(k_th_distance, select)
    #k_th_index = k_th_output_index[minD_idx]
    return minD_idx, k_th_distance


###length: Generated negative Sample nunmbers
def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000,depth=342, normalize = True):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features
    if normalize:
        target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
        normed_target = target / target_norm
    else:
        normed_target =  target
    #start_time = time.time()
    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    target_new = target.view(length, -1, depth)
    #k_th_output_index = output_index[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    minD_idx = minD_idx.squeeze()
    point_list = []
    for i in range(minD_idx.shape[1]):
        point_list.append(i*length + minD_idx[:,i])
    #return torch.cat(point_list, dim=0)
    return target[torch.cat(point_list)]

def generate_outliers(ID, input_index, negative_samples, ID_points_num=2, K=20, select=1, cov_mat=0.1, sampling_ratio=1.0, pic_nums=30, depth=342, device = 'cpu'):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[0], int(normed_data.shape[0] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th_disance = Approx_KNN_dis_search(ID, index, K, select)
    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat([ID[i:i+1].repeat(length,1) for i in minD_idx])
    #negative_sample_cov = (torch.mm(negative_samples.cuda(), cov)*cov_mat).repeat(pic_nums,1)
    negative_sample_cov = cov_mat*negative_samples.to(device).repeat(pic_nums,1)
    #negative_sample_cov = (negative_samples.cuda()*cov_mat).repeat(select,1)
    negative_sample_list = negative_sample_cov + data_point_list.to(device)
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length,depth)

    index.reset()

    #return ID[minD_idx]
    return point


import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class CompLoss(nn.Module):
    def __init__(self, n_class, temperature=0.07, base_temperature=0.07, device = 'cpu'):
        super(CompLoss, self).__init__()
        self.n_class = n_class
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, prototypes, labels):
        proxy_labels = torch.arange(0, self.n_class).to(self.device)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, proxy_labels.view(1,-1)).float().to(self.device)

        # compute logits
        anchor_feature = features
        contrast_feature = prototypes / prototypes.norm(dim=-1, keepdim=True)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss
    
# n_class / feat_dim
class DispLoss(nn.Module):
    def __init__(self, n_class, feature_dim, proto_m, model, loader, temperature= 0.1, base_temperature=0.1, device = 'cpu',cifar=True):
        super(DispLoss, self).__init__()
        self.n_class = n_class
        self.feat_dim = feature_dim
        self.proto_m = proto_m
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(n_class,feature_dim))
        self.model = model
        self.loader = loader
        self.device = device
        self.init_class_prototypes(if_cifar=cifar)

    def forward(self, features, labels):
        prototypes = self.prototypes
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.proto_m + features[j]*(1-self.proto_m), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, self.n_class).to(self.device)
        labels = labels.contiguous().view(-1, 1)
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.view(1,-1)).float()).to(self.device)

        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(self.n_class).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = (self.temperature / self.base_temperature) * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self, if_cifar):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            prototypes = torch.zeros(self.n_class,self.feat_dim).to(self.device)
            prototype_counts = torch.zeros(self.n_class).to(self.device)

            for i, (input, target) in enumerate(tqdm(self.loader)):
                input, target = input.to(self.device), target.to(self.device)
                
                _, features = self.model.virtual_forward(input)

                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1

                ##when if_cifar==False, heuristic initilization of class prototypes is enabled.
                if(if_cifar==False):
                    sum = torch.sum(prototype_counts>0)
                    #print(sum)
                    if(sum==self.n_class):
                        break

            for cls in range(self.n_class):
                prototypes[cls] /=  prototype_counts[cls]
            #measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes


from sklearn.metrics import roc_curve
def Fpr95(arr1, arr2):
    true_label = np.concatenate([np.ones_like(arr1),
                                 np.zeros_like(arr2)])
    score = np.concatenate([arr1, arr2])
    fpr, tpr, thresholds = roc_curve(true_label, score)

    tpr_mask = tpr >= 0.95
    fpr95_score = fpr[tpr_mask][0]

    return fpr95_score

def extract_anomaly_score(model, d_dataloaders, device, ood = False):
    features = []
    scores = []
    labels = []
      
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for i,(x,y) in enumerate(d_dataloaders):
            x = x.to(device)
            z = model.encode(x)
            score = model.predict(x)    
            features.append(z.detach().cpu().numpy())
            scores.append(score.detach().cpu().numpy())
            if ood:
                labels.append(torch.ones_like(y).numpy())

            else:
                labels.append(torch.zeros_like(y).numpy())
                
    features = np.concatenate(features, axis=0)
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)

    d_result = {}
    d_result['features'] = features
    d_result['scores'] = scores
    d_result['labels'] = labels

    return d_result
