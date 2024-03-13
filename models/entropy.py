"""
non-parametric entropy estimation
"""
import torch
import numpy as np
from scipy.special import digamma, gamma

def volume_of_unit_ball(d):
    """Volume of a d-dimensional unit ball."""
    return np.pi**(d/2.0) / gamma(d/2.0 + 1)


def pairwise_distances(X):
    """Compute the pairwise distance matrix for X."""
    sum_X = torch.sum(torch.square(X), 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.T), sum_X).T, sum_X)
    return torch.sqrt(D)


def pairwise_distances2(X):
    pdist = (
    (X**2).sum(-1, keepdim=True)
    + (X**2).sum(-1, keepdim=True).t()
    - 2 * X @ X.t()
    )
    return pdist


def pairwise_distances2_target(X, Y):
    pdist = (
    (X**2).sum(-1, keepdim=True)
    + (Y**2).sum(-1, keepdim=True).t()
    - 2 * X @ Y.t()
    )
    return pdist


# def estimate_entropy_knn(data, k=1):
#     # Compute pairwise distances
#     distances = pairwise_distances(data)
#     
#     # Sort each row and take the k+1 smallest value (k-th nearest neighbor excluding itself)
#     kth_distances = torch.sort(distances, axis=1).values[:, k]
#     
#     N = len(data)
#     d = data.shape[1]  # number of dimensions
#     
#     # Compute the entropy estimate
#     entropy = digamma(N) - digamma(k) + (d/N) * torch.sum(torch.log(kth_distances)) + np.log(volume_of_unit_ball(d))
#     
#     return entropy


def estimate_entropy_knn(data, target=None, k=1, eps=1e-6, constant=True, reduce=True):
    """
    target: if None, compute entropy of data, otherwise compute logp of data w.r.t. target
            if target is True, you'd better use reduce=False.
    reduce: if True, return the mean entropy over the batch, otherwise return - log p(x)
    """
    # Compute pairwise distances
    # distances = pairwise_distances(data)
    if target is None:
        distances = pairwise_distances2(data)  # returns squared distance
        N = len(data)
    else:
        distances = pairwise_distances2_target(target, data)
        N = len(target)
    
    # Sort each row and take the k+1 smallest value (k-th nearest neighbor excluding itself)
    kth_distances = torch.sort(distances, axis=1).values[:, k].clamp(min=eps)
    assert torch.all(kth_distances >= 0.0)
    
    d = np.prod(data.shape[1:])  # number of dimensions
    
    # Compute the entropy estimate
    # entropy = (d/N) * torch.sum(torch.log(kth_distances + eps))
    if constant:
        bias_corr = digamma(N) - digamma(k) + np.log(volume_of_unit_ball(d))
        mlogp = d * torch.log(kth_distances + eps) * 0.5 + bias_corr  # 0.5 accounts for the squared distance
    else:
        mlogp = torch.log(kth_distances + eps) * 0.5  # 0.5 accounts for the squared distance
    
    if reduce:
        return mlogp.mean() 
    else:
        return mlogp

