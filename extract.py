import argparse
import os
import random
import sys
import argparse

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from gpu_utils import AutoGPUAllocation

from time import time

from utils import roc_btw_arr, batch_run, batch_run_grad, parse_unknown_args, parse_nested_args
from sklearn.covariance import EmpiricalCovariance


"""
Extract Features and needed statistics for each model

Example:
    python extract.py --model vim \
            --data cifar100 \
            --run dev \
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vim', help='score model')
    parser.add_argument('--data', default='cifar100', help='dataset')
    parser.add_argument('--run', default='dev', help='Path to image list')

    return parser.parse_args()

def run(args):
    if os.path.exists(f"./results/{args.data}_{args.model}/{args.run}/.hydra"):
        initialize(config_path=f"results/{args.data}_{args.model}/{args.run}/.hydra",  version_base="1.2")
        cfg = compose(config_name="config")
    else:
        initialize(config_path=f"configs",  version_base="1.2")
        cfg = compose(config_name="main", overrides=[f'model={args.data}/{args.model}', f'data={args.data}', 'training.config.n_epoch=0', f'run={args.run}'])
    # print(OmegaConf.to_yaml(cfg))

    """main evaluation function"""
    # Setup device
    device = cfg.device

    # log dir and writer
    ### config <data, model> needed ###
    print("Extract features in directory: ", cfg.model.feat_dir)
    logdir = cfg.model.feat_dir
    writer = SummaryWriter(logdir=logdir)

    # load trained model 
    model = instantiate(cfg.model).to(device)
    state_input = False
    for i in cfg:
        if i == 'model_state':
            state_input=True
    
    if state_input:
        modeldict_name =  cfg.model_state+'.pkl'
    elif os.path.exists(logdir+'/model_best.pkl'):
        modeldict_name = 'model_best.pkl'
    else:
        modeldict_name = 'not_exist'
    ckpt = os.path.join(logdir, modeldict_name)
    
    if os.path.exists(ckpt):
        ckpt_data = torch.load(ckpt)
        print(f'loading from {ckpt}')
    else:
        ckpt_data = None
        print(f'directory:{ckpt}, but nothing to be loaded')
    if hasattr(model, "get_transform"):
        model_specific_transform = model.get_transform()
    else:
        model_specific_transform = None
    if ckpt_data != None:
        if 'model_state' in ckpt_data:
            model.load_state_dict(ckpt_data['model_state'])
        else:
            model.load_state_dict(torch.load(ckpt))
    model.eval()
    model.to(device)

    # load dataset(inlier, outlier)
    ## inlier
    in_dl= {}
    for key, dataloader_cfg in cfg.data.items():
        if key == "train":
            in_dl = instantiate(dataloader_cfg)
        # use custom transform from model if available
            if model_specific_transform is not None:
                in_dl.dataset.transform = model_specific_transform
    # result
    time_s = time()
    # inlier inference
    features = model.extract_feature(in_dl, device)
    stat = model.extract_stat(features)
    #result = dict(features, **stat)
    result = stat
    torch.save(result, model.stat_dir)
    print(f'Feature statistic file saved at {model.stat_dir}')
    print(f'Extraction complete after {time()-time_s} seconds')


if __name__ == "__main__":
    args = parse_args()
    run(args)