import argparse
import os
import random
import sys
import argparse

import hydra
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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
    python extract_feature.py --model backbone_sphere \
            --data cifar100 \
            --run finetune \
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', '-t', default='cifar100_backbone_sphere', help='trained model dir')
    parser.add_argument('--run', default='finetune', help='run name')
    parser.add_argument('--data', '-d', default=None, help='dataset')
    parser.add_argument('--device', default=0, type=int, help='device number')

    return parser.parse_args()

def run(args):
    if os.path.exists(f"./results/{args.trained}/{args.run}"):
        root_dir = os.path.join('./results', args.trained, args.run)
        trained_cfg = OmegaConf.load(root_dir+'/.hydra/config.yaml')
        if args.data:
            initialize(config_path=f"configs/data",  version_base="1.2")
            cfg = compose(config_name=f"{args.data}")
        else:
            cfg = trained_cfg
    else:
        raise Exception("No trained model to extract features")
    
    

    """main evaluation function"""
    # Setup device
    device = args.device

    # log dir and writer
    ### config <data, model> needed ###
    print("Extract features in directory: ", root_dir)
    logdir = root_dir
    writer = SummaryWriter(logdir=logdir)

    # load trained model 
    model = instantiate(trained_cfg.model).to(device)

    if hasattr(model, "get_transform"):
        model_specific_transform = model.get_transform()
    else:
        model_specific_transform = None
    state_input = False
    for i in trained_cfg:
        if i == 'model_state':
            state_input=True
    
    if state_input:
        modeldict_name =  trained_cfg.model_state+'.pkl'
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
    if ckpt_data != None:
        if 'model_state' in ckpt_data:
            model.load_state_dict(ckpt_data['model_state'])
        else:
            model.load_state_dict(torch.load(ckpt))
    model.eval()
    model.to(device)

    # load dataset(inlier, outlier)
    ## inlier
    in_dl= None
    val_dl = None
    test_dl = None
    l_ood_dl = []
    for key, dataloader_cfg in cfg.data.items():
        if key == "name":
            continue
        dl = instantiate(dataloader_cfg)
        if model_specific_transform is not None:
            dl.dataset.transform = model_specific_transform
        if key == "train":
            in_dl = dl
        elif key == "val":
            val_dl = dl
        elif key == "test":
            test_dl = dl
        elif key.startswith("ood"):
            if dataloader_cfg['dataset']['split']=='validation':
                dl.name='val_'+key
            else:
                dl.name = key
            l_ood_dl.append(dl)
        else:
            #print(key)
            raise ValueError(f"Unknown dataset key: {key}")

            
    # result
    time_s = time()
    # inlier inference
    if in_dl:
        print('train data')
        features = model.extract_feature(in_dl, device, show_tqdm=True)
        stat = model.extract_stat(features)
        #result = dict(features, **stat)
        result = stat
        torch.save(result, model.stat_dir)
        feature_vec = features[model.name]
        labels = features[model.name+'_labels']
        #feature_vec = feature_vec.detach().numpy()
        #labels = labels.detach().numpy() 
        np.save(model.feat_dir+'/features.npy', feature_vec)
        np.save(model.feat_dir+'/labels.npy', labels)
        print(f'Feature statistic file saved at {model.stat_dir}')

    if val_dl:
        print('validation data')
        val_features = model.extract_feature(val_dl, device, show_tqdm=True)
        val_feature_vec = val_features[model.name]
        val_labels = val_features[model.name+'_labels']
        feat_file_name = model.feat_dir+'/val_features.npy'
        label_file_name = model.feat_dir+'/val_labels.npy'
        np.save(feat_file_name, val_feature_vec)
        np.save(label_file_name, val_labels)
    
    if test_dl:
        print('test data')
        test_features = model.extract_feature(test_dl, device, show_tqdm=True)
        test_feature_vec = test_features[model.name]
        test_labels = test_features[model.name+'_labels']
        feat_file_name = model.feat_dir+'/test_features.npy'
        label_file_name = model.feat_dir+'/test_labels.npy'
        np.save(feat_file_name, test_feature_vec)
        np.save(label_file_name, test_labels)

    l_ood_pred = []
    for dl in l_ood_dl:
        print(dl.name)
        ood_features = model.extract_feature(dl, device, show_tqdm=True)
        ood_feature_vec = ood_features[model.name]
        ood_labels = ood_features[model.name+'_labels']
        feat_file_name = model.feat_dir+'/'+ dl.name+'_features.npy'
        label_file_name = model.feat_dir+'/'+ dl.name+'_labels.npy'
        np.save(feat_file_name, ood_feature_vec)
        np.save(label_file_name, ood_labels)

    print(f'Extraction complete after {time()-time_s} seconds')


if __name__ == "__main__":
    args = parse_args()
    run(args)