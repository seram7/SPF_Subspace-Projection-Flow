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

from npos_utils import Fpr95


"""
evaluate OOD detection performance through AUROC score

Example:
    python evaluate.py --model vim \
            --data cifar100 \
            --run dev \
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', '-t', default='cifar100_flow_sphere_langevin', help='trained model dir')
    parser.add_argument('--run', default='dev', help='run name')
    parser.add_argument('--eval', default=None, help='ood datasets for evaluation')
    parser.add_argument('--device', default=0, type=int, help='device number')
    parser.add_argument('--iter', help='evaluate model at specific iter')
    parser.add_argument('--noload', default=False, action='store_true', help='do not load checkpoint')
    parser.add_argument('--strict', action='store_true', help='use strict option in load_state_dict')
    parser.add_argument('--vim', action='store_true', help='use vim score for ood detection')

    return parser.parse_args()

def run(args):
    if os.path.exists(f"./results/{args.trained}/{args.run}"):
        root_dir = os.path.join('./results', args.trained, args.run)
        trained_cfg = OmegaConf.load(root_dir+'/.hydra/config.yaml')
        if args.eval:
            initialize(config_path=f"configs/eval",  version_base="1.2")
            cfg = compose(config_name=f"{args.eval}")
        else:
            cfg = trained_cfg
    else:
        raise Exception("No trained model to extract features")

    """main evaluation function"""
    # Setup device
    device = args.device

    # log dir and writer
    ### config <data, model> needed ###
    print("Evaluate result in directory: ", root_dir)
    logdir = root_dir
    writer = SummaryWriter(logdir=logdir)

    # load trained model 
    model = instantiate(trained_cfg.model).to(device)
    # state_input = False
    # for i in trained_cfg:
    #     if i == 'model_state':
    #         state_input=True
    if hasattr(model, "get_transform"):
        model_specific_transform = model.get_transform()
    else:
        model_specific_transform = None
    
    if args.iter:
        modeldict_name =  f'model_iter_{args.iter}.pkl'
    elif os.path.exists(logdir+'/model_best.pkl'):
        modeldict_name = 'model_best.pkl'
    else:
        modeldict_name = 'not_exist'
    ckpt = os.path.join(logdir, modeldict_name)
    
    # load model checkpoint
    if not args.noload:
        if os.path.exists(ckpt):
            ckpt_data = torch.load(ckpt)
            print(f'loading from {ckpt}')
        else:
            ckpt_data = None
            print(f'directory:{ckpt}, but nothing to be loaded')
        if ckpt_data != None:
            if 'model_state' in ckpt_data:
                model.load_state_dict(ckpt_data['model_state'], strict=args.strict)
            else:
                model.load_state_dict(torch.load(ckpt), strict=args.strict)
        # if args.vim and 'alpha' not in ckpt_data['model_state']:
        #     alpha = os.path.join(model.ft_model_dir, f'alpha_{model.evr_dim}.npy')
        #     model.register_buffer('alpha', torch.tensor(np.load(alpha)))
    model.eval()
    model.to(device)

    ## load dataloaders 
    in_dl = None
    l_ood_dl = [] 
    for key, dataloader_cfg in cfg.eval.items():
        dl = instantiate(dataloader_cfg)
        if model_specific_transform is not None:
            dl.dataset.transform = model_specific_transform

        if key.startswith("in"):
            in_dl = dl

        elif key.startswith("ood"):
            dl.name = key
            l_ood_dl.append(dl)

        else:
            raise ValueError(f"Unknown dataset key: {key}")

    modeldict_basename = modeldict_name.split('.')[0]

    # method
    method = 'predict'
    if args.vim:
        method = 'vim_predict'  
      
    # result
    time_s = time()
    # inlier inference
    in_pred = batch_run(model, in_dl, device=device, method=method, no_grad=False, show_tqdm=True)
    print(f'{time() - time_s:.3f} sec for inlier inference')
    if args.iter:
        in_score_file = os.path.join(logdir, f'IN_score_{modeldict_basename}.pkl')
    else: # best model
        in_score_file = os.path.join(logdir, f'IN_score.pkl')
    torch.save(in_pred, in_score_file)

    # outlier prediction
    l_ood_pred = []
    for dl in l_ood_dl:
        print(dl.name)
        xx, _ = next(iter(dl))
        out_pred = batch_run(model, dl, method=method, device=device, no_grad=False, show_tqdm=True)
        l_ood_pred.append(out_pred)
        if args.iter:
            out_score_file = os.path.join(logdir, f'OOD_score_{dl.name}_{modeldict_basename}.pkl')
        else: # best model
            out_score_file = os.path.join(logdir, f'OOD_score_{dl.name}.pkl')
        torch.save(out_pred, out_score_file)
    
    # AUC
    l_ood_auc = []
    s_auc = ''
    for pred in l_ood_pred:
        l_ood_auc.append(roc_btw_arr(pred, in_pred))
    for dl, auc in zip(l_ood_dl, l_ood_auc):
        ood_name = dl.name
        if args.iter:
            with open(os.path.join(logdir, f'{ood_name}_auc_{modeldict_basename}.txt'), 'w') as f:
                f.write(str(auc))
        else: # best model
            with open(os.path.join(logdir, f'{ood_name}_auc.txt'), 'w') as f:
                f.write(str(auc))
        s_auc += f'{auc*100:.2f} '
        print("AUC of OOD: ", ood_name, auc)
    print(s_auc)

    #FPR
    l_ood_fpr = []
    for pred in l_ood_pred:
        l_ood_fpr.append(Fpr95(pred, in_pred))

    for dl, fpr in zip(l_ood_dl, l_ood_fpr):
        ood_name = dl.name
        if args.iter:
            with open(os.path.join(logdir, f'{ood_name}_fpr_{modeldict_basename}.txt'), 'w') as f:
                f.write(str(fpr))
        else: # best model
            with open(os.path.join(logdir, f'{ood_name}_fpr.txt'), 'w') as f:
                f.write(str(fpr))
        print("FPR95 of OOD: ", ood_name, fpr)

if __name__ == "__main__":
    args = parse_args()
    run(args)
