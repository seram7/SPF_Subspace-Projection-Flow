import argparse
import os
import random
import sys

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from gpu_utils import AutoGPUAllocation


@hydra.main(config_path="configs", config_name="main", version_base="1.2")
def run(cfg):
    """main training function"""
    # Setup seeds
    seed = cfg.get("seed", 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # log dir and writer
    print("Result directory: ", HydraConfig.get().run.dir)
    logdir = HydraConfig.get().run.dir
    writer = SummaryWriter(logdir=logdir)

    # Setup device
    if cfg.device == 'auto':
        gpu_allocation = AutoGPUAllocation()
        device = gpu_allocation.device
        cfg['device'] = device
    else:
        cfg['device'] = f'cuda:{cfg.device}'
        device = f'{cfg.device}'

    # Setup Model
    model = instantiate(cfg.model).to(device)
    trainer = instantiate(cfg.training, device)
    logger = instantiate(cfg.logger, writer)
    if hasattr(model, "get_transform"):
        model_specific_transform = model.get_transform()
    else:
        model_specific_transform = None

    # Setup Dataloader
    d_dataloaders = {}
    for key, dataloader_cfg in cfg.data.items():
        if key == "name":
            continue
        d_dataloaders[key] = instantiate(dataloader_cfg)

        # use custom transform from model if available
        if model_specific_transform is not None:
            d_dataloaders[key].dataset.transform = model_specific_transform

    # Setup optimizer
    # if hasattr(model, 'own_optimizer') and model.own_optimizer:
    #     optimizer, sch = model.get_optimizer(cfg.training.optimizer)
    # elif 'optimizer' not in cfg['training']:
    #     optimizer = None
    #     sch = None
    # else:
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    sch = None

    # lr scheduler
    # sch = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    model, train_result = trainer.train(
        model,
        optimizer,
        d_dataloaders,
        logger=logger,
        logdir=writer.file_writer.get_logdir(),
        scheduler=sch,
    )


if __name__ == "__main__":
    run()
