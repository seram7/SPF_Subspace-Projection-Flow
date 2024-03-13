import numpy as np
import time
from metrics import averageMeter
from trainers.base import BaseTrainer
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from utils import roc_btw_arr, batch_run
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from npos_utils import CompLoss, DispLoss,Fpr95


class VirtualTrainer_NPOS(BaseTrainer):
    def train(self, model, opt, d_dataloaders, logger=None, logdir='', scheduler = None, clip_grad=None):
        cfg = self.cfg
        self.logdir = logdir
        best_val_loss = np.inf
        best_auc = 0
        time_meter = averageMeter()
        i = 0
        indist_train_loader = d_dataloaders['train']
        indist_val_loader = d_dataloaders['val']

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0, last_epoch=- 1, verbose=False)

        criterion_disp = DispLoss(cfg.n_class,cfg.feature_dim, cfg.proto_m, model, indist_train_loader, cfg.disp_l_temp,cfg.disp_base_temp,self.device)
        criterion_comp = CompLoss(cfg.n_class, cfg.comp_l_temp,cfg.comp_base_temp,self.device)
        
        for i_epoch in range(cfg.n_epoch):

            for x, y in indist_train_loader:
                i += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()

                d_train = model.train_step(x, y, opt, cfg.sample_from, i_epoch, cfg.start_epoch, 
                   cfg.ID_points_num,cfg.K,cfg.select,cfg.cov_mat,cfg.sampling_ratio, cfg.pick_nums, criterion_disp,criterion_comp,
                   cfg.w_disp, cfg.w_comp, cfg.ur_loss_weight, self.device)

                
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if isinstance(scheduler, _LRScheduler):
                    if not (isinstance(scheduler, ReduceLROnPlateau)):
                        scheduler.step()

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()

                if i % cfg.val_interval == 0:
                    model.eval()
                    for val_x, val_y in indist_val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        d_val = model.validation_step(val_x, val_y, cfg.w_disp, cfg.w_comp)
                        logger.process_iter_val(d_val)
                        if cfg.get('val_once', False):
                            # no need to run the whole val set
                            break

                    '''AUC'''
                    #d_result = self.get_auc(model,d_dataloaders)
                    d_result = self.get_auc(model, d_dataloaders)
                    logger.d_val.update(d_result)

                    d_val = logger.summary_val(i)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = best_auc < d_result['result/auc_avg_']

                    if cfg.save_interval is not None and i % cfg.save_interval == 0 and best_model:
                        self.save_model(model, logdir, best=best_model, i_iter=i, i_epoch=i_epoch)
                        new_best = d_result['result/auc_avg_']
                        print(f'Iter [{i:d}] best model saved {new_best} <= {best_auc}')
                        best_auc = new_best
                    if isinstance(scheduler, _LRScheduler):
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()

            if 'save_interval_epoch' in cfg and i_epoch % cfg.save_interval_epoch == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch)

        '''AUC'''
        model.eval()

        #d_result = self.get_auc(model, d_dataloaders)
        #print(d_result)

        d_result = self.get_auc(model, d_dataloaders)
        print(d_result)

        return model, d_result 

    def predict(self, m, dl, device, flatten=False):
        """run prediction for the whole dataset"""
        l_result = []
        for x, _ in dl:
            if flatten:
               x = x.view(len(x), -1)
            pred = m.predict(x.cuda(device)).detach().cpu()
            l_result.append(pred)
        return torch.cat(l_result)

    def get_auc(self, model, d_dataloaders):
        indist_val_loader = d_dataloaders['val']
        in_pred = self.predict(model, indist_val_loader, self.device)
        dcase = False
        d_result = {}
        avg_auc = 0
        cnt = 0
        for k, v in d_dataloaders.items():
            if k.startswith('ood_'):
                ood_pred = batch_run(model, v, self.device)
                auc = roc_btw_arr(ood_pred, in_pred)
                k = k.replace('ood_', '')
                d_result[f'result/auc_{k}_'] = auc
                avg_auc += auc
                cnt += 1

        d_result['result/auc_avg_'] = avg_auc

        torch.save(d_result, self.logdir + '/val_auc.pkl')
        return d_result 
    

    def get_auc_fpr(self, model, d_dataloaders):
        indist_val_loader = d_dataloaders['val']
        in_pred = self.predict(model, indist_val_loader, self.device)
        dcase = False
        d_result = {}
        avg_auc = 0
        avg_fpr95 = 0
        cnt = 0
        for k, v in d_dataloaders.items():
            if k.startswith('ood_'):
                ood_pred = batch_run(model, v, self.device)
                auc = roc_btw_arr(ood_pred, in_pred)
                fpr95 = Fpr95(ood_pred,in_pred)
                k = k.replace('ood_', '')
                d_result[f'result/auc_{k}_'] = auc
                d_result[f'result/fpr95_{k}_'] = fpr95
                avg_auc += auc
                avg_fpr95 += fpr95
                cnt += 1
        
        avg_auc /= cnt
        avg_fpr95 /= cnt

        d_result['result/auc_avg_'] = avg_auc
        d_result['result/fpr95_avg_'] = avg_fpr95

        torch.save(d_result, self.logdir + '/val_auc.pkl')
        return d_result 