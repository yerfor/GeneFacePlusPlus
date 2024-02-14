import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import lpips
import matplotlib.pyplot as plt

from modules.radnerfs.radnerf import RADNeRF
from modules.radnerfs.radnerf_sr import RADNeRFwithSR
from modules.radnerfs.radnerf_torso_sr import RADNeRFTorsowithSR
from tasks.radnerfs.radnerf_sr import RADNeRFTask

from utils.commons.image_utils import to8b
from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from utils.nn.model_utils import print_arch, num_params, not_requires_grad, requires_grad
from utils.nn.schedulers import NoneSchedule
from utils.nn.grad import get_grad_norm

from tasks.radnerfs.dataset_utils import RADNeRFDataset
from tasks.radnerfs.radnerf_sr import FeatureMatchingLoss
from tasks.radnerfs.losses import PerceptualLoss
from modules.eg3ds.models.dual_discriminator import DualDiscriminator


class ExponentialScheduleForRADNeRFTorso(NoneSchedule):
    """
    Default Scheduler in RAD-NeRF
    RAD-NeRF has two groups of params with different lr
    for tileGrid embedding, the lr=5e-3
    for other network params, the lr=5e-4
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr # 0.0005
        self.warmup_updates = warmup_updates

        optimizer.param_groups[0]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        optimizer.param_groups[1]['lr'] = self.lr * 10 # for tileGrid, lr starts from 0.005
        optimizer.param_groups[2]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-5)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 200k steps
            self.lr = max(new_lrate, 1e-5)
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 10
        self.optimizer.param_groups[2]['lr'] = self.lr
        return self.lr
    

class RADNeRFTorsoTask(RADNeRFTask):
    def __init__(self):
        super().__init__()

    def build_model(self):
        self.model = RADNeRFTorsowithSR(hparams) 
        # todo: load state_dict from RADNeRFwith
        head_model = RADNeRFwithSR(hparams)
        load_ckpt(head_model, hparams['head_model_dir'], strict=True)
        print(f"Loaded Head Model from {hparams['head_model_dir']}")
        self.model.load_state_dict(head_model.state_dict(), strict=False)
        print(f"Loaded state_dict of Head Model to the RADNeRFTorso Model")
        del head_model

        self.torso_embedders_params = [p for k, p in self.model.named_parameters() if p.requires_grad and 'torso_embedder' in k]
        self.torso_network_params = [p for k, p in self.model.named_parameters() if (p.requires_grad and 'torso_embedder' not in k and 'torso' in k)]
        self.sr_net_params = [p for k, p in self.model.sr_net.named_parameters() if p.requires_grad]
        
        for k, p in self.model.named_parameters():
            if 'torso' not in k:
                not_requires_grad(p)
            if 'sr_net' in k:
                requires_grad(p)

        self.model.poses = self.train_dataset.poses
        self.model.lm68s = self.train_dataset.lm68s

        return self.model
            
    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.torso_network_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            eps=1e-15)
        self.optimizer.add_param_group({
            'params': self.torso_embedders_params,
            'lr': hparams['lr'] * 10,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })
        self.optimizer.add_param_group({
            'params': self.sr_net_params,
            'lr': hparams['lr'],
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })

        return self.optimizer
    
    def build_scheduler(self, optimizer):
        return ExponentialScheduleForRADNeRFTorso(optimizer, hparams['lr'], hparams['warmup_updates'])

    ##########################
    # forward the model
    ##########################
    def run_model(self, sample, infer=False):
        """
        render or train on a single-frame
        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return:
            if not infer:
                return losses, model_out
            if infer:
                return model_out
        """
        cond_wins = sample['cond_wins']
        rays_o = sample['rays_o'] # [B, N, 3]
        rays_d = sample['rays_d'] # [B, N, 3]
        bg_coords = sample['bg_coords'] # [1, N, 2]
        poses = sample['pose'] # [B, 6]
        idx = sample['idx'] # [B]
        bg_color = sample['bg_img']
        H, W = sample['H'], sample['W']
        eye_area_percent = sample['eye_area_percent']
        cond_inp = cond_wins

        if not infer:
            model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, upscale_torso=True, lm68=sample['lm68'], eye_area_percent=eye_area_percent, **hparams)
            # if hparams['torso_train_mode'] == 1:
            #     pred_rgb = model_out['torso_rgb_map'] 
            #     gt_rgb = sample['bg_torso_img'] # the target is bg_torso_img\
            # else:
            pred_rgb = model_out['rgb_map'] # todo: try whole image 
            pred_torso_rgb = model_out['torso_rgb_map']
            gt_rgb = sample['gt_img'] # todo: try gt_image
            gt_torso_rgb = sample['bg_torso_img'] # todo: try gt_image
            
            gt_rgb = gt_rgb.reshape([1,256,256,3]).permute(0, 3, 1, 2)
            gt_torso_rgb = gt_torso_rgb.reshape([1,256,256,3]).permute(0, 3, 1, 2)

            losses_out = {}

            losses_out['torso_mse_loss'] = torch.mean((pred_torso_rgb - gt_torso_rgb) ** 2) # [B, N, 3] -->  scalar
            alphas = model_out['torso_alpha_map'].clamp(1e-5, 1 - 1e-5) 
            losses_out['torso_weights_entropy_loss'] = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)) # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW

            if 'deform' in model_out.keys():
                deform = model_out['deform']
                losses_out['torso_deform_loss'] = deform.abs().mean()

            if self.global_step >= hparams['sr_start_iters']:
                sr_pred_rgb = model_out['sr_rgb_map']
                sr_pred_torso_rgb = model_out['sr_torso_rgb_map']
                gt_rgb_512 = sample['gt_img_512'].reshape([1,512,512,3]).permute(0, 3, 1, 2)
                gt_torso_rgb_512 = sample['bg_torso_img_512'].reshape([1,512,512,3]).permute(0, 3, 1, 2)
                losses_out['sr_torso_mse_loss'] = torch.mean((sr_pred_torso_rgb - gt_torso_rgb_512) ** 2) # [B, N, 3] -->  scalar
            
            if self.global_step >= hparams['lpips_start_iters']:
                losses_out['torso_lpips_loss'] = self.criterion_lpips(pred_torso_rgb, gt_torso_rgb).mean()
                losses_out['sr_torso_lpips_loss'] = self.criterion_lpips(sr_pred_torso_rgb, gt_torso_rgb_512).mean()

            return losses_out, model_out
            
        else:
            # infer phase, generate the whole image
            model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=False, force_all_rays=True, lm68=sample['lm68'],eye_area_percent=eye_area_percent, **hparams)
            # calculate val loss
            if 'gt_img' in sample:
                gt_rgb = sample['gt_img']
                gt_rgb = gt_rgb.reshape([1,256,256,3]).permute(0, 3, 1, 2)
                pred_rgb = model_out['rgb_map']
                model_out['mse_loss'] = torch.mean((pred_rgb - gt_rgb) ** 2) # [B, N, 3] -->  scalar
            return model_out

    ##########################
    # training 
    ##########################
    def _training_step(self, sample, batch_idx, optimizer_idx):
        outputs = {}
        self.model.on_train_torso_nerf() # not update sr in head_nerf

        self.train_dataset.global_step = self.global_step
        if self.global_step % hparams['update_extra_interval'] == 0:
            self.model.update_extra_state()

        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'torso_mse_loss': 1.0,
            'torso_weights_entropy_loss': hparams['lambda_weights_entropy'],
            'torso_deform_loss': hparams['lambda_torso_deform'] if self.global_step >= 10_0000 else 0,
            'sr_torso_mse_loss': 1.0 if self.global_step >= hparams['sr_start_iters'] else 0,
            
            'torso_lpips_loss': 0.5 * hparams['lambda_lpips_loss'] if self.global_step >= hparams['sr_start_iters'] else 0,
            'sr_torso_lpips_loss': 0.5 * hparams['lambda_lpips_loss'] if self.global_step >= hparams['sr_start_iters'] else 0,
        }
        total_loss = sum([loss_weights[k] * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        def mse2psnr(x): return -10. * torch.log(x+1e-10) / torch.log(torch.Tensor([10.])).to(x.device)
        loss_output['image_psnr'] = mse2psnr(loss_output['torso_mse_loss'].detach())
        outputs.update(loss_output)

        if (self.global_step+1) % hparams['tb_log_interval'] == 0:
            density_grid_info = {
                "density_grid_info/min_density_torso": self.model.density_grid_torso.min().item(),
                "density_grid_info/max_density_torso": self.model.density_grid_torso.max().item(),
                "density_grid_info/mean_density_torso": self.model.mean_density_torso,
                "density_grid_info/occupancy_rate_torso": (self.model.density_grid_torso > min(self.model.mean_density_torso, self.model.density_thresh_torso)).sum() / (128**3 * self.model.cascade), 
                "density_grid_info/step_mean_count_torso": self.model.mean_count
            }
            outputs.update(density_grid_info)
        return total_loss, outputs
    
    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/torso_embedders_params': get_grad_norm(self.torso_embedders_params),
            f'{prefix}/torso_network_params': get_grad_norm(self.torso_network_params ),
        }
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
        return grad_norm_dict
        
