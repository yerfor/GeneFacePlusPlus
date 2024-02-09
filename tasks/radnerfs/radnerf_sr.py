import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import cv2
import lpips
import matplotlib.pyplot as plt
import traceback

from modules.radnerfs.radnerf_sr import RADNeRFwithSR
from modules.radnerfs.utils import convert_poses, get_bg_coords, get_rays
from tasks.radnerfs.losses import PerceptualLoss
from modules.eg3ds.models.dual_discriminator import DualDiscriminator
from tasks.radnerfs.dataset_utils import RADNeRFDataset

from utils.commons.image_utils import to8b
from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from utils.nn.schedulers import NoneSchedule
from utils.nn.grad import get_grad_norm



class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hw2weight = {512: 0.03125, 256: 0.0625, 128: 0.125, 64: 0.25, 32: 1.0}

    def forward(self, fake_features, real_features):
        # INPUT: list of [b, c, h, w]
        num_feature_maps = len(fake_features)
        loss = 0
        for i in range(num_feature_maps):
            fake_feat = fake_features[i] # b, c, h, w
            hw = fake_feat.shape[-1]
            if hw < 32: # following VGG19, we don't compute on small feature maps, see modules.os_avatar.facev2v_warp.losses.PerceptualLoss
                continue
            real_feat = real_features[i]
            tmp_loss = (fake_feat - real_feat.detach()).abs().mean()
            loss += self.hw2weight[hw] * tmp_loss
        return loss


class ExponentialScheduleForRADNeRF(NoneSchedule):
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
        self.finetune_lips = hparams['finetune_lips']
        self.finetune_lips_start_iter = hparams['finetune_lips_start_iter']

        optimizer.param_groups[0]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        optimizer.param_groups[1]['lr'] = self.lr * 10 # for tileGrid, lr starts from 0.005
        optimizer.param_groups[2]['lr'] = self.lr * 5 # for Att Net, lr starts from 0.0025
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-5)
        else:
            if self.finetune_lips and num_updates > self.finetune_lips_start_iter:
                new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.05x for every 200k steps
            else:
                new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 200k steps

            self.lr = max(new_lrate, 1e-5)

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 10
        self.optimizer.param_groups[2]['lr'] = self.lr * 5
        return self.lr


class RADNeRFTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = RADNeRFDataset
        self.train_dataset = self.dataset_cls(prefix='train', training=True)
        self.val_dataset = self.dataset_cls(prefix='val', training=False)

        self.criterion_lpips = PerceptualLoss()
        self.finetune_lip_flag = False
    
    @property
    def device(self):
        return iter(self.model.parameters()).__next__().device
    
    def build_model(self):
        self.model = RADNeRFwithSR(hparams)
        self.embedders_params = []
        self.embedders_params += [p for k, p in self.model.named_parameters() if p.requires_grad and 'position_embedder' in k]
        self.embedders_params += [p for k, p in self.model.named_parameters() if p.requires_grad and 'ambient_embedder' in k]
        self.network_params = [p for k, p in self.model.named_parameters() if (p.requires_grad and 'position_embedder' not in k and 'ambient_embedder' not in k and 'cond_att_net' not in k)]
        self.att_net_params = [p for k, p in self.model.named_parameters() if p.requires_grad and 'cond_att_net' in k]
        # sr_net also belongs to the newtwork_params
                
        self.model.conds = self.train_dataset.conds
        self.model.mark_untrained_grid(self.train_dataset.poses, self.train_dataset.intrinsics)
    
        if hparams['lambda_dual_fm'] > 0:
            hparams['base_channel'] = 32768
            hparams['max_channel'] = 512
            hparams['final_resolution'] = 512
            hparams['num_fp16_layers_in_discriminator'] = 4
            hparams['group_size_for_mini_batch_std'] = 2
            hparams['disc_c_noise'] = 1
            self.dual_disc = DualDiscriminator()
            pretrained_eg3d_ckpt_path = 'checkpoints/geneface2_ckpts/eg3d_baseline_run2'
            load_ckpt(self.dual_disc, pretrained_eg3d_ckpt_path, strict=True, model_name='disc')
            self.dual_feature_matching_loss_fn = FeatureMatchingLoss()
        return self.model
   
    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.network_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            eps=1e-15)
        self.optimizer.add_param_group({
            'params': self.embedders_params,
            'lr': hparams['lr'] * 10,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })
        self.optimizer.add_param_group({
            'params': self.att_net_params,
            'lr': hparams['lr'] * 5,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })

        return self.optimizer

    def build_scheduler(self, optimizer):
        return ExponentialScheduleForRADNeRF(optimizer, hparams['lr'], hparams['warmup_updates'])

    @data_loader
    def train_dataloader(self):
        self.train_dl = torch.utils.data.DataLoader(self.train_dataset,collate_fn=self.train_dataset.collater,
                                            batch_size=1, shuffle=True, 
                                            # num_workers=0, pin_memory=True)
                                            num_workers=0, pin_memory=False)
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        self.val_dl = torch.utils.data.DataLoader(self.val_dataset,collate_fn=self.val_dataset.collater,
                                            batch_size=1, shuffle=True, 
                                            # num_workers=0, pin_memory=True)
                                            num_workers=0, pin_memory=False)
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        self.val_dl = torch.utils.data.DataLoader(self.val_dataset,collate_fn=self.val_dataset.collater,
                                            batch_size=1, shuffle=False, 
                                            # num_workers=0, pin_memory=True)
                                            num_workers=0, pin_memory=False)
        return self.val_dl
        
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
        eye_area_percent = sample['eye_area_percent'] # [B]
        bg_color = sample['bg_torso_img'] if 'bg_torso_img' in sample else sample['bg_img'] # treat torso as a part of background
        H, W = sample['H'], sample['W']
        cond_mask = sample.get('cond_mask', None)
        cond_inp = cond_wins

        if not infer:
            # training phase, sample rays from the image
            model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, cond_mask=cond_mask, eye_area_percent=eye_area_percent, **hparams)
            pred_rgb = model_out['rgb_map']
            losses_out = {}
            gt_rgb = sample['gt_img'].reshape([1,256,256,3]).permute(0, 3, 1, 2)
            # loss on img_raw
            losses_out['mse_loss'] = torch.mean((pred_rgb - gt_rgb) ** 2) # [B, N, 3] -->  scalar
            
            if self.model.training:
                # only avaliable at raymarching_train
                alphas = model_out['weights_sum'].clamp(1e-5, 1 - 1e-5)
                losses_out['weights_entropy_loss'] = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
                ambient = model_out['ambient'] # [N], abs sum
                face_mask = sample['face_mask'] # [B, N]

                losses_out['num_non_facemask'] = (~face_mask).sum().detach()
                if hparams.get("ambient_loss_mode", 'mae') == 'mae':
                    losses_out['ambient_loss'] = (ambient.abs() * (~face_mask.view(-1))).sum() / ((~face_mask).sum() + 1)
                else:
                    losses_out['ambient_loss'] = (ambient.pow(2) * (~face_mask.view(-1))).sum() / ((~face_mask).sum() + 1)
                if torch.any(torch.isnan(losses_out['ambient_loss'])).item():
                    losses_out['ambient_loss'] = 0
            # loss on img_sr
            if self.global_step >= hparams['sr_start_iters']:
                sr_pred_rgb = model_out['sr_rgb_map']
                gt_rgb_512 = sample['gt_img_512']
                losses_out['sr_mse_loss'] = torch.mean((sr_pred_rgb - gt_rgb_512) ** 2) # [B, N, 3] -->  scalar

            if self.global_step >= hparams['lpips_start_iters']:
                losses_out['lpips_loss'] = self.criterion_lpips(pred_rgb, gt_rgb).mean()
                losses_out['sr_lpips_loss'] = self.criterion_lpips(sr_pred_rgb, gt_rgb_512).mean()
                xmin, xmax, ymin, ymax = sample['lip_rect'] # in 256 resolution so need to x2
                losses_out['sr_lip_lpips_loss'] = self.criterion_lpips(sr_pred_rgb[:,:,xmin*2:xmax*2,ymin*2:ymax*2], gt_rgb_512[:,:,xmin*2:xmax*2,ymin*2:ymax*2]).mean()

            if self.global_step >= hparams['lpips_start_iters']  and hparams['lambda_dual_fm'] > 0:
                fake_image = {'image': sr_pred_rgb, 'image_raw': pred_rgb}
                real_image = {'image': gt_rgb_512, 'image_raw': gt_rgb}
                camera = sample['camera']
                fake_feature_maps, real_feature_maps = [], []
                with torch.cuda.amp.autocast(False): # Disc有部分不支持amp
                    fake_logit = self.dual_disc(fake_image, camera, feature_maps=fake_feature_maps)
                    with torch.no_grad():
                        real_logit = self.dual_disc(real_image, camera, feature_maps=real_feature_maps)
                losses_out['dual_feature_matching_loss'] =  self.dual_feature_matching_loss_fn(fake_feature_maps, real_feature_maps)
            
            losses_out['lambda_ambient'] = self.model.lambda_ambient.item()
            return losses_out, model_out
        else:
            # infer phase, generate the whole image
            model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=False, force_all_rays=True, cond_mask=cond_mask, eye_area_percent=eye_area_percent, **hparams)
            # calculate val loss
            if 'gt_img' in sample:
                gt_rgb = sample['gt_img'].reshape([1,256,256,3]).permute(0, 3, 1, 2)
                pred_rgb = model_out['rgb_map']
                model_out['mse_loss'] = torch.mean((pred_rgb - gt_rgb) ** 2) # [B, N, 3] -->  scalar
                model_out['lpips_loss'] = self.criterion_lpips(pred_rgb, gt_rgb).mean()
 
            return model_out

    ##########################
    # training 
    ##########################
    def _training_step(self, sample, batch_idx, optimizer_idx):
        outputs = {}

        self.train_dataset.global_step = self.global_step
        if self.global_step % hparams['update_extra_interval'] == 0:
            start_finetune_with_lpips = self.global_step > hparams['lpips_start_iters']

            if not start_finetune_with_lpips:
                # when finetuning with lpips, we don't update the density grid and bitfield.
                self.model.update_extra_state()

        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'mse_loss': 1.0,
            'lpips_loss': hparams['lambda_lpips_loss'] if self.global_step >= hparams['lpips_start_iters'] else 0,
            'weights_entropy_loss': hparams['lambda_weights_entropy'],
            'ambient_loss': self.model.lambda_ambient.item(), # adaptative lambda given the ambient_loss
            'sr_mse_loss': 1.0 if self.global_step >= hparams['sr_start_iters'] else 0,
            'sr_lpips_loss': 0.5 * hparams['lambda_lpips_loss'] if self.global_step >= hparams['lpips_start_iters'] else 0,
            'sr_lip_lpips_loss': 0.5 * hparams['lambda_lpips_loss'] if self.global_step >= hparams['lpips_start_iters'] else 0,
            'dual_feature_matching_loss': hparams['lambda_dual_fm'] if self.global_step >= hparams['lpips_start_iters'] else 0
        }
        total_loss = sum([loss_weights[k] * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)
        loss_output['head_psnr'] = mse2psnr(loss_output['mse_loss'].detach())
        outputs.update(loss_output)
        
        # log and update lambda_ambient
        try:
            target_ambient_loss = hparams['target_ambient_loss'] # 1e-7
            current_ambient_loss = loss_output['ambient_loss'].item()
            grad_lambda_ambient = (math.log10(current_ambient_loss+1e-15) - math.log10(target_ambient_loss+1e-15)) # 如果需要增大lambda_ambient， 则current_loss大于targt，grad值大于0
            lr_lambda_ambient = hparams['lr_lambda_ambient']
            self.model.lambda_ambient.data = self.model.lambda_ambient.data + grad_lambda_ambient * lr_lambda_ambient
            self.model.lambda_ambient.data.clamp_(0, 1000)
            outputs['lambda_ambient'] = self.model.lambda_ambient.data
        except:
            traceback.print_exc()
            print("| WARNING: ERROR calculating ambient loss")
        if (self.global_step+1) % hparams['tb_log_interval'] == 0:
            density_grid_info = {
                "density_grid_info/min_density": self.model.density_grid.min().item(),
                "density_grid_info/max_density": self.model.density_grid.max().item(),
                "density_grid_info/mean_density": self.model.mean_density,
                # "density_grid_info/occupancy_rate": (self.model.density_grid > 0.01).sum() / (128**3 * self.model.cascade), 
                "density_grid_info/occupancy_rate": (self.model.density_grid > min(self.model.mean_density, self.model.density_thresh)).sum() / (128**3 * self.model.cascade), 
                "density_grid_info/step_mean_count": self.model.mean_count
            }
            outputs.update(density_grid_info)
        return total_loss, outputs
    
    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/cond_att': get_grad_norm(self.att_net_params),
            f'{prefix}/embedders_params': get_grad_norm(self.embedders_params),
            f'{prefix}/network_params': get_grad_norm(self.network_params ),
        }
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
        return grad_norm_dict
        
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    #####################
    # Validation
    #####################
    def validation_start(self):
        if self.global_step % hparams['valid_infer_interval'] == 0:
            self.gen_dir = os.path.join(hparams['work_dir'], f'validation_results/validation_{self.trainer.global_step}')
            os.makedirs(self.gen_dir, exist_ok=True)
            os.makedirs(f'{self.gen_dir}/images', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/depth', exist_ok=True)

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = 1
        outputs = tensors_to_scalars(outputs)

        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            for k in list(sample):
                del sample[k]
            torch.cuda.empty_cache()
            num_val_samples = len(self.val_dataset)
            interval = (num_val_samples-1) // 4
            idx_lst = [i * interval for i in range(5)]
            if torch.distributed.is_initialized():
                sample = move_to_cuda(self.val_dataset[idx_lst[batch_idx]], gpu_id=torch.distributed.get_rank())
            else:
                sample = move_to_cuda(self.val_dataset[idx_lst[batch_idx]])
            infer_outputs = self.run_model(sample, infer=True)
            H, W = sample['H'], sample['W']
            img_pred = infer_outputs['rgb_map'].permute(0, 2,3,1).reshape([H, W, 3])
            img_pred_sr = infer_outputs['sr_rgb_map'].permute(0, 2,3,1).reshape([512, 512, 3])
            depth_pred = infer_outputs['depth_map'].reshape([H, W])
            
            base_fn = f"frame_{sample['idx']}"
            self.logger.add_figure(f"frame_{sample['idx']}/img_pred", self.rgb_to_figure(img_pred), self.global_step)
            self.logger.add_figure(f"frame_{sample['idx']}/depth_pred", self.rgb_to_figure(depth_pred), self.global_step)

            self.save_rgb_to_fname(img_pred, f"{self.gen_dir}/images/{base_fn}.png")
            self.save_rgb_to_fname(depth_pred, f"{self.gen_dir}/depth/{base_fn}.png")
            base_fn = f"frame_{sample['idx']}_sr"
            self.save_rgb_to_fname(img_pred_sr, f"{self.gen_dir}/images/{base_fn}.png")

            if hparams['save_gt']:
                img_gt = sample['gt_img'].reshape([H, W, 3])
                img_gt_512 = sample['gt_img_512'].permute(0,2,3,1).reshape([512, 512, 3])
                if self.global_step == hparams['valid_infer_interval']:
                    self.logger.add_figure(f"frame_{sample['idx']}/img_gt", self.rgb_to_figure(img_gt), self.global_step)
                base_fn = f"frame_{sample['idx']}_gt"
                self.save_rgb_to_fname(img_gt, f"{self.gen_dir}/images/{base_fn}.png")
                base_fn = f"frame_{sample['idx']}_gt_512"
                self.save_rgb_to_fname(img_gt_512, f"{self.gen_dir}/images/{base_fn}.png")

        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)

    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/images', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/depth', exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        outputs = self.run_model(sample, infer=True)
        rgb_pred = outputs['rgb_map']
        H, W = sample['H'], sample['W']
        img_pred = rgb_pred.reshape([H, W, 3])
        gen_dir = self.gen_dir
        base_fn = f"frame_{sample['idx']}"
        self.save_rgb_to_fname(img_pred, f"{gen_dir}/images/{base_fn}.png")
        self.save_rgb_to_fname(img_pred, f"{gen_dir}/depth/{base_fn}.png")
        target = sample['gt_img']
        img_gt = target.reshape([H, W, 3])
        if hparams['save_gt']:
            base_fn = f"frame_{sample['idx']}_gt"
            self.save_rgb_to_fname(img_gt, f"{gen_dir}/images/{base_fn}.png")
            
        outputs['losses'] = (img_gt - img_pred).mean()
        return outputs

    def test_end(self, outputs):
        pass

    #####################
    # Visualization utils
    #####################
    @staticmethod
    def rgb_to_figure(rgb):
        fig = plt.figure(figsize=(12, 6))
        rgb = convert_to_np(rgb * 255.).astype(np.uint8)
        plt.imshow(rgb)
        return fig
    
    @staticmethod
    def save_rgb_to_fname(rgb, fname):
        rgb = convert_to_np(rgb * 255.).astype(np.uint8)
        if rgb.ndim == 3:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{fname}", bgr)
        else:
            # gray image
            cv2.imwrite(f"{fname}", rgb)

    ### GUI utils
    def test_gui_with_editable_data(self, pose, intrinsics, W, H, cond_wins, index=0, bg_color=None, spp=1, downscale=1):
    # def test_gui_with_edited_data(self, pose, intrinsics, W, H, cond_wins, index=0, bg_color=None, downscale=1):
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        cond_wins = cond_wins.cuda()
        pose = torch.from_numpy(pose).unsqueeze(0).cuda()
        rays = get_rays(pose, intrinsics, rH, rW, -1)
        bg_coords = get_bg_coords(rH, rW, 'cuda')

        sample = {
            'rays_o': rays['rays_o'].cuda(),
            'rays_d': rays['rays_d'].cuda(),
            'H': rH,
            'W': rW,
            'cond_wins': cond_wins,
            'idx': [index], # support choosing index for individual codes
            'pose': convert_poses(pose),
            'bg_coords': bg_coords,
            'bg_img': bg_color.cuda()
        }

        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=hparams['amp']):
                # here spp is used as perturb random seed!
                # face: do not perturb for the first spp, else lead to scatters.
                infer_outputs = self.run_model(sample, infer=True)
            preds = infer_outputs['rgb_map'].reshape([1,rH, rW, 3])
            preds_depth = infer_outputs['depth_map'].reshape([1, rH, rW])

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    # [GUI] test with provided data
    def test_gui_with_data(self, sample, target_W, target_H):
        # prevent calculate loss, which increase costs.
        del sample['gt_img']
        del sample['lip_rect']

        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=hparams['amp']):
                # here spp is used as perturb random seed!
                # face: do not perturb for the first spp, else lead to scatters.
                infer_outputs = self.run_model(sample, infer=True)
        H, W = sample['H'], sample['W']
        preds = infer_outputs['rgb_map'].reshape([1,H, W, 3])
        preds_depth = infer_outputs['depth_map'].reshape([1,H, W])

        # the H/W in data may be differnt to GUI, so we still need to resize...
        preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(target_H, target_W), mode='bilinear').permute(0, 2, 3, 1).contiguous()
        preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(target_H, target_W), mode='nearest').squeeze(1)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

