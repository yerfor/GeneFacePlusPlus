import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import modules.radnerfs.raymarching as raymarching
from modules.radnerfs.encoders.encoding import get_encoder
from modules.radnerfs.renderer import NeRFRenderer
from modules.radnerfs.radnerf import RADNeRF
from modules.radnerfs.cond_encoder import AudioNet, AudioAttNet, MLP
from modules.radnerfs.utils import trunc_exp
from modules.radnerfs.utils import custom_meshgrid, convert_poses
from modules.radnerfs.radnerf_sr import Superresolution
from utils.commons.hparams import hparams


class RADNeRFTorsowithSR(RADNeRF):
    def __init__(self, hparams):
        super().__init__(hparams)
        density_grid_torso = torch.zeros([self.grid_size ** 2]) # [H * H]
        self.register_buffer('density_grid_torso', density_grid_torso)
        self.mean_density_torso = 0
        self.density_thresh_torso = hparams['density_thresh_torso']

        self.torso_individual_embedding_num = hparams['individual_embedding_num']
        self.torso_individual_embedding_dim = hparams['torso_individual_embedding_dim']
        if self.torso_individual_embedding_dim > 0:
            self.torso_individual_codes = nn.Parameter(torch.randn(self.torso_individual_embedding_num, self.torso_individual_embedding_dim) * 0.1) 
        
        # self.lm68_embedder = MLP(68*2, 16, 64, 3)
        self.lm68_embedder, self.lm68_embedding_dim = get_encoder('frequency', input_dim=7*2, multires=4)

        self.torso_pose_embedder, self.pose_embedding_dim = get_encoder('frequency', input_dim=6, multires=4)
        self.torso_deform_pos_embedder, self.torso_deform_pos_dim = get_encoder('frequency', input_dim=2, multires=10) # input 2D position
        self.torso_embedder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
        
        # deform_net_in_dim = self.torso_deform_pos_dim + self.pose_embedding_dim + self.torso_individual_embedding_dim
        # canonicial_net_in_dim = self.torso_in_dim + self.torso_deform_pos_dim + self.pose_embedding_dim + self.torso_individual_embedding_dim
        deform_net_in_dim = self.torso_deform_pos_dim + self.lm68_embedding_dim + self.torso_individual_embedding_dim
        canonicial_net_in_dim = self.torso_in_dim + self.torso_deform_pos_dim + self.lm68_embedding_dim + self.torso_individual_embedding_dim

        if hparams['torso_head_aware']:
            head_aware_out_dim = 16
            self.head_color_weights_encoder = nn.Sequential(*[
                nn.Linear(3+1, 16, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(16, 32, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(32, head_aware_out_dim, bias=True),
            ])
            deform_net_in_dim += head_aware_out_dim
            canonicial_net_in_dim += head_aware_out_dim

        self.torso_deform_net = MLP(deform_net_in_dim, 2, 64, 3)
        self.torso_canonicial_net = MLP(canonicial_net_in_dim, 4, 32, 3)
        self.sr_net = Superresolution(channels=3)

    def on_train_torso_nerf(self):
        self.requires_grad_(False)
        if self.torso_individual_embedding_dim > 0:
            self.torso_individual_codes.requires_grad_(True)
        self.torso_pose_embedder.requires_grad_(True)
        self.torso_deform_pos_embedder.requires_grad_(True)
        self.torso_embedder.requires_grad_(True)
        if hparams['torso_head_aware']:
            self.head_color_weights_encoder.requires_grad_(True)
        self.torso_deform_net.requires_grad_(True)
        self.torso_canonicial_net.requires_grad_(True)
        self.sr_net.requires_grad_(False)

    def on_train_superresolution(self):
        self.requires_grad_(False)
        self.sr_net.requires_grad_(True)

    def forward_torso(self, x, poses, c=None, image=None, weights_sum=None, lm68=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 6]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * hparams['torso_shrink']

        # deformation-based 
        enc_pose = self.torso_pose_embedder(poses)
        enc_x = self.torso_deform_pos_embedder(x)
        lm68 = lm68.reshape([1, 68, 2])[:, [5,6,7,8,9,10,11]].reshape([1, -1])
        enc_lm68 = self.lm68_embedder(lm68)

        if c is not None: # individual_code
            # h = torch.cat([enc_x, enc_pose.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
            h = torch.cat([enc_x, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            # h = torch.cat([enc_x, enc_pose.repeat(x.shape[0], 1)], dim=-1)
            h = torch.cat([enc_x], dim=-1)

        h = torch.cat([h, enc_lm68.repeat([x.shape[0],1])], dim=-1)

        if hparams['torso_head_aware']:
            if image is None: # used in update_extra_states
                image = torch.zeros([x.shape[0],3], dtype=h.dtype, device=h.device)
                weights_sum = torch.zeros([x.shape[0],1], dtype=h.dtype, device=h.device)
            head_color_weights_inp = torch.cat([image, weights_sum],dim=-1)
            head_color_weights_encoding = self.head_color_weights_encoder(head_color_weights_inp)
            h = torch.cat([h, head_color_weights_encoding],dim=-1)

        dx = self.torso_deform_net(h)
        x = (x + dx).clamp(-1, 1).float()
        x = self.torso_embedder(x, bound=1)
        h = torch.cat([x, h], dim=-1)
        h = self.torso_canonicial_net(h)
        alpha = torch.sigmoid(h[..., :1])
        color = torch.sigmoid(h[..., 1:])

        return alpha, color, dx

    def render(self, rays_o, rays_d, cond, bg_coords, poses, index=0, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, upscale_torso=False, lm68=None, eye_area_percent=None, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # cond: [B, 29, 16]
        # bg_coords: [1, N, 2]
        # return: pred_rgb: [B, N, 3]

        ### run head nerf with no_grad to get the renderred head
        with torch.no_grad():
            prefix = rays_o.shape[:-1]
            rays_o = rays_o.contiguous().view(-1, 3)
            rays_d = rays_d.contiguous().view(-1, 3)
            bg_coords = bg_coords.contiguous().view(-1, 2)
            N = rays_o.shape[0] # N = B * N, in fact
            device = rays_o.device
            results = {}
            # pre-calculate near far
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
            nears = nears.detach()
            fars = fars.detach()
            # encode audio
            cond_feat = self.cal_cond_feat(cond, eye_area_percent=eye_area_percent) # [1, 64]
            if self.individual_embedding_dim > 0:
                if self.training:
                    ind_code = self.individual_embeddings[index]
                # use a fixed ind code for the unknown test data.
                else:
                    ind_code = self.individual_embeddings[0]
            else:
                ind_code = None
            if self.training:
                # setup counter
                counter = self.step_counter[self.local_step % 16]
                counter.zero_() # set to 0
                self.local_step += 1
                xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
                # xyzs, dirs, deltas, rays, points2rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
                sigmas, rgbs, ambient = self(xyzs, dirs, cond_feat, ind_code)
                sigmas = self.density_scale * sigmas
                #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')
                weights_sum, ambient_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ambient.abs().sum(-1), deltas, rays)
                # for training only
                results['weights_sum'] = weights_sum
                results['ambient'] = ambient_sum
            else:
                dtype = torch.float32
                weights_sum = torch.zeros(N, dtype=dtype, device=device)
                depth = torch.zeros(N, dtype=dtype, device=device)
                image = torch.zeros(N, 3, dtype=dtype, device=device)
                n_alive = N
                rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
                rays_t = nears.clone() # [N]
                step = 0
                while step < max_steps:
                    # count alive rays 
                    n_alive = rays_alive.shape[0]
                    # exit loop
                    if n_alive <= 0:
                        break
                    # decide compact_steps
                    n_step = max(min(N // n_alive, 8), 1)
                    xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                    sigmas, rgbs, ambient = self(xyzs, dirs, cond_feat, ind_code)
                    sigmas = self.density_scale * sigmas
                    raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
                    rays_alive = rays_alive[rays_alive >= 0]
                    step += n_step
            # background
            if bg_color is None:
                bg_color = 1

        ### Start Rendering Torso
        if self.torso_individual_embedding_dim > 0:
            if self.training:
                torso_individual_code = self.torso_individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                torso_individual_code = self.torso_individual_codes[0]
        else:
            torso_individual_code = None

        # 2D density grid for acceleration...
        density_thresh_torso = min(self.density_thresh_torso, self.mean_density_torso)
        occupancy = F.grid_sample(self.density_grid_torso.view(1, 1, self.grid_size, self.grid_size), bg_coords.view(1, -1, 1, 2), align_corners=True).view(-1)
        mask = occupancy > density_thresh_torso

        # masked query of torso
        torso_alpha = torch.zeros([N, 1], device=device)
        torso_color = torch.zeros([N, 3], device=device)

        # image[weights_sum <= 0.95] = 0
        # weights_sum[weights_sum <= 0.95] = 0

        if mask.any():
            if hparams['torso_head_aware']:
                torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, torso_individual_code, image[mask], weights_sum.unsqueeze(-1)[mask], lm68=lm68)
                # if random.random() < 0.5:
                    # torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, torso_individual_code, image[mask], weights_sum.unsqueeze(-1)[mask])
                # else:
                    # torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, torso_individual_code, None, None)
            else:
                torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, torso_individual_code, lm68=lm68)
            torso_alpha[mask] = torso_alpha_mask.float()
            torso_color[mask] = torso_color_mask.float()
            results['deform'] = deform
        # first mix torso with background
        torso_bg_color = torso_color * torso_alpha + bg_color * (1 - torso_alpha)
        results['torso_alpha_map'] = torso_alpha
        results['torso_rgb_map'] = torso_bg_color.reshape([1, 256, 256, 3]).permute(0,3,1,2)
        # then mix the head image with the torso_bg
        image = image + (1 - weights_sum).unsqueeze(-1) * torso_bg_color
        image = image.view(*prefix, 3)
        image = image.clamp(0, 1)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)
        results['depth_map'] = depth
        results['rgb_map'] = image # head_image if train, else com_image
        rgb_image = results['rgb_map'].reshape([1, 256, 256, 3]).permute(0,3,1,2)
        torso_bg_color = torso_bg_color.reshape([1, 256, 256, 3]).permute(0,3,1,2)

        # use SR to upscale
        sr_rgb_image = self.sr_net(rgb_image.clone())
        sr_rgb_image = sr_rgb_image.clamp(0,1)
        results['rgb_map'] = rgb_image
        results['sr_rgb_map'] = sr_rgb_image
        if upscale_torso:
            sr_torso_bg_color = self.sr_net(torso_bg_color.clone())
            sr_torso_bg_color = sr_torso_bg_color.clamp(0,1)
            results['sr_torso_rgb_map'] = sr_torso_bg_color
        return results

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # forbid updating head if is training torso...
        # only update torso density grid
        tmp_grid_torso = torch.zeros_like(self.density_grid_torso)

        # random pose, random ind_code
        rand_idx = random.randint(0, self.poses.shape[0] - 1)
        pose = convert_poses(self.poses[[rand_idx]]).to(self.density_bitfield.device)
        lm68 = self.lm68s[[rand_idx]].to(self.density_bitfield.device).reshape([1, 68*2])

        if self.torso_individual_embedding_dim > 0:
            ind_code = self.torso_individual_codes[[rand_idx]]
        else:
            ind_code = None

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        half_grid_size = 1 / self.grid_size

        for xs in X:
            for ys in Y:
                xx, yy = custom_meshgrid(xs, ys)
                coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1) # [N, 2], in [0, 128)
                indices = (coords[:, 1] * self.grid_size + coords[:, 0]).long() # NOTE: xy transposed!
                xys = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 2] in [-1, 1]
                xys = xys * (1 - half_grid_size)
                # add noise in [-hgs, hgs]
                xys += (torch.rand_like(xys) * 2 - 1) * half_grid_size
                # query density
                alphas, _, _ = self.forward_torso(xys, pose, ind_code, lm68=lm68) # [N, 1]
                
                # assign 
                tmp_grid_torso[indices] = alphas.squeeze(1).float()

        # dilate
        tmp_grid_torso = tmp_grid_torso.view(1, 1, self.grid_size, self.grid_size)
        tmp_grid_torso = F.max_pool2d(tmp_grid_torso, kernel_size=5, stride=1, padding=2)
        tmp_grid_torso = tmp_grid_torso.view(-1)
        
        self.density_grid_torso = torch.maximum(self.density_grid_torso * decay, tmp_grid_torso)
        self.mean_density_torso = torch.mean(self.density_grid_torso).item()
