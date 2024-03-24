import os
import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import uuid
import traceback
# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_np, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478, index_lm131_from_lm478
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from utils.commons.pitch_utils import f0_to_coarse
# Dataset Related
from tasks.radnerfs.dataset_utils import RADNeRFDataset, get_boundary_mask, dilate_boundary_mask, get_lf_boundary_mask
# Method Related
from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
from modules.postnet.lle import compute_LLE_projection, find_k_nearest_neighbors
from modules.radnerfs.utils import get_audio_features, get_rays, get_bg_coords, convert_poses, nerf_matrix_to_ngp
from modules.radnerfs.radnerf import RADNeRF
from modules.radnerfs.radnerf_sr import RADNeRFwithSR
from modules.radnerfs.radnerf_torso import RADNeRFTorso
from modules.radnerfs.radnerf_torso_sr import RADNeRFTorsowithSR


face3d_helper = None
def vis_cano_lm3d_to_imgs(cano_lm3d, hw=512):
    cano_lm3d_ = cano_lm3d[:1, ].repeat([len(cano_lm3d),1,1])
    cano_lm3d_[:, 17:27] = cano_lm3d[:, 17:27] # brow
    cano_lm3d_[:, 36:48] = cano_lm3d[:, 36:48] # eye
    cano_lm3d_[:, 27:36] = cano_lm3d[:, 27:36] # nose
    cano_lm3d_[:, 48:68] = cano_lm3d[:, 48:68] # mouth
    cano_lm3d_[:, 0:17] = cano_lm3d[:, :17] # yaw
    
    cano_lm3d = cano_lm3d_

    cano_lm3d = convert_to_np(cano_lm3d)

    WH = hw
    cano_lm3d = (cano_lm3d * WH/2 + WH/2).astype(int)
    frame_lst = []
    for i_img in range(len(cano_lm3d)):
        # lm2d = cano_lm3d[i_img ,:, 1:] # [68, 2]
        lm2d = cano_lm3d[i_img ,:, :2] # [68, 2]
        img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            color = (255,0,0)
            img = cv2.circle(img, center=(x,y), radius=3, color=color, thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.flip(img, 0)
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            y = WH - y
            img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
        frame_lst.append(img)
    return frame_lst

def inject_blink_to_lm68(lm68, opened_eye_area_percent=0.6, closed_eye_area_percent=0.15):
    # [T, 68, 2]
    # lm68[:,36:48] = lm68[0:1,36:48].repeat([len(lm68), 1, 1])
    opened_eye_lm68 = copy.deepcopy(lm68)
    eye_area_percent = 0.6 * torch.ones([len(lm68), 1], dtype=opened_eye_lm68.dtype, device=opened_eye_lm68.device)

    eye_open_scale = (opened_eye_lm68[:, 41, 1] - opened_eye_lm68[:, 37, 1]) + (opened_eye_lm68[:, 40, 1] - opened_eye_lm68[:, 38, 1]) + (opened_eye_lm68[:, 47, 1] - opened_eye_lm68[:, 43, 1]) + (opened_eye_lm68[:, 46, 1] - opened_eye_lm68[:, 44, 1])
    eye_open_scale = eye_open_scale.abs()
    idx_largest_eye = eye_open_scale.argmax()
    # lm68[:, list(range(17,27))] = lm68[idx_largest_eye:idx_largest_eye+1, list(range(17,27))].repeat([len(lm68),1,1])
    # lm68[:, list(range(36,48))] = lm68[idx_largest_eye:idx_largest_eye+1, list(range(36,48))].repeat([len(lm68),1,1])
    lm68[:,[37,38,43,44],1] = lm68[:,[37,38,43,44],1] + 0.03
    lm68[:,[41,40,47,46],1] = lm68[:,[41,40,47,46],1] - 0.03
    closed_eye_lm68 = copy.deepcopy(lm68)
    closed_eye_lm68[:,37] = closed_eye_lm68[:,41] = closed_eye_lm68[:,36] * 0.67 + closed_eye_lm68[:,39] * 0.33
    closed_eye_lm68[:,38] = closed_eye_lm68[:,40] = closed_eye_lm68[:,36] * 0.33 + closed_eye_lm68[:,39] * 0.67
    closed_eye_lm68[:,43] = closed_eye_lm68[:,47] = closed_eye_lm68[:,42] * 0.67 + closed_eye_lm68[:,45] * 0.33
    closed_eye_lm68[:,44] = closed_eye_lm68[:,46] = closed_eye_lm68[:,42] * 0.33 + closed_eye_lm68[:,45] * 0.67
    
    T = len(lm68)
    period = 100
    # blink_factor_lst = np.array([0.4, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.4]) # * 0.9
    # blink_factor_lst = np.array([0.4, 0.7, 0.8, 1.0, 0.8, 0.6, 0.4]) # * 0.9
    blink_factor_lst = np.array([0.1, 0.5, 0.7, 1.0, 0.7, 0.5, 0.1]) # * 0.9
    dur = len(blink_factor_lst)
    for i in range(T):
        if (i + 25) % period == 0:
            for j in range(dur):
                idx = i+j
                if idx > T - 1: # prevent out of index error
                    break
                blink_factor = blink_factor_lst[j]
                lm68[idx, 36:48] = lm68[idx, 36:48] * (1-blink_factor) + closed_eye_lm68[idx, 36:48] * blink_factor
                eye_area_percent[idx] = opened_eye_area_percent * (1-blink_factor) + closed_eye_area_percent * blink_factor
    return lm68, eye_area_percent


class GeneFace2Infer:
    def __init__(self, audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.audio2secc_dir = audio2secc_dir
        self.postnet_dir = postnet_dir
        self.head_model_dir = head_model_dir
        self.torso_model_dir = torso_model_dir
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.postnet_model = self.load_postnet(postnet_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir)
        self.audio2secc_model.to(device).eval()
        if self.postnet_model is not None:
            self.postnet_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=True)
        hparams['infer_smooth_camera_path_kernel_size'] = 7

    def load_audio2secc(self, audio2secc_dir):
        set_hparams(f"{os.path.dirname(audio2secc_dir) if os.path.isfile(audio2secc_dir) else audio2secc_dir}/config.yaml")
        self.audio2secc_hparams = copy.deepcopy(hparams)
        if hparams["motion_type"] == 'id_exp':
            self.in_out_dim = 80 + 64
        elif hparams["motion_type"] == 'exp':
            self.in_out_dim = 64
        audio_in_dim = 1024
        if hparams.get("use_pitch", False) is True:
            self.model = PitchContourVAEModel(hparams, in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim)
        else:
            self.model = VAEModel(in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim)
        load_ckpt(self.model, f"{audio2secc_dir}", model_name='model', strict=True)
        return self.model

    def load_postnet(self, postnet_dir):
        if postnet_dir == '':
            return None
        from modules.postnet.models import PitchContourCNNPostNet
        set_hparams(f"{os.path.dirname(postnet_dir) if os.path.isfile(postnet_dir) else postnet_dir}/config.yaml")
        self.postnet_hparams = copy.deepcopy(hparams)
        in_out_dim = 68*3
        pitch_dim = 128
        self.model = PitchContourCNNPostNet(in_out_dim=in_out_dim, pitch_dim=pitch_dim)
        load_ckpt(self.model, f"{postnet_dir}", steps=20000, model_name='model', strict=True)
        return self.model

    def load_secc2video(self, head_model_dir, torso_model_dir):

        if torso_model_dir != '':
            set_hparams(f"{os.path.dirname(torso_model_dir) if os.path.isfile(torso_model_dir) else torso_model_dir}/config.yaml")
            self.secc2video_hparams = copy.deepcopy(hparams)
            if hparams.get("with_sr"):
                model = RADNeRFTorsowithSR(hparams)
            else:
                model = RADNeRFTorso(hparams)
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=True)
        else:
            set_hparams(f"{os.path.dirname(head_model_dir) if os.path.isfile(head_model_dir) else head_model_dir}/config.yaml")
            self.secc2video_hparams = copy.deepcopy(hparams)
            if hparams.get("with_sr"):
                model = RADNeRFwithSR(hparams)
            else:
                model = RADNeRF(hparams)
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
        self.dataset_cls = RADNeRFDataset # the dataset only provides head pose 
        self.dataset = self.dataset_cls('trainval', training=False)
        eye_area_percents = torch.tensor(self.dataset.eye_area_percents)
        self.closed_eye_area_percent = torch.quantile(eye_area_percents, q=0.03).item()
        self.opened_eye_area_percent = torch.quantile(eye_area_percents, q=0.97).item()
        try:
            model = torch.compile(model)
        except:
            traceback.print_exc()
        return model

    def infer_once(self, inp):
        self.inp = inp
        samples = self.prepare_batch_from_inp(inp)
        out_name = self.forward_system(samples, inp)
        return out_name
        
    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        sample = {}
        # Process Audio
        self.save_wav16k(inp['drv_audio_name'])
        hubert = self.get_hubert(self.wav16k_name)
        t_x = hubert.shape[0]
        x_mask = torch.ones([1, t_x]).float() # mask for audio frames
        y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
        f0 = self.get_f0(self.wav16k_name)
        if f0.shape[0] > len(hubert):
            f0 = f0[:len(hubert)]
        else:
            num_to_pad = len(hubert) - len(f0)
            f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))

        sample.update({
            'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
            'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
            'x_mask': x_mask.cuda(),
            'y_mask': y_mask.cuda(),
            })
        
        sample['audio'] = sample['hubert']
        sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()

        sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
        sample['mouth_amp'] = torch.ones([1, 1]).cuda() * float(inp['mouth_amp'])
        # sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:t_x//2]).cuda()
        sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:1]).cuda().repeat([t_x//2, 1])
        pose_lst = []
        euler_lst = []
        trans_lst = []
        rays_o_lst = []
        rays_d_lst = []

        if '-' in inp['drv_pose']:
            start_idx, end_idx = inp['drv_pose'].split("-")
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            ds_ngp_pose_lst = [self.dataset.poses[i].unsqueeze(0) for i in range(start_idx, end_idx)]
            ds_euler_lst = [torch.tensor(self.dataset.ds_dict['euler'][i]) for i in range(start_idx, end_idx)]
            ds_trans_lst = [torch.tensor(self.dataset.ds_dict['trans'][i]) for i in range(start_idx, end_idx)]

        for i in range(t_x//2):
            if inp['drv_pose'] == 'static':
                ngp_pose = self.dataset.poses[0].unsqueeze(0)
                euler = torch.tensor(self.dataset.ds_dict['euler'][0])
                trans = torch.tensor(self.dataset.ds_dict['trans'][0])
            elif '-' in inp['drv_pose']:
                mirror_idx = mirror_index(i, len(ds_ngp_pose_lst))
                ngp_pose = ds_ngp_pose_lst[mirror_idx]
                euler = ds_euler_lst[mirror_idx]
                trans = ds_trans_lst[mirror_idx]
            else:
                ngp_pose = self.dataset.poses[i].unsqueeze(0)
                euler = torch.tensor(self.dataset.ds_dict['euler'][i])
                trans = torch.tensor(self.dataset.ds_dict['trans'][i])

            rays = get_rays(ngp_pose.cuda(), self.dataset.intrinsics, self.dataset.H, self.dataset.W, N=-1)
            rays_o_lst.append(rays['rays_o'].cuda())
            rays_d_lst.append(rays['rays_d'].cuda())
            pose = convert_poses(ngp_pose).cuda()
            pose_lst.append(pose)
            euler_lst.append(euler)
            trans_lst.append(trans)
        sample['rays_o'] = rays_o_lst
        sample['rays_d'] = rays_d_lst
        sample['poses'] = pose_lst
        sample['euler'] = torch.stack(euler_lst).cuda()
        sample['trans'] = torch.stack(trans_lst).cuda()
        sample['bg_img'] = self.dataset.bg_img.reshape([1,-1,3]).cuda()
        sample['bg_coords'] = self.dataset.bg_coords.cuda()
        return sample

    @torch.no_grad()
    def get_hubert(self, wav16k_name):
        from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
        hubert = get_hubert_from_16k_wav(wav16k_name).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        hubert = hubert[:len(hubert)//8*8]
        # if len_mel % x_multiply == 0:
        #     num_to_pad = 0
        # else:
        #     num_to_pad = x_multiply - len_mel % x_multiply
        # hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0)))
        return hubert

    def get_f0(self, wav16k_name):
        from data_gen.utils.process_audio.extract_mel_f0 import extract_mel_from_fname, extract_f0_from_wav_and_mel
        wav, mel = extract_mel_from_fname(self.wav16k_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        f0 = f0.reshape([-1,1])
        return f0
    
    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        # forward the audio-to-motion
        ret = {}
        pred = self.audio2secc_model.forward(batch, ret=ret,train=False, temperature=inp['temperature'])
        if pred.shape[-1] == 144:
            id = ret['pred'][0][:,:80]
            exp = ret['pred'][0][:,80:]
        else:
            id = batch['id']
            exp = ret['pred'][0]

        # exp = smooth_features_xd(exp, kernel_size=3)
        batch['exp'] = exp

        # render the SECC given the id,exp.
        # note that SECC is only used for visualization
        zero_eulers = torch.zeros([id.shape[0], 3]).to(id.device)
        zero_trans = torch.zeros([id.shape[0], 3]).to(exp.device)
        if inp['debug']:
            with torch.no_grad():
                chunk_size = 50
                drv_secc_color_lst = []
                num_iters = len(id)//chunk_size if len(id)%chunk_size == 0 else len(id)//chunk_size+1
                for i in tqdm.trange(num_iters, desc="rendering secc"):
                    torch.cuda.empty_cache()
                    face_mask, drv_secc_color = self.secc_renderer(id[i*chunk_size:(i+1)*chunk_size], exp[i*chunk_size:(i+1)*chunk_size], zero_eulers[i*chunk_size:(i+1)*chunk_size], zero_trans[i*chunk_size:(i+1)*chunk_size])
                    drv_secc_color_lst.append(drv_secc_color.cpu())
            drv_secc_colors = torch.cat(drv_secc_color_lst, dim=0)
            _, src_secc_color = self.secc_renderer(id[0:1], exp[0:1], zero_eulers[0:1], zero_trans[0:1])
            _, cano_secc_color = self.secc_renderer(id[0:1]*0, exp[0:1]*0, zero_eulers[0:1], zero_trans[0:1])
            batch['drv_secc'] = drv_secc_colors.cuda()
            batch['src_secc'] = src_secc_color.cuda()
            batch['cano_secc'] = cano_secc_color.cuda()

        # get idexp_lm3d
        id_ds = torch.from_numpy(self.dataset.ds_dict['id']).float().cuda()
        exp_ds = torch.from_numpy(self.dataset.ds_dict['exp']).float().cuda()
        idexp_lm3d_ds = self.face3d_helper.reconstruct_idexp_lm3d(id_ds, exp_ds)
        idexp_lm3d_mean = idexp_lm3d_ds.mean(dim=0, keepdim=True)
        idexp_lm3d_std = idexp_lm3d_ds.std(dim=0, keepdim=True)
        if hparams.get("normalize_cond", True):
            idexp_lm3d_ds_normalized = (idexp_lm3d_ds - idexp_lm3d_mean) / idexp_lm3d_std
        else:
            idexp_lm3d_ds_normalized = idexp_lm3d_ds
        lower = torch.quantile(idexp_lm3d_ds_normalized, q=0.03, dim=0)
        upper = torch.quantile(idexp_lm3d_ds_normalized, q=0.97, dim=0)

        LLE_percent = inp['lle_percent']
            
        keypoint_mode = self.secc2video_model.hparams.get("nerf_keypoint_mode", "lm68")
        if self.postnet_model is not None:
            idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478]
            idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
            lower = lower[index_lm68_from_lm478]
            upper = upper[index_lm68_from_lm478]

            f0 = self.audio2secc_model.downsampler(batch['f0'].unsqueeze(-1)).squeeze(-1)
            pitch = self.audio2secc_model.pitch_embed(f0_to_coarse(f0))
            b,t,*_ = pitch.shape
            raw_pred_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id, exp).reshape([t, -1, 3])
            raw_pred_lm3d = raw_pred_lm3d[:, index_lm68_from_lm478].reshape([1,t,68*3])
            
            idexp_lm3d = self.postnet_model(raw_pred_lm3d, pitch[:, :exp.shape[0]*2, :]).reshape([t,68*3])

            idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
            feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d[:, :68*3], feat_database=idexp_lm3d_ds_lle[:, :68*3], K=10)
            # feat_fuse = smooth_features_xd(feat_fuse, kernel_size=3)
            idexp_lm3d[:, :68*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d[:,:68*3]
            idexp_lm3d = idexp_lm3d.reshape([t, 68, 3])
            idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std
            idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)

        else:
            idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id, exp)
            if keypoint_mode == 'lm68':
                idexp_lm3d = idexp_lm3d[:, index_lm68_from_lm478]
                idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478]
                idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
                lower = lower[index_lm68_from_lm478]
                upper = upper[index_lm68_from_lm478]
            elif keypoint_mode == 'lm131':
                idexp_lm3d = idexp_lm3d[:, index_lm131_from_lm478]
                idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm131_from_lm478]
                idexp_lm3d_std = idexp_lm3d_std[:, index_lm131_from_lm478]
                lower = lower[index_lm131_from_lm478]
                upper = upper[index_lm131_from_lm478]
            elif keypoint_mode == 'lm468':
                idexp_lm3d = idexp_lm3d
            else:
                raise NotImplementedError()
            idexp_lm3d = idexp_lm3d.reshape([-1, 68*3])
            idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
            feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d[:, :68*3], feat_database=idexp_lm3d_ds_lle[:, :68*3], K=10)
            # feat_fuse = smooth_features_xd(feat_fuse, kernel_size=3)
            
            idexp_lm3d[:, :68*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d[:,:68*3]
            idexp_lm3d = idexp_lm3d.reshape([-1, 68, 3])
            idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std
            # idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)

        cano_lm3d = (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized) / 10 + self.face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)
        eye_area_percent = self.opened_eye_area_percent * torch.ones([len(cano_lm3d), 1], dtype=cano_lm3d.dtype, device=cano_lm3d.device)
        
        if inp['blink_mode'] == 'period':
            cano_lm3d, eye_area_percent = inject_blink_to_lm68(cano_lm3d, self.opened_eye_area_percent, self.closed_eye_area_percent)
            print("Injected blink to idexp_lm3d by directly editting.")
        batch['eye_area_percent'] = eye_area_percent
        idexp_lm3d_normalized = ((cano_lm3d - self.face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)) * 10 - idexp_lm3d_mean) / idexp_lm3d_std
        idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)
        batch['cano_lm3d'] = cano_lm3d

        assert keypoint_mode == 'lm68'
        idexp_lm3d_normalized_ = idexp_lm3d_normalized[0:1, :].repeat([len(exp),1,1]).clone()
        idexp_lm3d_normalized_[:, 17:27] = idexp_lm3d_normalized[:, 17:27] # brow
        idexp_lm3d_normalized_[:, 36:48] = idexp_lm3d_normalized[:, 36:48] # eye
        idexp_lm3d_normalized_[:, 27:36] = idexp_lm3d_normalized[:, 27:36] # nose
        idexp_lm3d_normalized_[:, 48:68] = idexp_lm3d_normalized[:, 48:68] # mouth
        idexp_lm3d_normalized_[:, 0:17] = idexp_lm3d_normalized[:, :17] # yaw

        idexp_lm3d_normalized = idexp_lm3d_normalized_

        cond_win = idexp_lm3d_normalized.reshape([len(exp), 1, -1])
        cond_wins = [get_audio_features(cond_win, att_mode=2, index=idx) for idx in range(len(cond_win))]
        batch['cond_wins'] = cond_wins

        # face boundark mask, for cond mask
        smo_euler = smooth_features_xd(batch['euler'])
        smo_trans = smooth_features_xd(batch['trans'])
        lm2d = self.face3d_helper.reconstruct_lm2d_nerf(id, exp, smo_euler, smo_trans)
        lm68 = lm2d[:, index_lm68_from_lm478, :]
        batch['lm68'] = lm68.reshape([lm68.shape[0], 68*2])

        return batch

    @torch.no_grad()
    # todo: fp16 infer for faster inference. Now 192/128/64 are all 45fps
    def forward_secc2video(self, batch, inp=None):
        num_frames = len(batch['poses'])
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        cond_inp = batch['cond_wins']
        bg_coords = batch['bg_coords']
        bg_color = batch['bg_img']
        poses = batch['poses']
        lm68s = batch['lm68']
        eye_area_percent = batch['eye_area_percent']

        pred_rgb_lst = []

        # forward renderer
        if inp['low_memory_usage']:
            # save memory, when one image is rendered, write it into video
            try:
                os.makedirs(os.path.dirname(inp['out_name']), exist_ok=True)
            except: pass
            import imageio
            tmp_out_name = inp['out_name'].replace(".mp4", ".tmp.mp4")
            writer = imageio.get_writer(tmp_out_name, fps = 25, format='FFMPEG', codec='h264')

            with torch.cuda.amp.autocast(enabled=True):
                # forward neural renderer
                for i in tqdm.trange(num_frames, desc="GeneFace++ is rendering... "):
                    model_out = self.secc2video_model.render(rays_o[i], rays_d[i], cond_inp[i], bg_coords, poses[i], index=i, staged=False, bg_color=bg_color, lm68=lm68s[i], perturb=False, force_all_rays=False,
                                    T_thresh=inp['raymarching_end_threshold'], eye_area_percent=eye_area_percent[i],
                                    **hparams)
                    if self.secc2video_hparams.get('with_sr', False):
                        pred_rgb = model_out['sr_rgb_map'][0].cpu() # [c, h, w]
                    else:
                        pred_rgb = model_out['rgb_map'][0].reshape([512,512,3]).permute(2,0,1).cpu()
                    img = (pred_rgb.permute(1,2,0) * 255.).int().cpu().numpy().astype(np.uint8)
                    writer.append_data(img)
            writer.close()

        else:

            with torch.cuda.amp.autocast(enabled=True):
                # forward neural renderer
                for i in tqdm.trange(num_frames, desc="GeneFace++ is rendering... "):
                    model_out = self.secc2video_model.render(rays_o[i], rays_d[i], cond_inp[i], bg_coords, poses[i], index=i, staged=False, bg_color=bg_color, lm68=lm68s[i], perturb=False, force_all_rays=False,
                                    T_thresh=inp['raymarching_end_threshold'], eye_area_percent=eye_area_percent[i],
                                    **hparams)
                    if self.secc2video_hparams.get('with_sr', False):
                        pred_rgb = model_out['sr_rgb_map'][0].cpu() # [c, h, w]
                    else:
                        pred_rgb = model_out['rgb_map'][0].reshape([512,512,3]).permute(2,0,1).cpu()

                    pred_rgb_lst.append(pred_rgb)
            pred_rgbs = torch.stack(pred_rgb_lst).cpu()
            pred_rgbs = pred_rgbs * 2 - 1 # to -1~1 scale

            if inp['debug']:
                # prepare for output
                drv_secc_colors = batch['drv_secc']
                secc_img = torch.cat([torch.nn.functional.interpolate(drv_secc_colors[i:i+1], (512,512)) for i in range(num_frames)]).cpu()
                cano_lm3d_frame_lst = vis_cano_lm3d_to_imgs(batch['cano_lm3d'], hw=512)
                cano_lm3d_frames = convert_to_tensor(np.stack(cano_lm3d_frame_lst)).permute(0, 3, 1, 2) / 127.5 - 1
                imgs = torch.cat([pred_rgbs, cano_lm3d_frames, secc_img], dim=3) # [B, C, H, W]
            else:
                imgs = pred_rgbs
            imgs = imgs.clamp(-1,1)

            try:
                os.makedirs(os.path.dirname(inp['out_name']), exist_ok=True)
            except: pass
            import imageio
            tmp_out_name = inp['out_name'].replace(".mp4", ".tmp.mp4")
            out_imgs = ((imgs.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
            writer = imageio.get_writer(tmp_out_name, fps = 25, format='FFMPEG', codec='h264')
            for i in tqdm.trange(len(out_imgs), desc=f"ImageIO is saving video using FFMPEG(h264) to {tmp_out_name}"):
                writer.append_data(out_imgs[i])
            writer.close()

        cmd = f"ffmpeg -i {tmp_out_name} -i {self.wav16k_name} -y -shortest -c:v libx264 -pix_fmt yuv420p -b:v 2000k -y -v quiet -shortest {inp['out_name']}"
        ret = os.system(cmd)
        if ret == 0:
            print(f"Saved at {inp['out_name']}")
            os.system(f"rm {self.wav16k_name}")
            os.system(f"rm {tmp_out_name}")
        else:
            raise ValueError(f"error running {cmd}, please check ffmpeg installation, especially check whether it supports libx264!")

    @torch.no_grad()
    def forward_system(self, batch, inp):
        self.forward_audio2secc(batch, inp)
        self.forward_secc2video(batch, inp)
        return inp['out_name']

    @classmethod
    def example_run(cls, inp=None):
        inp_tmp = {
            'drv_audio_name': 'data/raw/val_wavs/zozo.wav',
            'src_image_name': 'data/raw/val_imgs/Macron.png'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp

        infer_instance = cls(inp['a2m_ckpt'], inp['postnet_ckpt'], inp['head_ckpt'], inp['torso_ckpt'])
        infer_instance.infer_once(inp)

    ##############
    # IO-related
    ##############
    def save_wav16k(self, audio_name):
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert audio_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = audio_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {audio_name} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {audio_name} to {wav16k_name}.")


if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/audio2motion_vae')  # checkpoints/0727_audio2secc/audio2secc_withlm2d100_randomframe
    parser.add_argument("--head_ckpt", default='') 
    parser.add_argument("--postnet_ckpt", default='') 
    parser.add_argument("--torso_ckpt", default='')
    parser.add_argument("--drv_aud", default='data/raw/val_wavs/MacronSpeech.wav')
    parser.add_argument("--drv_pose", default='nearest', help="目前仅支持源视频的pose,包括从头开始和指定区间两种,暂时不支持in-the-wild的pose")
    parser.add_argument("--blink_mode", default='none') # none | period
    # parser.add_argument("--blink_mode", default='period') # none | period
    parser.add_argument("--lle_percent", default=0.2) # nearest | random
    parser.add_argument("--temperature", default=0.2) # nearest | random
    parser.add_argument("--mouth_amp", default=0.4) # nearest | random
    parser.add_argument("--raymarching_end_threshold", default=0.01, help="increase it to improve fps") # nearest | random
    parser.add_argument("--debug", action='store_true') 
    parser.add_argument("--fast", action='store_true') 
    parser.add_argument("--out_name", default='tmp.mp4') 
    parser.add_argument("--low_memory_usage", action='store_true', help='write img to video upon generated, leads to slower fps, but use less memory')


    args = parser.parse_args()

    inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'postnet_ckpt': args.postnet_ckpt,
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'drv_audio_name': args.drv_aud,
            'drv_pose': args.drv_pose,
            'blink_mode': args.blink_mode,
            'temperature': float(args.temperature),
            'mouth_amp': args.mouth_amp,
            'lle_percent': float(args.lle_percent),
            'debug': args.debug,
            'out_name': args.out_name,
            'raymarching_end_threshold': args.raymarching_end_threshold,
            'low_memory_usage': args.low_memory_usage,
            }
    if args.fast:
        inp['raymarching_end_threshold'] = 0.05
    GeneFace2Infer.example_run(inp)
