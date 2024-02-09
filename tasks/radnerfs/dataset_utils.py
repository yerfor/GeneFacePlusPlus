import os
import tqdm
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic

from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from utils.commons.image_utils import load_image_as_uint8_tensor
from utils.commons.meters import Timer

from modules.radnerfs.utils import get_audio_features, get_rays, get_bg_coords, convert_poses, nerf_matrix_to_ngp
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.mp_feature_extractors.mp_segmenter import decode_segmap_mask_from_image
from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478

face3d_helper = None


def dilate(bin_img, ksize=5, mode='max_pool'):
    """
    mode: max_pool or avg_pool
    """
    # bin_img, [1, h, w]
    pad = (ksize-1)//2
    bin_img = F.pad(bin_img, pad=[pad,pad,pad,pad], mode='reflect')
    if mode == 'max_pool':
        out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    else:
        out = F.avg_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def get_hull_mask(points, h=512, w=512):
    """
    points: [N, 2], 0~1 float
    mask: [H, W], binary
    """
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices([w, h], np.int16)
    idx_2d = idx_2d.transpose(1, 2, 0)
    mask = np.zeros([w, h])
    s = deln.find_simplex(idx_2d)
    mask[(s != -1)] = 1
    mask = mask.transpose(1, 0)
    return mask

def get_lf_boundary_mask(lm468, h=512, w=512, index_mode='lm468'):
    """
    lm468: [N=468, 3]
    """
    lm468 = convert_to_np(lm468)
    # index_boundary_from_lm468 = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
    if index_mode == 'lm468':
        index_boundary_from_lm468 = list(range(468))
    elif index_mode == 'lm68':
        from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478
        # index_lf_from_lm68 = [4,5,6,7,8,9,10, 11,12] + list(range(48, 68))
        index_lf_from_lm68 = [4,5,6,7,8,9,10,11,12] + list(range(27, 36)) + list(range(48,68))

        index_boundary_from_lm68 = index_lf_from_lm68
    lm468 = lm468[..., :2] * np.array([[w, h]])
    lm68 = lm468[index_lm68_from_lm478].astype(int)
    boundary_kp = lm68[index_boundary_from_lm68].astype(int)
    # nose_mouth_kp = lm68[[33,51]].mean(axis=0).reshape([1,2])
    # boundary_kp = np.concatenate([boundary_kp, nose_mouth_kp])
    mask = get_hull_mask(boundary_kp, h, w) 
    return convert_to_tensor(mask)

def get_boundary_mask(lm468, h=512, w=512, index_mode='lm468'):
    """
    lm468: [N=468, 3]
    """
    lm468 = convert_to_np(lm468)
    # index_boundary_from_lm468 = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
    if index_mode == 'lm468':
        index_boundary_from_lm468 = list(range(468))
    elif index_mode == 'lm68':
        from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478
        index_boundary_from_lm468 = index_lm68_from_lm478
    lm468 = lm468[..., :2] * np.array([[w, h]])
    boundary_kp = lm468[index_boundary_from_lm468].astype(int)
    mask = get_hull_mask(boundary_kp, h, w)
    return convert_to_tensor(mask)

def dilate_boundary_mask(mask, ksize=11):
    mask = dilate(mask, ksize=ksize, mode='max_pool')
    mask = dilate(mask, ksize=ksize, mode='avg_pool')
    return mask

def make_coordinate_grid(spatial_size):
    w, h = spatial_size
    x = torch.arange(w)
    y = torch.arange(h)
    x = x / (w - 1)
    y = y / (h - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.stack([xx, yy], -1)
    return meshed

def get_ldm_heatmap(spatial_size, ldm, std=0.006):
    """
    spatial_size: [W,H]
    ldm: pixel scale landmark, [B,T,478,3]
    """
    ldm = ldm[..., :2] # [B,T,N,2]
    meshed = make_coordinate_grid(spatial_size).to(ldm.device)  # [W, H, 2]
    ldm_normed = ldm / torch.FloatTensor(spatial_size)[None, :].to(ldm.device)  # [N_kp, 2]
    heatmap = torch.exp(-((meshed[None, None, :, :, None, :] - ldm_normed[:, :, None, None, :, :]) ** 2).sum(-1) / 2 / (std ** 2))
    return heatmap # [B, T, W, H, C=N_points=478]
 
def transform_normed_lm_to_pixel_lm(normed_lms, img_w=512, img_h=512):
    """
    normed_lms: array [B,T,N_points=778,N_dim=3], landmarks normalized to 0~1
    """
    normed_lms = convert_to_np(normed_lms) # [B, T, 778, 3]
    ldms_px = normed_lms * np.array([[[img_w, img_h, img_w]]])  # [B, T, 778, 3] * [ 1, 3]
    x_min, y_min, z_min = np.min(ldms_px, axis=(0, 1))
    x_max, y_max, z_max = np.max(ldms_px, axis=(0, 1))
    margin_w = (x_max - x_min) * 0.1
    margin_h = (y_max - y_min) * 0.1
    x_max = int(np.around(x_max + margin_w, 0))
    x_min = int(np.max(np.around(x_min - margin_w), 0))
    y_max = int(np.around(y_max + margin_h))
    y_min = int(np.max(np.around(y_min - margin_h), 0))
    ldms_px = ldms_px - np.array([[[x_min, y_min, -128]]])
    return ldms_px

def smooth_camera_path(poses, kernel_size=7):
    # smooth the camera trajectory (i.e., translation)...
    # poses: [N, 4, 4], numpy array
    N = poses.shape[0]
    K = kernel_size // 2
    
    trans = poses[:, :3, 3].copy() # [N, 3]
    rots = poses[:, :3, :3].copy() # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        try:
            poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()
        except:
            if i == 0:
                poses[i, :3, :3] = rots[i]
            else:
                poses[i, :3, :3] = poses[i-1, :3, :3]
    return poses


class RADNeRFDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, data_dir=None, training=True):
        super().__init__()
        self.hparams = hparams
        self.data_dir = os.path.join(hparams['binary_data_dir'], hparams['video_id']) if data_dir is None else data_dir
        binary_file_name = os.path.join(self.data_dir, "trainval_dataset.npy")
        self.ds_dict = ds_dict = np.load(binary_file_name, allow_pickle=True).tolist()
        if prefix == 'train':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']]
        elif prefix == 'val':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        elif prefix == 'trainval':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']] + [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        else:
            raise ValueError("prefix should in train/val !")
        
        num_train_samples = hparams.get("num_train_samples", 0)
        if num_train_samples != 0:
            orig_len = len(self.samples)
            if orig_len >= num_train_samples:
                self.samples = self.samples[: num_train_samples]
                print(f"| WARNING: we are only using the first {num_train_samples} frames of total {orig_len} frames to train the model!")

        self.prefix = prefix
        self.cond_type = hparams['cond_type']
        self.H = ds_dict['H']
        self.W = ds_dict['W']
        if hparams.get("with_sr"):
            # relate to intrinsic and face_mask and lip_rect, forget this will lead to nan
            self.H = self.H // 2
            self.W = self.W // 2
        self.focal = ds_dict['focal']
        self.cx = ds_dict['cx']
        self.cy = ds_dict['cy']
        self.near = hparams['near'] # follow AD-NeRF, we dont use near-far in ds_dict
        self.far = hparams['far'] # follow AD-NeRF, we dont use near-far in ds_dict
        if hparams['infer_bg_img_fname'] == '':
            # use the default bg_img from dataset
            bg_img = torch.from_numpy(ds_dict['bg_img']).float() / 255.
            self.bg_img_512 = convert_to_tensor(bg_img).cuda()
            bg_img = F.interpolate(bg_img.unsqueeze(0).permute(0,3,1,2), mode='bilinear', size=(self.H,self.W), antialias=True).permute(0,2,3,1).reshape([self.H,self.W,3])
        elif hparams['infer_bg_img_fname'] == 'white': # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif hparams['infer_bg_img_fname'] == 'black': # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else: # load from a specificfile
            bg_img = cv2.imread(hparams['infer_bg_img_fname'], cv2.IMREAD_UNCHANGED) # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255 # [H, W, 3/4]
        self.bg_img = convert_to_tensor(bg_img).cuda()

        self.idexp_lm3d_mean = torch.from_numpy(ds_dict['idexp_lm3d_mean']).float()
        self.idexp_lm3d_std = torch.from_numpy(ds_dict['idexp_lm3d_std']).float()

        cx = self.H / 2
        cy = self.W / 2
        fl_x = self.focal * (cx/self.cx)
        fl_y = self.focal * (cy/self.cy)
        
        # 对ngp pose做smoot会导致90度正负奇异
        if not training and hparams['infer_smooth_camera_path']:
            c2w_arr = torch.stack([s['c2w'] for s in self.samples])
            smo_c2w = smooth_camera_path(c2w_arr.numpy(), kernel_size=hparams['infer_smooth_camera_path_kernel_size'])
            smo_c2w = torch.tensor(smo_c2w)
            for i in range(len(c2w_arr)):
                self.samples[i]['c2w'] = smo_c2w[i]
            print(f"{prefix}: Smooth head trajectory (rotation and translation) with a window size of {hparams['infer_smooth_camera_path_kernel_size']}")

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        self.poses = torch.from_numpy(np.stack([nerf_matrix_to_ngp(s['c2w'], scale=hparams['camera_scale'], offset=hparams['camera_offset']) for s in self.samples]))
        if hparams.get("use_mp_pose", False):
            self.poses = torch.from_numpy(np.stack([nerf_matrix_to_ngp(s['mp_c2w'], scale=hparams['camera_scale'], offset=hparams['camera_offset']) for s in self.samples]))

        if torch.any(torch.isnan(self.poses)):
            raise ValueError("Found NaN in transform_matrix, please check the face_tracker process!")
        
        self.bg_coords = get_bg_coords(self.H, self.W, 'cpu') # [1, H*W, 2] in [-1, 1]

        if self.cond_type == 'deepspeech':
            raise NotImplementedError("We no longer support DeepSpeech")
        elif self.cond_type == 'esperanto':
            self.conds = torch.tensor(self.ds_dict['esperanto']) # [B=1, T=16, C=44]
        elif self.cond_type == 'idexp_lm3d_normalized':
            global face3d_helper
            if face3d_helper is None:
                face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=False)
            from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478, index_lm131_from_lm478
            id, exp = convert_to_tensor(ds_dict['id']), convert_to_tensor(ds_dict['exp'])
            idexp_lm3d_arr = face3d_helper.reconstruct_idexp_lm3d(id, exp)
            idexp_lm3d_mean = idexp_lm3d_arr.mean(dim=0, keepdim=True)
            idexp_lm3d_std = idexp_lm3d_arr.std(dim=0, keepdim=True)
            idexp_lm3d_normalized = (idexp_lm3d_arr - idexp_lm3d_mean) / idexp_lm3d_std
            euler, trans = convert_to_tensor(ds_dict['euler']), convert_to_tensor(ds_dict['trans'])
            self.lm2ds = face3d_helper.reconstruct_lm2d_nerf(id, exp, euler, trans)
            self.eye_area_percents = convert_to_tensor(ds_dict['eye_area_percent'])
            if prefix == 'train':
                self.lm2ds = self.lm2ds[:len(self.samples)]
                self.eye_area_percents = self.eye_area_percents[:len(self.samples)]
            elif prefix == 'val':
                self.lm2ds = self.lm2ds[-len(self.samples):]
                self.eye_area_percents = self.eye_area_percents[-len(self.samples):]

            # if hparams.get("with_sr"):
            #     self.lm2ds = self.lm2ds / 2
            self.lm68s = torch.tensor(self.lm2ds[:, index_lm68_from_lm478, :])

            eg3d_camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler':euler, 'trans':trans})
            self.eg3d_cameras = convert_to_tensor(np.concatenate([eg3d_camera_ret['c2w'].reshape([-1,16]), eg3d_camera_ret['intrinsics'].reshape([-1,9])],axis=-1))

            self.keypoint_mode = keypoint_mode = hparams.get("nerf_keypoint_mode", "lm68")
            if keypoint_mode == 'lm68':
                idexp_lm3d_normalized = idexp_lm3d_normalized[:, index_lm68_from_lm478]
                self.keypoint_num = 68
            elif keypoint_mode == 'lm131':
                idexp_lm3d_normalized = idexp_lm3d_normalized[:, index_lm131_from_lm478]
                self.keypoint_num = 131
            elif keypoint_mode == 'lm468':
                idexp_lm3d_normalized = idexp_lm3d_normalized
                self.keypoint_num = 468
            else: raise NotImplementedError()
            idexp_lm3d_normalized_win = idexp_lm3d_normalized.reshape([-1, 1, self.keypoint_num * 3])
            self.conds = idexp_lm3d_normalized_win
            if self.prefix == 'train':
                self.conds = self.conds[:len(self.samples)]
            else:
                self.conds = self.conds[-len(self.samples):]
        else:
            raise NotImplementedError
        
        self.finetune_lip_flag = False
        self.lips_rect = [s['lip_rect'] for s in self.samples]
        if hparams.get("with_sr"):
            self.lips_rect = (np.array([s['lip_rect'] for s in self.samples]) / 2).astype(int).tolist()
        self.training = training
        self.global_step = 0

    @property
    def num_rays(self):
        return hparams['n_rays'] if self.training else -1

    def __getitem__(self, idx):

        raw_sample = self.samples[idx]
        
        if self.hparams.get("load_imgs_to_memory", True):
            # disable it to save memory usage.
            # for 5500 images, it takes 1 minutes to imread, by contrast, only 1s is needed to index them in memory. 
            # But it reuqires 15GB memory for caching 5500 images at 512x512 resolution.
            if 'torso_img' not in self.samples[idx].keys():
                self.samples[idx]['torso_img'] = load_image_as_uint8_tensor(self.samples[idx]['torso_img_fname'])
                self.samples[idx]['gt_img'] = load_image_as_uint8_tensor(self.samples[idx]['gt_img_fname'])
            torso_img = self.samples[idx]['torso_img']
            gt_img = self.samples[idx]['gt_img']
        else:
            torso_img = load_image_as_uint8_tensor(self.samples[idx]['torso_img_fname'])
            gt_img = load_image_as_uint8_tensor(self.samples[idx]['gt_img_fname'])

        sample = {
            'H': self.H,
            'W': self.W,
            'focal': self.focal,
            'cx': self.cx,
            'cy': self.cy,
            'near': self.near,
            'far': self.far,
            'idx': raw_sample['idx'],
            'face_rect': raw_sample['face_rect'],
            'lip_rect': self.lips_rect[idx],
            'bg_img': self.bg_img,
        }
        sample['c2w'] = raw_sample['c2w']
        # eg3d_dummy_intrinsic = torch.tensor([[2985.29/700, 0, 0.5], [0, 2985.29/700, 0.5], [0, 0, 1]])
        # sample['camera'] = torch.cat([c2w.reshape([16,]), eg3d_dummy_intrinsic.reshape([9,])]).reshape([1, 25])
        sample['camera'] = self.eg3d_cameras[idx].unsqueeze(0)
        
        sample['gt_img_512'] = gt_img.cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255. # b,c,h,w

        sample['cond_wins'] = get_audio_features(self.conds, att_mode=2, index=idx)
        sample['cond_wins_prev'] = get_audio_features(self.conds, att_mode=2, index=max(idx-1, 0))
        sample['cond_wins_next'] = get_audio_features(self.conds, att_mode=2, index=min(idx+1, len(self)-1))

        ngp_pose = self.poses[idx].unsqueeze(0)

        # if self.training is False:
        #     ngp_pose[0,1,3] = self.poses[0][1,3] # inference, fix z axis

        sample['pose'] = convert_poses(ngp_pose) # [B, 6]
        sample['pose_matrix'] = ngp_pose # [B, 4, 4]

        sample.update({
            'torso_img': torso_img.cuda().float() / 255.,
            'gt_img': gt_img.cuda().float() / 255.,
        })

        if hparams.get("with_sr"):
            # SR模式下,不管是train还是infer,都采样256x256的全图pixel
            rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=-1, rect=None)
        else:
            # 否则,沿用adnerf的策略
            if self.training:
                if self.finetune_lip_flag:
                    # the finetune_lip_flag is controlled by the task that use this dataset 
                    rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=-1, rect=sample['lip_rect'])
                else:
                    # training phase
                    if self.num_rays >= self.H * self.W:
                        rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=self.num_rays, rect=None)
                    else:
                        rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=self.num_rays, rect=None)
            else:
                # inference phase
                rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=-1)

        sample['rays_o'] = rays['rays_o']
        sample['rays_d'] = rays['rays_d']

        sample['eye_area_percent'] = self.eye_area_percents[idx]

        if hparams.get("polygon_face_mask", True):
            f_mask = dilate_boundary_mask(get_boundary_mask(self.lm2ds[idx], index_mode='lm68', h=self.H,w=self.W).unsqueeze(0).cuda(), ksize=3)
            f_mask = f_mask.reshape([-1]).bool() # 512*512
            face_mask = f_mask[rays['inds']]
            sample['face_mask'] = face_mask
        else:
            # RAD-NeRF
            xmin, xmax, ymin, ymax = raw_sample['face_rect']
            if hparams.get("with_sr"):
                xmin = xmin/2
                xmax = xmax/2
                ymin = ymin/2
                ymax = ymax/2
            assert xmin <= xmax
            assert ymin <= ymax
            face_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            sample['face_mask'] = face_mask

        sample['cond_mask'] = face_mask.reshape([-1,])

        bg_torso_img = bg_torso_img_512 = sample['torso_img']
        gt_img = sample['gt_img']
        if hparams.get("with_sr"):
            bg_torso_img = F.interpolate(bg_torso_img.cuda().view(1, 512,512, -1).permute(0,3,1,2), size=(self.H,self.W),mode='bilinear', antialias=True).permute(0,2,3,1) # treat torso as a part of background
            gt_img = F.interpolate(gt_img.view(1, 512,512, 3).permute(0,3,1,2), size=(self.H,self.W),mode='bilinear', antialias=True).permute(0,2,3,1).view(1, -1, 3) # treat torso as a part of background

        bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
        bg_torso_img = bg_torso_img.view(1, -1, 3) # treat torso as a part of background
        bg_img = self.bg_img.view(1, -1, 3)
        
        bg_torso_img_512 = bg_torso_img_512[..., :3] * bg_torso_img_512[..., 3:] + self.bg_img_512 * (1 - bg_torso_img_512[..., 3:])
        bg_torso_img_512 = bg_torso_img_512.view(1, -1, 3) # treat torso as a part of background

        C = sample['gt_img'].shape[-1]

        # if self.training:
        bg_img = torch.gather(bg_img.cuda(), 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
        bg_torso_img = torch.gather(bg_torso_img.cuda(), 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
        gt_img = torch.gather(gt_img.reshape(1, -1, C).cuda(), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        sample['gt_img'] = gt_img
        # else:
            # gt_img = torch.gather(sample['gt_img'].reshape(1, -1, C).cuda(), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            # sample['gt_img'] = sample['gt_img'].reshape([1,-1,C])
        sample['bg_img'] = bg_img
        sample['bg_torso_img'] = bg_torso_img
        sample['bg_torso_img_512'] = bg_torso_img_512

        sample['lm68'] = torch.tensor(self.lm68s[idx].reshape([68*2]))
        if self.training:
            bg_coords = torch.gather(self.bg_coords.cuda(), 1, torch.stack(2 * [rays['inds']], -1)) # [1, N, 2]
        else:
            bg_coords = self.bg_coords # [1, N, 2]
        sample['bg_coords'] = bg_coords

        return sample
        
    def __len__(self):
        return len(self.samples)

    def collater(self, samples):
        assert len(samples) == 1 # NeRF only take 1 image for each iteration
        return samples[0]
 
if __name__ == '__main__':
    set_hparams()
    ds = RADNeRFDataset('trainval', data_dir='data/binary/videos/May')
    for i in tqdm.trange(len(ds)):
        ds[i]
    print("done!")