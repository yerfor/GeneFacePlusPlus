import os
import numpy as np
import math
import json
import imageio
import torch
import tqdm
import cv2

from data_util.face3d_helper import Face3DHelper
from utils.commons.euler2rot import euler_trans_2_c2w, c2w_to_euler_trans
from data_gen.utils.process_video.euler2quaterion import euler2quaterion, quaterion2euler
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_gen.utils.process_video.extract_blink import get_eye_area_percent


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def rot2euler(rot_mat):
    batch_size = len(rot_mat)
    # we assert that y in in [-0.5pi, 0.5pi]
    cos_y = torch.sqrt(rot_mat[:, 1, 2] * rot_mat[:, 1, 2] + rot_mat[:, 2, 2] * rot_mat[:, 2, 2])
    theta_x = torch.atan2(-rot_mat[:, 1, 2], rot_mat[:, 2, 2])
    theta_y = torch.atan2(rot_mat[:, 2, 0], cos_y)
    theta_z = torch.atan2(rot_mat[:, 0, 1], rot_mat[:, 0, 0])
    euler_angles = torch.zeros([batch_size, 3])
    euler_angles[:, 0] = theta_x
    euler_angles[:, 1] = theta_y
    euler_angles[:, 2] = theta_z
    return euler_angles

index_lm68_from_lm468 = [127,234,93,132,58,136,150,176,152,400,379,365,288,361,323,454,356,70,63,105,66,107,336,296,334,293,300,168,197,5,4,75,97,2,326,305,
                         33,160,158,133,153,144,362,385,387,263,373,380,61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178]

def plot_lm2d(lm2d):
    WH = 512
    img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
    
    for i in range(len(lm2d)):
        x, y = lm2d[i]
        color = (255,0,0)
        img = cv2.circle(img, center=(int(x),int(y)), radius=3, color=color, thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(lm2d)):
        x, y = lm2d[i]
        img = cv2.putText(img, f"{i}", org=(int(x),int(y)), fontFace=font, fontScale=0.3, color=(255,0,0))
    return img

def get_face_rect(lms, h, w):
    """
    lms: [68, 2]
    h, w: int
    return: [4,]
    """
    assert len(lms) == 68
    # min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
    min_x, max_x = np.min(lms[:, 0]), np.max(lms[:, 0])
    cx = int((min_x+max_x)/2.0)
    cy = int(lms[27, 1])
    h_w = int((max_x-cx)*1.5)
    h_h = int((lms[8, 1]-cy)*1.15)
    rect_x = cx - h_w
    rect_y = cy - h_h
    if rect_x < 0:
        rect_x = 0
    if rect_y < 0:
        rect_y = 0
    rect_w = min(w-1-rect_x, 2*h_w)
    rect_h = min(h-1-rect_y, 2*h_h)
    # rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
    # rect = [rect_x, rect_y, rect_w, rect_h]
    rect = [rect_x, rect_x + rect_w, rect_y, rect_y + rect_h] # min_j,  max_j, min_i, max_i
    return rect # this x is width, y is height

def get_lip_rect(lms, h, w):
    """
    lms: [68, 2]
    h, w: int
    return: [4,]
    """
    # this x is width, y is height
    # for lms, lms[:, 0] is width, lms[:, 1] is height
    assert len(lms) == 68
    lips = slice(48, 60)
    lms = lms[lips]
    min_x, max_x = np.min(lms[:, 0]), np.max(lms[:, 0])
    min_y, max_y = np.min(lms[:, 1]), np.max(lms[:, 1])
    cx = int((min_x+max_x)/2.0)
    cy = int((min_y+max_y)/2.0)
    h_w = int((max_x-cx)*1.2)
    h_h = int((max_y-cy)*1.2)
    
    h_w = max(h_w, h_h)
    h_h = h_w

    rect_x = cx - h_w
    rect_y = cy - h_h
    rect_w = 2*h_w
    rect_h = 2*h_h
    if rect_x < 0:
        rect_x = 0
    if rect_y < 0:
        rect_y = 0
    
    if rect_x + rect_w > w:
        rect_x = w - rect_w
    if rect_y + rect_h > h:
        rect_y = h - rect_h

    rect = [rect_x, rect_x + rect_w, rect_y, rect_y + rect_h] # min_j,  max_j, min_i, max_i
    return rect # this x is width, y is height


# def get_lip_rect(lms, h, w):
#     """
#     lms: [68, 2]
#     h, w: int
#     return: [4,]
#     """
#     assert len(lms) == 68
#     lips = slice(48, 60)
#     # this x is width, y is height
#     xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
#     ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
#     # padding to H == W
#     cx = (xmin + xmax) // 2
#     cy = (ymin + ymax) // 2
#     l = max(xmax - xmin, ymax - ymin) // 2
#     xmin = max(0, cx - l)
#     xmax = min(h, cx + l)
#     ymin = max(0, cy - l)
#     ymax = min(w, cy + l)
#     lip_rect = [xmin, xmax, ymin, ymax]
#     return lip_rect

def get_win_conds(conds, idx, smo_win_size=8, pad_option='zero'):
    """
    conds: [b, t=16, h=29]
    idx: long, time index of the selected frame
    """
    idx = max(0, idx)
    idx = min(idx, conds.shape[0]-1)
    smo_half_win_size = smo_win_size//2
    left_i = idx - smo_half_win_size
    right_i = idx + (smo_win_size - smo_half_win_size)
    pad_left, pad_right = 0, 0
    if left_i < 0:
        pad_left = -left_i
        left_i = 0
    if right_i > conds.shape[0]:
        pad_right = right_i - conds.shape[0]
        right_i = conds.shape[0]
    conds_win = conds[left_i:right_i]
    if pad_left > 0:
        if pad_option == 'zero':
            conds_win = np.concatenate([np.zeros_like(conds_win)[:pad_left], conds_win], axis=0)
        elif pad_option == 'edge':
            edge_value = conds[0][np.newaxis, ...]
            conds_win = np.concatenate([edge_value] * pad_left + [conds_win], axis=0)
        else: 
            raise NotImplementedError
    if pad_right > 0:
        if pad_option == 'zero':
            conds_win = np.concatenate([conds_win, np.zeros_like(conds_win)[:pad_right]], axis=0)
        elif pad_option == 'edge':
            edge_value = conds[-1][np.newaxis, ...]
            conds_win = np.concatenate([conds_win] + [edge_value] * pad_right , axis=0)
        else: 
            raise NotImplementedError
    assert conds_win.shape[0] == smo_win_size
    return conds_win


def load_processed_data(processed_dir):
    # load necessary files
    background_img_name = os.path.join(processed_dir, "bg.jpg")
    assert os.path.exists(background_img_name)
    head_img_dir = os.path.join(processed_dir, "head_imgs")
    torso_img_dir = os.path.join(processed_dir, "inpaint_torso_imgs")
    gt_img_dir = os.path.join(processed_dir, "com_imgs")
    # gt_img_dir = os.path.join(processed_dir, "gt_imgs")

    hubert_npy_name = os.path.join(processed_dir, "aud_hubert.npy")
    mel_f0_npy_name = os.path.join(processed_dir, "aud_mel_f0.npy")
    coeff_npy_name = os.path.join(processed_dir, "coeff_fit_mp.npy")
    lm2d_npy_name = os.path.join(processed_dir, "lms_2d.npy")
    
    ret_dict = {}

    ret_dict['bg_img'] = imageio.imread(background_img_name)
    ret_dict['H'], ret_dict['W'] = ret_dict['bg_img'].shape[:2]
    ret_dict['focal'], ret_dict['cx'], ret_dict['cy'] = face_model.focal, face_model.center, face_model.center

    print("loading lm2d coeff ...")
    lm2d_arr = np.load(lm2d_npy_name)
    face_rect_lst = []
    lip_rect_lst = []
    for lm2d in lm2d_arr:
        if len(lm2d) in [468, 478]:
            lm2d = lm2d[index_lm68_from_lm468]
        face_rect = get_face_rect(lm2d, ret_dict['H'], ret_dict['W'])
        lip_rect = get_lip_rect(lm2d, ret_dict['H'], ret_dict['W'])
        face_rect_lst.append(face_rect)
        lip_rect_lst.append(lip_rect)
    face_rects = np.stack(face_rect_lst, axis=0) # [T, 4]

    print("loading fitted 3dmm coeff ...")
    coeff_dict = np.load(coeff_npy_name, allow_pickle=True).tolist()
    identity_arr = coeff_dict['id']
    exp_arr = coeff_dict['exp']
    eye_area_percent = get_eye_area_percent(torch.tensor(coeff_dict['id']), torch.tensor(coeff_dict['exp']), face3d_helper)
    ret_dict['eye_area_percent'] = eye_area_percent
    ret_dict['id'] = identity_arr
    ret_dict['exp'] = exp_arr
    euler_arr = ret_dict['euler'] = coeff_dict['euler']
    trans_arr = ret_dict['trans'] = coeff_dict['trans']
    print("calculating lm3d ...")
    idexp_lm3d_arr = face3d_helper.reconstruct_idexp_lm3d(torch.from_numpy(identity_arr), torch.from_numpy(exp_arr)).cpu().numpy().reshape([-1, 68*3])
    len_motion = len(idexp_lm3d_arr)
    video_idexp_lm3d_mean = idexp_lm3d_arr.mean(axis=0)
    video_idexp_lm3d_std = idexp_lm3d_arr.std(axis=0)
    ret_dict['idexp_lm3d'] = idexp_lm3d_arr
    ret_dict['idexp_lm3d_mean'] = video_idexp_lm3d_mean
    ret_dict['idexp_lm3d_std'] = video_idexp_lm3d_std
    
    # now we convert the euler_trans from deep3d convention to adnerf convention
    eulers = torch.FloatTensor(euler_arr)
    trans = torch.FloatTensor(trans_arr)
    rots = face_model.compute_rotation(eulers) # rotation matrix is a better intermediate for convention-transplan than euler

    # handle the camera pose to geneface's convention
    trans[:, 2] = 10 - trans[:, 2] # 抵消fit阶段的to_camera操作，即trans[...,2] = 10 - trans[...,2]
    rots = rots.permute(0, 2, 1)
    trans[:, 2] = - trans[:,2] # 因为intrinsic proj不同
    # below is the NeRF camera preprocessing strategy, see `save_transforms` in data_util/process.py 
    trans = trans / 10.0
    rots_inv = rots.permute(0, 2, 1)
    trans_inv = - torch.bmm(rots_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat([len_motion, 1, 1]) # [T, 4, 4]
    pose[:, :3, :3] = rots_inv
    pose[:, :3, 3] = trans_inv[:, :, 0]
    c2w_transform_matrices = pose.numpy()

    # process the audio features used for postnet training
    print("loading hubert ...")
    hubert_features = np.load(hubert_npy_name)
    print("loading Mel and F0 ...")
    mel_f0_features = np.load(mel_f0_npy_name, allow_pickle=True).tolist()

    ret_dict['hubert'] = hubert_features
    ret_dict['mel'] = mel_f0_features['mel']
    ret_dict['f0'] = mel_f0_features['f0']

    # obtaining train samples
    frame_indices = list(range(len_motion))
    num_train = len_motion // 11 * 10
    train_indices = frame_indices[:num_train]
    val_indices = frame_indices[num_train:]

    for split in ['train', 'val']:
        if split == 'train':
            indices = train_indices
            samples = []
            ret_dict['train_samples'] = samples
        elif split == 'val':
            indices = val_indices
            samples = []
            ret_dict['val_samples'] = samples
        
        for idx in indices:
            sample = {}
            sample['idx'] = idx
            sample['head_img_fname'] = os.path.join(head_img_dir,f"{idx:08d}.png")
            sample['torso_img_fname'] = os.path.join(torso_img_dir,f"{idx:08d}.png")
            sample['gt_img_fname'] = os.path.join(gt_img_dir,f"{idx:08d}.jpg")
            # assert os.path.exists(sample['head_img_fname']) and os.path.exists(sample['torso_img_fname']) and os.path.exists(sample['gt_img_fname'])
            sample['face_rect'] = face_rects[idx]
            sample['lip_rect'] = lip_rect_lst[idx]
            sample['c2w'] = c2w_transform_matrices[idx]
            samples.append(sample)
    return ret_dict


class Binarizer:
    def __init__(self):
        self.data_dir = 'data/'
        
    def parse(self, video_id):
        processed_dir = os.path.join(self.data_dir, 'processed/videos', video_id)
        binary_dir = os.path.join(self.data_dir, 'binary/videos', video_id)
        out_fname = os.path.join(binary_dir, "trainval_dataset.npy")
        os.makedirs(binary_dir, exist_ok=True)
        ret = load_processed_data(processed_dir)
        mel_name = os.path.join(processed_dir, 'aud_mel_f0.npy')
        mel_f0_dict = np.load(mel_name, allow_pickle=True).tolist()
        ret.update(mel_f0_dict)
        np.save(out_fname, ret, allow_pickle=True)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--video_id', type=str, default='May', help='')
    args = parser.parse_args()
    ### Process Single Long Audio for NeRF dataset
    video_id = args.video_id
    face_model = ParametricFaceModel(bfm_folder='deep_3drecon/BFM', 
                camera_distance=10, focal=1015)
    face_model.to("cpu")
    face3d_helper = Face3DHelper()

    binarizer = Binarizer()
    binarizer.parse(video_id)
    print(f"Binarization for {video_id} Done!")
