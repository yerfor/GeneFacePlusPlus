# This is a script for efficienct 3DMM coefficient extraction.
# It could reconstruct accurate 3D face in real-time.
# It is built upon BFM 2009 model and mediapipe landmark extractor.
# It is authored by ZhenhuiYe (zhenhuiye@zju.edu.cn), free to contact him for any suggestion on improvement!

from numpy.core.numeric import require
from numpy.lib.function_base import quantile
import torch
import torch.nn.functional as F
import copy
import numpy as np

import random
import pickle
import os
import sys
import cv2
import argparse
import tqdm
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from data_gen.utils.mp_feature_extractors.face_landmarker import MediapipeLandmarker, read_video_to_frames
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from deep_3drecon.secc_renderer import SECC_Renderer
from utils.commons.os_utils import multiprocess_glob


face_model = ParametricFaceModel(bfm_folder='deep_3drecon/BFM', 
            camera_distance=10, focal=1015, keypoint_mode='mediapipe')
face_model.to(torch.device("cuda:0"))

dir_path = os.path.dirname(os.path.realpath(__file__))


def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    # yaw = -yaw
    pitch = - pitch
    roll = - roll
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f) 
        
def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content

def cal_lap_loss(in_tensor):
    # [T, 68, 2]
    t = in_tensor.shape[0]
    in_tensor = in_tensor.reshape([t, -1]).permute(1,0).unsqueeze(1) # [c, 1, t]
    in_tensor = torch.cat([in_tensor[:, :, 0:1], in_tensor, in_tensor[:, :, -1:]], dim=-1)
    lap_kernel = torch.Tensor((-0.5, 1.0, -0.5)).reshape([1,1,3]).float().to(in_tensor.device) # [1, 1, kw]
    loss_lap = 0

    out_tensor = F.conv1d(in_tensor, lap_kernel)
    loss_lap += torch.mean(out_tensor**2)
    return loss_lap

def cal_vel_loss(ldm):
    # [B, 68, 2]
    vel = ldm[1:] - ldm[:-1]
    return torch.mean(torch.abs(vel))

def cal_lan_loss(proj_lan, gt_lan):
    # [B, 68, 2]
    loss = (proj_lan - gt_lan)** 2
    # use the ldm weights from deep3drecon, see deep_3drecon/deep_3drecon_models/losses.py
    weights = torch.zeros_like(loss)
    weights = torch.ones_like(loss)
    weights[:, 36:48, :] = 3 # eye 12 points
    weights[:, -8:, :] =  3 # inner lip 8 points
    weights[:, 28:31, :] =  3 # nose 3 points
    loss = loss * weights
    return torch.mean(loss)

def cal_lan_loss_mp(proj_lan, gt_lan, mean:bool=True):
    # [B, 68, 2]
    loss = (proj_lan - gt_lan).pow(2)
    # loss = (proj_lan - gt_lan).abs()
    unmatch_mask = [ 93, 127, 132, 234, 323, 356, 361, 454]
    upper_eye = [161,160,159,158,157] + [388,387,386,385,384]
    eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] + [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
    inner_lip = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
    outer_lip = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
    weights = torch.ones_like(loss)
    weights[:, eye] = 3
    weights[:, upper_eye] = 20
    weights[:, inner_lip] = 5
    weights[:, outer_lip] = 5
    weights[:, unmatch_mask] = 0
    loss = loss * weights
    if mean:
        loss = torch.mean(loss)
    return loss

def cal_acceleration_loss(trans):
    vel = trans[1:] - trans[:-1]
    acc = vel[1:] - vel[:-1]
    return torch.mean(torch.abs(acc))

def cal_acceleration_ldm_loss(ldm):
    # [B, 68, 2]
    vel = ldm[1:] - ldm[:-1]
    acc = vel[1:] - vel[:-1]
    lip_weight = 0.25 # we dont want smooth the lip too much
    acc[48:68] *= lip_weight
    return torch.mean(torch.abs(acc))
 
def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True

@torch.enable_grad()
def fit_3dmm_for_a_video(
    video_name, 
    nerf=False, # use the file name convention for GeneFace++
    id_mode='global', 
    debug=False, 
    keypoint_mode='mediapipe',
    large_yaw_threshold=9999999.9,
    save=True
) -> bool: # True: good, False: bad 
    assert video_name.endswith(".mp4"), "this function only support video as input"
    if id_mode == 'global':
        LAMBDA_REG_ID = 0.2
        LAMBDA_REG_EXP = 0.6
        LAMBDA_REG_LAP = 1.0
        LAMBDA_REG_VEL_ID = 0.0 # laplcaian is all you need for temporal consistency
        LAMBDA_REG_VEL_EXP = 0.0 # laplcaian is all you need for temporal consistency
    else:
        LAMBDA_REG_ID = 0.3
        LAMBDA_REG_EXP = 0.05
        LAMBDA_REG_LAP = 1.0
        LAMBDA_REG_VEL_ID = 0.0 # laplcaian is all you need for temporal consistency
        LAMBDA_REG_VEL_EXP = 0.0 # laplcaian is all you need for temporal consistency

    frames = read_video_to_frames(video_name) # [T, H, W, 3]
    img_h, img_w = frames.shape[1], frames.shape[2]
    assert img_h == img_w
    num_frames = len(frames)

    if nerf: # single video
        lm_name = video_name.replace("/raw/", "/processed/").replace(".mp4","/lms_2d.npy")
    else:
        lm_name = video_name.replace("/video/", "/lms_2d/").replace(".mp4", "_lms.npy")

    if os.path.exists(lm_name):
        lms = np.load(lm_name)
    else:
        print(f"lms_2d file not found, try to extract it from video... {lm_name}")
        try:
            landmarker = MediapipeLandmarker()
            img_lm478, vid_lm478 = landmarker.extract_lm478_from_frames(frames, anti_smooth_factor=20)
            lms = landmarker.combine_vid_img_lm478_to_lm478(img_lm478, vid_lm478)
        except Exception as e:
            print(e)
            return False
        if lms is None:
            print(f"get None lms_2d, please check whether each frame has one head, exiting... {lm_name}")
            return False
    lms = lms[:, :468, :]
    lms = torch.FloatTensor(lms).cuda()
    lms[..., 1] = img_h - lms[..., 1] # flip the height axis

    if keypoint_mode == 'mediapipe':
        # default
        cal_lan_loss_fn = cal_lan_loss_mp
        if nerf: # single video
            out_name = video_name.replace("/raw/", "/processed/").replace(".mp4", "/coeff_fit_mp.npy")
        else:
            out_name = video_name.replace("/video/", "/coeff_fit_mp/").replace(".mp4", "_coeff_fit_mp.npy")
    else:
        # lm68 is less accurate than mp
        cal_lan_loss_fn = cal_lan_loss
        if nerf: # single video
            out_name = video_name.replace("/raw/", "/processed/").replace(".mp4", "_coeff_fit_lm68.npy")
        else:
            out_name = video_name.replace("/video/", "/coeff_fit_lm68/").replace(".mp4", "_coeff_fit_lm68.npy")
    try:
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
    except:
        pass

    id_dim, exp_dim = 80, 64
    sel_ids = np.arange(0, num_frames, 40)

    h = w = face_model.center * 2
    img_scale_factor = img_h / h
    lms /= img_scale_factor # rescale lms into [0,224]

    if id_mode == 'global':
        # default choice by GeneFace++ and later works
        id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    elif id_mode == 'finegrained':
        # legacy choice by GeneFace1 (ICLR 2023)
        id_para = lms.new_zeros((num_frames, id_dim), requires_grad=True)
    else: raise NotImplementedError(f"id mode {id_mode} not supported! we only support global or finegrained.")
    exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans = lms.new_zeros((num_frames, 3), requires_grad=True)

    set_requires_grad([id_para, exp_para, euler_angle, trans])

    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=.1)

    # 其他参数初始化，先训练euler和trans
    for _ in range(200):
        if id_mode == 'global':
            proj_geo = face_model.compute_for_landmark_fit(
                id_para.expand((num_frames, id_dim)), exp_para, euler_angle, trans)
        else:
            proj_geo = face_model.compute_for_landmark_fit(
                id_para, exp_para, euler_angle, trans)
        loss_lan = cal_lan_loss_fn(proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_frame.step()

    # print(f"loss_lan: {loss_lan.item():.2f}, euler_abs_mean: {euler_angle.abs().mean().item():.4f}, euler_std: {euler_angle.std().item():.4f}, euler_min: {euler_angle.min().item():.4f}, euler_max: {euler_angle.max().item():.4f}")
    # print(f"trans_z_mean: {trans[...,2].mean().item():.4f}, trans_z_std: {trans[...,2].std().item():.4f}, trans_min: {trans[...,2].min().item():.4f}, trans_max: {trans[...,2].max().item():.4f}")

    for param_group in optimizer_frame.param_groups:
        param_group['lr'] = 0.1

    # "jointly roughly training id exp euler trans"
    for _ in range(200):
        ret = {}
        if id_mode == 'global':
            proj_geo = face_model.compute_for_landmark_fit(
                id_para.expand((num_frames, id_dim)), exp_para, euler_angle, trans, ret)
        else:
            proj_geo = face_model.compute_for_landmark_fit(
                id_para, exp_para, euler_angle, trans, ret)
        loss_lan = cal_lan_loss_fn(
            proj_geo[:, :, :2], lms.detach())
        # loss_lap = cal_lap_loss(proj_geo)
        # laplacian对euler影响不大，但是对trans的提升很大
        loss_lap = cal_lap_loss(id_para) + cal_lap_loss(exp_para) + cal_lap_loss(euler_angle) * 0.3 + cal_lap_loss(trans) * 0.3

        loss_regid = torch.mean(id_para*id_para) # 正则化
        loss_regexp = torch.mean(exp_para * exp_para)

        loss_vel_id = cal_vel_loss(id_para)
        loss_vel_exp = cal_vel_loss(exp_para)
        loss = loss_lan  + loss_regid * LAMBDA_REG_ID + loss_regexp * LAMBDA_REG_EXP  + loss_vel_id * LAMBDA_REG_VEL_ID + loss_vel_exp * LAMBDA_REG_VEL_EXP + loss_lap * LAMBDA_REG_LAP
        optimizer_idexp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_idexp.step()
        optimizer_frame.step()

    # print(f"loss_lan: {loss_lan.item():.2f}, loss_reg_id: {loss_regid.item():.2f},loss_reg_exp: {loss_regexp.item():.2f},")
    # print(f"euler_abs_mean: {euler_angle.abs().mean().item():.4f}, euler_std: {euler_angle.std().item():.4f}, euler_min: {euler_angle.min().item():.4f}, euler_max: {euler_angle.max().item():.4f}")
    # print(f"trans_z_mean: {trans[...,2].mean().item():.4f}, trans_z_std: {trans[...,2].std().item():.4f}, trans_min: {trans[...,2].min().item():.4f}, trans_max: {trans[...,2].max().item():.4f}")

    # start fine training, intialize from the roughly trained results
    if id_mode == 'global':
        id_para_ = lms.new_zeros((1, id_dim), requires_grad=False)
    else:
        id_para_ = lms.new_zeros((num_frames, id_dim), requires_grad=True)
    id_para_.data = id_para.data.clone()
    id_para = id_para_
    exp_para_ = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    exp_para_.data = exp_para.data.clone()
    exp_para = exp_para_
    euler_angle_ = lms.new_zeros((num_frames, 3), requires_grad=True)
    euler_angle_.data = euler_angle.data.clone()
    euler_angle = euler_angle_
    trans_ = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans_.data = trans.data.clone()
    trans = trans_
    
    batch_size = 50
    # "fine fitting the 3DMM in batches"
    for i in range(int((num_frames-1)/batch_size+1)):
        if (i+1)*batch_size > num_frames:
            start_n = num_frames-batch_size
            sel_ids = np.arange(max(num_frames-batch_size,0), num_frames)
        else:
            start_n = i*batch_size
            sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)
        sel_lms = lms[sel_ids]

        if id_mode == 'global':
            sel_id_para = id_para.expand((sel_ids.shape[0], id_dim))
        else:
            sel_id_para = id_para.new_zeros((batch_size, id_dim), requires_grad=True)
            sel_id_para.data = id_para[sel_ids].clone()
        sel_exp_para = exp_para.new_zeros(
            (batch_size, exp_dim), requires_grad=True)
        sel_exp_para.data = exp_para[sel_ids].clone()
        sel_euler_angle = euler_angle.new_zeros(
            (batch_size, 3), requires_grad=True)
        sel_euler_angle.data = euler_angle[sel_ids].clone()
        sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
        sel_trans.data = trans[sel_ids].clone()
        
        if id_mode == 'global':
            set_requires_grad([sel_exp_para, sel_euler_angle, sel_trans])
            optimizer_cur_batch = torch.optim.Adam(
                [sel_exp_para, sel_euler_angle, sel_trans], lr=0.005)
        else:
            set_requires_grad([sel_id_para, sel_exp_para, sel_euler_angle, sel_trans])
            optimizer_cur_batch = torch.optim.Adam(
                [sel_id_para, sel_exp_para, sel_euler_angle, sel_trans], lr=0.005)

        for j in range(50):
            ret = {}
            proj_geo = face_model.compute_for_landmark_fit(
                sel_id_para, sel_exp_para, sel_euler_angle, sel_trans, ret)
            loss_lan = cal_lan_loss_fn(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            
            # loss_lap = cal_lap_loss(proj_geo)
            loss_lap = cal_lap_loss(sel_id_para) + cal_lap_loss(sel_exp_para) + cal_lap_loss(sel_euler_angle) * 0.3 + cal_lap_loss(sel_trans) * 0.3
            loss_vel_id = cal_vel_loss(sel_id_para)
            loss_vel_exp = cal_vel_loss(sel_exp_para)
            log_dict = {
                'loss_vel_id': loss_vel_id,
                'loss_vel_exp': loss_vel_exp,
                'loss_vel_euler': cal_vel_loss(sel_euler_angle),
                'loss_vel_trans': cal_vel_loss(sel_trans),
            }
            loss_regid = torch.mean(sel_id_para*sel_id_para) # 正则化
            loss_regexp = torch.mean(sel_exp_para*sel_exp_para)
            loss = loss_lan + loss_regid * LAMBDA_REG_ID + loss_regexp * LAMBDA_REG_EXP + loss_lap * LAMBDA_REG_LAP + loss_vel_id * LAMBDA_REG_VEL_ID + loss_vel_exp * LAMBDA_REG_VEL_EXP

            optimizer_cur_batch.zero_grad()
            loss.backward()
            optimizer_cur_batch.step()
            
        if debug:
            print(f"batch {i} | loss_lan: {loss_lan.item():.2f}, loss_reg_id: {loss_regid.item():.2f},loss_reg_exp: {loss_regexp.item():.2f},loss_lap_ldm:{loss_lap.item():.4f}")
            print("|--------" + ', '.join([f"{k}: {v:.4f}" for k,v in log_dict.items()]))
        if id_mode != 'global':
            id_para[sel_ids].data = sel_id_para.data.clone()
        exp_para[sel_ids].data = sel_exp_para.data.clone()
        euler_angle[sel_ids].data = sel_euler_angle.data.clone()
        trans[sel_ids].data = sel_trans.data.clone()

    coeff_dict = {'id': id_para.detach().cpu().numpy(), 'exp': exp_para.detach().cpu().numpy(),
                'euler': euler_angle.detach().cpu().numpy(), 'trans': trans.detach().cpu().numpy()}

    # filter data by side-view pose    
    # bad_yaw = False
    # yaws = [] # not so accurate
    # for index in range(coeff_dict["trans"].shape[0]):
    #     yaw = coeff_dict["euler"][index][1]
    #     yaw = np.abs(yaw)
    #     yaws.append(yaw)
    #     if yaw > large_yaw_threshold:
    #         bad_yaw = True
    
    if debug:
        import imageio
        from utils.visualization.vis_cam3d.camera_pose_visualizer import CameraPoseVisualizer
        from data_util.face3d_helper import Face3DHelper
        from data_gen.utils.process_video.extract_blink import get_eye_area_percent
        face3d_helper = Face3DHelper('deep_3drecon/BFM', keypoint_mode='mediapipe')

        t = coeff_dict['exp'].shape[0]
        if len(coeff_dict['id']) == 1:
            coeff_dict['id'] = np.repeat(coeff_dict['id'], t, axis=0)
        idexp_lm3d = face3d_helper.reconstruct_idexp_lm3d_np(coeff_dict['id'], coeff_dict['exp']).reshape([t, -1])
        cano_lm3d = idexp_lm3d / 10 + face3d_helper.key_mean_shape.squeeze().reshape([1, -1]).cpu().numpy()
        cano_lm3d = cano_lm3d.reshape([t, -1, 3])
        WH = 512
        cano_lm3d = (cano_lm3d * WH/2 + WH/2).astype(int)

        with torch.no_grad():
            rot = ParametricFaceModel.compute_rotation(euler_angle)
            extrinsic = torch.zeros([rot.shape[0], 4, 4]).to(rot.device)
            extrinsic[:, :3,:3] = rot
            extrinsic[:, :3, 3] = trans # / 10
            extrinsic[:, 3, 3] = 1
        extrinsic = extrinsic.cpu().numpy()

        xy_camera_visualizer = CameraPoseVisualizer(xlim=[extrinsic[:,0,3].min().item()-0.5,extrinsic[:,0,3].max().item()+0.5],ylim=[extrinsic[:,1,3].min().item()-0.5,extrinsic[:,1,3].max().item()+0.5], zlim=[extrinsic[:,2,3].min().item()-0.5,extrinsic[:,2,3].max().item()+0.5], view_mode='xy')
        xz_camera_visualizer = CameraPoseVisualizer(xlim=[extrinsic[:,0,3].min().item()-0.5,extrinsic[:,0,3].max().item()+0.5],ylim=[extrinsic[:,1,3].min().item()-0.5,extrinsic[:,1,3].max().item()+0.5], zlim=[extrinsic[:,2,3].min().item()-0.5,extrinsic[:,2,3].max().item()+0.5], view_mode='xz')

        if nerf:
            debug_name = video_name.replace("/raw/", "/processed/").replace(".mp4", "/debug_fit_3dmm.mp4")
        else:
            debug_name = video_name.replace("/video/", "/coeff_fit_debug/").replace(".mp4", "_debug.mp4")
        try:
            os.makedirs(os.path.dirname(debug_name), exist_ok=True)
        except: pass
        writer = imageio.get_writer(debug_name, fps=25)
        if id_mode == 'global':
            id_para = id_para.repeat([exp_para.shape[0], 1])
        proj_geo = face_model.compute_for_landmark_fit(id_para, exp_para, euler_angle, trans)
        lm68s = proj_geo[:,:,:2].detach().cpu().numpy()  # [T, 68,2]
        lm68s = lm68s * img_scale_factor
        lms = lms * img_scale_factor
        lm68s[..., 1] = img_h - lm68s[..., 1] # flip the height axis
        lms[..., 1] = img_h - lms[..., 1] # flip the height axis
        lm68s = lm68s.astype(int)
        for i in tqdm.trange(min(250, len(frames)), desc=f'rendering debug video to {debug_name}..'):
            xy_cam3d_img = xy_camera_visualizer.extrinsic2pyramid(extrinsic[i], focal_len_scaled=0.25)
            xy_cam3d_img = cv2.resize(xy_cam3d_img, (512,512))
            xz_cam3d_img = xz_camera_visualizer.extrinsic2pyramid(extrinsic[i], focal_len_scaled=0.25)
            xz_cam3d_img = cv2.resize(xz_cam3d_img, (512,512))
            
            img = copy.deepcopy(frames[i])
            img2 = copy.deepcopy(frames[i])

            img = draw_axes(img, euler_angle[i,0].item(), euler_angle[i,1].item(), euler_angle[i,2].item(), lm68s[i][4][0].item(), lm68s[i, 4][1].item(), size=50)

            gt_lm_color = (255, 0, 0)
                
            for lm in lm68s[i]:
                img = cv2.circle(img, lm, 1, (0, 0, 255), thickness=-1) # blue
            for gt_lm in lms[i]:
                img2 = cv2.circle(img2, gt_lm.cpu().numpy().astype(int), 2, gt_lm_color, thickness=1)
            
            cano_lm3d_img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
            for j in range(len(cano_lm3d[i])):
                x, y, _ = cano_lm3d[i, j]
                color = (255,0,0)
                cano_lm3d_img = cv2.circle(cano_lm3d_img, center=(x,y), radius=3, color=color, thickness=-1)
            cano_lm3d_img = cv2.flip(cano_lm3d_img, 0)

            _, secc_img = secc_renderer(id_para[0:1], exp_para[i:i+1], euler_angle[i:i+1]*0, trans[i:i+1]*0)
            secc_img = (secc_img +1)*127.5
            secc_img = F.interpolate(secc_img, size=(img_h, img_w))
            secc_img = secc_img.permute(0, 2,3,1).int().cpu().numpy()[0]
            out_img1 = np.concatenate([img, img2, secc_img], axis=1).astype(np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            out_img2 = np.concatenate([xy_cam3d_img, xz_cam3d_img, cano_lm3d_img], axis=1).astype(np.uint8)
            out_img = np.concatenate([out_img1, out_img2], axis=0)
            writer.append_data(out_img)
        writer.close()
        
    # if bad_yaw:
    #     print(f"Skip {video_name} due to TOO LARGE YAW")
    #     return False

    if save:
        np.save(out_name, coeff_dict, allow_pickle=True) 
    return coeff_dict

def out_exist_job(vid_name):
    out_name = vid_name.replace("/video/", "/coeff_fit_mp/").replace(".mp4","_coeff_fit_mp.npy") 
    lms_name = vid_name.replace("/video/", "/lms_2d/").replace(".mp4","_lms.npy") 
    if os.path.exists(out_name) or not os.path.exists(lms_name):
        return None
    else:
        return vid_name

def get_todo_vid_names(vid_names):
    if len(vid_names) == 1: # single video, nerf
        return vid_names
    todo_vid_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, vid_names, num_workers=16):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names


if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    # parser.add_argument("--vid_dir", default='/home/tiger/datasets/raw/CelebV-HQ/video')
    parser.add_argument("--vid_dir", default='data/raw/videos/May_10s.mp4')
    parser.add_argument("--ds_name", default='nerf') # 'nerf' | 'CelebV-HQ' | 'TH1KH_512' | etc
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--id_mode", default='global', type=str) # global | finegrained
    parser.add_argument("--keypoint_mode", default='mediapipe', type=str)
    parser.add_argument("--large_yaw_threshold", default=9999999.9, type=float) # could be 0.7
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--load_names", action="store_true")

    args = parser.parse_args()
    vid_dir = args.vid_dir
    ds_name = args.ds_name
    load_names = args.load_names
    
    print(f"args {args}")
    
    if ds_name.lower() == 'nerf': # 处理单个视频
        vid_names = [vid_dir]
        out_names = [video_name.replace("/raw/", "/processed/").replace(".mp4","_coeff_fit_mp.npy") for video_name in vid_names]
    else: # 处理整个数据集
        if ds_name in ['lrs3_trainval']:
            vid_name_pattern = os.path.join(vid_dir, "*/*.mp4")
        elif ds_name in ['TH1KH_512', 'CelebV-HQ']:
            vid_name_pattern = os.path.join(vid_dir, "*.mp4")
        elif ds_name in ['lrs2', 'lrs3', 'voxceleb2', 'CMLR']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        elif ds_name in ["RAVDESS", 'VFHQ']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        else:
            raise NotImplementedError()
        
        vid_names_path = os.path.join(vid_dir, "vid_names.pkl")
        if os.path.exists(vid_names_path) and load_names:
            print(f"loading vid names from {vid_names_path}")
            vid_names = load_file(vid_names_path)
        else:
            vid_names = multiprocess_glob(vid_name_pattern)
        vid_names = sorted(vid_names)
        print(f"saving vid names to {vid_names_path}")
        save_file(vid_names_path, vid_names)
        out_names = [video_name.replace("/video/", "/coeff_fit_mp/").replace(".mp4","_coeff_fit_mp.npy") for video_name in vid_names]

    print(vid_names[:10])
    random.seed(args.seed)
    random.shuffle(vid_names)

    face_model = ParametricFaceModel(bfm_folder='deep_3drecon/BFM', 
                camera_distance=10, focal=1015, keypoint_mode=args.keypoint_mode)
    face_model.to(torch.device("cuda:0"))
    secc_renderer = SECC_Renderer(512)
    secc_renderer.to("cuda:0")
    
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            vid_names = vid_names[process_id * num_samples_per_process : ]
        else:
            vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]

    if not args.reset:
        vid_names = get_todo_vid_names(vid_names)

    failed_img_names = []
    for i in tqdm.trange(len(vid_names), desc=f"process {process_id}: fitting 3dmm ..."):
        img_name = vid_names[i]
        try:
            is_person_specific_data = ds_name=='nerf'
            success = fit_3dmm_for_a_video(img_name, is_person_specific_data, args.id_mode, args.debug, large_yaw_threshold=args.large_yaw_threshold)
            if not success:
                failed_img_names.append(img_name)   
        except Exception as e:
            print(img_name, e)
            failed_img_names.append(img_name)
        print(f"finished {i + 1} / {len(vid_names)} = {(i + 1) / len(vid_names):.4f}, failed {len(failed_img_names)} / {i + 1} = {len(failed_img_names) / (i + 1):.4f}")
        sys.stdout.flush()
    print(f"all failed image names: {failed_img_names}")
    print(f"All finished!")