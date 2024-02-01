import numpy as np
import torch
import copy
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel


def _fix_intrinsics(intrinsics):
    """
    intrinsics: [3,3], not batch-wise
    """
    # unnormalized                                normalized

    # [[ f_x, s=0,    x_0]             [[ f_x/size_x,   s=0,            x_0/size_x=0.5]
    #  [ 0,   f_y,  y_0]      ->      [ 0,            f_y/size_y,   y_0/size_y=0.5]
    #  [ 0,   0,    1  ]]             [ 0,            0,            1         ]]
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0,0] = 2985.29/700
    intrinsics[1,1] = 2985.29/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    return intrinsics

# Used in original submission
def _fix_pose_orig(pose):
    """
    pose: [4,4], not batch-wise
    """
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3]/radius * 2.7
    return pose


def get_eg3d_convention_camera_pose_intrinsic(item):
    """
    item: a dict during binarize

    """
    if item['euler'].ndim == 1:
        angle = convert_to_tensor(copy.copy(item['euler']))
        trans = copy.deepcopy(item['trans'])

        # handle the difference of euler axis between eg3d and ours
        # see data_gen/process_ffhq_for_eg3d/transplant_eg3d_ckpt_into_our_convention.ipynb
        # angle += torch.tensor([0, 3.1415926535, 3.1415926535], device=angle.device)
        R = ParametricFaceModel.compute_rotation(angle.unsqueeze(0))[0].cpu().numpy()
        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3,:3] = R
        c *= 0.27 # normalize camera radius
        c[1] += 0.006 # additional offset used in submission
        c[2] += 0.161 # additional offset used in submission
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2]

        focal = 2985.29 # = 1015*1024/224*(300/466.285),
        # todo： 如果修改了fit 3dmm阶段的camera intrinsic，这里也要跟着改
        pp = 512#112
        w = 1024#224
        h = 1024#224

        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0
        convention_K = _fix_intrinsics(K)

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot) # permute axes
        convention_pose = _fix_pose_orig(pose)

        item['c2w'] = pose
        item['convention_c2w'] = convention_pose
        item['intrinsics'] = convention_K
        return item
    else:
        num_samples = len(item['euler'])
        eulers_all = convert_to_tensor(copy.deepcopy(item['euler'])) # [B, 3]
        trans_all = copy.deepcopy(item['trans']) # [B, 3]

        # handle the difference of euler axis between eg3d and ours
        # see data_gen/process_ffhq_for_eg3d/transplant_eg3d_ckpt_into_our_convention.ipynb
        # eulers_all += torch.tensor([0, 3.1415926535, 3.1415926535], device=eulers_all.device).unsqueeze(0).repeat([eulers_all.shape[0],1])

        intrinsics = []
        poses = []
        convention_poses = []
        for i in range(num_samples):
            angle = eulers_all[i]
            trans = trans_all[i]
            R = ParametricFaceModel.compute_rotation(angle.unsqueeze(0))[0].cpu().numpy()
            trans[2] += -10
            c = -np.dot(R, trans)
            pose = np.eye(4)
            pose[:3,:3] = R
            c *= 0.27 # normalize camera radius
            c[1] += 0.006 # additional offset used in submission
            c[2] += 0.161 # additional offset used in submission
            pose[0,3] = c[0]
            pose[1,3] = c[1]
            pose[2,3] = c[2]

            focal = 2985.29 # = 1015*1024/224*(300/466.285),
            # todo： 如果修改了fit 3dmm阶段的camera intrinsic，这里也要跟着改
            pp = 512#112
            w = 1024#224
            h = 1024#224

            K = np.eye(3)
            K[0][0] = focal
            K[1][1] = focal
            K[0][2] = w/2.0
            K[1][2] = h/2.0
            convention_K = _fix_intrinsics(K)
            intrinsics.append(convention_K)

            Rot = np.eye(3)
            Rot[0, 0] = 1
            Rot[1, 1] = -1
            Rot[2, 2] = -1        
            pose[:3, :3] = np.dot(pose[:3, :3], Rot)
            convention_pose = _fix_pose_orig(pose)
            convention_poses.append(convention_pose)
            poses.append(pose)

        intrinsics = np.stack(intrinsics) # [B, 3, 3]
        poses = np.stack(poses) # [B, 4, 4]
        convention_poses = np.stack(convention_poses) # [B, 4, 4]
        item['intrinsics'] = intrinsics
        item['c2w'] = poses
        item['convention_c2w'] = convention_poses
        return item
