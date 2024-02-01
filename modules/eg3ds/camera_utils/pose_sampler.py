# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from modules.eg3ds.volumetric_rendering import math_utils


class UnifiedCameraPoseSampler():
    """
    A unified class for obtain camera pose, a 25 dimension vector that consists of camera2world matrix (4x4) and camera intrinsic (3,3)
        it utilize the samplers constructed below.
    """
    def get_camera_pose(self, pitch, yaw, lookat_location=None, distance_to_orig=2.7, batch_size=1, device='cpu', roll=None):
        if lookat_location is None:
            lookat_location = torch.tensor([0., 0., -0.2], device=device)

        c2w = LookAtPoseSampler.sample(yaw, pitch, lookat_location, 0, 0, distance_to_orig, batch_size, device, roll=roll).reshape([batch_size, 16])
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device).reshape([9,]).unsqueeze(0).repeat([batch_size, 1])
        # intrinsics = FOV_to_intrinsics(fov_degrees, device=device).reshape([9,]).unsqueeze(0).repeat([batch_size, 1])
        camera = torch.cat([c2w, intrinsics], dim=1) # [batch, 25]
        return camera
    

class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        """
        horizontal_mean: 偏转角, 也叫方位角， -pi/2 denotes camera at left, 0 denotes forward, pi/2 denotes right,
        vertical_mean: 俯仰角, 0 denotes up, -pi/2 denotes camera at up, 0 means horizontal, pi/2 denotes down. however, 0.2 is a good choice for front face.
        """ 
        assert horizontal_mean < np.pi/2 + 1e-5 and horizontal_mean > - np.pi/2 - 1e-5
        assert vertical_mean < np.pi/2 + 1e-5 and vertical_mean > - np.pi/2 - 1e-5
        horizontal_mean += np.pi/2
        vertical_mean += np.pi/2
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins) # the direction the camera is pointing, pointing to origin in this func
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu', roll=None):
        """
        horizontal_mean: 偏转角, 也叫方位角， -pi/2 denotes camera at left, 0 denotes forward, pi/2 denotes right,
        vertical_mean: 俯仰角, 0 denotes up, -pi/2 denotes camera at up, 0 means horizontal, pi/2 denotes down. however, 0.2 is a good choice for front face.
        """ 
        # assert horizontal_mean < np.pi + 1e-5 and horizontal_mean > - np.pi - 1e-5
        # assert vertical_mean < np.pi + 1e-5 and vertical_mean > - np.pi - 1e-5
        horizontal_mean += np.pi/2
        vertical_mean += np.pi/2

        # if horizontal_mean < -np.pi:
        #     horizontal_mean += 2*np.pi
        # if vertical_mean < -np.pi:
        #     vertical_mean += 2*np.pi
        # if horizontal_mean > np.pi:
        #     horizontal_mean -= 2*np.pi
        # if vertical_mean > np.pi:
        #     vertical_mean -= 2*np.pi

        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h # 球坐标系里的滚转角
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        # radius*torch.sin(phi) 是球半径在水平平面上的投影，随后再根据yaw角来分别计算x和y
        # radius*torch.cos(phi)则是纵轴的分量
        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position.to(device) - camera_origins)  # the direction the camera is pointing, pointing to the lookat_position
        return create_cam2world_matrix(forward_vectors, camera_origins, roll)


class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a UNIFORM distribution with range +-[horizontal/vertical]_stddev, instead of a GAUSSIAN distribution.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        """
        horizontal_mean: 偏转角, 也叫方位角， -pi/2 denotes camera at left, 0 denotes forward, pi/2 denotes right,
        vertical_mean: 俯仰角, 0 denotes up, -pi/2 denotes camera at up, 0 means horizontal, pi/2 denotes down. however, 0.2 is a good choice for front face.
        """ 
        assert horizontal_mean < np.pi/2 + 1e-5 and horizontal_mean > - np.pi/2 - 1e-5
        assert vertical_mean < np.pi/2 + 1e-5 and vertical_mean > - np.pi/2 - 1e-5
        horizontal_mean += np.pi/2
        vertical_mean += np.pi/2
    
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device) # the location of camera

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins) # the direction the camera is pointing, pointing to origin in this func
        return create_cam2world_matrix(forward_vectors, camera_origins)    


def create_cam2world_matrix(forward_vector, origin, roll=None):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up.
    Modified by yerfor to support roll controll
    roll: Default None, leads to 0 roll; or Tensor([Batch_size, 1]), with radian in [-pi, pi]
    """

    batch_size = len(forward_vector)
    forward_vector = math_utils.normalize_vecs(forward_vector)
    # up_vector 代表相机的正上方方向向量，所以可以通过旋转它来控制roll
    up_vector = torch.zeros([batch_size, 3], dtype=forward_vector.dtype, device=forward_vector.device)
    if roll is None:
        roll = torch.zeros([batch_size, 1], dtype=forward_vector.dtype, device=forward_vector.device)
    else:
        roll = roll.reshape([batch_size, 1])

    up_vector[:, 0] = torch.sin(roll)
    up_vector[:, 1] = torch.cos(roll)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees=18.837, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics