import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

from deep_3drecon.deep_3drecon_models.bfm import perspective_projection


class Face3DHelper(nn.Module):
    def __init__(self, bfm_dir='deep_3drecon/BFM', keypoint_mode='lm68', use_gpu=True):
        super().__init__()
        self.keypoint_mode = keypoint_mode # lm68 | mediapipe
        self.bfm_dir = bfm_dir
        self.load_3dmm()
        if use_gpu: self.to("cuda")
            
    def load_3dmm(self):
        model = loadmat(os.path.join(self.bfm_dir, "BFM_model_front.mat"))
        self.register_buffer('mean_shape',torch.from_numpy(model['meanshape'].transpose()).float()) # mean face shape. [3*N, 1], N=35709, xyz=3, ==> 3*N=107127
        mean_shape = self.mean_shape.reshape([-1, 3])
        # re-center
        mean_shape = mean_shape - torch.mean(mean_shape, dim=0, keepdims=True)
        self.mean_shape = mean_shape.reshape([-1, 1])
        self.register_buffer('id_base',torch.from_numpy(model['idBase']).float()) # identity basis. [3*N,80], we have 80 eigen faces for identity
        self.register_buffer('exp_base',torch.from_numpy(model['exBase']).float()) # expression basis. [3*N,64], we have 64 eigen faces for expression
        
        self.register_buffer('mean_texure',torch.from_numpy(model['meantex'].transpose()).float()) # mean face texture. [3*N,1] (0-255)
        self.register_buffer('tex_base',torch.from_numpy(model['texBase']).float()) # texture basis. [3*N,80], rgb=3
        
        self.register_buffer('point_buf',torch.from_numpy(model['point_buf']).float()) # triangle indices for each vertex that lies in. starts from 1. [N,8] (1-F)
        self.register_buffer('face_buf',torch.from_numpy(model['tri']).float()) # vertex indices in each triangle. starts from 1. [F,3] (1-N)
        if self.keypoint_mode == 'mediapipe':
            self.register_buffer('key_points', torch.from_numpy(np.load("deep_3drecon/BFM/index_mp468_from_mesh35709.npy").astype(np.int64)))
            unmatch_mask = self.key_points < 0
            self.key_points[unmatch_mask] = 0
        else:
            self.register_buffer('key_points',torch.from_numpy(model['keypoints'].squeeze().astype(np.int_)).long()) # vertex indices of 68 facial landmarks. starts from 1. [68,1]
        

        self.register_buffer('key_mean_shape',self.mean_shape.reshape([-1,3])[self.key_points,:])
        self.register_buffer('key_id_base', self.id_base.reshape([-1,3,80])[self.key_points, :, :].reshape([-1,80])) 
        self.register_buffer('key_exp_base', self.exp_base.reshape([-1,3,64])[self.key_points, :, :].reshape([-1,64])) 
        self.key_id_base_np = self.key_id_base.cpu().numpy()
        self.key_exp_base_np = self.key_exp_base.cpu().numpy()

        self.register_buffer('persc_proj', torch.tensor(perspective_projection(focal=1015, center=112))) 
    def split_coeff(self, coeff):
        """
        coeff: Tensor[B, T, c=257] or [T, c=257]
        """
        ret_dict = {
            'identity': coeff[..., :80],  # identity, [b, t, c=80] 
            'expression': coeff[..., 80:144],  # expression, [b, t, c=80]
            'texture': coeff[..., 144:224],  # texture, [b, t, c=80]
            'euler': coeff[..., 224:227],  # euler euler for pose, [b, t, c=3]
            'translation':  coeff[..., 254:257], # translation, [b, t, c=3]
            'gamma': coeff[..., 227:254] # lighting, [b, t, c=27]
        }
        return ret_dict
    
    def reconstruct_face_mesh(self, id_coeff, exp_coeff):
        """
        Generate a pose-independent 3D face mesh!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.key_id_base.device)
        exp_coeff = exp_coeff.to(self.key_id_base.device)
        mean_face = self.mean_shape.squeeze().reshape([1, -1]) # [3N, 1] ==> [1, 3N]
        id_base, exp_base = self.id_base, self.exp_base # [3*N, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3N] ==> [t,3N]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3N] ==> [t,3N]
        
        face = mean_face + identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        # re-centering the face with mean_xyz, so the face will be in [-1, 1]
        # mean_xyz = self.mean_shape.squeeze().reshape([-1,3]).mean(dim=0) # [1, 3]
        # face_mesh = face - mean_xyz.unsqueeze(0) # [t,N,3]
        return face

    def reconstruct_cano_lm3d(self, id_coeff, exp_coeff):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.key_id_base.device)
        exp_coeff = exp_coeff.to(self.key_id_base.device)
        mean_face = self.key_mean_shape.squeeze().reshape([1, -1]) # [3*68, 1] ==> [1, 3*68]
        id_base, exp_base = self.key_id_base, self.key_exp_base # [3*68, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = mean_face + identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        # re-centering the face with mean_xyz, so the face will be in [-1, 1]
        # mean_xyz = self.key_mean_shape.squeeze().reshape([-1,3]).mean(dim=0) # [1, 3]
        # lm3d = face - mean_xyz.unsqueeze(0) # [t,N,3]
        return face

    def reconstruct_lm3d(self, id_coeff, exp_coeff, euler, trans, to_camera=True):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.key_id_base.device)
        exp_coeff = exp_coeff.to(self.key_id_base.device)
        mean_face = self.key_mean_shape.squeeze().reshape([1, -1]) # [3*68, 1] ==> [1, 3*68]
        id_base, exp_base = self.key_id_base, self.key_exp_base # [3*68, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = mean_face + identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        # re-centering the face with mean_xyz, so the face will be in [-1, 1]
        rot = self.compute_rotation(euler)
        # transform
        lm3d = face @ rot + trans.unsqueeze(1) # [t, N, 3]
        # to camera
        if to_camera:
            lm3d[...,-1] = 10 - lm3d[...,-1] 
        return lm3d

    def reconstruct_lm2d_nerf(self, id_coeff, exp_coeff, euler, trans):
        lm2d = self.reconstruct_lm2d(id_coeff, exp_coeff, euler, trans, to_camera=False)
        lm2d[..., 0] = 1 - lm2d[..., 0]
        lm2d[..., 1] = 1 - lm2d[..., 1]
        return lm2d

    def reconstruct_lm2d(self, id_coeff, exp_coeff, euler, trans, to_camera=True):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        is_btc_flag = True if id_coeff.ndim == 3 else False
        if is_btc_flag:
            b,t,_ = id_coeff.shape
            id_coeff = id_coeff.reshape([b*t,-1])
            exp_coeff = exp_coeff.reshape([b*t,-1])
            euler = euler.reshape([b*t,-1])
            trans = trans.reshape([b*t,-1])
        id_coeff = id_coeff.to(self.key_id_base.device)
        exp_coeff = exp_coeff.to(self.key_id_base.device)
        mean_face = self.key_mean_shape.squeeze().reshape([1, -1]) # [3*68, 1] ==> [1, 3*68]
        id_base, exp_base = self.key_id_base, self.key_exp_base # [3*68, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = mean_face + identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        # re-centering the face with mean_xyz, so the face will be in [-1, 1]
        rot = self.compute_rotation(euler)
        # transform
        lm3d = face @ rot + trans.unsqueeze(1) # [t, N, 3]
        # to camera
        if to_camera:
            lm3d[...,-1] = 10 - lm3d[...,-1] 
        # to image_plane
        lm3d = lm3d @ self.persc_proj
        lm2d = lm3d[..., :2] / lm3d[..., 2:]
        # flip
        lm2d[..., 1] = 224 - lm2d[..., 1]
        lm2d /= 224
        if is_btc_flag:
            return lm2d.reshape([b,t,-1,2])
        return lm2d
    
    def compute_rotation(self, euler):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            euler           -- torch.tensor, size (B, 3), radian
        """

        batch_size = euler.shape[0]
        euler = euler.to(self.key_id_base.device)
        ones = torch.ones([batch_size, 1]).to(self.key_id_base.device)
        zeros = torch.zeros([batch_size, 1]).to(self.key_id_base.device)
        x, y, z = euler[:, :1], euler[:, 1:2], euler[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)
    
    def reconstruct_idexp_lm3d(self, id_coeff, exp_coeff):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.key_id_base.device)
        exp_coeff = exp_coeff.to(self.key_id_base.device)
        id_base, exp_base = self.key_id_base, self.key_exp_base # [3*68, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        lm3d = face * 10
        return lm3d
    
    def reconstruct_idexp_lm3d_np(self, id_coeff, exp_coeff):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_base, exp_base = self.key_id_base_np, self.key_exp_base_np # [3*68, C]
        identity_diff_face = np.dot(id_coeff, id_base.T) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = np.dot(exp_coeff, exp_base.T) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        lm3d = face * 10
        return lm3d
    
    def get_eye_mouth_lm_from_lm3d(self, lm3d):
        eye_lm = lm3d[:, 17:48] # [T, 31, 3]
        mouth_lm = lm3d[:, 48:68] # [T, 20, 3]
        return eye_lm, mouth_lm
    
    def get_eye_mouth_lm_from_lm3d_batch(self, lm3d):
        eye_lm = lm3d[:, :, 17:48] # [T, 31, 3]
        mouth_lm = lm3d[:, :, 48:68] # [T, 20, 3]
        return eye_lm, mouth_lm
    
    def close_mouth_for_idexp_lm3d(self, idexp_lm3d, freeze_as_first_frame=True):
        idexp_lm3d = idexp_lm3d.reshape([-1, 68,3])
        num_frames = idexp_lm3d.shape[0]
        eps = 0.0
        # [n_landmarks=68,xyz=3], x 代表左右，y代表上下，z代表深度
        idexp_lm3d[:,49:54, 1] = (idexp_lm3d[:,49:54, 1] + idexp_lm3d[:,range(59,54,-1), 1])/2 + eps * 2
        idexp_lm3d[:,range(59,54,-1), 1] = (idexp_lm3d[:,49:54, 1] + idexp_lm3d[:,range(59,54,-1), 1])/2 - eps * 2

        idexp_lm3d[:,61:64, 1] = (idexp_lm3d[:,61:64, 1] + idexp_lm3d[:,range(67,64,-1), 1])/2 + eps
        idexp_lm3d[:,range(67,64,-1), 1] = (idexp_lm3d[:,61:64, 1] + idexp_lm3d[:,range(67,64,-1), 1])/2 - eps

        idexp_lm3d[:,49:54, 1] += (0.03 - idexp_lm3d[:,49:54, 1].mean(dim=1) + idexp_lm3d[:,61:64, 1].mean(dim=1)).unsqueeze(1).repeat([1,5])
        idexp_lm3d[:,range(59,54,-1), 1] += (-0.03 - idexp_lm3d[:,range(59,54,-1), 1].mean(dim=1) + idexp_lm3d[:,range(67,64,-1), 1].mean(dim=1)).unsqueeze(1).repeat([1,5])

        if freeze_as_first_frame:
            idexp_lm3d[:, 48:68,] = idexp_lm3d[0, 48:68].unsqueeze(0).clone().repeat([num_frames, 1,1])*0
        return idexp_lm3d.cpu()

    def close_eyes_for_idexp_lm3d(self, idexp_lm3d):
        idexp_lm3d = idexp_lm3d.reshape([-1, 68,3])
        eps = 0.003
        idexp_lm3d[:,37:39, 1] = (idexp_lm3d[:,37:39, 1] + idexp_lm3d[:,range(41,39,-1), 1])/2 + eps
        idexp_lm3d[:,range(41,39,-1), 1] = (idexp_lm3d[:,37:39, 1] + idexp_lm3d[:,range(41,39,-1), 1])/2 - eps

        idexp_lm3d[:,43:45, 1] = (idexp_lm3d[:,43:45, 1] + idexp_lm3d[:,range(47,45,-1), 1])/2 + eps
        idexp_lm3d[:,range(47,45,-1), 1] = (idexp_lm3d[:,43:45, 1] + idexp_lm3d[:,range(47,45,-1), 1])/2 - eps
        
        return idexp_lm3d

if __name__ == '__main__':
    import cv2
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_mesh_helper = Face3DHelper('deep_3drecon/BFM')
    coeff_npy = 'data/coeff_fit_mp/crop_nana_003_coeff_fit_mp.npy'
    coeff_dict = np.load(coeff_npy, allow_pickle=True).tolist()
    lm3d = face_mesh_helper.reconstruct_lm2d(torch.tensor(coeff_dict['id']).cuda(), torch.tensor(coeff_dict['exp']).cuda(), torch.tensor(coeff_dict['euler']).cuda(), torch.tensor(coeff_dict['trans']).cuda() )

    WH = 512
    lm3d = (lm3d * WH).cpu().int().numpy()
    eye_idx = list(range(36,48))
    mouth_idx = list(range(48,68))
    import imageio
    debug_name = 'debug_lm3d.mp4'
    writer = imageio.get_writer(debug_name, fps=25)
    for i_img in range(len(lm3d)):
        lm2d = lm3d[i_img ,:, :2] # [68, 2]
        img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            if i in eye_idx:
                color = (0,0,255)
            elif i in mouth_idx:
                color = (0,255,0)
            else:
                color = (255,0,0)
            img = cv2.circle(img, center=(x,y), radius=3, color=color, thickness=-1)
            img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
        writer.append_data(img)
    writer.close()
