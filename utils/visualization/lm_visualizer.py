import numpy as np
import cv2
from data_util.face3d_helper import Face3DHelper
from utils.visualization.ffmpeg_utils import imgs_to_video
import os

face3d_helper = Face3DHelper('deep_3drecon/BFM', keypoint_mode='mediapipe')
# lrs3_stats = np.load('data/binary/lrs3/stats.npy',allow_pickle=True).tolist()
# lrs3_idexp_mean = lrs3_stats['idexp_lm3d_mean'].reshape([1,204])
# lrs3_idexp_std = lrs3_stats['idexp_lm3d_std'].reshape([1,204])


def render_idexp_npy_to_lm_video(npy_name, out_video_name, audio_name=None):
    try:
        idexp_lm3d = np.load(npy_name)
    except:
        coeff = np.load(npy_name, allow_pickle=True).tolist()
        t = coeff['exp'].shape[0]
        # print(coeff['id'][0]-coeff['id'][1])
        if len(coeff['id']) == 1:
            coeff['id'] = np.repeat(coeff['id'], t, axis=0)
        idexp_lm3d = face3d_helper.reconstruct_idexp_lm3d_np(coeff['id'], coeff['exp']).reshape([t, -1])
    lm3d = idexp_lm3d / 10 + face3d_helper.key_mean_shape.squeeze().reshape([1, -1]).cpu().numpy()
    lm3d = lm3d.reshape([t, -1, 3])
    # lm3d[..., 0] = 0.5 # lm3d[:,:1, 0].repeat(lm3d.shape[1], axis=1)

    tmp_img_dir = os.path.join(os.path.dirname(out_video_name), "tmp_lm3d_imgs")
    os.makedirs(tmp_img_dir, exist_ok=True)

    WH = 512
    lm3d = (lm3d * WH/2 + WH/2).astype(int)
    # eye_idx = list(range(36,48))
    # mouth_idx = list(range(48,68))
    for i_img in range(len(lm3d)):
        lm2d = lm3d[i_img ,:, :2] # [68, 2]
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
        
        out_name = os.path.join(tmp_img_dir, f'{format(i_img, "05d")}.png')
        cv2.imwrite(out_name, img)
    imgs_to_video(tmp_img_dir, out_video_name, audio_name)
    os.system(f"rm -r {tmp_img_dir}")
    print(f"landmark video saved at {out_video_name}")

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--npy_name', type=str, default="infer_out/May/pred_lm3d/zozo.npy", help='the path of landmark .npy')
    argparser.add_argument('--audio_name', type=str, default="data/raw/val_wavs/zozo.wav", help='the path of audio file')
    argparser.add_argument('--out_path', type=str, default="infer_out/May/visualized_lm3d/zozo.mp4", help='the path to save visualization results')
    args = argparser.parse_args()
    render_idexp_npy_to_lm_video(args.npy_name, args.out_path, audio_name=args.audio_name)