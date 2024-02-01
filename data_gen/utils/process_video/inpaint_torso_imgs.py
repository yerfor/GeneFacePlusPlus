import cv2
import os
import numpy as np
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from scipy.ndimage import binary_erosion, binary_dilation

from tasks.eg3ds.loss_utils.segment_loss.mp_segmenter import MediapipeSegmenter
seg_model = MediapipeSegmenter()

def inpaint_torso_job(video_name, idx=None, total=None):
    raw_img_dir = video_name.replace(".mp4", "").replace("/video/","/gt_imgs/")
    img_names = glob.glob(os.path.join(raw_img_dir, "*.jpg"))

    for image_path in tqdm.tqdm(img_names):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        segmap = seg_model._cal_seg_map(cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB))
        head_part = (segmap[1] + segmap[3] + segmap[5]).astype(np.bool)
        torso_part = (segmap[4]).astype(np.bool)
        neck_part = (segmap[2]).astype(np.bool)
        bg_part = segmap[0].astype(np.bool)
        head_image = cv2.imread(image_path.replace("/gt_imgs/", "/head_imgs/"), cv2.IMREAD_UNCHANGED) # [H, W, 3]
        torso_image = cv2.imread(image_path.replace("/gt_imgs/", "/torso_imgs/"), cv2.IMREAD_UNCHANGED) # [H, W, 3]
        bg_image = cv2.imread(image_path.replace("/gt_imgs/", "/bg_imgs/"), cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # head_part = (head_image[...,0] != 0) & (head_image[...,1] != 0) & (head_image[...,2] != 0)
        # torso_part = (torso_image[...,0] != 0) & (torso_image[...,1] != 0) & (torso_image[...,2] != 0)
        # bg_part = (bg_image[...,0] != 0) & (bg_image[...,1] != 0) & (bg_image[...,2] != 0)

        # get gt image
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = 0
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0]) # [N, 2]
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None
            
        # neck part "vertical" in-painting...
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)] 
        
        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)] # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0) # [L, m, 3]
        darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite("0.png", np.concatenate([torso_image, torso_alpha], axis=-1))

    print(f'[INFO] ===== extracted torso and gt images =====')


def out_exist_job(vid_name):
    out_dir1 = vid_name.replace("/video/", "/inpaint_torso_imgs/").replace(".mp4","") 
    out_dir2 = vid_name.replace("/video/", "/inpaint_torso_with_bg_imgs/").replace(".mp4","") 
    out_dir3 = vid_name.replace("/video/", "/torso_imgs/").replace(".mp4","") 
    out_dir4 = vid_name.replace("/video/", "/torso_with_bg_imgs/").replace(".mp4","") 
    
    if os.path.exists(out_dir1) and os.path.exists(out_dir1) and os.path.exists(out_dir2) and os.path.exists(out_dir3) and os.path.exists(out_dir4):
        num_frames = len(os.listdir(out_dir1))
        if len(os.listdir(out_dir1)) == num_frames and len(os.listdir(out_dir2)) == num_frames and len(os.listdir(out_dir3)) == num_frames and len(os.listdir(out_dir4)) == num_frames:
            return None
        else:
            return vid_name
    else:
        return vid_name
    
def get_todo_vid_names(vid_names):
    todo_vid_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, vid_names, num_workers=16):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names

if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='/home/tiger/datasets/raw/CelebV-HQ/video')
    parser.add_argument("--ds_name", default='CelebV-HQ')
    parser.add_argument("--num_workers", default=48, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action='store_true')
    
    inpaint_torso_job('/home/tiger/datasets/raw/CelebV-HQ/video/dgdEr-mXQT4_8.mp4')
    # args = parser.parse_args()
    # vid_dir = args.vid_dir
    # ds_name = args.ds_name
    # if ds_name in ['lrs3_trainval']:
    #     mp4_name_pattern = os.path.join(vid_dir, "*/*.mp4")
    # if ds_name in ['TH1KH_512', 'CelebV-HQ']:
    #     vid_names = glob.glob(os.path.join(vid_dir, "*.mp4"))
    # elif ds_name in ['lrs2', 'lrs3', 'voxceleb2']:
    #     vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
    #     vid_names = glob.glob(vid_name_pattern)
    # vid_names = sorted(vid_names)
    # random.seed(args.seed)
    # random.shuffle(vid_names)

    # process_id = args.process_id
    # total_process = args.total_process
    # if total_process > 1:
    #     assert process_id <= total_process -1
    #     num_samples_per_process = len(vid_names) // total_process
    #     if process_id == total_process:
    #         vid_names = vid_names[process_id * num_samples_per_process : ]
    #     else:
    #         vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    # if not args.reset:
    #     vid_names = get_todo_vid_names(vid_names)
    # print(f"todo videos number: {len(vid_names)}")

    # fn_args = [(vid_name,i,len(vid_names)) for i, vid_name in enumerate(vid_names)]
    # for vid_name in multiprocess_run_tqdm(inpaint_torso_job ,fn_args, desc=f"Root process {args.process_id}: extracting segment images", num_workers=args.num_workers):
    #     pass