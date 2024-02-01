import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import glob
import cv2
import pickle
import tqdm
import numpy as np
import mediapipe as mp
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.commons.os_utils import multiprocess_glob
from data_gen.utils.mp_feature_extractors.face_landmarker import MediapipeLandmarker
import warnings
import traceback

warnings.filterwarnings('ignore')

"""
基于Face_aligment的lm68已被弃用,因为其：
1. 对眼睛部位的预测精度极低
2. 无法在大偏转角度时准确预测被遮挡的下颚线, 导致大角度时3dmm的GT label就是有问题的, 从而影响性能
我们目前转而使用基于mediapipe的lm68
"""
# def extract_landmarks(ori_imgs_dir):

#     print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')

#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#     image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.png'))
#     for image_path in tqdm.tqdm(image_paths):
#         out_name = image_path.replace("/images_512/", "/lms_2d/").replace(".png",".lms")
#         if os.path.exists(out_name):
#             continue
#         input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
#         input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
#         preds = fa.get_landmarks(input)
#         if preds is None:
#             print(f"Skip {image_path} for no face detected")
#             continue
#         if len(preds) > 0:
#             lands = preds[0].reshape(-1, 2)[:,:2]
#             os.makedirs(os.path.dirname(out_name), exist_ok=True)
#             np.savetxt(out_name, lands, '%f')
#     del fa
#     print(f'[INFO] ===== extracted face landmarks =====')

def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f) 
        
def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content


face_landmarker = None
    
def extract_landmark_job(video_name, nerf=False):
    try:
        if nerf:
            out_name = video_name.replace("/raw/", "/processed/").replace(".mp4","/lms_2d.npy")
        else:
            out_name = video_name.replace("/video/", "/lms_2d/").replace(".mp4","_lms.npy")
        if os.path.exists(out_name):
            # print("out exists, skip...")
            return
        try:
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
        except:
            pass
        global face_landmarker
        if face_landmarker is None:
            face_landmarker = MediapipeLandmarker()
        img_lm478, vid_lm478 = face_landmarker.extract_lm478_from_video_name(video_name)
        lm478 = face_landmarker.combine_vid_img_lm478_to_lm478(img_lm478, vid_lm478)
        np.save(out_name, lm478)
        return True
        # print("Hahaha, solve one item!!!")
    except Exception as e:
        traceback.print_exc()
        return False
        
def out_exist_job(vid_name):
    out_name = vid_name.replace("/video/", "/lms_2d/").replace(".mp4","_lms.npy") 
    if os.path.exists(out_name):
        return None
    else:
        return vid_name
    
def get_todo_vid_names(vid_names):
    if len(vid_names) == 1: # nerf
        return vid_names
    todo_vid_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, vid_names, num_workers=128):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names

if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='nerf')
    parser.add_argument("--ds_name", default='data/raw/videos/May.mp4')
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--load_names", action="store_true")

    args = parser.parse_args()
    vid_dir = args.vid_dir
    ds_name = args.ds_name
    load_names = args.load_names

    if ds_name.lower() == 'nerf': # 处理单个视频
        vid_names = [vid_dir]
        out_names = [video_name.replace("/raw/", "/processed/").replace(".mp4","/lms_2d.npy") for video_name in vid_names]
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
        if not load_names:
            print(f"saving vid names to {vid_names_path}")
            save_file(vid_names_path, vid_names)
        out_names = [video_name.replace("/video/", "/lms_2d/").replace(".mp4","_lms.npy") for video_name in vid_names]

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
    print(f"todo videos number: {len(vid_names)}")

    fail_cnt = 0
    job_args = [(vid_name, ds_name=='nerf') for vid_name in vid_names]
    for (i, res) in multiprocess_run_tqdm(extract_landmark_job, job_args, num_workers=args.num_workers, desc=f"Root {args.process_id}: extracing MP-based landmark2d"): 
        if res is False:
            fail_cnt += 1
        print(f"finished {i + 1} / {len(vid_names)} = {(i + 1) / len(vid_names):.4f}, failed {fail_cnt} / {i + 1} = {fail_cnt / (i + 1):.4f}")
        sys.stdout.flush()
        pass