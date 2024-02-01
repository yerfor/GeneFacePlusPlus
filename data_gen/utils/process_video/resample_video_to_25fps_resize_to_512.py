import os, glob
import cv2
from utils.commons.os_utils import multiprocess_glob
from utils.commons.multiprocess_utils import multiprocess_run_tqdm

def get_video_infos(video_path):
    vid_cap = cv2.VideoCapture(video_path)
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {'height': height, 'width': width, 'fps': fps, 'total_frames':total_frames}

def extract_img_job(video_name:str):
    out_path = video_name.replace("/video_raw/","/video/",1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_path = "/usr/bin/ffmpeg"
    vid_info = get_video_infos(video_name)
    assert vid_info['width'] == vid_info['height']
    cmd = f'{ffmpeg_path} -i {video_name} -vf fps={25},scale=w=512:h=512 -q:v 1 -c:v libx264 -pix_fmt yuv420p -b:v 2000k -v quiet -y {out_path}'
    os.system(cmd)

def extract_img_job_crop(video_name:str):
    out_path = video_name.replace("/video_raw/","/video/",1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_path = "/usr/bin/ffmpeg"
    vid_info = get_video_infos(video_name)
    wh = min(vid_info['width'], vid_info['height'])
    cmd = f'{ffmpeg_path} -i {video_name} -vf fps={25},crop={wh}:{wh},scale=w=512:h=512 -q:v 1 -c:v libx264 -pix_fmt yuv420p -b:v 2000k -v quiet -y {out_path}'
    os.system(cmd)

def extract_img_job_crop_ravdess(video_name:str):
    out_path = video_name.replace("/video_raw/","/video/",1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_path = "/usr/bin/ffmpeg"
    cmd = f'{ffmpeg_path} -i {video_name} -vf fps={25},crop=720:720,scale=w=512:h=512 -q:v 1 -c:v libx264 -pix_fmt yuv420p -b:v 2000k -v quiet -y {out_path}'
    os.system(cmd)

if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='/home/tiger/datasets/raw/CelebV-HQ/video_raw/')
    parser.add_argument("--ds_name", default='CelebV-HQ')
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    args = parser.parse_args()
    print(f"args {args}")

    vid_dir = args.vid_dir
    ds_name = args.ds_name
    if ds_name in ['lrs3_trainval']:
        mp4_name_pattern = os.path.join(vid_dir, "*/*.mp4")
    elif ds_name in ['TH1KH_512', 'CelebV-HQ']:
        vid_names = multiprocess_glob(os.path.join(vid_dir, "*.mp4"))
    elif ds_name in ['lrs2', 'lrs3', 'voxceleb2', 'CMLR']:
        vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        vid_names = multiprocess_glob(vid_name_pattern)
    elif ds_name in ["RAVDESS", 'VFHQ']:
        vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        vid_names = multiprocess_glob(vid_name_pattern)
    else:
        raise NotImplementedError()
    vid_names = sorted(vid_names)
    print(f"total video number : {len(vid_names)}")
    print(f"first {vid_names[0]} last {vid_names[-1]}")
    # exit()
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            vid_names = vid_names[process_id * num_samples_per_process : ]
        else:
            vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    if ds_name == "RAVDESS":
        for i, res in multiprocess_run_tqdm(extract_img_job_crop_ravdess, vid_names, num_workers=args.num_workers, desc="resampling videos"):
            pass
    elif ds_name == "CMLR":
        for i, res in multiprocess_run_tqdm(extract_img_job_crop, vid_names, num_workers=args.num_workers, desc="resampling videos"):
            pass
    else:
        for i, res in multiprocess_run_tqdm(extract_img_job, vid_names, num_workers=args.num_workers, desc="resampling videos"):
            pass

