import os, glob
from utils.commons.multiprocess_utils import multiprocess_run_tqdm

from data_gen.utils.path_converter import PathConverter, pc

# mp4_names = glob.glob("/home/tiger/datasets/raw/CelebV-HQ/video/*.mp4")

def extract_img_job(video_name, raw_img_dir=None):
    if raw_img_dir is not None:
        out_path = raw_img_dir
    else:
        out_path = pc.to(video_name.replace(".mp4", ""), "vid", "gt")
    os.makedirs(out_path, exist_ok=True)
    ffmpeg_path = "/usr/bin/ffmpeg"
    cmd = f'{ffmpeg_path} -i {video_name} -vf fps={25},scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 -v quiet {os.path.join(out_path, "%8d.jpg")}'
    os.system(cmd)

if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='/home/tiger/datasets/raw/CelebV-HQ/video')
    parser.add_argument("--ds_name", default='CelebV-HQ')
    parser.add_argument("--num_workers", default=64, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    args = parser.parse_args()
    vid_dir = args.vid_dir
    ds_name = args.ds_name
    if ds_name in ['lrs3_trainval']:
        mp4_name_pattern = os.path.join(vid_dir, "*/*.mp4")
    elif ds_name in ['TH1KH_512', 'CelebV-HQ']:
        vid_names = glob.glob(os.path.join(vid_dir, "*.mp4"))
    elif ds_name in ['lrs2', 'lrs3', 'voxceleb2']:
        vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        vid_names = glob.glob(vid_name_pattern)
    elif ds_name in ["RAVDESS", 'VFHQ']:
        vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        vid_names = glob.glob(vid_name_pattern)
    vid_names = sorted(vid_names)
    
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            vid_names = vid_names[process_id * num_samples_per_process : ]
        else:
            vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    for i, res in multiprocess_run_tqdm(extract_img_job, vid_names, num_workers=args.num_workers, desc="extracting images"):
        pass

