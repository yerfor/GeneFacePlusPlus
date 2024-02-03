## Tips: You can run step by step of all the steps below to verify your environment is ready. After that, you can directly execute the `bash run.sh ${VIDEO_ID}` to finish all the steps in one run.

[中文文档](./guide-zh.md)

# Step0. Crop videos to 512 * 512 and 25 FPS, ensure there is face in every frame.
```
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/${VIDEO_ID}_512.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4
```
# step1: Extract audio features, such as mel, f0, hubuert and esperanto
```
export CUDA_VISIBLE_DEVICES=0
export VIDEO_ID=May
export PYTHONPATH=./
mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}
```

# Step2. Extract images
TIPS: add `--force_single_process` if you discover that multiprocessing does not work well with Mediapipe
```
export PYTHONPATH=./
export VIDEO_ID=May
export CUDA_VISIBLE_DEVICES=0
mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background
```

# Step3. Extract lm2d_mediapipe
### Extract 2D landmark for Fit 3DMM later
### num_workers: number of CPU workers；total_process: number of machines to be used；process_id: machine ID 

```
export PYTHONPATH=./
export VIDEO_ID=May
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
```

# Step3. Fit 3DMM
```
export VIDEO_ID=May
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset  --debug --id_mode=global
```

# Step4. Binarize
```
export PYTHONPATH=./
export VIDEO_ID=May
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}
```
You can check `data/binary/videos/May` for generated dataset.
