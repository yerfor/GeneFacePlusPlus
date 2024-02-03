## 温馨提示：第一次执行可以先一步步跑完下面的命令行，把环境跑通后，之后可以直接运行同目录的`bash run.sh ${VIDEO_ID}`，一键完成下面的所有步骤。
[English Doc](./guide.md)

# 步骤0. 将视频Crop到512x512分辨率，25FPS，确保每一帧都有目标人脸
```
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/${VIDEO_ID}_512.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4
```
# 步骤1: 提取音频特征, 如mel, f0, hubuert
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./
export VIDEO_ID=May
mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav 
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}
```

# 步骤2. 提取图片
```
export PYTHONPATH=./
export VIDEO_ID=May
export CUDA_VISIBLE_DEVICES=0
mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background
```

# 步骤3. 提取lm2d_mediapipe
### 提取2D landmark用于之后Fit 3DMM
### num_workers是本机上的CPU worker数量；total_process是使用的机器数；process_id是本机的编号

```
export PYTHONPATH=./
export VIDEO_ID=May
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
```

# 步骤3. Fit 3DMM
```
export PYTHONPATH=./
export VIDEO_ID=May
export CUDA_VISIBLE_DEVICES=0
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset  --debug --id_mode=global
```

# 步骤4. Binarize（将数据打包）
```
export PYTHONPATH=./
export VIDEO_ID=May
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}
```
可以看到在`data/binary/videos/May`目录下得到了数据集。
