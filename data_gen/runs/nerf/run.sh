# usage: CUDA_VISIBLE_DEVICES=0 bash data_gen/runs/nerf/run.sh <VIDEO_ID>
# please place video to data/raw/videos/${VIDEO_ID}.mp4 
VIDEO_ID=$1
echo Processing $VIDEO_ID

echo Resizing the video to 512x512
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -y data/raw/videos/${VIDEO_ID}_512.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4
echo Done
echo The old video is moved to data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4

echo mkdir -p data/processed/videos/${VIDEO_ID}
mkdir -p data/processed/videos/${VIDEO_ID}
echo Done

# extract audio file from the training video
echo ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 -v quiet -y data/processed/videos/${VIDEO_ID}/aud.wav 
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 -v quiet -y data/processed/videos/${VIDEO_ID}/aud.wav 
echo Done

# extract hubert_mel_f0 from audio
echo python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
echo python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}
echo Done

# extract segment images
echo mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
echo ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 -v quiet data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 -v quiet data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
echo Done

echo python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background
echo Done

echo python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
echo Done

pkill -f void*
echo python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset --debug --id_mode=global
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset --debug --id_mode=global
echo Done

echo python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}
echo Done