# Get Audio2Motion Model
[中文文档](./guide-zh.md)

You can download our pre-trained Audio-to-Motion model (pretrained on voxceleb2, a 2000-hour lip reading dataset) in this [Google Drive](https://drive.google.com/drive/folders/1FqvNbQgOSkvVO8i-vCDJmKM4ppPZjUpL?usp=sharing) or in this [BaiduYun Disk](https://pan.baidu.com/s/19UZxMrO-ZvkOeYzUkOKsTQ?pwd=9cqp) (password 9cqp)

Place the model in the directory `checkpoints/audio2motion_vae`.

# Train Motion2Video Renderer
We suppose you have prepared the dataset following `docs/prepare_data/guide.md` and you can find a binarized `.npy` file in `data/binary/videos/{Video_ID}/trainval_dataset.npy` (Video_ID is your training video name, here we use `May` provided in this repo as an example.)

```
# Train the Head NeRF
# model and tensorboard will be saved at `checkpoints/<exp_name>`
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/May/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/may_head --reset

# Train the Torso NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/May/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/may_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/may_head --reset
```
You can also download our pre-trained renderer in this [Google Drive](https://drive.google.com/drive/folders/1SwZ7uRa5ESzzq_Cd21-Lk5heAZxa9oZO?usp=sharing) or in this [BaiduYun Disk](https://pan.baidu.com/s/1zR914cBQcGOAl4o4XInBNw?pwd=e1a3) password e1a3, place the model in the directory `checkpoints/motion2video_nerf`.

## How to train on your own video: 
Suppose you have a video named `{Video_ID}.mp4`
- Step1: crop your video to 512x512 and 25fps, then place it into `data/raw/videos/{Video_ID}.mp4`
- Step2: copy a config folder `egs/datasets/{Video_ID}` following `egs/datasets/May`, remind to change `video: May` to `video: {Video_ID}`
- Step3: Process the video following `docs/process_data/guide.md`, then you can get a `data/binary/videos/{Video_ID}/trainval_dataset.npy`
- Step4: Use the commandlines above to train the NeRF.

# Inference
## commandline inference
```
# we provide a inference script.
CUDA_VISIBLE_DEVICES=0  python inference/genefacepp_infer.py --head_ckpt= --torso_ckpt=motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav

# --debug option could visualize intermediate steps during inference
CUDA_VISIBLE_DEVICES=0  python inference/genefacepp_infer.py --head_ckpt= --torso_ckpt=motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --debug
```

## WebGUI Demo 
```
CUDA_VISIBLE_DEVICES=0 python inference/app_genefacepp.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=motion2video_nerf/may_torso
```

<p align="center">
    <br>
    <img src="../../assets/webui.png" width="100%"/>
    <br>
</p>
