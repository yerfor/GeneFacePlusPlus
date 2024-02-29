# GeneFace++: Generalized and Stable Real-Time 3D Talking Face Generation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2305.00787)| [![GitHub Stars](https://img.shields.io/github/stars/yerfor/GeneFacePlusPlus)](https://github.com/yerfor/GeneFacePlusPlus) | [中文文档](./README-zh.md)

This is the official implementation of GeneFace++ [Paper](https://arxiv.org/abs/2301.13430) with Pytorch, which enables high lip-sync, high video-reality and high system-efficiency 3D talking face generation. You can visit our [Demo Page](https://genefaceplusplus.github.io/) to watch demo videos and learn more details.

<p align="center">
    <br>
    <img src="assets/geneface++.png" width="100%"/>
    <br>
</p>

# Note
The eye blink control is an experimental feature, and we are currently working on improving its robustness. Thanks for your patience.

## You may also interested in 
- We release Real3D-portrait (ICLR 2024 Spotlight), ([https://github.com/yerfor/Real3DPortrait](https://github.com/yerfor/Real3DPortrait)), a NeRF-based one-shot talking face system. Only upload one image and enjoy realistic talking face!

## Quick Start!
We provide a guide for a quick start in GeneFace++.

- Step 1: Follow the steps in `docs/prepare_env/install_guide.md`, create a new python environment named `geneface`, and download 3DMM files into `deep_3drecib/BFM`.

- Step 2: Download pre-processed dataset of May([Google Drive](https://drive.google.com/drive/folders/1SwZ7uRa5ESzzq_Cd21-Lk5heAZxa9oZO?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/1U_FalVoxgb9sAb9FD1cZEw?pwd=98n4) with password 98n4), and place it here `data/binary/videos/May/trainval_dataset.npy`

- Step 3: Download pre-trained audio-to-motino model `audio2motion_vae.zip` ([Google Drive](https://drive.google.com/drive/folders/1M6CQH52lG_yZj7oCMaepn3Qsvb-8W2pT?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/19UZxMrO-ZvkOeYzUkOKsTQ?pwd=9cqp) with password 9cqp) and motion-to-video checkpoint `motion2video_nerf.zip`, which is specific to May (in this [Google Drive](https://drive.google.com/drive/folders/1M6CQH52lG_yZj7oCMaepn3Qsvb-8W2pT?usp=sharing) or in this[BaiduYun Disk](https://pan.baidu.com/s/1U_FalVoxgb9sAb9FD1cZEw?pwd=98n4) with password 98n4), and unzip them to `./checkpoints/`

After these steps，your directories `checkpoints` and `data` should be like this：

```
> checkpoints
    > audio2motion_vae
    > motion2video_nerf
        > may_head
        > may_torso
> data
    > binary
        > videos
            > May
                trainval_dataset.npy
```

- Step 4: activate `geneface` Python environment, and execute: 
```bash
export PYTHONPATH=./
python inference/genefacepp_infer.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=may_demo.mp4
```
Or you can play with our Gradio WebUI: 
```bash
export PYTHONPATH=./
python inference/app_genefacepp.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso
```
Or use our provided [Google Colab](https://colab.research.google.com/github/yerfor/GeneFacePlusPlus/blob/main/inference/genefacepp_demo.ipynb) and run all cells in it.

## Train GeneFace++ with your own videos
Please refer to details in  `docs/process_data` and `docs/train_and_infer`.

Below are answers to frequently asked questions when training GeneFace++ on custom videos:
- Please make sure that the head segment occupies a relatively large region in the video (e.g., similar to the provided `May.mp4`). Or you need to hand-crop your training video. [issue](https://github.com/yerfor/GeneFacePlusPlus/issues/30)
- Make sure that the talking person appears in every frame of the video, otherwise the data preprocessing pipeline may be failed.
- We only tested our code on Liunx (Ubuntu/CentOS). It is welcome that someone who are willing to share their installation guide on Windows/MacOS.


## ToDo
- [x] **Release Inference Code of Audio2Motion and Motion2Video.**
- [x] **Release Pre-trained weights of Audio2Motion and Motion2Video.**
- [x] **Release Training Code of Motino2Video Renderer.**
- [x] **Release Gradio Demo.**
- [x] **Release Google Colab.**
- [ ] **Release Training Code of Audio2Motion and Post-Net. (Maybe 2024.06.01) **

## Citation
If you found this repo helpful to your work, please consider cite us:
```
@article{ye2023geneface,
  title={GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis},
  author={Ye, Zhenhui and Jiang, Ziyue and Ren, Yi and Liu, Jinglin and He, Jinzheng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.13430},
  year={2023}
}
@article{ye2023geneface++,
  title={GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation},
  author={Ye, Zhenhui and He, Jinzheng and Jiang, Ziyue and Huang, Rongjie and Huang, Jiawei and Liu, Jinglin and Ren, Yi and Yin, Xiang and Ma, Zejun and Zhao, Zhou},
  journal={arXiv preprint arXiv:2305.00787},
  year={2023}
}
```
