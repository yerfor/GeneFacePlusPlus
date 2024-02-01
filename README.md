# GeneFace++: Generalized and Stable Real-Time 3D Talking Face Generation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2305.00787)| [![GitHub Stars](https://img.shields.io/github/stars/yerfor/GeneFacePlusPlus)](https://github.com/yerfor/GeneFacePlusPlus) | [![downloads](https://img.shields.io/github/downloads/yerfor/GeneFacePlusPlus/total.svg)](https://github.com/yerfor/GeneFacePlusPlus/releases) | ![visitors](https://visitor-badge.glitch.me/badge?page_id=yerfor/GeneFacePlusPlus)

[中文文档](./README-zh.md)

This is the official implementation of GeneFace++ [Paper](https://arxiv.org/abs/2301.13430) with Pytorch，which enables high lip-sync, high video-reality and high system-efficiency 3D talking face generation. You can visit our [Demo Page](https://genefaceplusplus.github.io/) to watch demo videos and learn more details.

<p align="center">
    <br>
    <img src="assets/geneface++.png" width="100%"/>
    <br>
</p>

## Quick Start!
We provide a quick guide to try GeneFace++ here.

- Step 1: Follow the steps in `docs/prepare_env/install_guide.md`, create a new python environment named `geneface`, and download 3DMM features we need.

- Step 2: Download pre-processed dataset of May([Google Drive](https://drive.google.com/drive/folders/1SwZ7uRa5ESzzq_Cd21-Lk5heAZxa9oZO?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/1zR914cBQcGOAl4o4XInBNw?pwd=e1a3) with password e1a3), and place it here `data/binary/videos/May/trainval_dataset.npy`

- Step 3: Download pre-trained audio-to-motino model ([Google Drive](https://drive.google.com/drive/folders/1FqvNbQgOSkvVO8i-vCDJmKM4ppPZjUpL?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/19UZxMrO-ZvkOeYzUkOKsTQ?pwd=9cqp) with password 9cqp) and motion-to-video which is specific to May, and unzip them to `./checkpoints/`

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
python inference/genefacepp_infer.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=may_demo.mp4
```
Or you can play with our Gradio WebUI: 
```bash
python inference/app_genefacepp.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso
```

## Train GeneFace++ with your own videos
Please refer to details in  `docs/process_data` and `docs/train_and_infer`.

## ToDo
- [x] **Release Inference Code of Audio2Motion and Motion2Video.**
- [x] **Release Pre-trained weights of Audio2Motion and Motion2Video.**
- [x] **Release Training Code of Motino2Video Renderer.**
- [x] **Release Gradio Demo.**
- [ ] **Release Training Code of Audio2Motion and Post-Net.**

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
