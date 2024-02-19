#Docker installation notes

It would be much easier to install the application in a docker container.
Here the basic steps to run the application in Docker.

## Build the base image, and the running image

```shell
docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

docker buildx build -t ubuntu22.04-cu118-conda:torch2.0.1-py39 -f Dockerfile.cu118.torch2.0.1.py39   .

docker buildx build -t genfaceplus:0219 -f Dockerfile.genface .
```

## Preparing the model checkpoints

Notes:
* https://github.com/yerfor/GeneFacePlusPlus/blob/main/docs/prepare_env/install_guide-zh.md
* https://github.com/yerfor/GeneFacePlusPlus/blob/main/README-zh.md
* Put trainval_dataset.npy to data/binary/videos/May/trainval_dataset.npy
* Unzip audio2motion_vae.zip, motion2video_nerf.zip to checkpoints directory.
* Copy 8 files from BFM2009 to deep_3drecon/BFM/

```shell
# Assume that all model files has been downloaded into ~/GeneFace++Models
mkdir -p data/binary/videos/May/
cp ~/GeneFace++Models/trainval_dataset.npy data/binary/videos/May/
(cd checkpoints && unzip ~/GeneFace++Models/audio2motion_vae.zip)
(cd checkpoints && unzip ~/GeneFace++Models/motion2video_nerf.zip)
cp ~/GeneFace++Models/BFM2009/* deep_3drecon/BFM/
```

## Start a docker container
```shell
docker run -it --name geneface -p 7869:7860 --gpus all -v ~/.cache:/root/.cache -v ~/workspace/GeneFacePlusPlus:/data/geneface/  genfaceplus:0219 /bin/bash
```

## Activate the inference environment including download other necessary models

Run the following commands in the docker container.

```shell
source ~/.bashrc
conda activate pytorch

cd /data/geneface/
export PYTHONPATH=./

export HF_ENDPOINT=https://hf-mirror.com
python inference/genefacepp_infer.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=may_demo.mp4

```

## Finally, start the Gradio demo app
Run the following command in the docker container.
```shell
export PYTHONPATH=./

python inference/app_genefacepp.py --server 0.0.0.0 --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso
```

And after that, you can access the server at http://127.0.0.1:7869

And you can quit the docker container at anytime.

## Restart the container, and start the service

After the container stopped, you can restart the container to run gradio service at anytime.

``` bash
docker start geneface
docker exec -it geneface /bin/bash

source ~/.bashrc
conda activate pytorch

cd /data/geneface/
export PYTHONPATH=./

python inference/app_genefacepp.py --server 0.0.0.0 --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso
```
