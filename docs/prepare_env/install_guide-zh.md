# 环境配置
[English Doc](./install_guide.md)

本文档陈述了搭建GeneFace++ Python环境的步骤，我们使用了Conda来管理依赖。

以下配置已在 A100/V100 + CUDA11.7 中进行了验证。


# 1. 安装CUDA
我们使用了CUDA extensions [torch-ngp](https://github.com/ashawkey/torch-ngp)，建议手动从[官方](https://developer.nvidia.com/cuda-toolkit)渠道安装CUDA。我们推荐安装CUDA `11.7`，其他CUDA版本（例如`10.2`）也可能有效。 请确保你的CUDA path(一般是 `/usr/local/cuda`) 指向了你需要的CUDA版本（例如 `/usr/local/cuda-11.7`）. 需要注意的是，我们目前不支持CUDA 12或者更高版本。

# 2. 安装Python依赖
```
cd <GeneFaceRoot>
source <CondaRoot>/bin/activate
conda create -n geneface python=3.9
conda activate geneface
conda install conda-forge::ffmpeg # ffmpeg with libx264 codec to turn images to video

# 我们推荐安装torch2.0.1+cuda11.7. 已经发现 torch=2.1+cuda12.1 会导致 torch-ngp 错误
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# 从源代码安装，需要比较长的时间 (如果遇到各种time-out问题，建议使用代理)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# MMCV安装
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

# 其他依赖项
sudo apt-get install libasound2-dev portaudio19-dev
pip install -r docs/prepare_env/requirements.txt -v

# 构建torch-ngp
bash docs/prepare_env/install_ext.sh 
```

# 3. 准备3DMM模型（BFM2009） 以及其他数据
你可以从这里下载 [Google Drive](https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=drive_link) 或 [BaiduYun Disk](https://pan.baidu.com/s/1-mbPr2_0F0jTU0z169yhyg?pwd=r8ux) (密码 r8ux)。 解压缩后, `BFM` 文件夹中将包含8个文件。 移动这些文件到 `<GeneFaceRoot>/deep_3drecon/BFM/`。该文件夹结构应该如下：
```
deep_3drecon/BFM/
├── 01_MorphableModel.mat
├── BFM_exp_idx.mat
├── BFM_front_idx.mat
├── BFM_model_front.mat
├── Exp_Pca.bin
├── facemodel_info.mat
├── index_mp468_from_mesh35709.npy
├── mediapipe_in_bfm53201.npy
└── std_exp.txt
```