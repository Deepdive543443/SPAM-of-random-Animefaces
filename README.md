# SPAM-of-random-Animefaces
A ProGAN implementation trained on anime face dataset and a Docker with trained generator hosted on Raspberry pi(64 bit Ubuntu Server 20.04.4 LTS).

Start training from 8x8 resolution, finished at 128x128 resolution(step 5).

Randomly generate anime faces in each 20-30 seconds on Raspberry pi.

The dataset is provided by https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset . 


# Install
1. Download and attach trained weights and wheel of PyTorch and Torchvision compiled for Ubuntu 20.04 to "stream" folder
https://drive.google.com/drive/folders/1pI_oUTL1QL2v20clE2jFICu5iUqBsHUV?usp=sharing

Or download wheels from Q-engineering's page
https://github.com/Qengineering/PyTorch-Raspberry-Pi-64-OS

2. Run:
```
docker build --no-cache -t anime_cans .
docker run -it --name anime_cans_container -p 5000:5000 anime_cans
```
to build and start the container

# Preview

http://120.24.218.136/
![image](https://user-images.githubusercontent.com/83911295/180642653-190857d6-65a6-4c46-86f8-3aad38f510c9.png)


