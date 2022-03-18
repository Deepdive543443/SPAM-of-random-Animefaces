# SPAM-of-random-Animefaces
A ProGAN implementation trained on anime face dataset and a Docker with trained generator hosted on Raspberry pi(64 bit Ubuntu Server 20.04.4 LTS).

The dataset is provided by https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset . 

Start training from 8x8 resolution, finished at 128x128 resolution(step 5).

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
![9ZDM_RMYF96HCRR`C57DQF4](https://user-images.githubusercontent.com/83911295/159079410-6de8c218-06ef-489b-a002-7dc336cb4e44.png)
![IMG_0793](https://user-images.githubusercontent.com/83911295/159079426-474360a5-33f4-43a8-b6ba-8946c0169d41.JPG)

