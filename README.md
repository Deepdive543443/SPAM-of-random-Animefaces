# SPAM-of-random-Animefaces
A ProGAN implementation trained on anime face dataset

The dataset is provided by https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset . 

# Demo
- NCNN demo:

- Web demo (Docker required)
  Clone this repo, then run
```
cd Docker
docker build --no-cache -t TWNE_MINI .
docker run -it --name anime_cans_container -p 5000:5000 TWNE_MINI
```
Open the website by: http://localhost:8000

BEWARE OF CURSED OUTPUT


![image](https://github.com/Deepdive543443/SPAM-of-random-Animefaces/assets/83911295/3a4e0e3e-d0db-4af6-b995-f3138bb35ee3)
![image](https://github.com/Deepdive543443/SPAM-of-random-Animefaces/assets/83911295/0c9a8ef0-6c7e-40c5-8163-e8dfc4d3f378)


