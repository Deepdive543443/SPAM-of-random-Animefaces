import torch
from math import log2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Path
DATASET = 'Anime256'
TRANSFORM = True

#Model
SAVE_MODEL = True
LOAD_MODEL = True
FLOAT16 = True

#Dataset
TEST_SPLIT = None#240
SHUFFLE = True
FACTORS = [1, 1, 1, 2, 2, 2, 2, 2]#[1, 1, 1, 1, 1, 2, 2, 2]  #4, 8, 16, 32, 64, 128, 256, 512, 1024
BATCH_SIZE = [64, 64, 64, 32, 16, 4, 3, 2, 2] #4, 8, 16, 32, 64, 128, 256, 512, 1024
START_IMAGESIZE = 8
TARGET_IMAGESIZE = 256


NUM_WORKERS = 0
EPOCH = 20

#Hyper parameters
LEARNING_RATE = 1e-3
NOISE_DIM = 512
ALPHA = 1e-5
BETA1 = 0
BETA2 = 0.99
LAMBDA_GP = 10


