#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --priority rse-com6012
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=6G
#SBATCH --mail-user=qfeng10@sheffield.ac.uk
#SBATCH --output=output.%j.test.out
#SBATCH --cpus-per-task=4

module load Anaconda3/2022.10
module load cuDNN/8.0.4.30-CUDA-11.1.1
source activate torch2

python train.py