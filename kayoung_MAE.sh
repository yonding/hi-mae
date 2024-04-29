#!/bin/bash

#SBATCH --j=marry-mae_wine
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=50G
#SBATCH -w augi3
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o logs/%N_%x_%j.out
#SBTACH -e %x_%j.err

source /data/kayoung/init.sh
conda activate hi-vae

python /data/kayoung/hi-mae/main.py \
--dataset_name wine