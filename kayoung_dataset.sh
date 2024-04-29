#!/bin/bash

#SBATCH --j=marry-mae_save_missing_data
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=250G
#SBATCH -w augi2
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o logs/%N_%x_%j.out
#SBTACH -e %x_%j.err

source /data/kayoung/init.sh
conda activate hi-vae

python /data/kayoung/hi-mae/save_missing_data.py \
--dataset_name ortho \
--min_remove_count 1 \
--max_remove_count 20 \
--missing_pattern multiple