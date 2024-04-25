#!/bin/bash

#SBATCH --j=marry-mae_multiple_1_3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH -w augi3
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o logs/%N_%x_%j.out
#SBTACH -e %x_%j.err

source /data/kayoung/init.sh
conda activate hi-vae

python /data/kayoung/hi-mae/save_missing_data.py \
--dataset_name wine \
--min_remove_count 1 \
--max_remove_count 12 \
--missing_pattern multiple