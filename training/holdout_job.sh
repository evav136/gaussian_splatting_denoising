#!/bin/bash
#SBATCH --job-name=holdout_eval
#SBATCH --output=holdout_%j.log
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd ~/nrg/seminarska_naloga/gaussian_splatting_rendering_denoising/training

python -u holdout_eval.py \
    --data    ../test_holdout \
    --checkpoint checkpoints/best.pt \
    --device  cuda \
    --out     holdout_results.csv
