#!/bin/bash
#SBATCH --job-name=nrg_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Zacetek: $(date)"

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

cd ~/nrg/seminarska_naloga/gaussian_splatting_rendering_denoising/training

mkdir -p logs

python neural_eval.py \
    --data      ../training_data \
    --checkpoint checkpoints/best.pt \
    --device    cuda \
    --csv       neural_results.csv

echo "Konec: $(date)"
