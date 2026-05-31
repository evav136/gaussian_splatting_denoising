#!/bin/bash
#SBATCH --job-name=nrg_denoise
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Zacetek: $(date)"

# Nalozi PyTorch modul (CUDA 12.1)
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Preverimo da je CUDA vidna
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ni')"

# Pojdi v pravo mapo
cd ~/seminarska_naloga/gaussian_splatting_rendering_denoising/training

# Ustvari mapi za loge in checkpointe ce ne obstajata
mkdir -p logs
mkdir -p checkpoints

# Pozeni trening
python train.py \
    --data    ../training_data \
    --epochs  100 \
    --batch_size 16 \
    --lr      1e-4 \
    --device  cuda \
    --save_dir checkpoints

echo "Konec: $(date)"
