#!/bin/bash

#SBATCH --job-name=ft-protrans-lora      # Nom du job
#SBATCH --output=protrans-lora.out           # Fichier de sortie standard (stdout)
#SBATCH --error=protrans-lora.err             # Fichier d'erreur (stderr)
#SBATCH --partition=gpuA100x4              # Demander 4 GPU NVIDIA A100
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbtw-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00 

module reset
module load anaconda3_gpu
module list
which python3

export TRANSFORMERS_CACHE=/scratch/bbtw/mbelcaid/thibaut/huggingface_cache/
mkdir -p $TRANSFORMERS_CACHE
export HF_HOME=/scratch/bbtw/mbelcaid/thibaut/huggingface_cache/
mkdir -p $HF_HOME
export TORCH_HOME=/scratch/bbtw/mbelcaid/thibaut/torch_cache/
mkdir -p $TORCH_HOME
export TRITON_CACHE_DIR=/scratch/bbtw/mbelcaid/thibaut/triton_cache/
mkdir -p $TRITON_CACHE_DIR
export XDG_CACHE_HOME=/scratch/bbtw/mbelcaid/thibaut/general_cache/
mkdir -p $XDG_CACHE_HOME

pip install wandb
pip install torch
pip install fair-esm
pip install einops
pip install transformers
pip install peft
pip install biopython
pip install evaluate

cd /scratch/bbtw/mbelcaid/thibaut/protrans/
python3 fine-tune.py
