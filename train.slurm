#!/bin/bash
#SBATCH --job-name=MARL_Variant5
#SBATCH --output=logs/marl_%j.out
#SBATCH --error=logs/marl_%j.err

#SBATCH --partition=debug
#SBATCH --time=24:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

# Single MI300X GPU is plenty for JAX compilation and CTDE training
#SBATCH --gres=gpu:1

# Load server modules
module load python/3.11
module load rocm/6.2.4

# Initialize conda for non-interactive SLURM shell
eval "$(conda shell.bash hook)"
conda activate marl_rocm

# ROCm Path configurations
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export LLVM_PATH=/opt/rocm/llvm
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/amdgcn/bitcode
export DEVICE_LIB_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/amdgcn/bitcode
export PATH=$LLVM_PATH/bin:$PATH

# Optional: JAX-specific XLA tuning for AMD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Ensure logs directory exists for SLURM stdout/stderr
mkdir -p logs

# Run the training script (using train_film.py for Variant 5)
cd ~/Marl/final_marl
python train_film.py
