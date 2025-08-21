#!/bin/bash
#SBATCH -J dna
#SBATCH -o /home/alex.tschantz/dna/logs/job.%A.out
#SBATCH -e /home/alex.tschantz/dna/logs/job.%A.err

# ---- request exactly 1 node with 8 GPUs ----
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8              
#SBATCH --cpus-per-task=16        
#SBATCH --time=1-00:00:00

set -euo pipefail

echo "Running on $(hostname)"
source env/bin/activate

# ----------------- JAX/XLA envs -----------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CPP_MIN_LOG_LEVEL=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -u test.py
