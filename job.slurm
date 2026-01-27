#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bffh-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=job
#SBATCH --time=03:00:00      # hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=1

set -euo pipefail

# print node status
echo "job is starting on `hostname`"
nvidia-smi

# activate conda env
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate sglang

# run single prompt
srun python main.py

# # start sglang server
# echo "starting sglang server"
# srun python -m sglang.launch_server \
#   --model-path inclusionAI/LLaDA2.0-mini \
#   --trust-remote-code \
#   --dllm-algorithm LowConfidence \
#   --dllm-algorithm-config ./config.yaml \
#   --host 127.0.0.1 \
#   --port 30000 \
#   --log-level info