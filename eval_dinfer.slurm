#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8 gpuH200x8
#SBATCH --account=bffh-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=eval_dinfer
#SBATCH --output=out/%x-%j.out
#SBATCH --time=40:00:00      # hh:mm:ss for the job
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

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=0

length=1024 # generate length
block_length=32 # block length
model_path='inclusionAI/LLaDA2.0-mini'  # your model path
threshold=0.95 # threshold for parallel decoding
ouput_dir='./out' # your custom output directory
save_samples=True # save samples
eos_early_stop=True
predictive_expert_offload=True
collect_stats=False
max_gpu_experts_per_layer=128
jump_steps=1
task=mbpp_sanitized_llada_moe
# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada moe use tasks gsm8k_llada_moe mbpp_sanitized_llada_moe
output_path=${ouput_dir}/${task}

python eval_dinfer.py --tasks ${task} \
--confirm_run_unsafe_code --model dInfer_eval \
--model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},show_speed=True,save_dir=${output_path},save_samples=${save_samples},eos_early_stop=${eos_early_stop},predictive_expert_offload=${predictive_expert_offload},collect_stats=${collect_stats},max_gpu_experts_per_layer=${max_gpu_experts_per_layer},jump_steps=${jump_steps} \
--output_path ${output_path} --include_path ./tasks --apply_chat_template
