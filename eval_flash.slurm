#!/bin/bash
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuH200x8      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8 gpuH200x8
#SBATCH --account=bfgy-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=flash_96
#SBATCH --output=out/%x-%j.out
#SBATCH --time=48:00:00      # hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=1

set -euo pipefail

# print node status
echo "job is starting on $(hostname)"
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

# Pin OpenMP threads to physical cores to avoid migration overhead across the
# many small parallel regions in CPU-offloaded MoE expert matmuls.
export OMP_PROC_BIND=close

length=256
block_length=32
model_path="inclusionAI/LLaDA2.0-flash"
threshold=0.95
output_dir="./out"
save_samples=True
eos_early_stop=True
predictive_expert_offload=True
collect_stats=True
limit=50
task=mbpp_sanitized_llada_moe
do_sample=True

# Edit these two arrays to sweep the combinations you want.
# jump_steps_list=(1 2 3 4 5 6 7 8 9 10)
# jump_steps_list=(10 11 12 13 14 15 16 17 18 19 20)
jump_steps_list=(20 21 22 23 24 25 26 27 28 29 30 31)
max_gpu_experts_per_layer_list=(96)

# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada moe use tasks gsm8k_llada_moe mbpp_sanitized_llada_moe

echo "${model_path}"

for max_gpu_experts_per_layer in "${max_gpu_experts_per_layer_list[@]}"; do
  for jump_steps in "${jump_steps_list[@]}"; do
    run_name="jump_${jump_steps}_max_gpu_experts_${max_gpu_experts_per_layer}"
    output_path="${output_dir}/${task}/${run_name}"

    mkdir -p "${output_path}"

    echo "Running ${task} with jump_steps=${jump_steps}, max_gpu_experts_per_layer=${max_gpu_experts_per_layer}"

    python eval_dinfer.py --tasks "${task}" \
      --confirm_run_unsafe_code --model dInfer_eval \
      --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},show_speed=True,save_dir=${output_path},save_samples=${save_samples},eos_early_stop=${eos_early_stop},predictive_expert_offload=${predictive_expert_offload},collect_stats=${collect_stats},max_gpu_experts_per_layer=${max_gpu_experts_per_layer},jump_steps=${jump_steps},do_sample=${do_sample} \
      --limit "${limit}" \
      --output_path "${output_path}" --include_path ./tasks --apply_chat_template
  done
done
