set -euo pipefail

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=0

length=1024
model_path="inclusionAI/LLaDA2.0-mini"
output_dir="./out"
save_samples=True
eos_early_stop=False
predictive_expert_offload=True
collect_stats=True
task=mbpp_sanitized_llada_moe
do_sample=True

jump_steps_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
block_length=32
max_gpu_experts_per_layer=128
threshold=0.95

echo "${model_path}"

for jump_steps in "${jump_steps_list[@]}"; do
    run_name="jump_${jump_steps}_max_gpu_experts_per_layer_${max_gpu_experts_per_layer}"
    output_path="${output_dir}/${task}/${run_name}"

    mkdir -p "${output_path}"

    echo "Running ${task} with jump_steps=${jump_steps}, threshold=${threshold}"

    python eval_dinfer.py --tasks "${task}" \
        --confirm_run_unsafe_code --model dInfer_eval \
        --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},show_speed=True,save_dir=${output_path},save_samples=${save_samples},eos_early_stop=${eos_early_stop},predictive_expert_offload=${predictive_expert_offload},collect_stats=${collect_stats},max_gpu_experts_per_layer=${max_gpu_experts_per_layer},jump_steps=${jump_steps},do_sample=${do_sample} \
        --output_path "${output_path}" --include_path ./tasks --apply_chat_template
done
