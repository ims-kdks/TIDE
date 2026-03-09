import torch
from model.modeling_llada2_moe import LLaDA2MoeModelLM
from transformers import AutoTokenizer

model_path = "inclusionAI/LLaDA2.0-mini"
base_device = "cuda:0"
model: LLaDA2MoeModelLM = LLaDA2MoeModelLM.from_pretrained(
    model_path, trust_remote_code=True, dtype=torch.bfloat16
).eval()
offload_info = model.enable_predictive_expert_offload(
    predictive_expert_offload=True,
    collect_stats=True,
    max_gpu_experts_per_layer=128,
    jump_steps=8,
)
print(offload_info)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "Write a brief introduction of the great wall"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
# input_ids = tokenizer.encode([prompt])
generated_tokens = model.generate(
    input_ids=input_ids,
    eos_early_stop=True,
    gen_length=1024,
    block_length=32,
    steps=32,
    temperature=0.0,
)
generated_answer = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
)
print(generated_answer)
