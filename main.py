import torch
import torch.nn.functional as F
from model.modeling_llada2_moe import LLaDA2MoeModelLM
from transformers import AutoTokenizer

model_path = "inclusionAI/LLaDA2.0-mini"
device = "cuda:0"
model = LLaDA2MoeModelLM.from_pretrained(
    model_path, trust_remote_code=True, dtype=torch.bfloat16
).to(device).eval()

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
    inputs=input_ids,
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
