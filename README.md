# TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload

## Setup

```bash
pip install -r requirements.txt
```

The code expects a CUDA GPU.

## Run

Run a minimal generation example:

```bash
python main.py
```

Run MBPP evaluation:

```bash
./eval.sh
```

`eval.sh` evaluates `mbpp_sanitized_llada_moe` for multiple `jump_steps` ($\tau$) values and writes outputs under `./out`.

## Acknowledgements

This project builds on [inclusionAI/LLaDA2.0](https://huggingface.co/collections/inclusionAI/llada20).

The evaluation code is adapted from [dInfer](https://github.com/inclusionAI/dInfer).
