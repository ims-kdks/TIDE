# TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2605.20179)
[![arXiv](https://img.shields.io/badge/🌊TIDE%20Project%20Page-blue?style=for-the-badge)](https://tide-paper.vercel.app/)

<img width="1847" height="839" alt="figure1-2" src="https://github.com/user-attachments/assets/502f54f0-738d-40e8-8de4-996522f3bfe5" />

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

## Citation
If you find our work useful, please consider citing our paper or giving it a star! ⭐️
```
@article{chen2026tide,
      title={TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload}, 
      author={Zhiben Chen and Youpeng Zhao and Yang Sui and Jun Wang and Yuzhang Shang},
      year={2026},
      eprint={2605.20179},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.20179}, 
}
```
