## What this project studies

The core hypothesis is that MoE routing decisions are stable enough across denoising steps that we do not need to refresh expert placement on every denoising step. The practical baseline in the current code is:

1. keep non-expert modules on GPU,
2. keep only a fixed per-layer budget of experts resident on GPU, and
3. move the most useful experts between CPU and GPU just in time before expert computation.

When `jump_steps == 1`, the system refreshes expert placement at every step. In the current setting, that is the strongest reference policy, because CPU expert execution is much slower than GPU expert execution, so it is usually worth paying the transfer cost to move the right experts onto GPU before computing tokens.

The key observation is that routing is stable enough that we do not need to pay that transfer cost every step. When `jump_steps > 1`, the system refreshes expert placement less often, which reduces CPU<->GPU bandwidth usage while still maintaining a high GPU expert hit rate on the skipped steps.

This idea is conceptually inspired by dKV-Cache, but it targets MoE routing rather than attention KV states. dKV-Cache studies step-to-step stability in diffusion-model internals; this project applies the same general intuition to expert selection.

## Current implementation

The current implementation is centered in `./model/modeling_llada2_moe.py`.

- `enable_predictive_expert_offload(...)` moves all non-expert modules to CUDA and computes a per-layer GPU expert budget from available VRAM.
- Each sparse MoE layer then keeps only that many experts on GPU; the remaining experts stay on CPU.
- During generation, the router runs before expert execution. On steps where `step % jump_steps == 0`, `jit_moe_infer(...)` counts how often each expert is selected in the current block/step, promotes the most-used experts to GPU up to the budget, demotes a matched number of stale experts, and then executes the expert MLPs.
- When `jump_steps == 1`, this happens at every denoising step and serves as the full-refresh reference policy.
- On steps skipped by `jump_steps > 1`, inference reuses the current expert placement without refreshing it. The method works only if routing remains stable enough that most needed experts are still already on GPU.
- Optional statistics track CPU/GPU token counts, CPU/GPU expert calls, promotions, and demotions.

One important nuance: the implementation is not purely predicting future experts from the previous step. It refreshes expert placement from the current router output whenever a refresh step is allowed, then relies on cross-step routing stability when `jump_steps > 1`.

## Current experiment setup

1. Model: `inclusionAI/LLaDA2.0-mini`.
2. Initialization: move non-expert modules to GPU and keep at most `max_gpu_experts_per_layer=128` experts per layer on GPU; the rest remain on CPU.
3. Generation: use block diffusion decoding, typically with `block_length=32` and `steps=32`.
4. Offloading policy: refresh expert placement every `jump_steps` denoising steps. `jump_steps=1` means every step and is the full-refresh baseline. `jump_steps=2` means every other step, `jump_steps=4` means every fourth step, and so on.
5. Evaluation: `./eval_dinfer.py` runs an `lm_eval` harness on `mbpp_sanitized`, so the reported "accuracy" is more precisely `pass@1` on sanitized MBPP code-generation tasks.

Related files:

- `./main.py`: minimal generation example with predictive expert offloading enabled.
- `./eval_dinfer.py`: evaluation harness adapted from dInfer-style evaluation code.
- `./eval_dinfer.slurm`: current cluster launch script for MBPP experiments.
- `./todo.md`: current implementation follow-ups.
