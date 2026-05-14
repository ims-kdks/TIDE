'''
Code adopted from https://github.com/inclusionAI/dInfer
This file is inspired by the code from https://github.com/NVlabs/Fast-dLLM
'''
import torch
import random
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm, trange
import random
import numpy as np
import json
import time
import datasets
import json
import time
import datasets
import os
from transformers import AutoTokenizer, AutoConfig
import torch.multiprocessing as mp
from multiprocessing import Process
from lm_eval.api.model import LM
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from model.modeling_llada2_moe import LLaDA2MoeModelLM
from dataclasses import dataclass

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
datasets.config.DOWNLOAD_TIMEOUT = 180 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

bucket_size = 32
used_buckets = []


@dataclass
class EvalConfig:
    model_name: str = ''
    batch_size: int = 1
    gen_len: int = 1024
    block_length: int = 64
    threshold: float = 0.9
    use_tp: bool = False
    save_path: str = ''
    config: int = 0
    port_offset: int = 0
    all_input_ids = None
    padded_gen_lens = None
    use_shift: bool = False
    save_dir: str = './res'
    save_samples: bool = False
    speed_path: str = ''


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("dInfer_eval")
class DInferEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        device="cuda",
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=True,
        gen_length=1024,
        block_length=32,
        save_dir=None,
        show_speed=False,
        threshold: float=0.9,
        use_shift = False,
        save_samples = False,
        eos_early_stop = False,
        collect_stats = False,
        max_gpu_experts_per_layer = 128,
        jump_steps = 1,
        temperature = 0.0,
        do_sample = True,
        **kwargs
    ):

        super().__init__()
        
        self.model_path = model_path
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.gen_length = gen_length
        self.block_length = block_length
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.threshold = threshold
        self.kwargs = kwargs
        self.use_shift = use_shift
        self.save_samples = save_samples
        self.eos_early_stop = eos_early_stop
        self.do_sample = do_sample
        self.temperature = temperature

        self.mask_id = 156895
        self.eos_id = 156892
        
        self.device= torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # load model
        self.model = LLaDA2MoeModelLM.from_pretrained(model_path, dtype="bfloat16", config=config).eval()
        # self.model = self.model.to(self.device)
        offload_info = self.model.enable_predictive_expert_offload(
            max_gpu_experts_per_layer=max_gpu_experts_per_layer,
            collect_stats=collect_stats,
            jump_steps=jump_steps,
        )
        print(offload_info)


    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def tokenizer_name(self) -> str:
        return self.model_path
    
    def apply_chat_template(self, chat_history, **kwargs) -> str:
        if "tokenize" not in kwargs:
            kwargs["tokenize"] = False
        return self.tokenizer.apply_chat_template(chat_history, **kwargs)

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, f'rank_{self.rank}.jsonl')
            print(f"save_path: {self.save_path}")
            self.speed_path = os.path.join(self.save_dir, f'results.txt')
        

        def get_bucket_length(length):
            bucket_length = bucket_size*(length//bucket_size)
            if bucket_length not in used_buckets:
                used_buckets.append(bucket_length)
            return bucket_length

        def load_inputs(prompts, tokenizer):
            all_input_ids = []
            for id, prompt in enumerate(prompts):
                input_ids = tokenizer(prompt.args[0])['input_ids']
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                all_input_ids.append(input_ids)
            return all_input_ids

        def cal_bucket_len(gen_len, all_input_ids):
            max_prompt_length = 0
            padded_gen_lens = []

            for i in range(len(all_input_ids)):
                input_ids = all_input_ids[i]
                if input_ids.shape[1] > max_prompt_length:
                    max_prompt_length = input_ids.shape[1]
                padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
                padded_gen_lens.append(padded_length - input_ids.shape[1])
            return padded_gen_lens


        all_input_ids = load_inputs(requests, self.tokenizer)
        padded_gen_lens = cal_bucket_len(self.gen_length, all_input_ids)
        
        answers = []
        outputs = []
        tpss = []
        total_token = 0
        token_numbers = []
        total_stats = []
        total_time = 0
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            input_ids = all_input_ids[i]
            padded_gen_len = padded_gen_lens[i]
            inner_start = time.perf_counter()
            input_ids = input_ids.to(self.device)
            out, stats = self.model.generate(
                input_ids,
                gen_length=padded_gen_len,
                block_length=self.block_length,
                eos_early_stop=self.eos_early_stop,
                temperature=self.temperature,
                do_sample=self.do_sample,
            )
            inner_stop = time.perf_counter()
            sample_time = inner_stop - inner_start
            total_time += sample_time
            outputs.append(out)
            answer = (self.tokenizer.decode(out[0], skip_special_tokens=True))
            answers.append(answer)
            token_number = out.numel()
            token_numbers.append(token_number)
            tps = token_number/sample_time
            tpss.append(tps)
            total_token += token_number

            # stats
            total_stats.append({
                "demotions": [step["offload_stats"]["demotions"] for step in stats],
                "promotions": [step["offload_stats"]["promotions"] for step in stats],
                "cpu_tokens": [step["offload_stats"]["cpu_tokens"] for step in stats],
                "gpu_tokens": [step["offload_stats"]["gpu_tokens"] for step in stats],
                "cpu_calls": [step["offload_stats"]["cpu_calls"] for step in stats],
                "gpu_calls": [step["offload_stats"]["gpu_calls"] for step in stats],
                "gpu_compute_time": [step["offload_stats"]["gpu_compute_time"] for step in stats],
                "cpu_compute_time": [step["offload_stats"]["cpu_compute_time"] for step in stats],
                "gpu_tokens_move_time": [step["offload_stats"]["gpu_tokens_move_time"] for step in stats],
                "cpu_tokens_move_time": [step["offload_stats"]["cpu_tokens_move_time"] for step in stats],
                "experts_move_time": [step["offload_stats"]["experts_move_time"] for step in stats],
                "transferred_tokens": [step["transferred_tokens"] for step in stats],
                "elapsed_ms": [step["elapsed_ms"] for step in stats],
                "step_count": [len(stats)],
            })

        if total_stats:
            sum_stats = {}
            for req_stats in total_stats:
                for key, value in req_stats.items():
                    sum_stats[key] = sum_stats.get(key, 0) + sum(value)
            print(
                f"transferred_tokens={sum_stats['transferred_tokens']}|"
                f"elapsed_ms={sum_stats['elapsed_ms']}|"
                f"hit_ratio={sum_stats['gpu_tokens'] / (sum_stats['gpu_tokens'] + sum_stats['cpu_tokens'])}|"
                f"gpu_tokens_per_call={sum_stats['gpu_tokens'] / sum_stats['gpu_calls']}|"
                f"cpu_tokens_per_call={sum_stats['cpu_tokens'] / sum_stats['cpu_calls']}|"
                f"promotion_utilization={sum_stats['gpu_tokens'] / sum_stats['promotions']}|"
                f"gpu_compute_time={sum_stats['gpu_compute_time'] / sum_stats['step_count']}|"
                f"cpu_compute_time={sum_stats['cpu_compute_time'] / sum_stats['step_count']}|"
                f"cpu_tokens_move_time={sum_stats['cpu_tokens_move_time'] / sum_stats['step_count']}|"
                f"experts_move_time={sum_stats['experts_move_time'] / sum_stats['step_count']}|"
                f"promotions={sum_stats['promotions'] / sum_stats['step_count']}|"
                f"cpu_tokens={sum_stats['cpu_tokens'] / sum_stats['step_count']}|"
            )

        print(f'Time: {total_time}, TPS: {np.mean(tpss)}, {total_token=}, step_count: {sum_stats["step_count"]}, step/sec: {sum_stats["step_count"] / total_time}')

        if self.show_speed and self.save_dir is not None:
            with open (self.save_dir+f'/rank{self.rank}_results.jsonl', 'w', encoding='utf-8') as file:
                data={'rank':f'rank{self.rank}',
                    'tokens per second': np.mean(tpss),
                    'average generated length': total_token / len(all_input_ids)
                    }
                file.write(json.dumps(data, ensure_ascii=False) + '\n')
        return answers


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
