"""
测试模型的速度，包括prefill、decode、throughput、memory footprint
使用的prompt是DirectRequest的前5个prompt，重复三次
注意要放在HarmBench的根目录下运行
时间的单位是秒
"""
import transformers
import json
import argparse
import os
import time
from tqdm import tqdm 
import torch

# Ensure project root in path
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
sys.path.insert(0, PROJECT_ROOT)

from baselines import get_template, load_model_and_tokenizer
from functools import partial
import yaml


# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SLM efficiency: memory, prefill, decode, throughput")
    parser.add_argument("--model_name", type=str,
                        help="The name of the model in the models config file")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--test_cases_path", type=str, default="/data2/zwh/HarmBench/results_full_70/DirectRequest/default/test_cases/test_cases.json",
                        help="The path to the test cases file containing prompts (JSON)")
    parser.add_argument("--prompt_num", type=int, default=5,
                        help="The number of prompts to test")
    parser.add_argument("--save_path", type=str, default="./cost_analysis/slm_speed_metrics",
                        help="Directory path to save results JSON as <model_name>.json")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens to generate for decode timing")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for timing (prefill/decode)")
    parser.add_argument("--repeat_times", type=int, default=3,
                        help="Repeat count for averaging metrics")
    parser.add_argument("--mem_seq_len", type=int, default=512,
                        help="Sequence length used for memory footprint measurement")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results JSON if present")
    args = parser.parse_args()
    return args


def load_prompts(test_cases_path: str, prompt_num: int):
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Accept formats: list[str], list[dict{"test_case": str}], or dict[str, list[str]]
    prompts = []
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict) and 'test_case' in data[0]:
            prompts = [d['test_case'] for d in data]
        else:
            prompts = [str(x) for x in data]
    elif isinstance(data, dict):
        for _k, vals in data.items():
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, dict) and 'test_case' in v:
                        prompts.append(v['test_case'])
                    else:
                        prompts.append(str(v))
    else:
        raise ValueError("Unsupported prompts format in test_cases_path; expected JSON list or dict")
    # Filter empties
    prompts = [p for p in prompts if isinstance(p, str) and len(p.strip()) > 0]
    prompts = prompts[:prompt_num]
    return prompts


def measure_gpu_memory_bytes(model, tokenizer, device, batch_size: int, seq_len: int, max_new_tokens: int):
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Create synthetic batch of length seq_len
    vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is None:
        vocab_size = 32000
    input_ids = torch.randint(low=10, high=max(11, vocab_size - 1), size=(batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False)

    mem_bytes = torch.cuda.max_memory_allocated(device)
    return int(mem_bytes)


def compute_metrics(model, tokenizer, template, prompts, batch_size: int, repeat_times: int, max_new_tokens: int, device):
    all_prefill_speeds = []
    all_decode_speeds = []
    all_throughputs = []
    per_prompt_records = []

    # Pre-tokenize lengths to avoid repeated formatting overhead in timing; formatting/tokenization excluded from timing
    formatted_prompts = [template['prompt'].format(instruction=p) for p in prompts]

    for repeat_index in range(repeat_times):
        for i in range(0, len(formatted_prompts), batch_size):
            batch_texts = formatted_prompts[i:i+batch_size]
            prompt_indices = list(range(i, min(i + batch_size, len(formatted_prompts))))
            enc = tokenizer(batch_texts, return_tensors='pt', padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}

            # Prefill timing
            start_prefill = time.time()
            with torch.no_grad():
                _ = model(enc['input_ids'], attention_mask=enc['attention_mask'])
            prefill_time = time.time() - start_prefill
            prefill_tokens = enc['input_ids'].numel()
            if prefill_time > 0:
                all_prefill_speeds.append(prefill_tokens / prefill_time)

            # Decode timing
            start_decode = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            # Move only shapes; avoid .cpu() data transfer cost for timing
            decode_time = time.time() - start_decode

            # Per-sequence new token counts (accounting for early stopping)
            try:
                input_lengths = enc['attention_mask'].sum(dim=1).tolist()
            except Exception:
                input_lengths = [int(enc['input_ids'].shape[1])] * enc['input_ids'].shape[0]
            output_seq_len = int(outputs.shape[1])
            new_tokens_per_seq_list = [max(0, int(output_seq_len - int(inp_len))) for inp_len in input_lengths]
            total_new_tokens = int(sum(new_tokens_per_seq_list))
            if decode_time > 0 and total_new_tokens > 0:
                all_decode_speeds.append(total_new_tokens / decode_time)

            # Throughput counts new output tokens over end-to-end time
            total_time = prefill_time + decode_time
            if total_time > 0 and total_new_tokens > 0:
                all_throughputs.append(total_new_tokens / total_time)

            # Decode full texts for this batch
            try:
                decoded_full_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            except Exception:
                decoded_full_texts = ["" for _ in range(outputs.shape[0])]

            # Per-prompt records within this batch (times are batch-level, token counts per sequence where available)
            for local_idx, prompt_idx in enumerate(prompt_indices):
                # Decode only the newly generated portion for this sequence
                start_new = int(input_lengths[local_idx]) if local_idx < len(input_lengths) else 0
                try:
                    gen_ids = outputs[local_idx, start_new:]
                    decoded_new_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
                except Exception:
                    decoded_new_text = ""

                per_prompt_records.append({
                    "repeat_index": int(repeat_index),
                    "prompt_index": int(prompt_idx),
                    "instruction": str(prompts[prompt_idx]) if prompt_idx < len(prompts) else None,
                    "formatted_prompt": str(formatted_prompts[prompt_idx]) if prompt_idx < len(formatted_prompts) else None,
                    "prefill_time_s": float(prefill_time),
                    "decode_time_s": float(decode_time),
                    "total_time_s": float(total_time),
                    "input_tokens": int(input_lengths[local_idx]) if local_idx < len(input_lengths) else None,
                    "new_tokens": int(new_tokens_per_seq_list[local_idx]) if local_idx < len(new_tokens_per_seq_list) else None,
                    "generated_text": str(decoded_full_texts[local_idx]) if local_idx < len(decoded_full_texts) else "",
                    "generated_new_text": str(decoded_new_text),
                })

    def _safe_avg(values):
        return float(sum(values) / len(values)) if len(values) > 0 else 0.0

    return dict(
        prefill_speed=_safe_avg(all_prefill_speeds),
        decode_speed=_safe_avg(all_decode_speeds),
        throughput=_safe_avg(all_throughputs),
        per_prompt=per_prompt_records,
    )


if __name__ == "__main__":
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # Determine save path early and handle overwrite/skip
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{args.model_name}.json")
    if os.path.exists(save_file) and not args.overwrite:
        print(f"Results already exist at {save_file}. Use --overwrite to overwrite.")
        exit(0)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    model_config = model_configs[args.model_name]['model']

    # ========== load prompts ========== #
    prompts = load_prompts(args.test_cases_path, args.prompt_num)
    if len(prompts) == 0:
        print("No prompts found in test_cases_path")
        exit()
    print(f"Loaded {len(prompts)} prompts")
    print(prompts)

    # ========== load model/tokenizer/template ========== #
    model_name_or_path = model_config['model_name_or_path']
    model, tokenizer = load_model_and_tokenizer(**model_config)
    device = model.device
    template = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))

    # ========== metrics ========== #
    print("Measuring GPU memory footprint...")
    mem_bytes = measure_gpu_memory_bytes(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.mem_seq_len,
        max_new_tokens=args.max_new_tokens,
    )

    print("Timing prefill/decode/throughput...")
    metrics = compute_metrics(
        model=model,
        tokenizer=tokenizer,
        template=template,
        prompts=prompts,
        batch_size=args.batch_size,
        repeat_times=args.repeat_times,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    result = {
        "model": args.model_name,
        "mem_bytes": int(mem_bytes),
        "prefill_speed": float(metrics["prefill_speed"]),
        "decode_speed": float(metrics["decode_speed"]),
        "throughput": float(metrics["throughput"]),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "mem_seq_len": int(args.mem_seq_len),
        "num_prompts": int(len(prompts)),
        "per_prompt": metrics.get("per_prompt", []),
    }

    # ========== save ========== #
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics to {save_file}")

