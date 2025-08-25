#!/usr/bin/env python3

"""
测量两种防御的显存占用：PPL_Calculator 与 Llama Guard 3 1B。
- 遍历所有攻击方法下的 `llama3_2_1b_instruct` 目录里的 `test_cases.json`
- 取前 10 个 prompt，分别对两个防御进行推理以触发显存使用
- 每种防御重复 3 次，记录 `torch.cuda.max_memory_allocated`
- 参考 `test_model_speed.py` 的风格
注意：请在 HarmBench 根目录下运行。
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import time

# Ensure project root in path
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE)
sys.path.insert(0, PROJECT_ROOT)

from my_scripts.model_and_method_list import method_list as ALL_METHODS  # noqa: E402
from defense_baselines.PPL.ppl_calculator import PPL_Calculator  # noqa: E402
from defense_baselines.LlamaGuard3.detect import Llama3_guard_Predictor  # noqa: E402
from defense_baselines.Retokenization.bpe import load_subword_nmt_table, BpeOnlineTokenizer  # noqa: E402


def resolve_base_result_path(base_path: str, method: str, model: str) -> str:
    """Mirror path rules used by other analysis scripts."""
    if method == 'DirectRequest':
        return os.path.join(base_path, method, 'default')
    if method == 'HumanJailbreaks':
        return os.path.join(base_path, method, 'random_subset_5')
    if method == 'PAP':
        return os.path.join(base_path, method, 'top_5')
    return os.path.join(base_path, method, model)


def load_prompts(test_cases_path: str, limit: int) -> List[str]:
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts: List[str] = []
    if isinstance(data, dict):
        for _k, vals in data.items():
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, dict) and 'test_case' in v:
                        prompts.append(str(v['test_case']))
                    else:
                        prompts.append(str(v))
    elif isinstance(data, list):
        for v in data:
            if isinstance(v, dict) and 'test_case' in v:
                prompts.append(str(v['test_case']))
            else:
                prompts.append(str(v))
    prompts = [p for p in prompts if isinstance(p, str) and len(p.strip()) > 0]
    return prompts[:limit]


def reset_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_mem_allocated() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


# New: summary printer similar to test_defense_speed.py
def print_summary(records: List[Dict[str, Any]]) -> None:
    """Print averages across all models/methods for each defender (memory MB and time s)."""
    if not records:
        print('No records to summarize.')
        return
    # Collect per-defender metrics
    mem_values: Dict[str, List[float]] = {
        'ppl_calculator': [],
        'retokenization': [],
        'llama_guard_3_1B': [],
    }
    time_values: Dict[str, List[float]] = {
        'ppl_calculator': [],
        'retokenization': [],
        'llama_guard_3_1B': [],
    }
    for rec in records:
        # memory avgs are ints (bytes), time avgs are floats (seconds)
        for defender_key, mem_key, time_key in [
            ('ppl_calculator', 'ppl_calculator_mem_bytes_avg', 'ppl_calculator_time_sec_avg'),
            ('retokenization', 'retokenization_mem_bytes_avg', 'retokenization_time_sec_avg'),
            ('llama_guard_3_1B', 'llama_guard_3_1B_mem_bytes_avg', 'llama_guard_3_1B_time_sec_avg'),
        ]:
            mem_v = rec.get(mem_key)
            time_v = rec.get(time_key)
            try:
                if isinstance(mem_v, (int, float)) and mem_v >= 0:
                    mem_values[defender_key].append(float(mem_v) / (1024.0 * 1024.0))  # bytes -> MB
            except Exception:
                pass
            try:
                if isinstance(time_v, (int, float)) and time_v >= 0.0:
                    time_values[defender_key].append(float(time_v))
            except Exception:
                pass
    # Print
    has_any = False
    print('\nSummary: average GPU memory (MB) and time (s) by defender (across all models/methods)')
    for defender_key in ['ppl_calculator', 'retokenization', 'llama_guard_3_1B']:
        mem_list = mem_values[defender_key]
        time_list = time_values[defender_key]
        if not mem_list and not time_list:
            continue
        has_any = True
        avg_mem_mb = sum(mem_list) / len(mem_list) if mem_list else 0.0
        avg_time_s = sum(time_list) / len(time_list) if time_list else 0.0
        print(f"  - {defender_key}: mem={avg_mem_mb:.2f} MB, time={avg_time_s:.6f} s")
    if not has_any:
        print('No records to summarize.')


def measure_ppl_memory(prompts: List[str], ppl_model_path: str) -> Tuple[int, float]:
    """Instantiate PPL_Calculator, run once over prompts, return (peak bytes, elapsed seconds)."""
    reset_cuda()
    scorer = PPL_Calculator(ppl_model_path)
    start_time_sec = time.perf_counter()
    try:
        _ = scorer.get_perplexity(prompts)
    finally:
        elapsed_sec = time.perf_counter() - start_time_sec
        # Ensure cleanup before reading peak memory
        mem_bytes = get_peak_mem_allocated()
        del scorer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return mem_bytes, elapsed_sec


def measure_llamaguard_memory(prompts: List[str], guard_path: str) -> Tuple[int, float]:
    """Instantiate Llama Guard (1B), run predict once over prompts, return (peak bytes, elapsed seconds)."""
    reset_cuda()
    # Mirror generate_defense_completions.py: do not pass explicit device; let implementation handle it
    guard = Llama3_guard_Predictor(guard_path)
    start_time_sec = time.perf_counter()
    try:
        _raw, _labels, _cats = guard.predict(prompts)
    finally:
        elapsed_sec = time.perf_counter() - start_time_sec
        mem_bytes = get_peak_mem_allocated()
        del guard
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return mem_bytes, elapsed_sec


def measure_retokenization_memory(prompts: List[str], merge_table_path: str, bpe_dropout_rate: float) -> Tuple[int, float]:
    """Apply Retokenization BPE-dropout to prompts, return (peak bytes, elapsed seconds)."""
    reset_cuda()
    merge_table = load_subword_nmt_table(merge_table_path)
    tokenizer = BpeOnlineTokenizer(bpe_dropout_rate=bpe_dropout_rate, merge_table=merge_table)
    start_time_sec = time.perf_counter()
    try:
        _retok = [
            tokenizer(p,
                      sentinels=['', '</w>'],
                      regime='end',
                      bpe_symbol=' ')
            for p in prompts
        ]
    finally:
        elapsed_sec = time.perf_counter() - start_time_sec
        mem_bytes = get_peak_mem_allocated()
        del tokenizer
        del merge_table
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return mem_bytes, elapsed_sec


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze defender GPU memory usage on jailbreak prompts.')
    parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_50', help='Base results directory containing method/model outputs and test_cases.')
    parser.add_argument('--model', type=str, default='llama3_2_1b_instruct', help='Target model to locate test_cases.json under each method.')
    parser.add_argument('--prompt_num', type=int, default=50, help='Number of prompts to load per method.')
    parser.add_argument('--repeat_times', type=int, default=3, help='How many repeats per defender per method.')
    parser.add_argument('--ppl_calculator', type=str, default='/data2/zwh/models/gpt2', help='Path or alias for PPL_Calculator language model (e.g., gpt2).')
    parser.add_argument('--llama_guard_3_1B_path', type=str, default='/data2/zwh/models/Llama-Guard-3-1B', help='Local path to Llama Guard 3 1B model.')
    parser.add_argument('--save_path', type=str, default='/data2/zwh/HarmBench/cost_analysis/defense_memory_metrics.json', help='Output JSON path for metrics.')
    parser.add_argument('--BPO_dropout_rate', type=float, default=0.2, help='BPE Dropout rate for Retokenization defense.')
    parser.add_argument('--retokenization_merge_table', type=str, default=os.path.join(PROJECT_ROOT, 'defense_baselines', 'Retokenization', 'subword_nmt.voc'), help='Path to subword_nmt merge table for Retokenization defense.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite metrics file instead of reusing if it exists.')
    args = parser.parse_args()

    methods = list(ALL_METHODS)

    results: List[Dict[str, Any]] = []

    # 改成名称里面有prompt_num的
    args.save_path = f'/data2/zwh/HarmBench/cost_analysis/defense_memory_metrics_prompt_num_{args.prompt_num}.json'

    # Early exit: if file exists and not overwriting, print summary of existing and exit
    if (not args.overwrite) and os.path.exists(args.save_path):
        try:
            with open(args.save_path, 'r', encoding='utf-8') as f_in:
                existing = json.load(f_in)
            if not isinstance(existing, list):
                existing = []
        except Exception as e:
            print(f"Warning: failed to load existing metrics for summary: {e}")
            existing = []
        print_summary(existing)
        print(f"Metrics file exists at {args.save_path}. Use --overwrite to recompute and replace.")
        return

    for method in methods:
        print(f"Processing {method}")
        base_result_path = resolve_base_result_path(args.base_path, method, args.model)
        test_cases_path = os.path.join(base_result_path, 'test_cases', 'test_cases.json')
        if not os.path.exists(test_cases_path):
            print(f"[WARN] No {test_cases_path} found for {method}")
            continue
        try:
            prompts = load_prompts(test_cases_path, args.prompt_num)
        except Exception:
            continue
        if len(prompts) == 0:
            print(f"[WARN] No prompts found for {method}")
            continue

        # Measure PPL_Calculator memory, repeated
        print(f"[INFO] Measuring PPL_Calculator memory for {method}")
        ppl_mem_list: List[int] = []
        ppl_time_list: List[float] = []
        for _ in range(args.repeat_times):
            try:
                mem_b, elapsed_s = measure_ppl_memory(prompts, args.ppl_calculator)
            except Exception as e:
                mem_b, elapsed_s = -1, -1.0
                print(f"[WARN] PPL memory/time measure failed for {method}: {e}")
            ppl_mem_list.append(int(mem_b))
            ppl_time_list.append(float(elapsed_s))

        # Measure Retokenization memory, repeated
        print(f"[INFO] Measuring Retokenization memory for {method}")
        retok_mem_list: List[int] = []
        retok_time_list: List[float] = []
        for _ in range(args.repeat_times):
            try:
                mem_b, elapsed_s = measure_retokenization_memory(prompts, args.retokenization_merge_table, args.BPO_dropout_rate)
            except Exception as e:
                mem_b, elapsed_s = -1, -1.0
                print(f"[WARN] Retokenization memory/time measure failed for {method}: {e}")
            retok_mem_list.append(int(mem_b))
            retok_time_list.append(float(elapsed_s))

        # Measure Llama Guard 3 1B memory, repeated
        print(f"[INFO] Measuring Llama Guard 3 1B memory for {method}")
        lg_mem_list: List[int] = []
        lg_time_list: List[float] = []
        for _ in range(args.repeat_times):
            try:
                mem_b, elapsed_s = measure_llamaguard_memory(prompts, args.llama_guard_3_1B_path)
            except Exception as e:
                mem_b, elapsed_s = -1, -1.0
                print(f"[WARN] LlamaGuard-1B memory/time measure failed for {method}: {e}")
            lg_mem_list.append(int(mem_b))
            lg_time_list.append(float(elapsed_s))

        record = {
            'method': method,
            'model': args.model,
            'num_prompts': len(prompts),
            'ppl_calculator_mem_bytes': ppl_mem_list,
            'ppl_calculator_mem_bytes_avg': int(sum([b for b in ppl_mem_list if isinstance(b, int) and b >= 0]) / max(1, len([b for b in ppl_mem_list if isinstance(b, int) and b >= 0])) ) if any(b >= 0 for b in ppl_mem_list) else -1,
            'ppl_calculator_time_sec': ppl_time_list,
            'ppl_calculator_time_sec_avg': (sum([t for t in ppl_time_list if isinstance(t, float) and t >= 0.0]) / max(1, len([t for t in ppl_time_list if isinstance(t, float) and t >= 0.0]))) if any(t >= 0.0 for t in ppl_time_list) else -1.0,
            'retokenization_mem_bytes': retok_mem_list,
            'retokenization_mem_bytes_avg': int(sum([b for b in retok_mem_list if isinstance(b, int) and b >= 0]) / max(1, len([b for b in retok_mem_list if isinstance(b, int) and b >= 0])) ) if any(b >= 0 for b in retok_mem_list) else -1,
            'retokenization_time_sec': retok_time_list,
            'retokenization_time_sec_avg': (sum([t for t in retok_time_list if isinstance(t, float) and t >= 0.0]) / max(1, len([t for t in retok_time_list if isinstance(t, float) and t >= 0.0]))) if any(t >= 0.0 for t in retok_time_list) else -1.0,
            'llama_guard_3_1B_mem_bytes': lg_mem_list,
            'llama_guard_3_1B_mem_bytes_avg': int(sum([b for b in lg_mem_list if isinstance(b, int) and b >= 0]) / max(1, len([b for b in lg_mem_list if isinstance(b, int) and b >= 0])) ) if any(b >= 0 for b in lg_mem_list) else -1,
            'llama_guard_3_1B_time_sec': lg_time_list,
            'llama_guard_3_1B_time_sec_avg': (sum([t for t in lg_time_list if isinstance(t, float) and t >= 0.0]) / max(1, len([t for t in lg_time_list if isinstance(t, float) and t >= 0.0]))) if any(t >= 0.0 for t in lg_time_list) else -1.0,
            'test_cases_path': test_cases_path,
        }
        results.append(record)
        print(f"[Method={method}] PPL avg={record['ppl_calculator_mem_bytes_avg']} | Retok avg={record['retokenization_mem_bytes_avg']} | LG1B avg={record['llama_guard_3_1B_mem_bytes_avg']} | PPL time avg={record['ppl_calculator_time_sec_avg']:.4f}s | Retok time avg={record['retokenization_time_sec_avg']:.4f}s | LG1B time avg={record['llama_guard_3_1B_time_sec_avg']:.4f}s")

    # Save results
    # Print summary for the newly computed results
    print_summary(results)

    save_dir = os.path.dirname(args.save_path) or '.'
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved defense memory metrics to {args.save_path} ({len(results)} methods)")


if __name__ == "__main__":
    main()
