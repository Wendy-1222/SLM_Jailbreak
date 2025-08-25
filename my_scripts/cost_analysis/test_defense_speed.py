"""
对比无防御和不同防御的耗时
注意要放在HarmBench的根目录下运行
时间的单位是妙
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Any

# Ensure project root in path
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
sys.path.insert(0, PROJECT_ROOT)

from my_scripts.model_and_method_list import method_list as ALL_METHODS, model_list as ALL_MODELS  # noqa: E402

# Reuse defense/generation utilities
from generate_defense_completions import (  # noqa: E402
    load_generation_function,
    get_defense_completions,
)
from defense_baselines.LlamaGuard3.detect import Llama3_guard_Predictor  # noqa: E402


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Print average overhead_per_prompt_s per defender across provided records."""
    if not records:
        print('No records to summarize.')
        return
    overheads_by_defender: Dict[str, List[float]] = {}
    for rec in records:
        defender_name = rec.get('defender')
        overhead = rec.get('overhead_per_prompt_s')
        if defender_name is None or overhead is None:
            continue
        try:
            overhead_value = float(overhead)
        except Exception:
            continue
        overheads_by_defender.setdefault(defender_name, []).append(overhead_value)
    if not overheads_by_defender:
        print('No records to summarize.')
        return
    print('\nSummary: average overhead_per_prompt_s by defender (across all models/methods)')
    for defender_name in sorted(overheads_by_defender.keys()):
        values = overheads_by_defender[defender_name]
        avg_overhead = sum(values) / len(values) if len(values) > 0 else 0.0
        print(f"  - {defender_name}: {avg_overhead:.6f} s")


def resolve_base_result_path(base_path: str, method: str, model: str) -> str:
    if method == 'DirectRequest':
        return os.path.join(base_path, method, 'default', model)
    if method == 'HumanJailbreaks':
        return os.path.join(base_path, method, 'random_subset_5', model)
    if method == 'PAP':
        return os.path.join(base_path, method, 'top_5', model)
    return os.path.join(base_path, method, model)


def load_test_cases(test_cases_path: str, limit: int) -> List[str]:
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
    if limit and limit > 0:
        prompts = prompts[:limit]
    return prompts

def measure_times_for_combo(
    defender: str,
    model_name: str,
    model_config: Dict[str, Any],
    test_cases_path: str,
    prompts: List[str],
    max_new_tokens: int,
    generate_with_vllm: bool,
    defender_args: argparse.Namespace,
) -> float:
    """Return defense_total_s with a single timing run for the specified defender."""
    # Build generation function once per combo
    gen_fn = load_generation_function(
        model_config=model_config,
        max_new_tokens=max_new_tokens,
        test_cases_path=test_cases_path,
        generate_with_vllm=generate_with_vllm,
    )

    # Prepare args-like object for defense call (mimic generate_defense_completions)
    class _A:
        pass
    a = _A()
    a.defender = defender
    a.ppl_calculator = defender_args.ppl_calculator
    a.ppl_threshold = defender_args.ppl_threshold
    a.BPO_dropout_rate = defender_args.BPO_dropout_rate
    a.llama_guard_3_path = defender_args.llama_guard_3_path
    a.llama_guard_3_1B_path = defender_args.llama_guard_3_1B_path

    # Preload Llama Guard models as in reference script to avoid measuring one-time init repeatedly
    if defender == 'llama_guard_3':
        a.llama_guard_3 = Llama3_guard_Predictor(a.llama_guard_3_path)
    elif defender == 'llama_guard_3_1B':
        a.llama_guard_3_1B = Llama3_guard_Predictor(a.llama_guard_3_1B_path)

    # Timing (single run)
    t0 = time.time()
    _def_outputs, _gens = get_defense_completions(prompts, gen_fn, a, model_config)
    def_total = time.time() - t0

    return def_total


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure defense-induced latency over jailbreak prompts from results directories.')
    parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_50', help='Base results directory containing method/model outputs and test_cases.')
    parser.add_argument('--methods', type=str, default='all', help="Comma-separated methods or 'all'.")
    parser.add_argument('--models', type=str, default='llama3_2_1b_instruct', help="Comma-separated models or 'all'.")
    parser.add_argument('--defenders', type=str, default='no_defense,ppl,retokenization,self-reminder,llama_guard_3_1B', help="Comma-separated defenders to measure (e.g., 'no_defense,ppl,retokenization,self-reminder,llama_guard_3').")
    parser.add_argument('--models_config_file', type=str, default='./configs/model_configs/models.yaml', help='Path to models.yaml.')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max new tokens for generation.')
    parser.add_argument('--prompt_num', type=int, default=50, help='Number of prompts per method/model to time.')
    parser.add_argument('--generate_with_vllm', action='store_true', help='Use vLLM for generation if applicable.')
    parser.add_argument('--save_path', type=str, default='/data2/zwh/HarmBench/cost_analysis/defense_speed_metrics.json', help='Where to save metrics JSON.')
    # defender specific
    parser.add_argument('--ppl_calculator', type=str, default='/data2/zwh/models/gpt2')
    parser.add_argument('--llama_guard_3_path', type=str, default='/data2/zwh/models/Llama-Guard-3-8B')
    parser.add_argument('--llama_guard_3_1B_path', type=str, default='/data2/zwh/models/Llama-Guard-3-1B')
    parser.add_argument('--ppl_threshold', type=float, default=415.88)
    parser.add_argument('--BPO_dropout_rate', type=float, default=0.2)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite metrics file instead of appending if it exists.')
    args = parser.parse_args()

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

    # Build method/model sets
    if args.methods.strip().lower() == 'all':
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    if args.models.strip().lower() == 'all':
        models = list(ALL_MODELS)
    else:
        models = [m.strip() for m in args.models.split(',') if m.strip()]

    defenders = [d.strip() for d in args.defenders.split(',') if d.strip()]

    # Load model configs
    import yaml  # local import to avoid global dependency cost
    with open(args.models_config_file, 'r') as f:
        model_configs = yaml.full_load(f)

    results: List[Dict[str, Any]] = []

    for method in methods:
        print(f"[Method] {method}")
        for model in models:
            if model not in model_configs:
                # Skip models not configured
                continue
            if method == 'PAP':
                base_result_path = os.path.join(args.base_path, 'PAP', 'top_5')
            else:
                base_result_path = resolve_base_result_path(args.base_path, method, model)
            test_cases_dir = os.path.join(base_result_path, 'test_cases')
            test_cases_path = os.path.join(test_cases_dir, 'test_cases.json')
            if not os.path.exists(test_cases_path):
                # Skip if prompts not found
                continue
            try:
                prompts = load_test_cases(test_cases_path, args.prompt_num)
            except Exception:
                continue
            if len(prompts) == 0:
                continue

            model_config = dict(model_configs[model]['model'])

            # Measure each defender independently
            defender_times: Dict[str, float] = {}
            for defender in defenders:
                print(f"  - Model={model} | Defender={defender} | Prompts={len(prompts)}")
                try:
                    def_s = measure_times_for_combo(
                        defender=defender,
                        model_name=model,
                        model_config=model_config,
                        test_cases_path=test_cases_path,
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                        generate_with_vllm=args.generate_with_vllm,
                        defender_args=args,
                    )
                except Exception as e:
                    print(f"    ! Error timing {method}/{model} with {defender}: {e}")
                    continue
                defender_times[defender] = def_s

            # Compute records using 'no_defense' as baseline if available
            base_s = defender_times.get('no_defense')
            n = len(prompts)
            for defender, def_s in defender_times.items():
                record = {
                    'method': method,
                    'model': model,
                    'defender': defender,
                    'num_prompts': n,
                    'defense_total_s': round(def_s, 6),
                    'defense_per_prompt_s': round(def_s / n, 6),
                    'test_cases_path': test_cases_path,
                }
                if base_s is not None:
                    record.update({
                        'baseline_total_s': round(base_s, 6),
                        'overhead_total_s': round(def_s - base_s, 6),
                        'baseline_per_prompt_s': round(base_s / n, 6),
                        'overhead_per_prompt_s': round((def_s - base_s) / n, 6),
                    })
                else:
                    record.update({
                        'baseline_total_s': None,
                        'overhead_total_s': None,
                        'baseline_per_prompt_s': None,
                        'overhead_per_prompt_s': None,
                    })
                results.append(record)
                if base_s is not None:
                    print(f"    -> overhead_per_prompt_s={record['overhead_per_prompt_s']}")
                else:
                    print(f"    -> defense_per_prompt_s={record['defense_per_prompt_s']}")

    # Save aggregated results
    # Print summary over newly computed results, then save (overwrite)
    print_summary(results)

    save_dir = os.path.dirname(args.save_path) or '.'
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved defense speed metrics to {args.save_path} ({len(results)} records)")


if __name__ == '__main__':
    main() 