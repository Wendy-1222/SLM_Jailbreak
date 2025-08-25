import os
import json
import argparse
import sys
import csv
from typing import Dict, List, Optional, Tuple
import numpy as np

# 动态添加项目根目录到 Python 路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

from my_scripts.model_and_method_list import method_list as ALL_METHODS, model_list as ALL_MODELS

EXCLUDED_METHODS = {'DirectRequest', 'HumanJailbreaks'}
BATCH_5_METHODS = {'AutoDAN', 'PAP', 'PEZ', 'UAT', 'GBDA'}


def resolve_base_result_path(base_path: str, method: str, model: str) -> str:
    """Resolve the base directory for a given method and model, mirroring existing layout rules.

    Some methods store results under an extra subdirectory (e.g., 'default', 'random_subset_5', 'top_5').
    """
    if method == 'DirectRequest':
        return os.path.join(base_path, method, 'default', model)
    if method == 'HumanJailbreaks':
        return os.path.join(base_path, method, 'random_subset_5', model)
    if method == 'PAP':
        return os.path.join(base_path, method, 'top_5', model)
    return os.path.join(base_path, method, model)


def list_advbench_ids(test_cases_dir: str) -> List[int]:
    """List numeric IDs from directories named 'advbench_subset_{id}'."""
    if not os.path.isdir(test_cases_dir):
        return []
    ids: List[int] = []
    for name in os.listdir(test_cases_dir):
        if not name.startswith('advbench_subset_'):
            continue
        try:
            ids.append(int(name.split('_')[-1]))
        except ValueError:
            continue
    ids.sort()
    return ids


def read_rounds_and_prompt_count(logs_path: str, case_key: str, method: str) -> Tuple[Optional[float], Optional[int]]:
    """Return (rounds, prompt_count) inferred from logs.json per method.

    - AutoDAN: rounds = len(data[case_key][0])
    - GCG/AutoPrompt/UAT: rounds = len(data[case_key][0]["all_losses"]) 
    - PEZ/GBDA: rounds = average(len(elem["all_losses"]) for elem in data[case_key]); prompt_count = len(data[case_key])
    - Other methods: fallback to AutoDAN style if list-of-list

    For non-PEZ/GBDA, prompt_count is 1.
    """
    if not os.path.exists(logs_path):
        return None, None
    try:
        with open(logs_path, 'r') as f:
            data = json.load(f)
        value = data.get(case_key)
        if value is None:
            return None, None

        method_upper = method
        # PEZ/GBDA: multiple prompts per case
        if method_upper in {'PEZ', 'GBDA'}:
            if not isinstance(value, list) or len(value) == 0:
                return None, None
            prompt_count = len(value)
            assert prompt_count == 5, f"prompt_count should be 5, but got {prompt_count}"
            lengths: List[int] = []
            for elem in value:
                if isinstance(elem, dict) and isinstance(elem.get('all_losses'), list):
                    lengths.append(len(elem['all_losses']))
            if not lengths:
                return None, prompt_count
            avg_rounds = float(sum(lengths)) / float(len(lengths))
            return avg_rounds, prompt_count

        # GCG / AutoPrompt / UAT: first item dict with all_losses
        if method_upper in {'GCG', 'AutoPrompt', 'UAT'}:
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                all_losses = value[0].get('all_losses')
                if isinstance(all_losses, list):
                    return float(len(all_losses)), 1
            return None, 1

        # AutoDAN or default: first item is a list of rounds
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            return float(len(value[0])), 1

        return None, 1
    except Exception:
        return None, None


def get_mtime_seconds(path: Optional[str]) -> Optional[float]:
    try:
        if path is None:
            return None
        return os.path.getmtime(path)
    except Exception:
        return None


def compute_case_timings(test_cases_dir: str, case_id: int, method: str, prompt_count: int) -> Tuple[Optional[float], Optional[float]]:
    """Compute (total_time_seconds, per_round_seconds) for a case using mtime diffs and rounds.

    - If method in BATCH_5_METHODS, diff current group's mtime against previous group's boundary (every 5 cases),
      then divide by 5 (group size) to attribute equally to each case.
    - For PEZ/GBDA, further divide by prompt_count=len(data[case_key]) because multiple prompts are optimized per case.
    - Otherwise, diff mtime(logs.json of id) - mtime(logs.json of id-1)
    """
    case_dir = os.path.join(test_cases_dir, f"advbench_subset_{case_id}")
    logs_path = os.path.join(case_dir, 'logs.json')

    group_size = 5 if method in BATCH_5_METHODS else 1

    if method in BATCH_5_METHODS:
        prev_anchor_id = ((case_id - 1) // 5) * 5
        prev_logs_path = os.path.join(test_cases_dir, f"advbench_subset_{prev_anchor_id}", 'logs.json') if prev_anchor_id >= 1 else None
    else:
        prev_logs_path = os.path.join(test_cases_dir, f"advbench_subset_{case_id - 1}", 'logs.json')

    this_mtime = get_mtime_seconds(logs_path)
    prev_mtime = get_mtime_seconds(prev_logs_path)

    if this_mtime is None or prev_mtime is None:
        return None, None

    diff = this_mtime - prev_mtime
    total_time = diff if diff >= 0 else 0.0

    # divide by batch group size if applicable
    if group_size > 1:
        total_time = total_time / float(group_size)

    # For PEZ/GBDA, divide by number of prompts optimized for this case
    if method in {'PEZ', 'GBDA'} and prompt_count and prompt_count > 0:
        total_time = total_time / float(prompt_count)

    return total_time, None  # per-round computed by caller using rounds


def remove_outliers_and_negatives(values: List[float]) -> List[float]:
    """Remove outliers using IQR method and filter out non-positive values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Filtered list with outliers and non-positive values removed
    """
    if not values:
        return []
    
    # First filter out non-positive values
    positive_values = [v for v in values if v > 0]
    
    if len(positive_values) < 4:  # Need at least 4 values to detect outliers
        return positive_values
    
    # Convert to numpy array for easier calculations
    arr = np.array(positive_values)
    
    # Calculate Q1, Q3 and IQR
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers
    filtered_values = [v for v in positive_values if lower_bound <= v <= upper_bound]
    
    return filtered_values


def iqr_bounds_for_positive(values: List[Optional[float]]) -> Optional[Tuple[float, float]]:
    """Return (lower_bound, upper_bound) for outlier detection using IQR on positive values.

    If fewer than 4 positive values are available, returns None to indicate no bounds.
    """
    positive_values = [v for v in values if v is not None and v > 0]
    if len(positive_values) < 4:
        return None
    arr = np.array(positive_values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def analyze_pap(base_path: str) -> Dict[str, object]:
    """Special-case analysis for PAP.

    PAP shares the same test cases across models under: PAP/top_5/test_cases/test_cases_individual_behaviors.
    There are no turns to report; we only compute total_time using file mtime-based diffs with batch-of-5 grouping.
    """
    base_result_path = os.path.join(base_path, 'PAP', 'top_5')
    test_cases_dir = os.path.join(base_result_path, 'test_cases', 'test_cases_individual_behaviors')

    ids = list_advbench_ids(test_cases_dir)
    case_rows: List[Dict[str, object]] = []

    total_times: List[Optional[float]] = []

    for case_id in ids:
        case_dir = os.path.join(test_cases_dir, f"advbench_subset_{case_id}")
        logs_path = os.path.join(case_dir, 'logs.json')

        # Determine prompt_count for PAP if available (len(data[0])), though time attribution remains per case
        prompt_count: int = 1
        # print(logs_path)
        try:
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get(f'advbench_subset_{case_id}'), list):
                    prompt_count = len(data[f'advbench_subset_{case_id}'])
        except Exception:
            prompt_count = 1

        assert prompt_count == 5, f"prompt_count should be 5, but got {prompt_count}"
        total_time, _ = compute_case_timings(test_cases_dir, case_id, 'PAP', prompt_count)

        total_times.append(total_time if total_time is not None else None)

        case_rows.append({
            'case_id': case_id,
            'logs_path': logs_path,
            'rounds': -1,
            'total_time_minutes': round((total_time / 60.0), 3) if total_time is not None else -1,
            'per_round_minutes': -1,
        })

    # Valid cases: total_time > 0 and not an outlier by IQR
    bounds = iqr_bounds_for_positive(total_times)
    valid_indices: List[int] = []
    for idx in range(len(ids)):
        tt = total_times[idx]
        if tt is None or tt <= 0:
            continue
        if bounds is not None:
            lower_bound, upper_bound = bounds
            if tt < lower_bound or tt > upper_bound:
                continue
        valid_indices.append(idx)

    valid_total_times: List[float] = [total_times[i] for i in valid_indices]

    summary = {
        'method': 'PAP',
        'model': 'shared',
        'num_cases_found': len(ids),
        'num_valid_cases': len(valid_indices),
        'average_rounds': -1,
        'average_total_time_minutes': round((sum(valid_total_times) / len(valid_total_times)) / 60.0, 3) if valid_total_times else -1,
        'average_per_round_minutes': -1,
        'sum_total_time_minutes': round((sum(valid_total_times)) / 60.0, 3) if valid_total_times else 0,
        'cases': case_rows,
        'base_result_path': base_result_path,
        'test_cases_dir': test_cases_dir,
    }
    return summary


def analyze_method_model(base_path: str, method: str, model: str) -> Dict[str, object]:
    """Analyze a method+model pair and return detailed and summary stats."""
    base_result_path = resolve_base_result_path(base_path, method, model)

    # We focus on the iterative optimization under test_cases/test_cases_individual_behaviors
    test_cases_dir = os.path.join(base_result_path, 'test_cases', 'test_cases_individual_behaviors')

    ids = list_advbench_ids(test_cases_dir)
    case_rows: List[Dict[str, object]] = []

    total_times: List[Optional[float]] = []
    per_round_times: List[Optional[float]] = []
    rounds_list: List[Optional[float]] = []

    for case_id in ids:
        case_dir = os.path.join(test_cases_dir, f"advbench_subset_{case_id}")
        logs_path = os.path.join(case_dir, 'logs.json')

        rounds, prompt_count = read_rounds_and_prompt_count(logs_path, f"advbench_subset_{case_id}", method)
        if prompt_count is None:
            prompt_count = 1

        total_time, _ = compute_case_timings(test_cases_dir, case_id, method, prompt_count)

        per_round: Optional[float] = None
        if total_time is not None and rounds and rounds > 0:
            per_round = total_time / float(rounds)

        rounds_list.append(float(rounds) if rounds is not None else None)
        total_times.append(total_time if total_time is not None else None)
        per_round_times.append(per_round if per_round is not None else None)

        case_rows.append({
            'case_id': case_id,
            'logs_path': logs_path,
            'rounds': round(float(rounds), 4) if rounds is not None else -1,
            'total_time_minutes': round((total_time / 60.0), 3) if total_time is not None else -1,
            'per_round_minutes': round((per_round / 60.0), 3) if per_round is not None else -1,
        })

    # Determine valid cases using three conditions (case-level filter):
    # 1) rounds > 0  2) total_time > 0  3) total_time is not an outlier by IQR
    bounds = iqr_bounds_for_positive(total_times)
    valid_indices: List[int] = []
    for idx in range(len(ids)):
        r = rounds_list[idx]
        tt = total_times[idx]
        if r is None or r <= 0:
            continue
        if tt is None or tt <= 0:
            continue
        if bounds is not None:
            lower_bound, upper_bound = bounds
            if tt < lower_bound or tt > upper_bound:
                continue
        valid_indices.append(idx)

    valid_rounds_list: List[float] = [rounds_list[i] for i in valid_indices]
    valid_total_times: List[float] = [total_times[i] for i in valid_indices]
    # per_round is derived from total_time / rounds, but compute from stored list for consistency
    valid_per_round_times: List[float] = [per_round_times[i] for i in valid_indices if per_round_times[i] is not None]

    summary = {
        'method': method,
        'model': model,
        'num_cases_found': len(ids),
        'num_valid_cases': len(valid_indices),
        'average_rounds': round(sum(valid_rounds_list) / len(valid_rounds_list), 4) if valid_rounds_list else -1,
        'average_total_time_minutes': round((sum(valid_total_times) / len(valid_total_times)) / 60.0, 3) if valid_total_times else -1,
        'average_per_round_minutes': round((sum(valid_per_round_times) / len(valid_per_round_times)) / 60.0, 3) if valid_per_round_times else -1,
        'sum_total_time_minutes': round((sum(valid_total_times)) / 60.0, 3) if valid_total_times else 0,
        'cases': case_rows,
        'base_result_path': base_result_path,
        'test_cases_dir': test_cases_dir,
    }
    return summary


def write_csv(rows: List[Dict[str, object]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['method', 'model', 'case_id', 'rounds', 'total_time_minutes', 'per_round_minutes', 'logs_path']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze attack optimization cost (rounds and time in minutes).')
    parser.add_argument('--base_path', type=str, default='/data2/zwh/HarmBench/results_full_50', help='Base path of the results directory.')
    parser.add_argument('--methods', type=str, default='all', help="Comma-separated methods to analyze, or 'all' for all known methods.")
    parser.add_argument('--models', type=str, default='all', help="Comma-separated models to analyze, or 'all' for all known models.")
    parser.add_argument('--output_csv', type=str, default='/data2/zwh/HarmBench/cost_analysis/attack_cost_analysis.csv', help='Optional CSV output path to store per-case rows.')
    parser.add_argument('--output_json', type=str, default='', help='Optional JSON output path to store summaries.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files.')

    args = parser.parse_args()

    if args.overwrite:
        if os.path.exists(args.output_csv):
            os.remove(args.output_csv)
        if os.path.exists(args.output_json):
            os.remove(args.output_json)
    else:
        if os.path.exists(args.output_csv):
            print(f"Output CSV file {args.output_csv} already exists. Use --overwrite to overwrite.")
            return
        if os.path.exists(args.output_json):
            print(f"Output JSON file {args.output_json} already exists. Use --overwrite to overwrite.")
            return

    # Build method/model sets
    if args.methods.strip().lower() == 'all':
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    # Always exclude methods that are not needed
    methods = [m for m in methods if m not in EXCLUDED_METHODS]

    if args.models.strip().lower() == 'all':
        models = list(ALL_MODELS)
    else:
        models = [m.strip() for m in args.models.split(',') if m.strip()]

    all_summaries: List[Dict[str, object]] = []
    all_rows_csv: List[Dict[str, object]] = []

    for method in methods:
        print(f"Processing method {method}")
        if method == 'PAP':
            summary = analyze_pap(args.base_path)
            all_summaries.append(summary)
            for case in summary['cases']:
                all_rows_csv.append({
                    'method': method,
                    'model': summary['model'],
                    'case_id': case['case_id'],
                    'rounds': case['rounds'],
                    'total_time_minutes': case['total_time_minutes'],
                    'per_round_minutes': case['per_round_minutes'],
                    'logs_path': case['logs_path'],
                })
            continue
        for model in models:
            summary = analyze_method_model(args.base_path, method, model)
            all_summaries.append(summary)
            # extend CSV rows
            for case in summary['cases']:
                all_rows_csv.append({
                    'method': method,
                    'model': model,
                    'case_id': case['case_id'],
                    'rounds': case['rounds'],
                    'total_time_minutes': case['total_time_minutes'],
                    'per_round_minutes': case['per_round_minutes'],
                    'logs_path': case['logs_path'],
                })

    # Save outputs if requested
    if args.output_csv:
        write_csv(all_rows_csv, args.output_csv)
        print(f"Per-case CSV saved to {args.output_csv}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Summaries JSON saved to {args.output_json}")

    # Print concise summary to stdout
    for summary in all_summaries:
        method = summary['method']
        model = summary['model']
        avg_rounds = summary['average_rounds']
        avg_total = summary['average_total_time_minutes']
        avg_per_round = summary['average_per_round_minutes']
        num_cases = summary['num_cases_found']
        num_valid_cases = summary.get('num_valid_cases', -1)
        print(f"[Summary] {method} | {model} | cases={num_cases}(valid={num_valid_cases}) | avg_rounds={avg_rounds} | avg_total_min={avg_total} | avg_per_round_min={avg_per_round}")

    # Final per-method aggregation over models (ignore non-positive values and outliers)
    method_to_rounds: Dict[str, List[float]] = {}
    method_to_totals: Dict[str, List[float]] = {}
    for summary in all_summaries:
        method = summary['method']
        avg_rounds = summary.get('average_rounds', -1)
        avg_total = summary.get('average_total_time_minutes', -1)
        if isinstance(avg_rounds, (int, float)) and avg_rounds > 0:
            method_to_rounds.setdefault(method, []).append(float(avg_rounds))
        if isinstance(avg_total, (int, float)) and avg_total > 0:
            method_to_totals.setdefault(method, []).append(float(avg_total))

    for method in sorted(set(method_to_rounds.keys()).union(set(method_to_totals.keys()))):
        rounds_list = method_to_rounds.get(method, [])
        totals_list = method_to_totals.get(method, [])
        
        # Apply outlier removal to method-level aggregation as well
        filtered_rounds_list = remove_outliers_and_negatives(rounds_list)
        filtered_totals_list = remove_outliers_and_negatives(totals_list)
        
        avg_rounds_over_models = round(sum(filtered_rounds_list) / len(filtered_rounds_list), 4) if filtered_rounds_list else -1
        avg_total_over_models = round(sum(filtered_totals_list) / len(filtered_totals_list), 3) if filtered_totals_list else -1
        if (isinstance(avg_rounds_over_models, (int, float)) and avg_rounds_over_models > 0 and
            isinstance(avg_total_over_models, (int, float)) and avg_total_over_models > 0):
            avg_per_round_over_models = round(avg_total_over_models / avg_rounds_over_models, 3)
        else:
            avg_per_round_over_models = -1
        print(f"[Final Method Summary] {method} | models_used_rounds={len(filtered_rounds_list)} | models_used_time={len(filtered_totals_list)} | avg_rounds_over_models={avg_rounds_over_models} | avg_total_min_over_models={avg_total_over_models} | avg_per_round_min_over_models={avg_per_round_over_models}")


if __name__ == '__main__':
    main()
