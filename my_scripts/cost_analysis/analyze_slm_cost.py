#!/usr/bin/env python3

import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Set

INPUT_DIR = "/data2/zwh/HarmBench/cost_analysis/slm_speed_metrics"
OUTPUT_XLSX = "/data2/zwh/HarmBench/cost_analysis/slm_speed_metrics.xlsx"
EXCLUDE_MODELS: Set[str] = {"llama2_7b", "vicuna_7b_v1_5"}
FIELDS_TO_KEEP = ["model", "mem_bytes", "prefill_speed", "decode_speed", "throughput"]


def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(text)


def collect_records(input_dir: str, exclude: Set[str]) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        try:
            r = load_json(path)
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue
        model = r.get("model")
        if model and model not in exclude:
            # 只保留需要的字段
            records.append({k: r.get(k) for k in FIELDS_TO_KEEP})
    return sorted(records, key=lambda r: str(r.get("model")))


def excel_col_name(n: int) -> str:
    name = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        name = chr(65 + rem) + name
    return name


def write_xlsx(records: List[Dict[str, Any]], columns: List[str], output: str):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "slm_speed_metrics"
    ws.append(columns)

    for r in records:
        ws.append([r.get(c, None) for c in columns])

    for i, col in enumerate(columns, 1):
        max_len = max(len(str(r.get(col, ""))) for r in records)
        max_len = max(max_len, len(col)) + 2
        max_len = min(60, max(10, max_len))
        ws.column_dimensions[excel_col_name(i)].width = max_len

    os.makedirs(os.path.dirname(output), exist_ok=True)
    wb.save(output)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Input directory missing: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    records = collect_records(INPUT_DIR, EXCLUDE_MODELS)
    if not records:
        print("No valid records found.", file=sys.stderr)
        sys.exit(1)

    write_xlsx(records, FIELDS_TO_KEEP, OUTPUT_XLSX)
    print(f"Wrote {len(records)} rows × {len(FIELDS_TO_KEEP)} cols to {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
