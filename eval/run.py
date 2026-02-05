#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight RAG evaluation metrics on a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset JSONL")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional predictions JSONL keyed by `id` with `retrieved_doc_ids` and/or `answer`",
    )
    parser.add_argument("--k", type=int, default=5, help="Cutoff K for Recall@K and MRR@K")
    parser.add_argument("--min-recall-at-k", type=float, default=None, help="Fail if Recall@K is below this value")
    parser.add_argument("--min-grounding-pass-rate", type=float, default=None, help="Fail if grounding pass rate is low")
    parser.add_argument("--max-p95-latency-ms", type=float, default=None, help="Fail if p95 latency exceeds this value")
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_num}")
            rows.append(obj)
    return rows


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _first_rank(relevant: List[str], retrieved: List[str], k: int) -> Optional[int]:
    relevant_set = set(relevant)
    for idx, doc_id in enumerate(retrieved[: max(0, k)], start=1):
        if doc_id in relevant_set:
            return idx
    return None


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _merge_predictions(rows: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pred_by_id: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        key = str(pred.get("id") or "").strip()
        if key:
            pred_by_id[key] = pred

    merged: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        row_id = str(row.get("id") or "").strip()
        pred = pred_by_id.get(row_id)
        if pred:
            if "retrieved_doc_ids" in pred:
                out["retrieved_doc_ids"] = pred.get("retrieved_doc_ids")
            if "answer" in pred:
                out["answer"] = pred.get("answer")
            if "latency_ms" in pred:
                out["latency_ms"] = pred.get("latency_ms")
        merged.append(out)
    return merged


def evaluate(rows: Iterable[Dict[str, Any]], *, k: int) -> Dict[str, Any]:
    total = 0
    recall_hits = 0
    reciprocal_sum = 0.0
    grounding_hits = 0
    grounding_total = 0
    latencies: List[float] = []

    for row in rows:
        total += 1
        relevant = _as_str_list(row.get("relevant_doc_ids"))
        retrieved = _as_str_list(row.get("retrieved_doc_ids"))
        rank = _first_rank(relevant, retrieved, k)
        if rank is not None:
            recall_hits += 1
            reciprocal_sum += 1.0 / float(rank)

        expected_snippet = str(row.get("expected_snippet") or "").strip()
        if expected_snippet:
            grounding_total += 1
            answer = str(row.get("answer") or "")
            if expected_snippet.lower() in answer.lower():
                grounding_hits += 1

        latency_ms = row.get("latency_ms")
        if latency_ms is not None:
            try:
                latencies.append(float(latency_ms))
            except Exception:
                pass

    recall_at_k = (recall_hits / total) if total else 0.0
    mrr_at_k = (reciprocal_sum / total) if total else 0.0
    grounding_pass_rate = (grounding_hits / grounding_total) if grounding_total else 0.0

    latencies_sorted = sorted(latencies)
    latency_metrics = {
        "count": len(latencies_sorted),
        "p50_ms": _percentile(latencies_sorted, 50.0) if latencies_sorted else 0.0,
        "p95_ms": _percentile(latencies_sorted, 95.0) if latencies_sorted else 0.0,
        "mean_ms": (sum(latencies_sorted) / len(latencies_sorted)) if latencies_sorted else 0.0,
        "median_ms": median(latencies_sorted) if latencies_sorted else 0.0,
    }

    return {
        "samples": total,
        "k": int(k),
        "retrieval": {
            "recall_at_k": recall_at_k,
            "mrr_at_k": mrr_at_k,
            "hits_at_k": recall_hits,
        },
        "grounding": {
            "pass_rate": grounding_pass_rate,
            "pass_count": grounding_hits,
            "total_with_expected_snippet": grounding_total,
        },
        "latency": latency_metrics,
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    rows = read_jsonl(dataset_path)
    if args.predictions:
        pred_path = Path(args.predictions).expanduser().resolve()
        rows = _merge_predictions(rows, read_jsonl(pred_path))

    report = evaluate(rows, k=max(1, int(args.k)))
    failures: List[str] = []

    recall_at_k = float(report["retrieval"]["recall_at_k"])
    grounding_rate = float(report["grounding"]["pass_rate"])
    p95_latency = float(report["latency"]["p95_ms"])

    if args.min_recall_at_k is not None and recall_at_k < float(args.min_recall_at_k):
        failures.append(
            f"Recall@{report['k']} {recall_at_k:.4f} < required {float(args.min_recall_at_k):.4f}"
        )
    if args.min_grounding_pass_rate is not None and grounding_rate < float(args.min_grounding_pass_rate):
        failures.append(
            f"Grounding pass rate {grounding_rate:.4f} < required {float(args.min_grounding_pass_rate):.4f}"
        )
    if args.max_p95_latency_ms is not None and p95_latency > float(args.max_p95_latency_ms):
        failures.append(
            f"Latency p95 {p95_latency:.2f}ms > allowed {float(args.max_p95_latency_ms):.2f}ms"
        )

    output = {
        "ok": len(failures) == 0,
        "report": report,
        "threshold_failures": failures,
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
