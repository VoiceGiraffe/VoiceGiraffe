"""End-to-end evaluation: load benchmark -> run model -> compute metrics -> dump report.

Usage:
    python scripts/run_eval.py \
        --jsonl benchmark/benchmark_1500.jsonl \
        --audio-dir audios \
        --model qwen2.5-omni \
        --model-path Qwen/Qwen2.5-Omni-7B \
        --out-dir results/qwen_omni \
        --limit 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from audio_bench.eval.evaluator import PredRecord, evaluate, format_report  # noqa: E402
from audio_bench.models import get_model  # noqa: E402
from audio_bench.models.base import QAInput  # noqa: E402
from audio_bench.utils.dataset import load_benchmark  # noqa: E402
from audio_bench.utils.io import write_json, write_jsonl  # noqa: E402


def _lang_to_short(language_type: str) -> str:
    """Map jsonl `language_type` to short code used by model prompts."""
    if not language_type:
        return "en"
    if "中" in language_type or "zh" in language_type.lower() or "chinese" in language_type.lower():
        return "zh"
    return "en"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--audio-dir", required=True,
                        help="Directory containing <youtube_id>.wav files")
    parser.add_argument("--model", required=True,
                        help="Registered model name, e.g. 'qwen2.5-omni' or 'dummy'")
    parser.add_argument("--model-path", default=None,
                        help="HF repo id or local model path (forwarded to backend)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help=">0 = only run on first N samples (debug)")
    parser.add_argument("--skip-missing-audio", action="store_true",
                        help="If set, samples whose audio is missing locally are skipped (counted as parse_failure).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] loading benchmark: {args.jsonl}")
    samples = load_benchmark(args.jsonl)
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"[INFO] {len(samples)} samples")

    # Build model.
    backend_kwargs = {
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.model_path:
        backend_kwargs["model_path"] = args.model_path
    print(f"[INFO] loading model backend: {args.model} ({backend_kwargs})")
    model = get_model(args.model, **backend_kwargs)

    # Eager-load so that load failures fail fast.
    try:
        model.load()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] model.load() raised: {e}; will retry per-sample.")

    pred_records: list[PredRecord] = []
    raw_rows: list[dict] = []
    start = time.time()

    for i, s in enumerate(samples, 1):
        local_audio = s.resolve_local_audio(args.audio_dir) or ""
        if not local_audio:
            if args.skip_missing_audio:
                rec = PredRecord(
                    sample_id=s.sample_id, youtube_id=s.youtube_id,
                    qa_type=s.qa_type, qa_level=s.qa_level,
                    language_type=s.language_type,
                    gold=s.answer, pred=None, raw_text="",
                    error="audio_missing_local",
                )
                pred_records.append(rec)
                raw_rows.append({**asdict(rec), "question": s.question_full})
                continue
            # otherwise still hand it to the backend and let it return audio_not_found
            local_audio = s.audio_path

        qa_in = QAInput(
            audio_path=local_audio,
            question=s.question_full,
            language=_lang_to_short(s.language_type),
            sample_id=s.sample_id,
        )
        out = model.predict(qa_in)
        rec = PredRecord(
            sample_id=s.sample_id, youtube_id=s.youtube_id,
            qa_type=s.qa_type, qa_level=s.qa_level,
            language_type=s.language_type,
            gold=s.answer, pred=out.pred_letter, raw_text=out.raw_text,
            error=out.error,
        )
        pred_records.append(rec)
        raw_rows.append({**asdict(rec), "question": s.question_full,
                         "audio_local": local_audio})

        if i % 20 == 0 or i == len(samples):
            elapsed = time.time() - start
            print(f"[PROGRESS] {i}/{len(samples)} elapsed={elapsed:.1f}s", flush=True)

    # Persist raw predictions.
    pred_path = os.path.join(args.out_dir, "predictions.jsonl")
    write_jsonl(pred_path, raw_rows)
    print(f"[INFO] predictions written to {pred_path}")

    # Compute and dump metrics.
    metrics = evaluate(pred_records)
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    write_json(metrics_path, {"model": args.model, "model_path": args.model_path,
                              **metrics})
    report = format_report(metrics, model_name=f"{args.model} ({args.model_path})")
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n{report}\n")
    print(f"[INFO] metrics: {metrics_path}")
    print(f"[INFO] report : {report_path}")


if __name__ == "__main__":
    main()
