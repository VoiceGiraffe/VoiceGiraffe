"""FastAPI web app: visualize benchmark + per-sample model predictions.

Run:
    cd audio_bench_release
    pip install fastapi uvicorn jinja2
    python web/app.py \
        --jsonl benchmark/benchmark_1500.jsonl \
        --audio-dir audios \
        --results-dir results \
        --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

# Allow running this script directly without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from audio_bench.utils.dataset import load_benchmark  # noqa: E402


# ---------------- Globals (filled in main) ----------------
APP_STATE: dict = {
    "jsonl": "",
    "audio_dir": "",
    "results_dir": "",
    "samples": [],
    "loaded_model": None,       # BaseAudioQAModel instance or None
    "loaded_model_name": "",    # str
    "human_answers": {},        # dict[sample_id -> {"letter": str, "correct": bool}]
}


def _load_results(results_dir: str) -> dict:
    """Discover result subdirectories. Each child dir of results_dir is treated
    as one model run, expected to contain `predictions.jsonl` and `metrics.json`."""
    runs: dict = {}
    if not results_dir or not os.path.isdir(results_dir):
        return runs
    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        preds_path = os.path.join(run_dir, "predictions.jsonl")
        if not os.path.exists(metrics_path) and not os.path.exists(preds_path):
            continue

        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:  # noqa: BLE001
                metrics = {}

        preds_by_id: dict = {}
        if os.path.exists(preds_path):
            with open(preds_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    sid = row.get("sample_id")
                    if sid:
                        preds_by_id[sid] = row

        runs[name] = {"metrics": metrics, "preds": preds_by_id}
    return runs


def build_app() -> FastAPI:
    app = FastAPI(title="AudioBench Viewer")

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=templates_dir)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"title": "AudioBench Viewer"},
        )

    @app.get("/api/summary")
    def api_summary():
        samples = APP_STATE["samples"]
        runs = _load_results(APP_STATE["results_dir"])

        qt = Counter(s.qa_type for s in samples)
        ql = Counter(s.qa_level for s in samples)
        lt = Counter(s.language_type for s in samples)

        # How many audios are present locally.
        present = 0
        for s in samples:
            if s.resolve_local_audio(APP_STATE["audio_dir"]) is not None:
                present += 1

        return {
            "jsonl": APP_STATE["jsonl"],
            "audio_dir": APP_STATE["audio_dir"],
            "results_dir": APP_STATE["results_dir"],
            "total_samples": len(samples),
            "unique_audios": len({s.youtube_id for s in samples if s.youtube_id}),
            "audios_present_locally": present,
            "qa_type": dict(qt),
            "qa_level": dict(ql),
            "language_type": dict(lt),
            "runs": [
                {
                    "name": name,
                    "overall_accuracy": run["metrics"].get("overall_accuracy"),
                    "total": run["metrics"].get("total"),
                    "groups": run["metrics"].get("groups", {}),
                }
                for name, run in runs.items()
            ],
        }

    @app.get("/api/samples")
    def api_samples(
        offset: int = 0,
        limit: int = 50,
        qa_type: Optional[str] = None,
        qa_level: Optional[str] = None,
        language_type: Optional[str] = None,
        run: Optional[str] = None,
        only_wrong: bool = False,
    ):
        samples = APP_STATE["samples"]

        def _match(s):
            if qa_type and s.qa_type != qa_type:
                return False
            if qa_level and s.qa_level != qa_level:
                return False
            if language_type and s.language_type != language_type:
                return False
            return True

        filtered = [s for s in samples if _match(s)]

        run_data = None
        if run:
            runs = _load_results(APP_STATE["results_dir"])
            run_data = runs.get(run)

        if only_wrong and run_data is not None:
            preds = run_data["preds"]
            filtered = [
                s for s in filtered
                if (preds.get(s.sample_id, {}).get("pred")
                    or "").upper() != s.answer
            ]

        page = filtered[offset:offset + limit]
        rows = []
        for s in page:
            local_audio = s.resolve_local_audio(APP_STATE["audio_dir"])
            row = {
                "sample_id": s.sample_id,
                "youtube_id": s.youtube_id,
                "question_stem": s.question_stem,
                "question_full": s.question_full,
                "options": s.options,
                "answer": s.answer,
                "qa_type": s.qa_type,
                "qa_level": s.qa_level,
                "language_type": s.language_type,
                "audio_local_available": bool(local_audio),
                "audio_url": (f"/audio/{s.youtube_id}" if local_audio else None),
            }
            if run_data is not None:
                pr = run_data["preds"].get(s.sample_id) or {}
                row["pred"] = pr.get("pred")
                row["pred_correct"] = (
                    bool(pr.get("pred")) and
                    str(pr.get("pred", "")).upper() == s.answer
                )
                row["raw_text"] = pr.get("raw_text", "")
                row["error"] = pr.get("error")
            rows.append(row)

        return {"total": len(filtered), "offset": offset, "limit": limit, "rows": rows}

    @app.get("/api/quiz_chains")
    def api_quiz_chains(
        qa_type: Optional[str] = None,
        qa_level: Optional[str] = None,
        language_type: Optional[str] = None,
    ):
        """Return quiz chains grouped by youtube_id.

        Each chain is one audio + its list of questions, so the user can
        listen to the audio continuously while answering multiple questions
        about the same clip (multi-hop reasoning).
        """
        from collections import OrderedDict

        samples = APP_STATE["samples"]

        def _match(s):
            if qa_type and s.qa_type != qa_type:
                return False
            if qa_level and s.qa_level != qa_level:
                return False
            if language_type and s.language_type != language_type:
                return False
            return True

        filtered = [s for s in samples if _match(s)]

        # Group by youtube_id preserving encounter order
        groups: OrderedDict[str, list] = OrderedDict()
        for s in filtered:
            yid = s.youtube_id or "__no_audio__"
            groups.setdefault(yid, []).append(s)

        chains = []
        for yid, grp in groups.items():
            local_audio = grp[0].resolve_local_audio(APP_STATE["audio_dir"])
            questions = []
            for s in grp:
                questions.append({
                    "sample_id": s.sample_id,
                    "question_stem": s.question_stem,
                    "question_full": s.question_full,
                    "options": s.options,
                    "answer": s.answer,
                    "qa_type": s.qa_type,
                    "qa_level": s.qa_level,
                    "language_type": s.language_type,
                })
            chains.append({
                "youtube_id": yid,
                "audio_local_available": bool(local_audio),
                "audio_url": (f"/audio/{yid}" if local_audio else None),
                "question_count": len(questions),
                "questions": questions,
            })

        return {"total_chains": len(chains), "chains": chains}

    @app.get("/audio/{youtube_id}")
    def serve_audio(youtube_id: str):
        # Find the audio with any allowed extension.
        for ext in (".wav", ".m4a", ".mp3", ".flac", ".opus"):
            cand = os.path.join(APP_STATE["audio_dir"], youtube_id + ext)
            if os.path.exists(cand):
                return FileResponse(cand)
        raise HTTPException(status_code=404, detail="audio not found")

    @app.post("/api/download")
    def api_download(payload: dict):
        """Trigger a download for one YouTube id (calls scripts.download_audios.download_one)."""
        from scripts.download_audios import download_one  # noqa: WPS433

        yid = (payload or {}).get("youtube_id", "").strip()
        if not yid:
            raise HTTPException(400, "missing youtube_id")
        os.makedirs(APP_STATE["audio_dir"], exist_ok=True)
        _id, ok, msg = download_one(yid, APP_STATE["audio_dir"])
        return JSONResponse({"youtube_id": yid, "ok": ok, "msg": msg})

    # -------- Model Management --------

    @app.get("/api/models")
    def api_models():
        """列出所有已注册的模型 + 当前加载状态"""
        from audio_bench.models.registry import MODEL_REGISTRY, _bootstrap_builtin_models
        _bootstrap_builtin_models()
        return {
            "available": sorted(MODEL_REGISTRY.keys()),
            "loaded": APP_STATE["loaded_model_name"] or None,
        }

    @app.post("/api/models/load")
    def api_load_model(payload: dict):
        """加载指定模型到 GPU"""
        from audio_bench.models.registry import get_model
        name = (payload or {}).get("model_name", "").strip()
        if not name:
            raise HTTPException(400, "missing model_name")
        # 如果已加载同名模型则跳过
        if APP_STATE["loaded_model_name"] == name and APP_STATE["loaded_model"] is not None:
            return {"status": "already_loaded", "model": name}
        # 卸载旧模型
        APP_STATE["loaded_model"] = None
        APP_STATE["loaded_model_name"] = ""
        # 加载新模型
        kwargs = {}
        if payload.get("model_path"):
            kwargs["model_path"] = payload["model_path"]
        if payload.get("dtype"):
            kwargs["dtype"] = payload["dtype"]
        if payload.get("device"):
            kwargs["device"] = payload["device"]
        try:
            model = get_model(name, **kwargs)
            model.load()
            APP_STATE["loaded_model"] = model
            APP_STATE["loaded_model_name"] = name
            return {"status": "loaded", "model": name}
        except Exception as e:
            raise HTTPException(500, f"Failed to load model '{name}': {e}")

    # -------- Inference --------

    @app.post("/api/infer")
    def api_infer(payload: dict):
        """对指定 sample 跑单条推理"""
        from audio_bench.models.base import QAInput
        model = APP_STATE["loaded_model"]
        if model is None:
            raise HTTPException(400, "No model loaded. Call /api/models/load first.")
        sample_id = (payload or {}).get("sample_id", "").strip()
        if not sample_id:
            raise HTTPException(400, "missing sample_id")
        # 查找 sample
        sample = None
        for s in APP_STATE["samples"]:
            if s.sample_id == sample_id:
                sample = s
                break
        if sample is None:
            raise HTTPException(404, f"sample_id '{sample_id}' not found")
        # 确认音频存在
        audio_path = sample.resolve_local_audio(APP_STATE["audio_dir"])
        if not audio_path:
            raise HTTPException(400, f"Audio not available locally for {sample.youtube_id}")
        # 推理
        inp = QAInput(
            audio_path=audio_path,
            question=sample.question_full,
            language=sample.language_type,
            sample_id=sample.sample_id,
        )
        output = model.predict(inp)
        return {
            "sample_id": sample_id,
            "raw_text": output.raw_text,
            "pred_letter": output.pred_letter,
            "correct": (output.pred_letter or "").upper() == sample.answer,
            "gold": sample.answer,
            "error": output.error,
        }

    # -------- Human Answer --------

    @app.post("/api/human_answer")
    def api_human_answer(payload: dict):
        """记录用户答案"""
        sample_id = (payload or {}).get("sample_id", "").strip()
        letter = (payload or {}).get("letter", "").strip().upper()
        if not sample_id or letter not in ("A", "B", "C", "D"):
            raise HTTPException(400, "missing sample_id or invalid letter")
        # 查找正确答案
        sample = None
        for s in APP_STATE["samples"]:
            if s.sample_id == sample_id:
                sample = s
                break
        if sample is None:
            raise HTTPException(404, f"sample_id '{sample_id}' not found")
        correct = letter == sample.answer
        APP_STATE["human_answers"][sample_id] = {"letter": letter, "correct": correct}
        # 累计统计
        total = len(APP_STATE["human_answers"])
        correct_count = sum(1 for v in APP_STATE["human_answers"].values() if v["correct"])
        return {
            "sample_id": sample_id,
            "submitted": letter,
            "gold": sample.answer,
            "correct": correct,
            "stats": {
                "total": total,
                "correct": correct_count,
                "accuracy": correct_count / total if total > 0 else 0,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    APP_STATE["jsonl"] = os.path.abspath(args.jsonl)
    APP_STATE["audio_dir"] = os.path.abspath(args.audio_dir)
    APP_STATE["results_dir"] = os.path.abspath(args.results_dir)
    APP_STATE["samples"] = load_benchmark(args.jsonl)
    print(f"[INFO] loaded {len(APP_STATE['samples'])} samples from {args.jsonl}")
    print(f"[INFO] audio_dir   = {APP_STATE['audio_dir']}")
    print(f"[INFO] results_dir = {APP_STATE['results_dir']}")
    app = build_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
