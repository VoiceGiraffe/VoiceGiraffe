# VoiceGiraffe 🦒

**VoiceGiraffe** is a bilingual, hour-scale audio understanding benchmark for Large Audio Language Models (LALMs). It comprises **1,500** rigorously curated multiple-choice QA pairs over **126** real-world long-form recordings (~116 hours total), structured into a dual-level taxonomy:

| Level | Tasks | Description |
|-------|-------|-------------|
| **Single-Hop Perception** | SC · AE · PL · TL | Semantic Content, Acoustic Event, Paralinguistic, Temporal Localisation |
| **Multi-Hop Reasoning** | CE · ST | Causal / Event-Tracking across non-contiguous segments |

All items are available in both **Chinese** and **English**, sourced from diverse domains (podcasts, sports, gaming, news, etc.). This repository ships:

- **Data** — a JSONL of QA pairs (`benchmark/`) referencing YouTube IDs.
- **Downloader** — pulls audio for every referenced video via `yt-dlp`.
- **Inference** — pluggable model backends; ships with a working **Qwen2.5-Omni** runner and a `dummy` baseline for pipeline validation.
- **Evaluator** — overall accuracy + per–`qa_type` / `qa_level` / `language_type` / `qa_type × language` breakdowns, output as `metrics.json` and a human-readable `report.txt`.
- **Web Viewer** — a single-command dashboard to browse the dataset, listen to audio, inspect model predictions, and visualize per-subtask accuracy.

---

## Repository layout

```
audio_bench_release/
├── audio_bench/                  # Python package
│   ├── models/                   # Backends (registry + Qwen2.5-Omni + dummy)
│   ├── eval/                     # Evaluator
│   └── utils/                    # Dataset loader, IO, output parsing
├── scripts/
│   ├── download_audios.py        # YouTube id -> .wav
│   └── run_eval.py               # benchmark -> predictions -> metrics
├── web/
│   ├── app.py                    # FastAPI server
│   ├── templates/index.html
│   └── static/{style.css,app.js}
├── benchmark/                    # put your benchmark JSONL here
├── audios/                       # downloaded .wav files (gitignored)
├── results/                      # metrics + predictions per run
└── requirements.txt
```

## Benchmark JSONL format

One sample per line, fields:

| field | type | description |
|------|------|-------------|
| `question` | str | full prompt with options inlined as `Q\nA: ...\nB: ...\nC: ...\nD: ...` |
| `answer` | str | gold letter, one of `A` / `B` / `C` / `D` |
| `audio` | str | path to source audio; basename without extension is the YouTube ID |
| `qa_type` | str | one of `SC`, `AE`, `PL`, `TL`, `CE`, `ST` |
| `qa_level` | str | `Single-Hop` or `Multi-Hop` |
| `language_type` | str | `中文` or `英文` |

## Setup

```bash
cd audio_bench_release
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# yt-dlp also needs ffmpeg available in PATH
sudo apt-get install -y ffmpeg   # or: brew install ffmpeg
```

## 1. Download audios

```bash
python scripts/download_audios.py \
  --jsonl benchmark/benchmark_1500.jsonl \
  --out-dir audios \
  --workers 8
```

- Skips files already present in `audios/`.
- Retries up to 3 times per id.
- Failures (private/region-locked/etc.) are logged to `audios/_download_failures.txt`.
- Downloads as 16 kHz mono `.wav` by default (configurable via flags).

## 2. Run a model

```bash
# 1) sanity check the pipeline (no GPU needed)
python scripts/run_eval.py \
  --jsonl benchmark/benchmark_1500.jsonl \
  --audio-dir audios \
  --model dummy \
  --out-dir results/dummy

# 2) Qwen2.5-Omni-7B (needs GPU, transformers>=4.52)
python scripts/run_eval.py \
  --jsonl benchmark/benchmark_1500.jsonl \
  --audio-dir audios \
  --model qwen2.5-omni \
  --model-path Qwen/Qwen2.5-Omni-7B \
  --device cuda --dtype bfloat16 \
  --out-dir results/qwen_omni
```

Each run writes:
- `results/<run>/predictions.jsonl`
- `results/<run>/metrics.json`
- `results/<run>/report.txt`

### Adding a new model backend

Create `audio_bench/models/my_model.py`:

```python
from .base import BaseAudioQAModel, QAInput, QAOutput
from .registry import register_model
from ..utils.parsing import parse_letter

@register_model("my-model")
class MyModel(BaseAudioQAModel):
    def __init__(self, model_path: str = "...", **kw): ...
    def load(self): ...
    def predict(self, item: QAInput) -> QAOutput:
        # call your model
        text = "..."
        return QAOutput(raw_text=text, pred_letter=parse_letter(text),
                        sample_id=item.sample_id)
```

Then run with `--model my-model`.

## 3. Launch the visualizer

```bash
python web/app.py \
  --jsonl benchmark/benchmark_1500.jsonl \
  --audio-dir audios \
  --results-dir results \
  --host 0.0.0.0 --port 7860
```

Open <http://localhost:7860>. The viewer auto-discovers every subdirectory of
`results/` as a model run; pick one from the top-right dropdown to see:

- overall accuracy + per-subtask bar charts + qa_type×language radar,
- a paged sample browser with embedded audio player,
- option highlighting (gold answer + model prediction),
- a one-click *Download* button for any audio missing locally.

## Metrics

`metrics.json`:

```json
{
  "model": "qwen2.5-omni",
  "total": 1500,
  "correct": 957,
  "overall_accuracy": 0.638,
  "infer_failures": 12,
  "parse_failures": 4,
  "groups": {
    "qa_type":            {"Causal": {...}, "Timestamp": {...}},
    "qa_level":           {"T1": {...}, "T2": {...}},
    "language_type":      {"中文": {...}, "英文": {...}},
    "qa_type_x_language": {"Causal|中文": {...}, ...}
  }
}
```


## Notes

- Commit the JSONL but **not** the audios; instead ship the downloader and a
  failure list. This stays under file-size limits and respects YouTube licensing.
- Pin everything in `requirements.txt`; declare `ffmpeg` as an external dependency.
- For reproducibility, fix `--dtype`, `--device`, and `--max-new-tokens` and
  log them in `metrics.json` (already done by `run_eval.py`).
- The `dummy` backend gives ~25% accuracy — a useful sanity check that the
  pipeline is neither always-correct nor always-wrong.
