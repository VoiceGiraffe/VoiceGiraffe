"""Download YouTube audios listed by a benchmark JSONL.

The `audio` field in each JSONL row points to a path whose filename (without
extension) is the YouTube video id. We download each unique video id as a
.wav file into --out-dir, skipping ones that are already present.

Requires: yt-dlp + ffmpeg installed in PATH.

Usage:
    python scripts/download_audios.py \
        --jsonl benchmark/benchmark_1500.jsonl \
        --out-dir audios \
        --workers 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow running this script directly without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from audio_bench.utils.dataset import iter_unique_youtube_ids, load_benchmark  # noqa: E402


def _ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"required tool '{name}' not found in PATH. "
            "Install yt-dlp (pip install yt-dlp) and ffmpeg first."
        )


def download_one(youtube_id: str, out_dir: str, audio_format: str = "wav",
                 sample_rate: int = 16000, retries: int = 3,
                 cookies_file: str | None = None,
                 proxy: str | None = None,
                 impersonate: bool = False) -> tuple[str, bool, str]:
    """Download one YouTube audio. Returns (id, ok, message)."""
    out_path = os.path.join(out_dir, f"{youtube_id}.{audio_format}")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return youtube_id, True, "skipped (already exists)"

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    last_err = ""
    for attempt in range(1, retries + 1):
        cmd = [
            "yt-dlp",
            "-x",  # extract audio
            "--audio-format", audio_format,
            "--audio-quality", "0",
            "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
            "-o", out_template,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
        ]
        if impersonate:
            cmd += ["--impersonate", "Chrome"]
        if proxy:
            cmd += ["--proxy", proxy]
        if cookies_file:
            cmd += ["--cookies", cookies_file]
        cmd.append(url)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0 and os.path.exists(out_path):
                return youtube_id, True, f"ok (attempt {attempt})"
            last_err = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
            last_err = last_err[0]
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"

        time.sleep(2 * attempt)

    return youtube_id, False, last_err or "unknown error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Benchmark JSONL path")
    parser.add_argument("--out-dir", required=True, help="Where to save .wav files")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audio-format", default="wav", choices=["wav", "m4a", "mp3", "flac", "opus"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cookies", default=None, help="Optional cookies.txt for age/region-restricted videos")
    parser.add_argument("--proxy", default=None, help="Proxy URL, e.g. http://127.0.0.1:7890")
    parser.add_argument("--impersonate", action="store_true", help="Enable Chrome impersonation (disabled by default)")
    parser.add_argument("--limit", type=int, default=0, help=">0 means only download the first N unique ids")
    args = parser.parse_args()

    _ensure_tool("yt-dlp")
    _ensure_tool("ffmpeg")

    os.makedirs(args.out_dir, exist_ok=True)

    samples = load_benchmark(args.jsonl)
    ids = [yid for yid in iter_unique_youtube_ids(samples) if yid]
    if args.limit > 0:
        ids = ids[: args.limit]
    print(f"[INFO] {len(ids)} unique YouTube ids to consider")

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                download_one, yid, args.out_dir,
                args.audio_format, args.sample_rate, args.retries,
                args.cookies, args.proxy, args.impersonate,
            ): yid
            for yid in ids
        }
        for i, fut in enumerate(as_completed(futs), 1):
            yid, ok, msg = fut.result()
            if ok and msg.startswith("skipped"):
                skip_cnt += 1
            elif ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                failures.append((yid, msg))
            if i % 10 == 0 or i == len(futs):
                print(
                    f"[{i}/{len(futs)}] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt}",
                    flush=True,
                )

    print(f"\n[DONE] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt}")
    if failures:
        log_path = os.path.join(args.out_dir, "_download_failures.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for yid, msg in failures:
                f.write(f"{yid}\t{msg}\n")
        print(f"[WARN] failed ids logged to {log_path}")


if __name__ == "__main__":
    main()
