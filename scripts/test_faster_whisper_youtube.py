#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import error, request


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (project_root() / ".env")
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue

        val = value.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ[key] = val


def read_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


load_dotenv()

# Set the test target directly here when you want to run the script.
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZYbKUUrbatI"

BASE_URL = read_env("STT_BASE_URL", "http://localhost:8001").rstrip("/")
MODEL = read_env("STT_MODEL", "whisper-1")
LANGUAGE = read_env("STT_LANGUAGE", "ko")
RETRIES = int(read_env("STT_HEALTH_RETRIES", "20"))
BACKOFF_SECONDS = float(read_env("STT_HEALTH_BACKOFF_SEC", "2"))
BEAM_SIZE = int(read_env("STT_BEAM_SIZE", "5"))
STT_CHUNK_SECONDS = float(read_env("STT_CHUNK_SECONDS", "600"))
STT_CHUNK_OVERLAP_SECONDS = float(read_env("STT_CHUNK_OVERLAP_SECONDS", "2"))
OUTPUT_DIR = Path(read_env("STT_OUTPUT_DIR", str(project_root() / "scripts" / "outputs")))


def http_json(method: str, url: str, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    req = request.Request(url=url, method=method, headers=headers or {})
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[HTTP ERROR] status={exc.code} url={url}")
        print(body)
        raise
    except error.URLError as exc:
        print(f"[NETWORK ERROR] url={url} reason={exc.reason}")
        raise


def wait_for_health() -> None:
    health_url = f"{BASE_URL}/health"
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            status, payload = http_json("GET", health_url)
            if status == 200 and payload.get("status") == "ok":
                print(f"[INFO] health ready: {payload}")
                return
            print(f"[WARN] Unexpected /health payload: {payload}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WAIT] attempt={attempt}/{RETRIES} failed, retrying...")

        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS)

    print("[FAIL] /health did not become ready in time.")
    if last_error is not None:
        raise last_error
    raise RuntimeError("STT health endpoint unavailable")


def ensure_command(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Required command not found in PATH: {name}")


def download_audio(youtube_url: str, target_dir: Path) -> Path:
    if not youtube_url:
        raise RuntimeError(
            "YouTube URL is empty. Set YOUTUBE_URL at the top of this script."
        )

    ensure_command("yt-dlp")

    outtmpl = str(target_dir / "%(title).200s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--print",
        "after_move:filepath",
        "-o",
        outtmpl,
        youtube_url,
    ]
    print("[INFO] Downloading full audio with yt-dlp...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if proc.stdout.strip():
            print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr)
        raise RuntimeError(f"yt-dlp failed with exit code {proc.returncode}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("yt-dlp did not report the downloaded file path.")

    audio_path = Path(lines[-1])
    if not audio_path.is_file():
        raise RuntimeError(f"Downloaded file not found: {audio_path}")

    print(f"[INFO] downloaded_audio={audio_path}")
    return audio_path


def probe_duration(audio_path: Path) -> float:
    ensure_command("ffprobe")
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed with exit code {proc.returncode}: {proc.stderr.strip()}")

    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {proc.stdout!r}") from exc

    if duration <= 0:
        raise RuntimeError(f"Invalid audio duration: {duration}")
    return duration


def build_chunk_plan(total_duration: float) -> list[dict[str, float]]:
    if STT_CHUNK_SECONDS <= 0:
        raise RuntimeError("STT_CHUNK_SECONDS must be greater than 0.")

    plans: list[dict[str, float]] = []
    logical_start = 0.0

    while logical_start < total_duration:
        logical_end = min(logical_start + STT_CHUNK_SECONDS, total_duration)
        extract_start = max(0.0, logical_start - STT_CHUNK_OVERLAP_SECONDS)
        extract_end = min(total_duration, logical_end + STT_CHUNK_OVERLAP_SECONDS)
        plans.append(
            {
                "logical_start": logical_start,
                "logical_end": logical_end,
                "extract_start": extract_start,
                "extract_end": extract_end,
            }
        )
        logical_start = logical_end

    return plans


def extract_chunk_audio(
    source_audio_path: Path,
    chunk_plan: dict[str, float],
    chunk_idx: int,
    target_dir: Path,
) -> Path:
    ensure_command("ffmpeg")
    chunk_path = target_dir / f"chunk_{chunk_idx:04d}.wav"
    duration = chunk_plan["extract_end"] - chunk_plan["extract_start"]

    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{chunk_plan['extract_start']:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source_audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg chunk extraction failed: {proc.stderr.strip()}")
    return chunk_path


def request_transcription(audio_path: Path) -> dict[str, Any]:
    ensure_command("curl")
    url = f"{BASE_URL}/v1/audio/transcriptions"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as resp_tmp:
        response_path = Path(resp_tmp.name)

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-o",
                str(response_path),
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                "-F",
                f"file=@{audio_path}",
                "-F",
                f"model={MODEL}",
                "-F",
                f"language={LANGUAGE}",
                "-F",
                "response_format=verbose_json",
                "-F",
                f"beam_size={BEAM_SIZE}",
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60 * 60,
        )
        elapsed = time.perf_counter() - t0
        print(f"[INFO] transcription_elapsed_sec={elapsed:.3f}")
        if proc.returncode != 0:
            raise RuntimeError(f"curl failed with exit code {proc.returncode}: {proc.stderr.strip()}")

        status_code = proc.stdout.strip()
        raw = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        if status_code != "200":
            raise RuntimeError(f"STT request failed with status={status_code}: {raw}")
        payload = json.loads(raw) if raw else {}
    finally:
        if response_path.exists():
            response_path.unlink()

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected transcription response type: {type(payload).__name__}")
    return payload


def adjust_word_timestamps(words: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    adjusted_words: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        item = dict(word)
        if item.get("start") is not None:
            item["start"] = float(item["start"]) + offset
        if item.get("end") is not None:
            item["end"] = float(item["end"]) + offset
        adjusted_words.append(item)
    return adjusted_words


def merge_chunk_payloads(
    chunk_payloads: list[dict[str, Any]],
    chunk_plans: list[dict[str, float]],
    total_duration: float,
) -> dict[str, Any]:
    merged_segments: list[dict[str, Any]] = []
    detected_language = LANGUAGE

    for idx, (payload, plan) in enumerate(zip(chunk_payloads, chunk_plans, strict=True)):
        if idx == 0 and payload.get("language"):
            detected_language = str(payload["language"])

        logical_start = plan["logical_start"]
        logical_end = plan["logical_end"]
        extract_start = plan["extract_start"]
        is_last_chunk = idx == len(chunk_plans) - 1

        for seg in payload.get("segments", []):
            if not isinstance(seg, dict):
                continue

            seg_start = float(seg.get("start", 0.0)) + extract_start
            seg_end = float(seg.get("end", seg.get("start", 0.0))) + extract_start
            midpoint = (seg_start + seg_end) / 2.0

            in_window = logical_start <= midpoint if is_last_chunk else logical_start <= midpoint < logical_end
            if not in_window:
                continue

            item = dict(seg)
            item["start"] = seg_start
            item["end"] = seg_end
            item["words"] = adjust_word_timestamps(item.get("words", []), extract_start)
            merged_segments.append(item)

    for idx, seg in enumerate(merged_segments):
        seg["id"] = idx

    merged_text = "".join(str(seg.get("text", "")) for seg in merged_segments).strip()
    return {
        "task": "transcribe",
        "language": detected_language,
        "duration": total_duration,
        "text": merged_text,
        "segments": merged_segments,
    }


def transcribe_full_audio(audio_path: Path) -> dict[str, Any]:
    total_duration = probe_duration(audio_path)
    chunk_plans = build_chunk_plan(total_duration)
    print(
        f"[INFO] full_audio_duration_sec={total_duration:.3f} "
        f"chunk_sec={STT_CHUNK_SECONDS:.3f} overlap_sec={STT_CHUNK_OVERLAP_SECONDS:.3f} "
        f"chunk_count={len(chunk_plans)}"
    )

    chunk_payloads: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stt-chunks-") as chunk_dir:
        chunk_root = Path(chunk_dir)
        for idx, plan in enumerate(chunk_plans, start=1):
            print(
                f"[INFO] chunk {idx}/{len(chunk_plans)} "
                f"logical={plan['logical_start']:.3f}-{plan['logical_end']:.3f} "
                f"extract={plan['extract_start']:.3f}-{plan['extract_end']:.3f}"
            )
            chunk_audio_path = extract_chunk_audio(audio_path, plan, idx, chunk_root)
            payload = request_transcription(chunk_audio_path)
            chunk_payloads.append(payload)

    return merge_chunk_payloads(chunk_payloads, chunk_plans, total_duration)


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "??:??:??.???"

    value = max(float(seconds), 0.0)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = int(value % 60)
    millis = int(round((value - int(value)) * 1000))

    if millis == 1000:
        millis = 0
        secs += 1
    if secs == 60:
        secs = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1

    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def print_segments(payload: dict[str, Any]) -> None:
    text = payload.get("text", "")
    segments = payload.get("segments", [])

    print("\n=== FULL TEXT ===")
    print(text.strip() if isinstance(text, str) else text)

    print("\n=== TIMESTAMPED SEGMENTS ===")
    if not isinstance(segments, list) or not segments:
        print("[WARN] No segments returned.")
        return

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = format_timestamp(seg.get("start"))
        end = format_timestamp(seg.get("end"))
        seg_text = str(seg.get("text", "")).strip()
        print(f"[{start} - {end}] {seg_text}")


def save_outputs(audio_path: Path, payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    json_path = OUTPUT_DIR / f"{base_name}.verbose.json"
    txt_path = OUTPUT_DIR / f"{base_name}.segments.txt"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    for seg in payload.get("segments", []):
        if not isinstance(seg, dict):
            continue
        start = format_timestamp(seg.get("start"))
        end = format_timestamp(seg.get("end"))
        seg_text = str(seg.get("text", "")).strip()
        lines.append(f"[{start} - {end}] {seg_text}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INFO] saved_json={json_path}")
    print(f"[INFO] saved_segments={txt_path}")


def main() -> int:
    try:
        print(f"[INFO] base_url={BASE_URL}")
        print(f"[INFO] model={MODEL}")
        print(f"[INFO] language={LANGUAGE}")
        wait_for_health()

        with tempfile.TemporaryDirectory(prefix="yt-stt-") as tmp_dir:
            work_dir = Path(tmp_dir)
            audio_path = download_audio(YOUTUBE_URL, work_dir)
            payload = transcribe_full_audio(audio_path)
            print_segments(payload)
            save_outputs(audio_path, payload)

        print("[SUCCESS] YouTube -> full audio -> chunked sequential STT completed.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
