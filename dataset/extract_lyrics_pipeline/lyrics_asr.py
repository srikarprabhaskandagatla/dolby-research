"""
lyrics.py
─────────
Node-parallel Whisper ASR pipeline. Designed to run as a SLURM job array
(--array=0-9) across 10 nodes, each processing 5 000 tracks.

Node ID is read from $SLURM_ARRAY_TASK_ID automatically.
Can also be passed manually:  python lyrics.py <node_id>

Per-node paths (node N):
  WAV dir    : .../batch_1/node_N/
  Status CSV : .../batch_1/download_status_node_N.csv
  Output CSV : .../lyrics/master_lyrics_node_N.csv

Track ID ranges:
  node 0 → tracks   0 –  4 999
  node 1 → tracks   5 000 –  9 999
  ...
  node 9 → tracks  45 000 – 49 999
  (Track IDs come directly from each node's download_status CSV.)

Quality-check signals
  1. Whisper segment confidence  (no_speech_prob, avg_logprob)
  2. Text coherence heuristics   (word count, repetition, noise chars)
  3. Duration-aware coverage      (words-per-minute relative to song length)

source_tag values in CSV
  "whisper_full"    – full-track transcription passed all QC gates
  "whisper_window"  – full track failed; a random 30 s window passed
  "whisper_failed"  – both failed; longer text returned as best-effort
"""

import argparse
import os
import re
import sys
import random
import tempfile
import traceback

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

# ── Node ID ───────────────────────────────────────────────────────────────────
# Priority: --node_id CLI flag → $SLURM_ARRAY_TASK_ID → default 0

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--node_id", type=int, default=None)
_args, _ = _parser.parse_known_args()

if _args.node_id is not None:
    NODE_ID = _args.node_id
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    NODE_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
else:
    NODE_ID = 0

# ── Paths (all derived from NODE_ID) ─────────────────────────────────────────

_BATCH_ROOT = "/scratch3/workspace/skandagatla_umass_edu-dolby/raw_audio_files/batch_1"
_OUTPUT_ROOT = "/scratch3/workspace/skandagatla_umass_edu-dolby/lyrics"

WAV_DIR    = os.path.join(_BATCH_ROOT, f"node_{NODE_ID}")
STATUS_CSV = os.path.join(_BATCH_ROOT, f"download_status_node_{NODE_ID}.csv")
OUTPUT_DIR = _OUTPUT_ROOT
LYRICS_CSV = os.path.join(OUTPUT_DIR, f"master_lyrics_node_{NODE_ID}.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE = "large-v3-turbo"   # faster-whisper model; much better multilingual

# ── Quality-check thresholds ──────────────────────────────────────────────────

_NO_SPEECH_PROB_MAX = 0.60
_AVG_LOGPROB_MIN    = -1.00
_BAD_SEG_RATIO_MAX  = 0.35

_UNIQUE_WORD_RATIO  = 0.25
_NOISE_CHAR_RATIO   = 0.15

_MIN_WORDS_PER_MIN  = 8
_ABSOLUTE_MIN_WORDS = 15

_WINDOW_SEC      = 30
_SAFE_START_FRAC = 0.20
_SAFE_END_FRAC   = 0.70

# ── Internal helpers ──────────────────────────────────────────────────────────

def _write_and_transcribe(
    model,
    audio_np: np.ndarray,
    sr: int = 16000,
    language: str | None = None,
) -> dict:
    """
    Transcribe using faster-whisper. Returns a dict shaped like the original
    whisper output so the rest of the pipeline (quality check, etc.) is unchanged.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, audio_np, sr)
        segments_iter, info = model.transcribe(
            tmp_path,
            task="transcribe",        # always transcribe in native language, never translate
            language=language,        # None → auto-detect per segment
            vad_filter=True,          # skip non-speech (instrumentals) → faster + fewer hallucinations
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
        )
        # Materialise the lazy iterator
        segments = [
            {
                "text":           s.text,
                "start":          s.start,
                "end":            s.end,
                "avg_logprob":    s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in segments_iter
        ]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    full_text = " ".join(s["text"].strip() for s in segments)
    return {
        "text":     full_text,
        "language": info.language,
        "segments": segments,
    }


def _quality_check(result: dict, duration_sec: float) -> tuple:
    segments = result.get("segments", [])

    if not segments:
        return False, "no_segments_returned"

    n_bad = sum(
        1 for s in segments
        if s.get("no_speech_prob", 0.0) > _NO_SPEECH_PROB_MAX
        or s.get("avg_logprob",    0.0) < _AVG_LOGPROB_MIN
    )
    bad_ratio = n_bad / len(segments)
    if bad_ratio >= _BAD_SEG_RATIO_MAX:
        return False, f"too_many_bad_segs ({n_bad}/{len(segments)}, ratio={bad_ratio:.2f})"

    text  = result.get("text", "").strip()
    words = text.split()

    min_words = max(_ABSOLUTE_MIN_WORDS, int(duration_sec / 60 * _MIN_WORDS_PER_MIN))
    if len(words) < min_words:
        return False, (
            f"too_short — {len(words)} words for {duration_sec:.0f}s track "
            f"(need ≥ {min_words})"
        )

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < _UNIQUE_WORD_RATIO:
        return False, f"repetitive_hallucination (unique_ratio={unique_ratio:.2f})"

    noise_ratio = len(re.findall(r"[^a-zA-Z\s'\-\u0000-\uFFFF]", text)) / max(len(text), 1)
    if noise_ratio > _NOISE_CHAR_RATIO:
        return False, f"high_noise_chars (noise_ratio={noise_ratio:.2f})"

    return True, f"ok (bad_seg_ratio={bad_ratio:.2f}, words={len(words)})"


# ── Public API ────────────────────────────────────────────────────────────────

def get_lyrics(
    audio_16k: np.ndarray,
    model,
    sr: int = 16000,
) -> tuple:
    """
    Transcribe audio with language auto-detection and quality-gated fallback.

    Returns
    -------
    (lyrics_text, source_tag, detected_language)
      source_tag: "whisper_full" | "whisper_window" | "whisper_failed"
    """
    duration = len(audio_16k) / sr

    try:
        # faster-whisper exposes detect_language directly on the model
        clip = audio_16k[: sr * 30].astype(np.float32)
        tmp_fd, tmp_clip_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_clip_path, clip, sr)
        lang_probs, _ = model.detect_language(tmp_clip_path)
        os.unlink(tmp_clip_path)
        detected_lang = max(lang_probs, key=lang_probs.get)
        lang_conf     = lang_probs[detected_lang]
        print(f"    [LANG] detected={detected_lang!r}  confidence={lang_conf:.2f}")
    except Exception as exc:
        detected_lang = None
        print(f"    [LANG] detection failed ({exc}) — will auto-detect during transcription")

    # Stage 1: Full-track
    full_result    = _write_and_transcribe(model, audio_16k, sr, language=detected_lang)
    full_text      = full_result.get("text", "").strip()
    passed, reason = _quality_check(full_result, duration)

    if passed:
        print(f"    [QC-FULL] PASS — {reason}")
        return full_text, "whisper_full", detected_lang or full_result.get("language", "unknown")

    print(f"    [QC-FULL] FAIL — {reason} → trying random window")

    # Stage 2: Random-window fallback
    window_samples = int(_WINDOW_SEC * sr)
    safe_start     = int(_SAFE_START_FRAC * duration * sr)
    safe_end       = int(_SAFE_END_FRAC   * duration * sr) - window_samples

    if safe_end <= safe_start:
        mid          = len(audio_16k) // 2
        start_sample = max(0, mid - window_samples // 2)
    else:
        start_sample = random.randint(safe_start, safe_end)

    end_sample   = min(start_sample + window_samples, len(audio_16k))
    window_audio = audio_16k[start_sample:end_sample]
    window_dur   = (end_sample - start_sample) / sr

    print(f"    [QC-WIN]  Window [{start_sample/sr:.1f}s – {end_sample/sr:.1f}s]")

    win_result         = _write_and_transcribe(model, window_audio, sr, language=detected_lang)
    win_text           = win_result.get("text", "").strip()
    w_passed, w_reason = _quality_check(win_result, window_dur)

    if w_passed and win_text:
        print(f"    [QC-WIN]  PASS — {w_reason}")
        return win_text, "whisper_window", detected_lang or win_result.get("language", "unknown")

    # Stage 3: Best-effort fallback
    fallback = full_text if len(full_text) >= len(win_text) else win_text
    print(f"    [QC-WIN]  FAIL — {w_reason} → best-effort fallback")
    return fallback, "whisper_failed", detected_lang or "unknown"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _append_lyrics_log(row: dict) -> None:
    write_header = not os.path.exists(LYRICS_CSV)
    pd.DataFrame([row]).to_csv(LYRICS_CSV, mode="a", header=write_header, index=False)


def _load_done_ids() -> set:
    if os.path.exists(LYRICS_CSV):
        return set(pd.read_csv(LYRICS_CSV)["track_index"].tolist())
    return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"{'=' * 64}")
    print(f"  NODE {NODE_ID}")
    print(f"  WAV dir    : {WAV_DIR}")
    print(f"  Status CSV : {STATUS_CSV}")
    print(f"  Output CSV : {LYRICS_CSV}")
    print(f"{'=' * 64}\n")

    if not os.path.exists(STATUS_CSV):
        print(f"[ERROR] Status CSV not found: {STATUS_CSV}")
        sys.exit(1)

    status_df = pd.read_csv(STATUS_CSV)
    track_df  = (
        status_df[status_df["download_success"] == True]
        [["track_index", "artist", "track"]]
        .drop_duplicates(subset=["track_index"])
        .reset_index(drop=True)
    )

    total = len(track_df)
    if total == 0:
        print("[WARN] No successfully downloaded tracks found in status CSV.")
        return

    track_id_min = track_df["track_index"].min()
    track_id_max = track_df["track_index"].max()
    print(f"Tracks to transcribe : {total}  (IDs {track_id_min} – {track_id_max})\n")

    done_ids = _load_done_ids()
    if done_ids:
        print(f"Resuming — {len(done_ids)} tracks already transcribed.\n")

    from faster_whisper import WhisperModel

    def _load_model(device: str) -> WhisperModel:
        """
        Try compute types in order. After loading, run a 1-second silent
        smoke-test to catch cudaErrorNoKernelImageForDevice (architecture
        mismatch) before processing any real audio.
        Falls back to CPU if CUDA kernels fail at runtime.
        """
        compute_types = ["int8_float16", "int8", "float32"] if device == "cuda" else ["int8"]
        for compute_type in compute_types:
            try:
                print(f"Loading faster-whisper {WHISPER_MODEL_SIZE!r} on {device.upper()} ({compute_type})...")
                m = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
                # Smoke-test: actually run the encoder so CUDA kernel errors surface here
                _tmp_fd, _tmp = tempfile.mkstemp(suffix=".wav")
                os.close(_tmp_fd)
                sf.write(_tmp, np.zeros(16000, dtype=np.float32), 16000)
                list(m.transcribe(_tmp)[0])   # materialise lazy iterator
                os.unlink(_tmp)
                print(f"faster-whisper ready — device={device.upper()}, compute_type={compute_type!r}\n")
                return m
            except ValueError as e:
                print(f"  {compute_type!r} rejected ({e}), trying next...")
            except RuntimeError as e:
                print(f"  CUDA runtime error ({e})")
                break   # no point trying other compute_types on same broken device

        if device == "cuda":
            print("  CUDA not usable on this GPU — falling back to CPU (int8).\n")
            return _load_model("cpu")

        raise RuntimeError("Could not load faster-whisper on any device.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _load_model(device)

    for idx, row in track_df.iterrows():
        track_id = int(row["track_index"])
        artist   = str(row["artist"])
        track    = str(row["track"])

        if track_id in done_ids:
            print(f"  [SKIP] [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
            continue

        print(f"\n{'─' * 64}")
        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        print(f"{'─' * 64}")

        wav_path = os.path.join(WAV_DIR, f"{track_id}.wav")
        if not os.path.exists(wav_path):
            print(f"  [WARN] WAV not found: {wav_path} — skipping.")
            continue

        try:
            audio_16k, _ = librosa.load(wav_path, sr=16000, mono=True)
            duration      = len(audio_16k) / 16000
            print(f"  Loaded — {duration:.1f}s")

            lyrics, lyrics_source, detected_lang = get_lyrics(audio_16k, model, sr=16000)
            preview = lyrics[:100].replace("\n", " ")
            print(f"  [{lyrics_source}] [{detected_lang}] {preview}{'...' if len(lyrics) > 100 else ''}")

            _append_lyrics_log({
                "track_index":       track_id,
                "artist_name":       artist,
                "track_name":        track,
                "duration_seconds":  round(duration, 2),
                "detected_language": detected_lang,
                "lyrics":            lyrics,
                "lyrics_source":     lyrics_source,
            })
            done_ids.add(track_id)
            print("  Saved.")

        except Exception as e:
            print(f"\n  [ERROR] track_{track_id}: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 64}")
    print(f"Node {NODE_ID} complete.  CSV: {LYRICS_CSV}")
    if os.path.exists(LYRICS_CSV):
        ldf = pd.read_csv(LYRICS_CSV)
        print(f"\nLyrics source breakdown ({len(ldf)} tracks):")
        for src, cnt in ldf["lyrics_source"].value_counts().items():
            print(f"  {src:25s} {cnt:6d}  ({100 * cnt / len(ldf):.1f} %)")


if __name__ == "__main__":
    main()
