"""
lyrics.py
─────────
Node-parallel lyrics pipeline. Designed to run as a SLURM job array
(--array=0-9) across 10 nodes, each processing 5 000 tracks.

Node ID is read from $SLURM_ARRAY_TASK_ID automatically.
Can also be passed manually:  python lyrics.py --node_id <N>

Per-node paths (node N):
  WAV dir    : .../batch_1/node_N/          (used for Whisper fallback)
  Status CSV : .../batch_1/download_status_node_N.csv
  Output CSV : .../lyrics/master_lyrics_node_N.csv

Lyrics source waterfall (per track, fully synchronous):
  Stage 1 — LRCLIB   (free, no auth, fast)
  Stage 2 — Genius   (free API key)
  Stage 3 — Whisper  (last resort; only when both APIs miss)

Language is detected from the returned lyrics text via langdetect.
Whisper reports its own detected language natively.

source_tag values in CSV
  "lrclib"          – lyrics from LRCLIB
  "genius"          – lyrics from Genius
  "whisper_full"    – Whisper full-track passed QC
  "whisper_window"  – Whisper windowed passed QC
  "whisper_failed"  – Whisper best-effort (QC failed)
  "instrumental"    – no vocals detected by Whisper (pure music/instrumental)
  "not_found"       – all three stages returned nothing
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
import requests
import soundfile as sf
import torch
import lyricsgenius
from langdetect import detect as _langdetect, LangDetectException

# ── Node ID ────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--node_id", type=int, default=None)
_args, _ = _parser.parse_known_args()

if _args.node_id is not None:
    NODE_ID = _args.node_id
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    NODE_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
else:
    NODE_ID = 8  # hardcoded for node_8 resume run

# ── Paths ──────────────────────────────────────────────────────────────────────
_BATCH_ROOT  = "/scratch3/workspace/skandagatla_umass_edu-dolby/raw_audio_files/batch_2"
_OUTPUT_ROOT = "/scratch3/workspace/skandagatla_umass_edu-dolby/lyrics_syn_3"

WAV_DIR    = os.path.join(_BATCH_ROOT, f"node_{NODE_ID}")
STATUS_CSV = os.path.join(_BATCH_ROOT, f"download_status_node_{NODE_ID}.csv")
OUTPUT_DIR = _OUTPUT_ROOT
LYRICS_CSV = os.path.join(OUTPUT_DIR, f"master_lyrics_node_{NODE_ID}.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
GENIUS_API_KEY     = "LBRWiW9W_l7j_8tHETkxvmbxKhLUFm0lm8ntZEnHGOz3filuDyDTbh4x12MSa066"
WHISPER_MODEL_SIZE = "large-v3-turbo"

# ── Whisper QC thresholds ──────────────────────────────────────────────────────
_NO_SPEECH_PROB_MAX = 0.60
_AVG_LOGPROB_MIN    = -1.00
_BAD_SEG_RATIO_MAX  = 0.35
_UNIQUE_WORD_RATIO  = 0.25
_MIN_WORDS_PER_MIN  = 8
_ABSOLUTE_MIN_WORDS = 15
_WINDOW_SEC         = 30
_SAFE_START_FRAC    = 0.20
_SAFE_END_FRAC      = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# Language detection from text
# ══════════════════════════════════════════════════════════════════════════════

def _detect_lang(text: str) -> str:
    """Detect language of a lyrics string. Returns ISO-639-1 code or 'unknown'."""
    if not text or len(text.strip()) < 15:
        return "unknown"
    try:
        return _langdetect(text)
    except LangDetectException:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LRCLIB  (synchronous, no auth)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_lrclib(artist: str, title: str) -> str | None:
    """Return plain lyrics from LRCLIB, or None on miss/error."""
    try:
        r = requests.get(
            "https://lrclib.net/api/get",
            params={"artist_name": artist, "track_name": title},
            timeout=10,
        )
        if r.status_code == 200:
            data  = r.json()
            plain = (data.get("plainLyrics") or "").strip()
            if plain:
                return plain
            synced = data.get("syncedLyrics") or ""
            plain_from_synced = re.sub(r"\[\d{2}:\d{2}\.\d+\]", "", synced).strip()
            return plain_from_synced or None
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Genius  (synchronous via lyricsgenius)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_genius(text: str) -> str:
    text = re.sub(r"\d*EmbedShare.*", "", text, flags=re.DOTALL)
    return text.strip()


def _fetch_genius(genius_client, artist: str, title: str) -> str | None:
    try:
        song = genius_client.search_song(title, artist, get_full_info=False)
        if song and song.lyrics:
            return _clean_genius(song.lyrics)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Whisper ASR  (synchronous, lazy-loaded)
# ══════════════════════════════════════════════════════════════════════════════

def _whisper_transcribe(model, audio_np: np.ndarray, sr: int, language=None) -> dict:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, audio_np, sr)
        segments_iter, info = model.transcribe(
            tmp_path,
            task="transcribe",
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = [
            {
                "text":           s.text,
                "avg_logprob":    s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in segments_iter
        ]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "text":     " ".join(s["text"].strip() for s in segments),
        "language": info.language,
        "segments": segments,
    }


# Threshold: if avg no_speech_prob across all segments exceeds this, treat as instrumental.
# VAD already filters silence; anything left with this high a no_speech_prob is music/noise.
_INSTRUMENTAL_NO_SPEECH_THRESHOLD = 0.85


def _is_instrumental(result: dict) -> bool:
    """
    Return True if Whisper's output strongly indicates no vocals.
    Two signals:
      1. No segments at all (VAD filtered everything — pure silence/music)
      2. All segments have very high no_speech_prob (music detected, no speech)
    """
    segments = result.get("segments", [])
    if not segments:
        return True
    avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
    return avg_no_speech > _INSTRUMENTAL_NO_SPEECH_THRESHOLD


def _whisper_qc(result: dict, duration_sec: float) -> tuple[bool, str]:
    segments = result.get("segments", [])
    if not segments:
        return False, "no_segments"

    n_bad     = sum(
        1 for s in segments
        if s.get("no_speech_prob", 0.0) > _NO_SPEECH_PROB_MAX
        or s.get("avg_logprob",    0.0) < _AVG_LOGPROB_MIN
    )
    bad_ratio = n_bad / len(segments)
    if bad_ratio >= _BAD_SEG_RATIO_MAX:
        return False, f"bad_segs({bad_ratio:.2f})"

    words = result.get("text", "").strip().split()
    min_w = max(_ABSOLUTE_MIN_WORDS, int(duration_sec / 60 * _MIN_WORDS_PER_MIN))
    if len(words) < min_w:
        return False, f"too_short({len(words)}<{min_w})"
    if len(set(words)) / max(len(words), 1) < _UNIQUE_WORD_RATIO:
        return False, "repetitive"

    return True, f"ok(words={len(words)})"


def _run_whisper(wav_path: str, model) -> tuple[str, str, str]:
    """
    Returns (lyrics, source_tag, detected_language).
    source_tag: whisper_full | whisper_window | whisper_failed | instrumental
    """
    audio_16k, _ = librosa.load(wav_path, sr=16000, mono=True)
    sr            = 16000
    duration      = len(audio_16k) / sr

    detected_lang = None
    try:
        clip = audio_16k[: sr * 30].astype(np.float32)
        tmp_fd, tmp_clip = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_clip, clip, sr)
        lang_probs, _ = model.detect_language(tmp_clip)
        os.unlink(tmp_clip)
        detected_lang = max(lang_probs, key=lang_probs.get)
        print(f"    [LANG] {detected_lang!r}")
    except Exception:
        pass

    full_res       = _whisper_transcribe(model, audio_16k, sr, detected_lang)
    full_text      = full_res["text"].strip()
    passed, reason = _whisper_qc(full_res, duration)

    if passed:
        print(f"    [WHISPER-FULL] PASS — {reason}")
        return full_text, "whisper_full", detected_lang or full_res.get("language", "unknown")

    print(f"    [WHISPER-FULL] FAIL — {reason} → window fallback")

    w_samp     = int(_WINDOW_SEC * sr)
    safe_start = int(_SAFE_START_FRAC * duration * sr)
    safe_end   = int(_SAFE_END_FRAC   * duration * sr) - w_samp
    start_s    = (
        random.randint(safe_start, safe_end) if safe_end > safe_start
        else max(0, len(audio_16k) // 2 - w_samp // 2)
    )
    end_s     = min(start_s + w_samp, len(audio_16k))
    win_audio = audio_16k[start_s:end_s]

    win_res         = _whisper_transcribe(model, win_audio, sr, detected_lang)
    win_text        = win_res["text"].strip()
    w_passed, w_rsn = _whisper_qc(win_res, (end_s - start_s) / sr)

    if w_passed and win_text:
        print(f"    [WHISPER-WIN] PASS — {w_rsn}")
        return win_text, "whisper_window", detected_lang or win_res.get("language", "unknown")

    # Both full-track and window failed — check if this is simply an instrumental
    if _is_instrumental(full_res) and _is_instrumental(win_res):
        print(f"    [INSTRUMENTAL] no speech detected in full track or window")
        return "", "instrumental", "n/a"

    fallback = full_text if len(full_text) >= len(win_text) else win_text
    print(f"    [WHISPER-WIN] FAIL — {w_rsn} → best-effort")
    return fallback, "whisper_failed", detected_lang or "unknown"


def _load_whisper() -> object:
    """Load faster-whisper with automatic compute_type and CPU fallback."""
    from faster_whisper import WhisperModel

    def _try_load(device: str):
        compute_types = ["int8_float16", "int8", "float32"] if device == "cuda" else ["int8"]
        for ct in compute_types:
            try:
                print(f"  [WHISPER] Loading {WHISPER_MODEL_SIZE!r} on {device.upper()} ({ct})...")
                m = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=ct)
                # Smoke-test to catch architecture mismatches before real audio
                tmp_fd, tmp = tempfile.mkstemp(suffix=".wav")
                os.close(tmp_fd)
                sf.write(tmp, np.zeros(16000, dtype=np.float32), 16000)
                list(m.transcribe(tmp)[0])
                os.unlink(tmp)
                print(f"  [WHISPER] Ready — {device.upper()}, {ct}\n")
                return m
            except ValueError as e:
                print(f"  [WHISPER] {ct!r} rejected: {e}")
            except RuntimeError as e:
                print(f"  [WHISPER] CUDA runtime error: {e}")
                break   # architecture mismatch — skip remaining compute types
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _try_load(device)
    if model is None and device == "cuda":
        print("  [WHISPER] Falling back to CPU...")
        model = _try_load("cpu")
    if model is None:
        raise RuntimeError("Could not load faster-whisper on any device.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def _append_row(row: dict) -> None:
    """Write one row to CSV immediately — visible to tail -f in real time."""
    row = dict(row)
    if row.get("lyrics"):
        row["lyrics"] = " ".join(row["lyrics"].splitlines())  # flatten to single line
    write_header = not os.path.exists(LYRICS_CSV)
    pd.DataFrame([row]).to_csv(LYRICS_CSV, mode="a", header=write_header, index=False)


def _load_done_ids() -> set:
    if os.path.exists(LYRICS_CSV):
        return set(pd.read_csv(LYRICS_CSV)["track_index"].tolist())
    return set()


# ══════════════════════════════════════════════════════════════════════════════
# Main  (fully synchronous — one track at a time, CSV written after each)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"{'=' * 64}")
    print(f"  NODE {NODE_ID}  —  Lyrics Pipeline")
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
        print("[WARN] No successfully downloaded tracks found.")
        return

    print(f"Tracks to process : {total}  "
          f"(IDs {track_df['track_index'].min()} – {track_df['track_index'].max()})\n")

    done_ids = _load_done_ids()
    if done_ids:
        print(f"Resuming — {len(done_ids)} tracks already done.\n")

    genius_client = lyricsgenius.Genius(
        GENIUS_API_KEY,
        verbose=False,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        timeout=10,
    )

    whisper_model = None   # lazy-loaded only if Whisper is ever needed

    for idx, row in track_df.iterrows():
        track_id = int(row["track_index"])
        artist   = str(row["artist"])
        track    = str(row["track"])

        if track_id in done_ids:
            print(f"  [SKIP] [{idx+1}/{total}] track_{track_id}")
            continue

        print(f"\n[{idx+1}/{total}] track_{track_id} | {artist} — {track}")

        lyrics = source = lang = None

        # ── Stage 1: LRCLIB ───────────────────────────────────────────────────
        lyrics = _fetch_lrclib(artist, track)
        if lyrics:
            source = "lrclib"
            lang   = _detect_lang(lyrics)
            print(f"  [LRCLIB]  lang={lang!r}  chars={len(lyrics)}")

        # ── Stage 2: Genius ───────────────────────────────────────────────────
        if not lyrics:
            lyrics = _fetch_genius(genius_client, artist, track)
            if lyrics:
                source = "genius"
                lang   = _detect_lang(lyrics)
                print(f"  [GENIUS]  lang={lang!r}  chars={len(lyrics)}")

        # ── Stage 3: Whisper ASR ──────────────────────────────────────────────
        if not lyrics:
            print(f"  [WHISPER] API miss — falling back to ASR...")
            wav_path = os.path.join(WAV_DIR, f"{track_id}.wav")
            if not os.path.exists(wav_path):
                print(f"    [WARN] WAV not found: {wav_path}")
                lyrics, source, lang = "", "not_found", "unknown"
            else:
                if whisper_model is None:
                    whisper_model = _load_whisper()
                try:
                    lyrics, source, lang = _run_whisper(wav_path, whisper_model)
                    # lang is already set by Whisper's own detect_language
                except Exception as e:
                    print(f"    [ERROR] {e}")
                    traceback.print_exc()
                    lyrics, source, lang = "", "not_found", "unknown"

        # ── Duration ──────────────────────────────────────────────────────────
        duration = 0.0
        wav_path = os.path.join(WAV_DIR, f"{track_id}.wav")
        if os.path.exists(wav_path):
            try:
                duration = round(sf.info(wav_path).duration, 2)
            except Exception:
                pass

        # ── Write to CSV immediately ───────────────────────────────────────────
        _append_row({
            "track_index":       track_id,
            "artist_name":       artist,
            "track_name":        track,
            "duration_seconds":  duration,
            "detected_language": lang,
            "lyrics":            lyrics,
            "lyrics_source":     source,
        })
        done_ids.add(track_id)
        preview = (lyrics or "")[:80].replace("\n", " ")
        print(f"  → saved  source={source!r}  lang={lang!r}  preview: {preview}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"Node {NODE_ID} complete.  CSV: {LYRICS_CSV}")
    if os.path.exists(LYRICS_CSV):
        ldf = pd.read_csv(LYRICS_CSV)
        print(f"\nLyrics source breakdown ({len(ldf)} tracks):")
        for src, cnt in ldf["lyrics_source"].value_counts().items():
            print(f"  {src:25s} {cnt:6d}  ({100 * cnt / len(ldf):.1f} %)")
        print(f"\nLanguage breakdown:")
        for lng, cnt in ldf["detected_language"].value_counts().head(15).items():
            print(f"  {lng:10s} {cnt:6d}  ({100 * cnt / len(ldf):.1f} %)")


if __name__ == "__main__":
    main()
