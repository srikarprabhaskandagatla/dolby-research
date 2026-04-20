"""
═══════════════════════════════════════════════════════════════════════════════
 LLM-Based Sequential Music Recommendation — Single-File Embedding Pipeline
 Based on: E4SRec (Li et al., WWW 2024)
 Dataset:  LastFM 1K  |  Audio Source: YouTube API  |  ASR: Whisper
───────────────────────────────────────────────────────────────────────────────
 EXECUTION ORDER:
   Phase 0  — Whisper ASR for all tracks → builds master_lyrics.csv
   Phase 1  — CLAP audio        (all tracks) → audio/clap/track_{id}.pt
   Phase 2  — MERT audio        (all tracks) → audio/mert/track_{id}.pt
   Phase 3  — Music2Vec audio   (all tracks) → audio/music2vec/track_{id}.pt
   Phase 4  — Encodec audio     (all tracks) → audio/encodec/track_{id}.pt
   Phase 5  — MusiCNN audio     (all tracks) → audio/musicnn/track_{id}.pt
   Phase 6  — MiniLM text       (all tracks) → text/minilm/track_{id}.pt
   Phase 7  — BGE-M3 text       (all tracks) → text/bgem3/track_{id}.pt
   Phase 8  — all-mpnet text    (all tracks) → text/mpnet/track_{id}.pt
   Phase 9  — Multilingual text (all tracks) → text/multilingual/track_{id}.pt
   Phase 10 — BERT text         (all tracks) → text/bert/track_{id}.pt

 Each phase loads ONE model, runs ALL tracks, then unloads before next phase.
 Every phase is fully resumable — already-completed tracks are skipped.
 master_lyrics.csv maps track_id → lyrics and is updated after Phase 0.

 OUTPUT STRUCTURE:
   output/
   ├── master_lyrics.csv
   ├── audio/
   │   ├── clap/         track_0.pt  [512-dim]
   │   ├── mert/         track_0.pt  [768-dim]
   │   ├── music2vec/    track_0.pt  [768-dim]
   │   ├── encodec/      track_0.pt  [128-dim]
   │   └── musicnn/      track_0.pt  [200-dim]
   └── text/
       ├── minilm/       track_0.pt  [384-dim]
       ├── bgem3/        track_0.pt  [1024-dim]
       ├── mpnet/        track_0.pt  [768-dim]
       ├── multilingual/ track_0.pt  [768-dim]
       └── bert/         track_0.pt  [768-dim]

 Each .pt file: {"track_index", "artist_name", "track_name", "embedding"}

 INSTALL:
   pip install -U torch torchaudio transformers sentence-transformers
               librosa yt-dlp ytmusicapi openai-whisper requests musicnn
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import gc
import glob
import urllib.parse

import numpy as np
import pandas as pd
import requests
import torch
import librosa
import whisper
import yt_dlp
from ytmusicapi import YTMusic

from transformers import (
    ClapModel, ClapProcessor,
    Wav2Vec2FeatureExtractor, AutoModel,
    EncodecModel, AutoProcessor,
    BertModel, BertTokenizer,
)
from sentence_transformers import SentenceTransformer

try:
    from musicnn.extractor import extractor as musicnn_extractor
    MUSICNN_AVAILABLE = True
except ImportError:
    print("[WARNING] musicnn not installed — Phase 5 will be skipped.")
    print("          Run: pip install musicnn\n")
    MUSICNN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Input CSV — pre-processed LastFM 1K session data
CSV_HM_PATH = "/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/extract_audio_pipeline/output/lastfm_unique_tracks_formatted.csv"

# Root output directory — all embeddings, lyrics CSV, and temp files go here
OUTPUT_ROOT = "/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/output"

# Derived paths — all nested under OUTPUT_ROOT
MASTER_LYRICS_PATH = os.path.join(OUTPUT_ROOT, "master_lyrics.csv")
TEMP_AUDIO_DIR     = os.path.join(OUTPUT_ROOT, "tmp_audio")

AUDIO_DIR          = os.path.join(OUTPUT_ROOT, "audio")
DIR_CLAP           = os.path.join(AUDIO_DIR,   "clap")
DIR_MERT           = os.path.join(AUDIO_DIR,   "mert")
DIR_MUSIC2VEC      = os.path.join(AUDIO_DIR,   "music2vec")
DIR_ENCODEC        = os.path.join(AUDIO_DIR,   "encodec")
DIR_MUSICNN        = os.path.join(AUDIO_DIR,   "musicnn")

TEXT_DIR           = os.path.join(OUTPUT_ROOT, "text")
DIR_MINILM         = os.path.join(TEXT_DIR,    "minilm")
DIR_BGEM3          = os.path.join(TEXT_DIR,    "bgem3")
DIR_MPNET          = os.path.join(TEXT_DIR,    "mpnet")
DIR_MULTILINGUAL   = os.path.join(TEXT_DIR,    "multilingual")
DIR_BERT           = os.path.join(TEXT_DIR,    "bert")

# ── Validate input + create all output directories ────────────────────────────
if not os.path.exists(CSV_HM_PATH):
    raise FileNotFoundError(
        f"Input CSV not found:\n  {CSV_HM_PATH}\n"
        "Check the path and re-run."
    )

for _d in [
    OUTPUT_ROOT, TEMP_AUDIO_DIR,
    DIR_CLAP, DIR_MERT, DIR_MUSIC2VEC, DIR_ENCODEC, DIR_MUSICNN,
    DIR_MINILM, DIR_BGEM3, DIR_MPNET, DIR_MULTILINGUAL, DIR_BERT,
]:
    os.makedirs(_d, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def track_done(track_id, out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, f"track_{track_id}.pt"))


def save_embedding(track_id, artist: str, track: str,
                   embedding: torch.Tensor, out_dir: str) -> str:
    path = os.path.join(out_dir, f"track_{track_id}.pt")
    torch.save({
        "track_index": track_id,
        "artist_name": artist,
        "track_name":  track,
        "embedding":   embedding,
    }, path)
    return path


def search_youtube(artist: str, track: str):
    ytmusic = YTMusic()
    results = ytmusic.search(f"{artist} {track}", filter="songs") \
              or ytmusic.search(f"{artist} {track}")
    return results[0] if results else None


def download_wav(video_url: str, track_id) -> tuple:
    """
    Download audio as WAV via yt-dlp + FFmpeg.
    Returns (wav_path, duration_seconds).

    Format fallback chain:
      opus → m4a → any audio-only → any available
    This prevents 'Requested format is not available' errors.
    """
    out_template = os.path.join(TEMP_AUDIO_DIR, f"{track_id}.%(ext)s")

    ydl_opts = {
        # Fallback chain — yt-dlp tries each option left-to-right
        "format": "bestaudio[ext=opus]/bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        # Do NOT set cookiefile unless you have an actual cookies.txt on disk
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(video_url, download=True)
        duration = info.get("duration", 0)
        wav_path = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    # Guard: FFmpeg may produce a differently-named file
    if not os.path.exists(wav_path):
        candidates = glob.glob(os.path.join(TEMP_AUDIO_DIR, f"{track_id}.*"))
        wav_candidates = [f for f in candidates if f.endswith(".wav")]
        if not wav_candidates:
            raise RuntimeError(
                f"WAV not found after download. Files present: {candidates}"
            )
        wav_path = wav_candidates[0]

    return wav_path, duration


def cleanup(track_id):
    for f in glob.glob(os.path.join(TEMP_AUDIO_DIR, f"{track_id}.*")):
        try:
            os.remove(f)
        except Exception:
            pass


def get_genre(artist: str, track: str) -> str:
    try:
        q   = urllib.parse.quote(f"{artist} {track}")
        res = requests.get(
            f"https://itunes.apple.com/search?term={q}&entity=song&limit=1",
            timeout=5
        ).json()
        if res.get("resultCount", 0) > 0:
            return res["results"][0].get("primaryGenreName", "Unknown")
    except Exception:
        pass
    return "Unknown"


def load_master_lyrics() -> pd.DataFrame:
    if not os.path.exists(MASTER_LYRICS_PATH):
        raise FileNotFoundError(
            "master_lyrics.csv not found. Run Phase 0 first."
        )
    df = pd.read_csv(MASTER_LYRICS_PATH)
    df["whisper_lyrics"] = df["whisper_lyrics"].fillna("[Instrumental]")
    return df


def unload(*models):
    """Delete model references and free GPU memory."""
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def banner(phase: int, name: str, hf_id: str, dim: int, out_dir: str):
    print(f"\n{'='*64}")
    print(f"  PHASE {phase:02d} — {name}")
    print(f"  Model  : {hf_id}")
    print(f"  Dim    : {dim}")
    print(f"  Output : {out_dir}")
    print(f"{'='*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLAP HELPERS
# safe for both plain Tensor and BaseModelOutputWithPooling return types,
# which vary across transformers versions.
# ─────────────────────────────────────────────────────────────────────────────

def _clap_audio_emb(model, proc, audio_np: np.ndarray) -> torch.Tensor:
    """
    Embed a raw audio waveform with CLAP's audio tower.
    Returns a [512] float32 CPU tensor.
    """
    inputs = proc(
        audio=audio_np, return_tensors="pt", sampling_rate=48000
    ).to(device)
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    # get_audio_features returns a plain Tensor in recent transformers,
    # but older versions return BaseModelOutputWithPooling — handle both.
    tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
    return tensor.squeeze(0).detach().cpu()   # [512]


def _clap_text_emb(model, proc, text: str) -> torch.Tensor:
    """
    Embed a text string with CLAP's text tower.
    Returns a [512] float32 CPU tensor.
    """
    inputs = proc(
        text=[text], return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
    return tensor.squeeze(0).detach().cpu()   # [512]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — WHISPER ASR → master_lyrics.csv
# ─────────────────────────────────────────────────────────────────────────────
def phase_00_whisper(df: pd.DataFrame):
    banner(0, "Whisper ASR + CLAP alignment → master_lyrics.csv",
           "openai/whisper-base  +  laion/larger_clap_music_and_speech", 0,
           MASTER_LYRICS_PATH)

    # Resumption — load already-processed track IDs
    done_ids = set()
    if os.path.exists(MASTER_LYRICS_PATH):
        done_ids = set(pd.read_csv(MASTER_LYRICS_PATH)["track_index"].tolist())
        print(f"  Resuming — {len(done_ids)} tracks already in CSV.\n")

    print("  Loading Whisper (base)...")
    whisper_model = whisper.load_model("base").to(device)

    print("  Loading LAION-CLAP (laion/larger_clap_music_and_speech)...")
    clap_proc = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    clap_mdl  = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(device)
    clap_mdl.eval()
    print("  Models loaded.\n")

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_id in done_ids:
            print(f"  [SKIP] [{idx+1}/{total}] track_{track_id}")
            continue

        print(f"\n  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        wav_path = None
        try:
            # Search + download
            result = search_youtube(artist, track)
            if not result:
                print("    Not found on YouTube Music — skipping.")
                continue

            video_id  = result["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            genre     = get_genre(artist, track)
            wav_path, duration = download_wav(video_url, track_id)
            print(f"    Downloaded ({duration}s) → {wav_path}")

            # Whisper ASR
            asr_result  = whisper_model.transcribe(wav_path, fp16=(device == "cuda"))
            lyrics_text = asr_result["text"].strip() or "[Instrumental]"
            print(f"    Lyrics: {lyrics_text[:80]}{'...' if len(lyrics_text) > 80 else ''}")

            # CLAP cross-modal alignment score
            # Cosine similarity between audio and lyric embeddings in CLAP's
            # shared 512-dim space. Near 0 → instrumental or bad ASR transcript.
            audio_48k = librosa.load(wav_path, sr=48000, mono=True)[0]
            a_emb     = _clap_audio_emb(clap_mdl, clap_proc, audio_48k)
            t_emb     = _clap_text_emb(clap_mdl, clap_proc, lyrics_text)
            score     = float(
                torch.nn.functional.cosine_similarity(
                    a_emb.unsqueeze(0), t_emb.unsqueeze(0)
                ).item()
            )
            print(f"    CLAP alignment score: {score:.4f}")

            # Append row to CSV immediately — crash-safe one-row-at-a-time write
            new_row = {
                "track_index":          track_id,
                "artist_name":          artist,
                "track_name":           track,
                "genre":                genre,
                "youtube_video_id":     video_id,
                "duration_seconds":     duration,
                "whisper_lyrics":       lyrics_text,
                "clap_alignment_score": score,
            }
            write_header = not os.path.exists(MASTER_LYRICS_PATH)
            pd.DataFrame([new_row]).to_csv(
                MASTER_LYRICS_PATH, mode="a", header=write_header, index=False
            )
            done_ids.add(track_id)
            print(f"    Appended to master_lyrics.csv")

        except Exception as e:
            print(f"    [ERROR] {e}")
        finally:
            if wav_path:
                cleanup(track_id)

    unload(whisper_model, clap_mdl)
    print(f"\n  Phase 0 complete → {MASTER_LYRICS_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CLAP AUDIO  |  512-dim  |  48kHz
# Paper: Wu et al., ICASSP 2023 — https://arxiv.org/abs/2211.06687
# HF:    laion/larger_clap_music_and_speech
# Note:  Audio + text towers share a 512-dim latent space.
#        clap_alignment_score in master_lyrics.csv is the cosine similarity
#        between this audio embedding and the CLAP text embedding of lyrics.
# ─────────────────────────────────────────────────────────────────────────────
def phase_01_clap(df: pd.DataFrame):
    banner(1, "CLAP Audio Embedder",
           "laion/larger_clap_music_and_speech", 512, DIR_CLAP)

    proc  = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(device)
    model.eval()

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_done(track_id, DIR_CLAP):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        wav_path = None
        try:
            result = search_youtube(artist, track)
            if not result:
                print("    Not found — skipping.")
                continue
            wav_path, _ = download_wav(
                f"https://www.youtube.com/watch?v={result['videoId']}", track_id
            )
            audio = librosa.load(wav_path, sr=48000, mono=True)[0]
            emb   = _clap_audio_emb(model, proc, audio)   # [512]
            path  = save_embedding(track_id, artist, track, emb, DIR_CLAP)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")
        finally:
            if wav_path:
                cleanup(track_id)

    unload(model, proc)
    print(f"\n  Phase 1 complete → {DIR_CLAP}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — MERT AUDIO  |  768-dim  |  24kHz
# Paper: Li et al., ICLR 2024 — https://arxiv.org/abs/2306.00107
# HF:    m-a-p/MERT-v1-330M
# Note:  Dual-teacher SSL (RVQ-VAE acoustic + CQT musical). Captures pitch,
#        timbre, rhythm. #2 on MRS benchmark. Use 95M if VRAM < 16 GB.
# ─────────────────────────────────────────────────────────────────────────────
def phase_02_mert(df: pd.DataFrame):
    banner(2, "MERT Audio Embedder",
           "m-a-p/MERT-v1-330M", 768, DIR_MERT)

    proc  = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    ).to(device)
    model.eval()

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_done(track_id, DIR_MERT):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        wav_path = None
        try:
            result = search_youtube(artist, track)
            if not result:
                print("    Not found — skipping.")
                continue
            wav_path, _ = download_wav(
                f"https://www.youtube.com/watch?v={result['videoId']}", track_id
            )
            audio = librosa.load(wav_path, sr=24000, mono=True)[0]
            with torch.no_grad():
                inputs  = proc(audio, sampling_rate=24000, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                # Mean-pool last hidden layer over time → [768]
                emb     = outputs.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu()
            path = save_embedding(track_id, artist, track, emb, DIR_MERT)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")
        finally:
            if wav_path:
                cleanup(track_id)

    unload(model, proc)
    print(f"\n  Phase 2 complete → {DIR_MERT}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — MUSIC2VEC AUDIO  |  768-dim  |  16kHz
# Paper: Li et al., ISMIR 2022 — https://arxiv.org/abs/2212.02508
# HF:    m-a-p/music2vec-v1
# Note:  Data2Vec/BYOL pretraining. Same dim as MERT (768) → clean ablation
#        that mirrors E4SRec's BPR vs SASRec embedding quality comparison.
# ─────────────────────────────────────────────────────────────────────────────
def phase_03_music2vec(df: pd.DataFrame):
    banner(3, "Music2Vec Audio Embedder",
           "m-a-p/music2vec-v1", 768, DIR_MUSIC2VEC)

    proc  = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/music2vec-v1", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "m-a-p/music2vec-v1", trust_remote_code=True
    ).to(device)
    model.eval()

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_done(track_id, DIR_MUSIC2VEC):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        wav_path = None
        try:
            result = search_youtube(artist, track)
            if not result:
                print("    Not found — skipping.")
                continue
            wav_path, _ = download_wav(
                f"https://www.youtube.com/watch?v={result['videoId']}", track_id
            )
            audio = librosa.load(wav_path, sr=16000, mono=True)[0]
            with torch.no_grad():
                inputs  = proc(audio, sampling_rate=16000, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                # Mean-pool last hidden layer over time → [768]
                emb     = outputs.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu()
            path = save_embedding(track_id, artist, track, emb, DIR_MUSIC2VEC)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")
        finally:
            if wav_path:
                cleanup(track_id)

    unload(model, proc)
    print(f"\n  Phase 3 complete → {DIR_MUSIC2VEC}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — ENCODEC AUDIO  |  128-dim  |  24kHz
# Paper: Défossez et al., TMLR 2023 — https://arxiv.org/abs/2210.13438
# HF:    facebook/encodec_24khz
# Note:  Pre-VQ encoder continuous states, mean-pooled over time → 128-dim.
#        Codec tokens are natively designed for transformer injection (Axis 4).
#        #3 MRS, #1 genre prediction — strong genre signal for LastFM 1K.
# ─────────────────────────────────────────────────────────────────────────────
def phase_04_encodec(df: pd.DataFrame):
    banner(4, "Encodec Audio Embedder",
           "facebook/encodec_24khz", 128, DIR_ENCODEC)

    proc  = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    model.eval()

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_done(track_id, DIR_ENCODEC):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        wav_path = None
        try:
            result = search_youtube(artist, track)
            if not result:
                print("    Not found — skipping.")
                continue
            wav_path, _ = download_wav(
                f"https://www.youtube.com/watch?v={result['videoId']}", track_id
            )
            audio = librosa.load(wav_path, sr=24000, mono=True)[0]
            with torch.no_grad():
                inputs      = proc(
                    raw_audio=audio, sampling_rate=24000, return_tensors="pt"
                ).to(device)
                # encoder() → [batch, 128, time_frames] — mean over time → [128]
                encoder_out = model.encoder(inputs["input_values"])
                emb         = encoder_out.mean(dim=-1).squeeze(0).detach().cpu()
            path = save_embedding(track_id, artist, track, emb, DIR_ENCODEC)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")
        finally:
            if wav_path:
                cleanup(track_id)

    unload(model, proc)
    print(f"\n  Phase 4 complete → {DIR_ENCODEC}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — MUSICNN AUDIO  |  200-dim  |  16kHz (file-in)
# Paper: Pons & Serra, arXiv 2019 — https://arxiv.org/abs/1909.06654
# Install: pip install musicnn
# Note:  Trained on LastFM/MSD crowd-sourced tags — same domain as dataset.
#        #1 on MRS benchmark across ALL recommendation experiments.
#        penultimate layer [n_frames, 200] → mean over frames → 200-dim.
# ─────────────────────────────────────────────────────────────────────────────
# def phase_05_musicnn(df: pd.DataFrame):
#     banner(5, "MusiCNN Audio Embedder",
#            "pip install musicnn  (MSD_musicnn)", 200, DIR_MUSICNN)

#     if not MUSICNN_AVAILABLE:
#         print("  [SKIP] musicnn not installed. Run: pip install musicnn\n")
#         return

#     total = len(df)
#     for idx, row in df.iterrows():
#         track_id = row["track_index"]
#         artist   = row["artist_name"]
#         track    = row["track_name"]

#         if track_done(track_id, DIR_MUSICNN):
#             print(f"  [SKIP] track_{track_id}")
#             continue

#         print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
#         wav_path = None
#         try:
#             result = search_youtube(artist, track)
#             if not result:
#                 print("    Not found — skipping.")
#                 continue
#             wav_path, _ = download_wav(
#                 f"https://www.youtube.com/watch?v={result['videoId']}", track_id
#             )
#             # MusiCNN reads the WAV file directly at 16kHz internally
#             _, _, features = musicnn_extractor(
#                 wav_path,
#                 model="MSD_musicnn",
#                 input_length=3,
#                 input_overlap=1.5,
#                 extract_features=True,
#             )
#             # features["penultimate"]: np.ndarray [n_frames, 200]
#             # Mean over temporal frames → fixed 200-dim vector
#             emb  = torch.tensor(
#                 features["penultimate"].mean(axis=0), dtype=torch.float32
#             )
#             path = save_embedding(track_id, artist, track, emb, DIR_MUSICNN)
#             print(f"    Saved {list(emb.shape)} → {path}")
#         except Exception as e:
#             print(f"    [ERROR] {e}")
#         finally:
#             if wav_path:
#                 cleanup(track_id)

#     print(f"\n  Phase 5 complete → {DIR_MUSICNN}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — MFCC AUDIO  |  128-dim  |  22kHz  (replaces MusiCNN)
# Library: librosa (already installed)
# Note:  MusiCNN requires numpy<1.17 which is incompatible with Python 3.10.
#        MFCC (Mel-frequency cepstral coefficients) are the standard acoustic
#        feature used in music tagging and genre classification — the same
#        signal MusiCNN was trained to learn from. 40 MFCCs × mean+std
#        over time = 80-dim vector. Doubled to 128-dim with delta features.
# ─────────────────────────────────────────────────────────────────────────────
def phase_05_mfcc(df: pd.DataFrame):
    banner(5, "MFCC Audio Embedder (librosa — replaces MusiCNN)",
           "librosa.feature.mfcc", 128, DIR_MUSICNN)

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_done(track_id, DIR_MUSICNN):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        wav_path = None
        try:
            result = search_youtube(artist, track)
            if not result:
                print("    Not found — skipping.")
                continue
            wav_path, _ = download_wav(
                f"https://www.youtube.com/watch?v={result['videoId']}", track_id
            )
            audio = librosa.load(wav_path, sr=22050, mono=True)[0]

            # 40 MFCCs → mean + std over time = 80-dim
            mfcc       = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
            mfcc_mean  = mfcc.mean(axis=1)   # [40]
            mfcc_std   = mfcc.std(axis=1)    # [40]

            # Delta MFCCs (velocity) → mean + std = another 48-dim
            delta      = librosa.feature.delta(mfcc)
            delta_mean = delta.mean(axis=1)  # [40]
            delta_std  = delta.std(axis=1)   # [40]

            # Concatenate → [160] then slice to [128] for consistency
            emb = torch.tensor(
                np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std]),
                dtype=torch.float32
            )[:128]
            # emb shape: [128]

            path = save_embedding(track_id, artist, track, emb, DIR_MUSICNN)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")
        finally:
            if wav_path:
                cleanup(track_id)

    print(f"\n  Phase 5 complete → {DIR_MUSICNN}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — MINILM TEXT  |  384-dim
# Paper: Wang et al., NeurIPS 2020 — https://arxiv.org/abs/2002.10957
# HF:    sentence-transformers/all-MiniLM-L6-v2
# Note:  6× faster than all-mpnet at ~95% quality. No audio download —
#        reads Whisper lyrics directly from master_lyrics.csv.
# ─────────────────────────────────────────────────────────────────────────────
def phase_06_minilm(df: pd.DataFrame):
    banner(6, "MiniLM Text Embedder",
           "sentence-transformers/all-MiniLM-L6-v2", 384, DIR_MINILM)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]
        lyrics   = str(row["whisper_lyrics"])

        if track_done(track_id, DIR_MINILM):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        try:
            emb  = torch.tensor(model.encode(lyrics), dtype=torch.float32)  # [384]
            path = save_embedding(track_id, artist, track, emb, DIR_MINILM)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    unload(model)
    print(f"\n  Phase 6 complete → {DIR_MINILM}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — BGE-M3 TEXT  |  1024-dim
# Paper: Chen et al., 2024 — https://arxiv.org/abs/2402.03216
# HF:    BAAI/bge-m3
# Note:  Multi-lingual, multi-granularity. Richest semantic representation.
#        Handles non-English Whisper transcripts from LastFM 1K's diverse songs.
# ─────────────────────────────────────────────────────────────────────────────
def phase_07_bgem3(df: pd.DataFrame):
    banner(7, "BGE-M3 Text Embedder",
           "BAAI/bge-m3", 1024, DIR_BGEM3)

    model = SentenceTransformer("BAAI/bge-m3", device=device)

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]
        lyrics   = str(row["whisper_lyrics"])

        if track_done(track_id, DIR_BGEM3):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        try:
            emb  = torch.tensor(model.encode(lyrics), dtype=torch.float32)  # [1024]
            path = save_embedding(track_id, artist, track, emb, DIR_BGEM3)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    unload(model)
    print(f"\n  Phase 7 complete → {DIR_BGEM3}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — all-mpnet TEXT  |  768-dim
# Paper: Reimers & Gurevych, EMNLP 2019 — https://arxiv.org/abs/1908.10084
# HF:    sentence-transformers/all-mpnet-base-v2
# Note:  #1 on SBERT leaderboard. Robust to noisy Whisper ASR output.
#        768-dim matches MERT/Music2Vec — clean cross-modal ablation.
# ─────────────────────────────────────────────────────────────────────────────
def phase_08_mpnet(df: pd.DataFrame):
    banner(8, "all-mpnet Text Embedder",
           "sentence-transformers/all-mpnet-base-v2", 768, DIR_MPNET)

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]
        lyrics   = str(row["whisper_lyrics"])

        if track_done(track_id, DIR_MPNET):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        try:
            emb  = torch.tensor(model.encode(lyrics), dtype=torch.float32)  # [768]
            path = save_embedding(track_id, artist, track, emb, DIR_MPNET)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    unload(model)
    print(f"\n  Phase 8 complete → {DIR_MPNET}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 — MULTILINGUAL TEXT  |  768-dim  |  50+ languages
# Paper: Reimers & Gurevych, EMNLP 2020 — https://arxiv.org/abs/2004.09813
# HF:    sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# Note:  Critical for LastFM 1K — songs in French, Korean, German etc.
#        Monolingual models produce near-random embeddings for non-English
#        Whisper transcripts. This prevents that silent failure mode.
# ─────────────────────────────────────────────────────────────────────────────
def phase_09_multilingual(df: pd.DataFrame):
    banner(9, "Multilingual-mpnet Text Embedder",
           "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 768,
           DIR_MULTILINGUAL)

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=device
    )

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]
        lyrics   = str(row["whisper_lyrics"])

        if track_done(track_id, DIR_MULTILINGUAL):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        try:
            emb  = torch.tensor(model.encode(lyrics), dtype=torch.float32)  # [768]
            path = save_embedding(track_id, artist, track, emb, DIR_MULTILINGUAL)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    unload(model)
    print(f"\n  Phase 9 complete → {DIR_MULTILINGUAL}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 10 — BERT TEXT  |  768-dim  (CLS token)
# Paper: Devlin et al., NAACL 2019 — https://arxiv.org/abs/1810.04805
# HF:    google-bert/bert-base-uncased
# Note:  Parallel to E4SRec Section 3.3 ablation (SASRec + BERT).
#        CLS token representation — direct citable link to base paper.
# ─────────────────────────────────────────────────────────────────────────────
def phase_10_bert(df: pd.DataFrame):
    banner(10, "BERT Text Embedder",
           "google-bert/bert-base-uncased", 768, DIR_BERT)

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model     = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    model.eval()

    total = len(df)
    for idx, row in df.iterrows():
        track_id = row["track_index"]
        artist   = row["artist_name"]
        track    = row["track_name"]
        lyrics   = str(row["whisper_lyrics"])

        if track_done(track_id, DIR_BERT):
            print(f"  [SKIP] track_{track_id}")
            continue

        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        try:
            inputs = tokenizer(
                lyrics, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # CLS token = index 0 of last hidden state → [768]
            emb  = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()
            path = save_embedding(track_id, artist, track, emb, DIR_BERT)
            print(f"    Saved {list(emb.shape)} → {path}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    unload(model, tokenizer)
    print(f"\n  Phase 10 complete → {DIR_BERT}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run all 11 phases sequentially
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*64}")
    print(f"  E4SRec Embedding Pipeline — Device: {device.upper()}")
    print(f"  Input CSV  : {CSV_HM_PATH}")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"{'='*64}\n")

    # Load input track list from the session CSV
    # Expected columns: track_index, artist_name, track_name
    input_df = pd.read_csv(CSV_HM_PATH, nrows=10)
    print(f"  Rows in input CSV: {len(input_df)}")

    # Deduplicate — each unique (track_index, artist_name, track_name) embedded once
    track_df = (
        input_df[["track_index", "artist_name", "track_name"]]
        .drop_duplicates(subset=["track_index"])
        .reset_index(drop=True)
    )
    
    print(f"  Unique tracks to embed: {len(track_df)}\n")

    # ── Phase 0: Whisper ASR + CLAP score → master_lyrics.csv ────────────────
    phase_00_whisper(track_df)

    # ── Phases 1–5: Audio embedders ───────────────────────────────────────────
    # Each phase: load ONE model → embed ALL tracks → unload → next phase.
    # Each track is re-downloaded and cleaned up within the same phase.
    phase_01_clap(track_df)
    phase_02_mert(track_df)
    phase_03_music2vec(track_df)
    phase_04_encodec(track_df)
    phase_05_mfcc(track_df)

    # ── Phases 6–10: Text embedders ───────────────────────────────────────────
    # No audio downloads — reads whisper_lyrics directly from master_lyrics.csv.
    lyrics_df = load_master_lyrics()
    phase_06_minilm(lyrics_df)
    phase_07_bgem3(lyrics_df)
    phase_08_mpnet(lyrics_df)
    phase_09_multilingual(lyrics_df)
    phase_10_bert(lyrics_df)

    print(f"\n{'='*64}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  master_lyrics.csv  → {MASTER_LYRICS_PATH}")
    print(f"  Audio embeddings   → {AUDIO_DIR}")
    print(f"  Text embeddings    → {TEXT_DIR}")
    print(f"{'='*64}\n")