import os, sys, queue, threading, traceback, librosa, torch, whisper, torch
import numpy as np
import pandas as pd

sys.path.append("/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/audio_embed_pipeline")

from paths import CSV_SESSION_UNIQUE_TRACKS_PATH, OUTPUT_DIR
from downloader import search_youtube, download_wav, get_genre, cleanup

from embedder import (
    PT_FILES, device,
    load_all_models,
    append_pt,
    embed_track,
    _clap_audio_emb,
    _clap_text_emb,
)

# CHUNK CONFIGURATION
TOTAL_CHUNKS = int(os.environ.get("TOTAL_CHUNKS", 50))
CHUNK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
QUEUE_SIZE = 3   # number of pre-downloaded tracks buffered in RAM

# Each chunk gets its own output subfolder
CHUNK_OUTPUT_DIR    = os.path.join(OUTPUT_DIR, f"chunk_{CHUNK_ID:02d}")
MASTER_LYRICS_PATH  = os.path.join(CHUNK_OUTPUT_DIR, "master_lyrics.csv")

# Per-chunk .pt files
CHUNK_PT_FILES = {
    k: os.path.join(CHUNK_OUTPUT_DIR, os.path.basename(v))
    for k, v in PT_FILES.items()
}

os.makedirs(CHUNK_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tmp_audio"), exist_ok=True)

# Sentinel - signals downloader is done
_DONE = object()


def downloader_worker(track_rows, done_ids, out_queue):
    for _, row in track_rows.iterrows():
        track_id = int(row["track_index"])
        artist   = str(row["artist_name"])
        track    = str(row["track_name"])

        if track_id in done_ids:
            print(f"  [SKIP] track_{track_id} | {artist} — {track}")
            continue

        wav_path = None
        try:
            # Search
            result = search_youtube(artist, track)
            if not result:
                print(f"  [SKIP] Not found: {artist} — {track}")
                continue

            video_id  = result["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Genre
            genre = get_genre(artist, track)

            # Download
            wav_path, duration = download_wav(video_url, track_id)

            # Load all sample rates into RAM immediately
            audio_48k = librosa.load(wav_path, sr=48000, mono=True)[0]
            audio_24k = librosa.load(wav_path, sr=24000, mono=True)[0]
            audio_16k = librosa.load(wav_path, sr=16000, mono=True)[0]

            # Delete WAV - audio is now in RAM
            cleanup(track_id)
            wav_path = None

            out_queue.put({
                "track_id":  track_id,
                "artist":    artist,
                "track":     track,
                "genre":     genre,
                "video_id":  video_id,
                "duration":  duration,
                "audio_48k": audio_48k,
                "audio_24k": audio_24k,
                "audio_16k": audio_16k,
            })

        except Exception as e:
            print(f"  [DOWNLOAD ERROR] track_{track_id} {artist} — {track}: {e}")
            traceback.print_exc()
            if wav_path:
                cleanup(track_id)

    out_queue.put(_DONE)
    print("  [DOWNLOADER] Done — all tracks queued.")

def embedder_worker(models, out_queue, done_ids, total):
    processed = 0

    while True:
        payload = out_queue.get()

        if payload is _DONE:
            print(f"  [EMBEDDER] Done — {processed} tracks embedded.")
            break

        track_id = payload["track_id"]
        artist   = payload["artist"]
        track    = payload["track"]
        genre    = payload["genre"]
        video_id = payload["video_id"]
        duration = payload["duration"]
        audio_48k = payload["audio_48k"]
        audio_24k = payload["audio_24k"]
        audio_16k = payload["audio_16k"]

        try:
            print(f"\n  [EMBED] track_{track_id} | {artist} — {track}")

            # Whisper ASR - write audio to temp file for Whisper
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_48k, 48000)
            asr    = models["whisper"].transcribe(tmp_path, fp16=(device == "cuda"))
            lyrics = asr["text"].strip() or "[Instrumental]"
            os.remove(tmp_path)
            print(f"    Lyrics: {lyrics[:60]}{'...' if len(lyrics) > 60 else ''}")

            # CLAP alignment score
            a_emb = _clap_audio_emb(models["clap_model"], models["clap_proc"], audio_48k)
            t_emb = _clap_text_emb(models["clap_model"], models["clap_proc"], lyrics)
            score = float(
                torch.nn.functional.cosine_similarity(
                    a_emb.unsqueeze(0), t_emb.unsqueeze(0)
                ).item()
            )
            print(f"    CLAP alignment: {score:.4f}")

            # All 10 embedders
            embeddings = embed_track(audio_48k, audio_24k, audio_16k, lyrics, models)

            # Save to per-chunk .pt files
            for name, emb in embeddings.items():
                append_pt(CHUNK_PT_FILES[name], track_id, artist, track, emb)

            # Save to per-chunk master_lyrics.csv
            new_row = {
                "track_index":          track_id,
                "artist_name":          artist,
                "track_name":           track,
                "genre":                genre,
                "youtube_video_id":     video_id,
                "duration_seconds":     duration,
                # "whisper_lyrics":       lyrics,
                "clap_alignment_score": score,
            }
            write_header = not os.path.exists(MASTER_LYRICS_PATH)
            pd.DataFrame([new_row]).to_csv(
                MASTER_LYRICS_PATH, mode="a", header=write_header, index=False
            )
            done_ids.add(track_id)
            processed += 1
            print(f"    Saved. [{processed}/{total}]")

        except Exception as e:
            print(f"  [EMBED ERROR] track_{track_id}: {e}")
            traceback.print_exc()


if __name__ == "__main__":

    print(f"\n{'='*64}")
    print(f"  E4SRec Parallel Embedding Pipeline")
    print(f"  Device      : {device.upper()}")
    print(f"  Chunk       : {CHUNK_ID} / {TOTAL_CHUNKS}")
    print(f"  Output dir  : {CHUNK_OUTPUT_DIR}")
    print(f"{'='*64}\n")

    full_df = pd.read_csv(CSV_SESSION_UNIQUE_TRACKS_PATH)
    full_df = (
        full_df[["track_index", "artist_name", "track_name"]]
        .drop_duplicates(subset=["track_index"])
        .reset_index(drop=True)
    )
    total_tracks = len(full_df)
    chunk_size   = (total_tracks + TOTAL_CHUNKS - 1) // TOTAL_CHUNKS
    start        = CHUNK_ID * chunk_size
    end          = min(start + chunk_size, total_tracks)
    chunk_df     = full_df.iloc[start:end].reset_index(drop=True)

    print(f"  Total tracks  : {total_tracks}")
    print(f"  This chunk    : rows {start} → {end} ({len(chunk_df)} tracks)\n")

    done_ids = set()
    if os.path.exists(MASTER_LYRICS_PATH):
        done_ids = set(pd.read_csv(MASTER_LYRICS_PATH)["track_index"].tolist())
        print(f"  Resuming — {len(done_ids)} tracks already done.\n")

    remaining = len(chunk_df) - len(
        chunk_df[chunk_df["track_index"].isin(done_ids)]
    )
    print(f"  Tracks remaining: {remaining}\n")

    models = load_all_models()

    track_queue = queue.Queue(maxsize=QUEUE_SIZE)

    dl_thread = threading.Thread(
        target=downloader_worker,
        args=(chunk_df, done_ids, track_queue),
        daemon=True
    )
    dl_thread.start()

    embedder_worker(models, track_queue, done_ids, remaining)

    dl_thread.join()

    print(f"\n{'-'*64}")
    print(f"CHUNK {CHUNK_ID} COMPLETE")
    print(f"Output: {CHUNK_OUTPUT_DIR}")
    for name, path in CHUNK_PT_FILES.items():
        if os.path.exists(path):
            store = torch.load(path, weights_only=False)
            n = len(store["track_ids"])
            shape = list(store["embeddings"].shape)
            print(f"  {name:22s} {shape}  ({n} tracks)")