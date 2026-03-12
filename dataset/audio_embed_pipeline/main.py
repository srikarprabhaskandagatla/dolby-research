import os, traceback, librosa, torch
import numpy as np
import pandas as pd

import sys
sys.path.append("/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/extract_audio_pipeline")

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

MASTER_LYRICS_PATH = os.path.join(OUTPUT_DIR, "master_lyrics.csv")


if __name__ == "__main__":
    input_df = pd.read_csv(CSV_SESSION_UNIQUE_TRACKS_PATH, nrows = 10) # remove nrows= for full run
    track_df = (
        input_df[["track_index", "artist_name", "track_name"]]
        .drop_duplicates(subset=["track_index"])
        .reset_index(drop=True)
    )
    total = len(track_df)
    print(f"Unique tracks to Embed: {total}\n")

    # Skip already-processed tracks
    done_ids = set()
    if os.path.exists(MASTER_LYRICS_PATH):
        done_ids = set(pd.read_csv(MASTER_LYRICS_PATH)["track_index"].tolist())
        print(f"  Resuming — {len(done_ids)} tracks already processed.\n")

    models = load_all_models()

    for idx, row in track_df.iterrows():
        track_id = int(row["track_index"])
        artist   = row["artist_name"]
        track    = row["track_name"]

        if track_id in done_ids:
            print(f"  [SKIP] [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
            continue

        print(f"\n{'─'*64}")
        print(f"  [{idx+1}/{total}] track_{track_id} | {artist} — {track}")
        print(f"{'─'*64}")

        wav_path = None
        try:
            # Step 1: Search YouTube Music
            print(f"  [1] Searching YouTube Music")
            result = search_youtube(artist, track)
            if not result:
                print(f"  [SKIP] Not found on YouTube Music.")
                continue

            video_id  = result["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"  [1] Found: {video_url}")

            # Step 2: Genre lookup
            genre = get_genre(artist, track)
            print(f"  [2] Genre: {genre}")

            # Step 3: Download WAV
            print(f"  [3] Downloading WAV")
            wav_path, duration = download_wav(video_url, track_id)
            print(f"  [3] Downloaded ({duration}s) → {wav_path}")

            # Step 3b: Pre-load audio into RAM at all 3 sample rates as no disk is needed after.
            print(f"  [3b] Pre-loading audio at 48 / 24 / 16 kHz into RAM")
            audio_48k = librosa.load(wav_path, sr=48000, mono=True)[0]
            audio_24k = librosa.load(wav_path, sr=24000, mono=True)[0]
            audio_16k = librosa.load(wav_path, sr=16000, mono=True)[0]
            print(f"  [3b] Audio in RAM.")

            # Step 4: Whisper ASR (still needs wav_path on disk)
            print(f"  [4] Whisper ASR...")
            asr    = models["whisper"].transcribe(wav_path, fp16=(device == "cuda"))
            lyrics = asr["text"].strip() or "[Instrumental]"
            print(f"  [4] Lyrics: {lyrics[:80]}{'...' if len(lyrics) > 80 else ''}")

            # Step 5: CLAP cross-modal alignment score
            # Cosine similarity in CLAP's 512-dim space.
            # Score near 0 then instrumental or poor ASR. Use as quality signal.
            print(f"  [5] CLAP alignment score")
            a_emb = _clap_audio_emb(models["clap_model"], models["clap_proc"], audio_48k)
            t_emb = _clap_text_emb(models["clap_model"], models["clap_proc"], lyrics)
            score = float(
                torch.nn.functional.cosine_similarity(
                    a_emb.unsqueeze(0), t_emb.unsqueeze(0)
                ).item()
            )
            print(f"  [5] CLAP alignment: {score:.4f}")

            # Step 6: Run all 10 embedders on pre-loaded audio + lyrics
            print(f"  [6] Running all 10 embedders...")
            embeddings = embed_track(audio_48k, audio_24k, audio_16k, lyrics, models)

            # Step 7: Append each embedding to its growing .pt file
            print(f"  [7] Appending to 10 .pt files...")
            for name, emb in embeddings.items():
                append_pt(PT_FILES[name], track_id, artist, track, emb)
                print(f"      {name:22s} {list(emb.shape)}")

            # Step 8: Append row to master_lyrics.csv
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
            print(f"  [8] Saved to master_lyrics.csv")

        except Exception as e:
            print(f"\n  [ERROR] track_{track_id}: {e}")
            traceback.print_exc()

        finally:
            # Step 9: Delete WAV, audio is in RAM, disk copy not needed
            if wav_path:
                cleanup(track_id)
                print(f"  [9] WAV deleted.")

    # Summary
    print(f"\n{'-'*64}")
    print(f"All Tracks Embedded Successfully!")
    print()
    for name, path in PT_FILES.items():
        if os.path.exists(path):
            store = torch.load(path, weights_only=False)
            shape = list(store["embeddings"].shape)
            print(f"  {name:22s} {shape}  ({len(store["track_ids"])} tracks)")