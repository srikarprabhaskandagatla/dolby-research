import os, time, random, argparse, yt_dlp, sys
import pandas as pd


# Scratch Space - /scratch3/workspace/skandagatla_umass_edu-dolby

# Append path if necessary
sys.path.append("/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/extract_audio_pipeline")
from paths import CSV_SESSION_UNIQUE_TRACKS_PATH, OUTPUT_DIR

# --- ARGUMENT PARSER FOR CLUSTER NODES ---
parser = argparse.ArgumentParser(description="Distributed YouTube Audio Downloader")
parser.add_argument("--node_id", type=int, required=True, help="Node ID (e.g., 0 to 49)")
parser.add_argument("--total_nodes", type=int, default=50, help="Total number of nodes running")
args = parser.parse_args()

NODE_ID = args.node_id
TOTAL_NODES = args.total_nodes

# --- CONFIGURATION ---
INPUT_CSV = CSV_SESSION_UNIQUE_TRACKS_PATH
BASE_OUTPUT_DIR = '/scratch3/workspace/skandagatla_umass_edu-dolby'
COOKIES_PATH = f"/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/extract_audio_pipeline/cookies/cookies_{NODE_ID}.txt"

NODE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"node_{NODE_ID}")
NODE_STATUS_CSV = os.path.join(BASE_OUTPUT_DIR, f"master_download_status_node_{NODE_ID}.csv")
BURST_LIMIT = random.randint(8, 15) # Randomize the burst limit to look less robotic

os.makedirs(NODE_OUTPUT_DIR, exist_ok=True)

def download_track(artist, track, track_id, output_folder):
    wav_path = os.path.join(output_folder, f"{track_id}.wav")
    
    # Skip if file already exists locally
    if os.path.exists(wav_path):
        return True, "Already exists"

    if not os.path.exists(COOKIES_PATH) or os.path.getsize(COOKIES_PATH) == 0:
        return False, f"Cookies file missing or empty: {COOKIES_PATH}"

    query = f"ytsearch1:{artist} {track} audio"
    
    ydl_opts = {
        'format': "bestaudio/best",
        'outtmpl': os.path.join(output_folder, str(track_id)), # Temp name
        'cookiefile': COOKIES_PATH,
        'noplaylist': True,
        'js_runtimes': {
            'node': {'path': '/home/skandagatla_umass_edu/.conda/envs/698ds/bin/node'}
        },
        'remote_components': ['ejs:github'],
        'extractor_args': {'youtube': {'player_client': ['ios', 'web']}},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        
        # Verify the file was actually written
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return True, "Success"
        else:
            return False, "yt-dlp returned no error but file was not created"
    except Exception as e:
        return False, str(e)



if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df = df.iloc[50000:110001]
    track_df = df.drop_duplicates(subset=["track_index"]).reset_index(drop=True)

    # --- CHUNKING LOGIC ---
    chunk_size = len(track_df) // TOTAL_NODES + 1
    start_idx = NODE_ID * chunk_size
    end_idx = min((NODE_ID + 1) * chunk_size, len(track_df))

    # Slice the dataframe for this specific node
    my_chunk = track_df.iloc[start_idx:end_idx]

    print("="*50)
    print(f"NODE {NODE_ID} INITIALIZED")
    print(f"Total Unique Tracks Global: {len(track_df)}")
    print(f"This node is handling index {start_idx} to {end_idx - 1}")
    print(f"Total tracks for this node: {len(my_chunk)}")
    print("="*50)

    # Resume logic for this specific node
    done_ids = set()
    if os.path.exists(NODE_STATUS_CSV):
        try:
            status_df = pd.read_csv(NODE_STATUS_CSV)
            done_ids = set(status_df[status_df['download_success'] == True]['track_index'].tolist())
            print(f"Resuming pipeline. {len(done_ids)} tracks already completed by Node {NODE_ID}.")
        except: pass

    burst_counter = 0

    for relative_idx, row in my_chunk.iterrows():
        # track_index preserves its global continuous ID
        track_id = int(row["track_index"])
        
        if track_id in done_ids:
            continue

        artist, track = row["artist_name"], row["track_name"]

        print(f"[Node {NODE_ID}] Processing Track ID {track_id}: {artist} - {track}")
        
        # Save directly to NODE_OUTPUT_DIR (no batch subfolders)
        success, log = download_track(artist, track, track_id, NODE_OUTPUT_DIR)
        
        # Log status immediately
        new_entry = {
            "track_index": track_id,
            "artist": artist,
            "track": track,
            "download_success": success,
            "error_log": log,
            "node_processed": NODE_ID
        }
        
        pd.DataFrame([new_entry]).to_csv(NODE_STATUS_CSV, mode='a', header=not os.path.exists(NODE_STATUS_CSV), index=False)

        if success:
            done_ids.add(track_id)
            burst_counter += 1
        
        # Random short delay between every song to avoid connection drops
        time.sleep(random.uniform(2.0, 5.0))

        # Burst cooldown to avoid "Bot" flag
        if burst_counter >= BURST_LIMIT:
            cooldown = random.uniform(15.0, 30.0)
            print(f"Node {NODE_ID} taking a burst cooldown: Sleeping {cooldown:.2f}s...")
            time.sleep(cooldown)
            # Reset counter and pick a new random burst limit
            burst_counter = 0
            BURST_LIMIT = random.randint(8, 15)

    print(f"Node {NODE_ID} Pipeline process finished.")