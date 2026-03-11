import sys, os, threading, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ytmusicapi import YTMusic
from pytubefix import YouTube
from paths import PROJECT_DIR, CSV_PATH

CHUNK_SIZE = 10000
array_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
start_row = array_id * CHUNK_SIZE
end_row = start_row + CHUNK_SIZE

ytmusic = YTMusic()
os.makedirs(PROJECT_DIR, exist_ok=True)
list_lock = threading.Lock()
metadata_log = []

def download_track(row):
    tid = row['track_index']
    target = os.path.join(PROJECT_DIR, f"{tid}.m4a")
    
    if os.path.exists(target): return None

    try:
        # Search YT Music
        res = ytmusic.search(f"{row['artist_name']} {row['track_name']}", filter="songs")
        if not res: return None
        vid = res[0]['videoId']
        
        # Metadata Extraction (Lyrics, Length, Genre)
        lyrics_text = "Not Found"
        try:
            watch = ytmusic.get_watch_playlist(videoId=vid)
            if 'lyrics' in watch and watch['lyrics']:
                lyrics_text = ytmusic.get_lyrics(watch['lyrics']).get('lyrics', 'Not Found')
        except: pass

        yt = YouTube(f"https://www.youtube.com/watch?v={vid}", client='TV', use_oauth=True, allow_oauth_cache=True)
        
        # Download
        yt.streams.get_audio_only().download(output_path=PROJECT_DIR, filename=f"{tid}.m4a")
        
        with list_lock:
            metadata_log.append({
                'track_index': tid,
                'duration_seconds': yt.length,
                'youtube_genre': yt.metadata[0].get('Genre', 'Unknown') if yt.metadata else 'Unknown',
                'lyrics': lyrics_text
            })
        return tid
    except: return None

df = pd.read_csv(CSV_PATH)
my_chunk = df.iloc[start_row:min(end_row, len(df))]

with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(download_track, row) for _, row in my_chunk.iterrows()]
    for future in as_completed(futures):
        future.result()

if metadata_log:
    meta_df = pd.DataFrame(metadata_log)
    meta_df.to_csv(os.path.join(PROJECT_DIR, f"meta_chunk_{array_id}.csv"), index=False)