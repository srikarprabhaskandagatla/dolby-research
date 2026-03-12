import os, glob, urllib.parse, requests, yt_dlp
from ytmusicapi import YTMusic
from paths import OUTPUT_DIR

TEMP_AUDIO_DIR = os.path.join(OUTPUT_DIR, "tmp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


def search_youtube(artist: str, track: str):
    # Returns the first result dict (contains videoId, title, etc.) or None
    ytmusic = YTMusic()
    results = ytmusic.search(f"{artist} {track}", filter="songs") \
              or ytmusic.search(f"{artist} {track}")
    return results[0] if results else None

def download_wav(video_url: str, track_id) -> tuple:
    # Download audio as WAV via yt-dlp + FFmpeg
    out_template = os.path.join(TEMP_AUDIO_DIR, f"{track_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[ext=opus]/bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(video_url, download=True)
        duration = info.get("duration", 0)
        wav_path = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    # FFmpeg sometimes names the file differently — search for it
    if not os.path.exists(wav_path):
        candidates     = glob.glob(os.path.join(TEMP_AUDIO_DIR, f"{track_id}.*"))
        wav_candidates = [f for f in candidates if f.endswith(".wav")]
        if not wav_candidates:
            raise RuntimeError(
                f"WAV not found after download. Files present: {candidates}"
            )
        wav_path = wav_candidates[0]

    return wav_path, duration

def get_genre(artist: str, track: str) -> str:
    # iTunes genre lookup
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

def cleanup(track_id):
    for f in glob.glob(os.path.join(TEMP_AUDIO_DIR, f"{track_id}.*")):
        try:
            os.remove(f)
        except Exception:
            pass