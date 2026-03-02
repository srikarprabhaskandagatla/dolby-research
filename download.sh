#!/bin/bash
#SBATCH --job-name=lastfm_download
#SBATCH --output=download_log_%j.txt
#SBATCH --error=download_error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=pi_dagarwal_umass_edu

# Navigate to your project directory
cd /work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/extract_audio_pipeline/

# Load Conda and activate your specific environment
module load conda/latest
conda activate 698ds

# Run the extraction script
echo "Starting Last.fm dataset download..."
python freq_rank_get_unique_tracks.py
echo "Pipeline finished!"