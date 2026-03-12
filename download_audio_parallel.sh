#!/bin/bash
#SBATCH --job-name=lastfm_full
#SBATCH --array=0-149
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/node_%a.out
#SBATCH --account=pi_dagarwal_umass_edu

cd /work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/new_dataset_creation/data_pipeline/
module load conda/latest
conda activate 698ds

mkdir -p logs

python extract_parallel_new.py $SLURM_ARRAY_TASK_ID