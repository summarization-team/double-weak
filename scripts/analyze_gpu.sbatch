#!/bin/bash

#SBATCH --job-name=distil_whisper
#SBATCH --mail-user=kharalds@uw.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --chdir=/mmfs1/gscratch/clmbr/kharalds/ling580/double-weak

#SBATCH --account=clmbr
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=logs/%x_%j.out # where STDOUT goes- %x is the job name above, and %j is the job ID, unique to each job
#SBATCH --error=logs/%x_%j.err # where STDERR goes

./scripts/analyze.sh