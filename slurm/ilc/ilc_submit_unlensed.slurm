#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --account="punim1922"
#SBATCH --time=24:0:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

python get_ilc_maps_unlensed.py

my-job-stats -a -n -s
