#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --account="punim1922"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G

python get_ilc_maps_summer-b_unlensed.py

myjob-stats -a -n -s
