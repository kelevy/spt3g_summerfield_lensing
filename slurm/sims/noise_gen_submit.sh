#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --account="punim1922"
#SBATCH --output=batch_jobs_noise_gen/job.o%j
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=24:0:00
#SBATCH --mem=25G

export SHELL=bash


python noise_gen_run.py


my-job-stats -a -n -s
