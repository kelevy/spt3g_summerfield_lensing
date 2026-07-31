# CMB Lensing Reconstruction Using Two Years of Temperature Data from the SPT-3G Summer Survey

Code accompanying the paper:

> K. Levy et al. (SPT-3G Collaboration), *"CMB Lensing Reconstruction Using Two Years of
> Temperature Data from the SPT-3G Summer Survey"*, submitted to JCAP (2026).
> [arXiv:2607.05784](https://arxiv.org/abs/2607.05784)

## Overview

This repository contains the code used to reconstruct the CMB lensing potential from
two years (2019–2021) of SPT-3G Summer survey temperature data, spanning three fields
(Summer-A, Summer-B, Summer-C; ~2640 deg² total). The pipeline implements a curved-sky
quadratic estimator (Okamoto & Hu 2003) for lensing reconstruction.


## Repository Structure

```
spt3g_summerfield_lensing/
├── config/       # Configuration yaml files to coadd and mask data, generate sims, create ILC maps
├── figures/      # Plotting outputs
├── healqest/     # HEALPix-based QE lensing module including the QE src code, the lensing reconstruction pipeline and corresponding yaml files
├── notebooks/    # Notebooks used to create the analysis plots
├── scripts/      # Run scripts for the data, sims, ILC stages
├── slurm/        # SLURM batch job submission scripts for the data, sims, ILC stages
├── src/          # Source code for the data, sims, ILC stages
└── README.md
```

## Key Results

- Combined lensing amplitude: **A = 1.015 ± 0.053** (50 < L < 2000), SNR ≈ 19
- Individual field amplitudes: Summer-A = 1.029 ± 0.078, Summer-B = 0.890 ± 0.115,
  Summer-C = 1.077 ± 0.093
- Baseline analysis choices: ℓmin = 500, ℓmax = 3000, mmin = 100

## Citation

This repository contains the analysis code used to produce the results in the paper
below. It is provided for transparency and reference rather than as a general-purpose
tool. If you refer to these results, please cite:

```bibtex
@article{Levy2026SPT3GSummerLensing,
  title   = {CMB Lensing Reconstruction Using Two Years of Temperature Data from the SPT-3G Summer Survey},
  author  = {Levy, K. and Raghunathan, S. and Guidi, F. and others (SPT-3G Collaboration)},
  journal = {arXiv preprint arXiv:2607.05784},
  year    = {2026}
}
```