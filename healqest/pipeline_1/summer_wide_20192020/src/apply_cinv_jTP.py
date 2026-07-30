import os, sys, argparse, yaml, shutil, datetime
import healpy as hp
import numpy as np
import logging as lg
from pathlib import Path

sys.path.insert(
    0,
    "/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/utils/",
)
import utils as butils

sys.path.insert(0, "/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/")
sys.path.insert(
    0, "/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/cinv/"
)
import maps
from cinv import cinv_hp as cinv
import healqest_utils as utils
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("file_yaml", default=None, type=str, help="main yaml")
parser.add_argument("seed", default=1, type=int, help="seed")
parser.add_argument("cmbset", default=1, type=int, help="cmbset")
parser.add_argument("--nops", default=False, dest="nops", action="store_true")
args = parser.parse_args()

file_yaml = args.file_yaml
seed = args.seed
cmbset = args.cmbset
nops = args.nops

config = yaml.safe_load(open(file_yaml))
nside = config["cinv"]["nside"]
lmax = config["cinv"]["lmax"]
lmin = config["cinv"]["lmin"]
mmin = config["cinv"]["mmin"]
eps_min = config["cinv"]["eps_min"]
dir_tmp = config["cinv"]["dir_tmp"].format(seed=seed, cmbset=cmbset)
file_mask = config["cinv"]["file_mask"]
file_alm = config["cinv"]["file_alm"]
file_noisefg = config["cinv"]["file_noisefg"]
file_out = config["cinv"]["file_out_jTP"]
file_cambcls = config["cls"]["file_lcmb"]
file_bconfig = config["base"]["config"]
runname = config["base"]["runname"]
notch = config["cinv"]["notch"]
nlev_t = config["cinv"]["nlev_t"]
nlev_p = config["cinv"]["nlev_p"]
scal_res_t = config["cinv"]["scal_res_t"]
scal_res_p = config["cinv"]["scal_res_p"]
file_ninv_t = config["cinv"]["ninv_t"]
file_ninv_p = config["cinv"]["ninv_p"]
ninv_fac_t = config["cinv"]["ninv_fac_t"]
ninv_fac_p = config["cinv"]["ninv_fac_p"]
file_pmask_t = config["cinv"]["pmask_t"]
file_pmask_p = config["cinv"]["pmask_p"]
eps_t = config["cinv"]["eps_t"]
eps_p = config["cinv"]["eps_p"]

suffix = ""

print("Removing dir_tmp:", dir_tmp)
shutil.rmtree(dir_tmp, ignore_errors=True)

print("nside--", nside)
pixarea_sqrad = hp.nside2pixarea(nside)

print("Output file:")
file_out = file_out.format(
    cmbset=cmbset,
    nside=nside,
    lmin=lmin,
    lmax=lmax,
    mmin=mmin,
    seed=seed,
    runname=runname,
    suffix=suffix,
)

print("Creating output directory")
print(f" - {file_out}")
p = pathlib.Path(file_out)
Path(p.parent).mkdir(parents=True, exist_ok=True)
p = pathlib.Path(dir_tmp)
Path(p).mkdir(parents=True, exist_ok=True)

print("Loading base config file")
print("- %s" % file_bconfig)
bconfig = yaml.safe_load(open(file_bconfig))

print("Setting scratch outputs")
print("- %s" % dir_tmp + "/outputs/")
shutil.rmtree(dir_tmp + "/outputs/", ignore_errors=True)

print("Setting seed number")
# Conversion between raw sims and lensing sim
# cmbset1: 1-
# cmbset2: 1001-
# cmbset3: 7001-
# cmbset4: 8001-
seedR = seed + {2: 1000, 3: 7000, 4: 8000}.get(cmbset, 0)
print(" - %s" % file_alm.format(seed=seedR))

print("Loading camb Cls")
print("- %s" % file_cambcls)
cl_len = utils.load_cambcls(file_cambcls, lmax=lmax, dict=True)

print("Loading mask")
print(" - %s" % file_mask)
mask = hp.read_map(file_mask)
if nside != 2048:
    mask = hp.ud_grade(mask, nside)


# Load 2d transfer function
print("Loading 2d transfer function")
tfblT_2d = butils.load_tf(bconfig, lmax=lmax, include_beam=True, freq=150)
tfblP_2d = butils.load_tf(bconfig, lmax=lmax, include_beam=True, freq=150)
tfblT_2d = tfblT_2d["2d"]["150T"]
tfblP_2d = tfblP_2d["2d"]["150P"]

tfblT_2d_nobeam = butils.load_tf(bconfig, lmax=lmax, include_beam=False, freq=150)
tfblP_2d_nobeam = butils.load_tf(bconfig, lmax=lmax, include_beam=False, freq=150)
tfblT_2d_nobeam = tfblT_2d_nobeam["2d"]["150T"]
tfblP_2d_nobeam = tfblP_2d_nobeam["2d"]["150P"]
tfblT_2d_nobeam[tfblT_2d_nobeam < 0.05] = 0.05
tfblP_2d_nobeam[tfblP_2d_nobeam < 0.05] = 0.05

# Load 1d transfer function
print("Loading 1d transfer function")
tfblT_1d = np.load("/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf_withbeam.npz")["tt"][: lmax + 1]
tfblT_1d[np.isnan(tfblT_1d)] = 1e-30
tfblT_1d[tfblT_1d < 0.01] = 1e20
tfblP_1d = np.load("/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf_withbeam.npz")["ee"][: lmax + 1]
tfblP_1d[np.isnan(tfblP_1d)] = 1e-30
tfblP_1d[tfblP_1d < 0.01] = 1e20
tfblT_1d[:lmin] = 1e20
tfblP_1d[:lmin] = 1e20

print("Computing 1d residual spectra")
lmaxs = 16000
ell, emm = hp.Alm.getlm(lmaxs)
ws = np.ones(hp.Alm.getsize(lmaxs), dtype=np.complex128)
ws[emm < 220] = 0

_tfblT_2d = butils.load_tf(bconfig, lmax=lmaxs, include_beam=True, freq=150)
_tfblP_2d = butils.load_tf(bconfig, lmax=lmaxs, include_beam=True, freq=150)
_tfblT_2d = _tfblT_2d["2d"]["150T"]
_tfblP_2d = _tfblP_2d["2d"]["150P"]

d = "/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/midell/sims/foregrounds/mcmc/emulator/emmtest2_map2alm/"
mcorr = np.load(d + "mcorr_map2alm.npz")["TT"][: lmax + 1]

nfg = np.load(
    "/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/almstack/ilc/noisefg2_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_0001_0500_withsignflipnoise_2dilc_crosstf_full_021125.npz"
)

_tlm = (nfg["almTT"] / nfg["nsims"]) ** 0.5
_elm = (nfg["almEE"] / nfg["nsims"]) ** 0.5
_blm = (nfg["almBB"] / nfg["nsims"]) ** 0.5

cltt_nfg = hp.alm2cl(_tlm * ws * _tfblT_2d)[: lmax + 1] / mcorr * nfg["nrm"]
clee_nfg = hp.alm2cl(_elm * ws * _tfblP_2d)[: lmax + 1] / mcorr * nfg["nrm"]
clbb_nfg = hp.alm2cl(_blm * ws * _tfblP_2d)[: lmax + 1] / mcorr * nfg["nrm"]


# alm-space mask
print("Setting almspace masking")
print("-lmin: %d" % lmin)
print("-mmin: %d" % mmin)

ell, emm = hp.Alm.getlm(lmax)
ww = np.ones_like(tfblT_2d, dtype=np.complex_)
ww[emm < mmin] = 0
ww[ell < lmin] = 0

# Load alm file. Note these are transfer function deconcolved.
print("Loading alm file")
print(" - %s" % file_alm.format(seed=seedR))
almin = np.load(file_alm.format(seed=seedR))

tlm = butils.reduce_lmax(almin["almT"], lmax=lmax)
elm = butils.reduce_lmax(almin["almE"], lmax=lmax)
blm = butils.reduce_lmax(almin["almB"], lmax=lmax)

# _aa = hp.alm2map([tlm, elm, blm], nside)
# _aa[0][mask == 0] = 0
# _aa[1][mask == 0] = 0
# _aa[2][mask == 0] = 0
# tlm, elm, blm = hp.map2alm(_aa, lmax=lmax, use_pixel_weights=True)

file_tmp = dir_tmp + "input_%d" % (seedR)

if seed == 0:
    # Add extra noise to data to match with sims
    beam = base_utils.load_bl(bconfig, lmax=lmax)
    alm = hp.synalm((np.pi / 180.0 / 60.0 * 11.0) ** 2 * beam["150T"] ** 16)
    _nn = hp.alm2map(alm, nside)
    nlm = hp.map2alm(_nn * mask, lmax=lmax)

    hp.write_alm(file_tmp, [ww * (tlm + nlm),
                            ww * elm,
                            ww * blm], overwrite=True)
else:
    hp.write_alm(file_tmp, [ww * (tlm),
                            ww * elm,
                            ww * blm], overwrite=True)


# Simulation dictionary
sim_dict = {}
sim_dict["nside"] = nside
sim_dict["ivf_lrange"] = [lmin, lmax]
sim_dict["dir_output"] = dir_tmp + "/output/"
sim_dict["dir_cinvT"] = dir_tmp + "/T/"
sim_dict["dir_cinvP"] = dir_tmp + "/P/"
sim_dict["file_mask"] = file_mask
sim_dict["file_signal"] = dir_tmp + "input_%d" % (seedR)
sim_dict["file_noise"] = None  # dir_tmp+'input_%d'%(seedR)
sim_dict["eps_min"] = eps_min
sim_dict["tf2d"] = None
sims = maps.maps(sim_dict)

print("Loading mask")
print(" - %s" % file_mask)

binmask = np.copy(mask)
binmask[binmask > 0] = 1
pixvar_t = (np.pi / 180.0 / 60.0 * nlev_t) ** 2 / pixarea_sqrad
pixvar_p = (np.pi / 180.0 / 60.0 * nlev_p) ** 2 / pixarea_sqrad
ninv_t = binmask * (1 / pixvar_t) * ninv_fac_t
ninv_p = binmask * (1 / pixvar_p) * ninv_fac_p
# NOTE: just because you use the exact same value for this it doesnt mean that the


if nops:
    pass
else:
    print("Loading pmask_t: ", file_pmask_t)
    mt = hp.read_map(file_pmask_t, partial=True)
    mt[mt == hp.UNSEEN] = 0
    mt = 1 - mt
    mt = hp.ud_grade(mt, nside_out=nside)
    mt[mt < 1] = 0

    mp = hp.read_map(file_pmask_p, partial=True)
    mp[mp == hp.UNSEEN] = 0
    mp = 1 - mp
    mp = hp.ud_grade(mp, nside_out=nside)
    mp[mp < 1] = 0

    ninv_t *= mt
    ninv_p *= mp

print("nlev_t--", nlev_t)
print("nlev_p--", nlev_p)

# Treat the residual part in harmonic space.
# This is known not to be perfect so some scaling
# might be needed (this is true even in an
# idealized set up).
_tt = (cltt_nfg - (np.pi / 180.0 / 60.0 * nlev_t) ** 2) * scal_res_t
_ee = (clee_nfg - (np.pi / 180.0 / 60.0 * nlev_p) ** 2) * scal_res_p
_bb = (clbb_nfg - (np.pi / 180.0 / 60.0 * nlev_p) ** 2) * scal_res_p
_tt[_tt < 0] = 0
_tt[np.isnan(_tt)] = 0
_ee[_ee < 0] = 0
_ee[np.isnan(_ee)] = 0
_bb[_bb < 0] = 0
_bb[np.isnan(_bb)] = 0

nl_res = {"tt": _tt, "ee": _ee, "bb": _bb}

cinv_tp = cinv.cinv_tp(
    dir_tmp + "/TP/",
    lmax,
    nside,
    cl_len,
    nl_res,
    [ninv_t, ninv_p],
    tfblT_1d,
    tfblP_1d,
    tf2d_t=ww * tfblT_2d,
    tf2d_p=ww * tfblP_2d,
    eps_min=eps_min,
)

print("Apply a lrange cut: [%d<ell<%d]" % (lmin, lmax))
lfilt = np.ones(lmax + 1) + 0.0
lfilt[:lmin] = 0
lfilt[lmax:] = 0

print("               ")
print("-------------------------------- Running C^-1 --------------------------------")
jivfs = cinv.library_cinv_jTP(dir_tmp + "outputs", sims, cinv_tp, cl_len, lfilt=lfilt)
ivf_t, ivf_e, ivf_b = jivfs.get_sim_teblmivf(seed)

print("Saving output file")
if nops:
    file_out = file_out[:-4] + "_nops.npz"
    print(" - %s" % file_out)
    np.savez(file_out, tlm=ivf_t, elm=ivf_e, blm=ivf_b)

else:
    print(" - %s" % file_out)
    np.savez(file_out, tlm=ivf_t, elm=ivf_e, blm=ivf_b)

shutil.rmtree(dir_tmp, ignore_errors=True)
