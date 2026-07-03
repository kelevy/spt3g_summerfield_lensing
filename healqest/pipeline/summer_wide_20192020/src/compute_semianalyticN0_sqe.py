# SQE semi-analytic N0
#
# call:
#   python src/compute_semianalyticN0_sqe.py yaml/YAML_FILE 1 100 TT -src_yaml yaml/SRCYAML_FILE

import os, sys, logging, yaml

p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../healqest/src/"))
sys.path.insert(0, p)
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import weights, qest
import healqest_utils as hutils
import weights, qest, resp, gmv_resp, profiles
import argparse
from pathlib import Path
from utils import utils as base_utils
#sys.path.insert(0,'//sdf/home/w/wlwu/repos/spt3g_software/scratch/yomori/utils/')
#import utils as base_utils


parser = argparse.ArgumentParser()
parser.add_argument("file_yaml", default=None, type=str, help="dir_base")
parser.add_argument("cmbset", default=1, type=int, help="cmbset")
parser.add_argument("seed", default=1, type=int, help="seed")
parser.add_argument("qetype", default="TT", type=str, help="qe")
parser.add_argument("-src_yaml", default=None, dest="src_yaml", help="srcprf yamlfile")
parser.add_argument("-prftype", default="tsz", dest="prftype", help="tsz or 1am")
parser.add_argument("--curl", default=False, dest="curl", action="store_true")
args = parser.parse_args()

file_yaml = args.file_yaml
cmbset = args.cmbset
seed = args.seed
qetype = args.qetype
prftype = args.prftype
curl = args.curl

print("Reading from yaml file: %s" % file_yaml)
config = hutils.load_yaml(file_yaml)

runname = config["base"]["runname"]
file_bconfig = config["base"]["config"]
Lmax = config["lensrec"]["Lmax"]
lminT = config["lensrec"]["lminT"]
lminP = config["lensrec"]["lminP"]
lmaxT = config["lensrec"]["lmaxT"]
lmaxP = config["lensrec"]["lmaxP"]
rmask = config["lensrec"]["mask"]
mmin = config["lensrec"]["mmin"]
rectype = config["lensrec"]["rectype"]
notch = config["lensrec"]["notch"]
bmask = config["pspec"]["mask_boundary"]
amask = config["pspec"]["mask_analysis"]
print("Lmax to: %d" % Lmax)
print("lmaxT to: %d" % lmaxT)
print("lmaxP to: %d" % lmaxP)

base_config = yaml.safe_load(open(file_bconfig))
### mod base_config for S3DF tf path
#base_config['data']['tf']['file'] = "/sdf/home/w/wlwu/data/spt3glens1920_gmvph/tf/tf2d_{freq}ghz_750sims.npz"


# Make save output directory
dir_plm = config["lensrec"]["dir_out"].format(
    runname=runname,
    rectype=rectype,
    lminT=lminT,
    lminP=lminP,
    lmaxT=lmaxT,
    lmaxP=lmaxP,
    mmin=mmin,
)

dir_out = dir_plm + "/SAN0/"

Path(dir_out).mkdir(parents=True, exist_ok=True)
print("Saving at:", dir_out)

# Load Cls
cls = hutils.get_lensedcls(config["cls"]["file_gcmb"], lmax=Lmax, dict=True)

# Load almbar
nside = config["cinv"]["nside"]
cinvlmax = config["cinv"]["lmax"]
cinvlmin = config["cinv"]["lmin"]
cinvmmin = config["cinv"]["mmin"]

if qetype[:3] == "GMV":
    file_cinv = config["cinv"]["file_out_jTP"]
else:
    file_cinv = config["cinv"]["file_out_sepTP"]

file_almbar = file_cinv.format(
    cmbset=cmbset,
    nside=nside,
    lmin=cinvlmin,
    lmax=cinvlmax,
    mmin=cinvmmin,
    seed=seed,
    runname=runname,
    suffix="",
)
print(file_almbar)


# Load masks. Uses ones if renslec mask not provided.
mb = hp.read_map(bmask)
mr = hp.read_map(rmask) if rmask is not None else 1

# Load almbar
print("Load almbar:", file_almbar)
d = np.load(file_almbar)
tlm, elm, blm = d["tlm"], d["elm"], d["blm"]

print("T almmask almlmax=%d mmin=%d lmin=%d lmax=%d" % (cinvlmax, mmin, lminT, lmaxT))
wT = hutils.make_almmask(cinvlmax, mmin=mmin, lmin=lminT, lmax=lmaxT)

print("P almmask almlmax=%d mmin=%d lmin=%d lmax=%d" % (cinvlmax, mmin, lminT, lmaxT))
wP = hutils.make_almmask(cinvlmax, mmin=mmin, lmin=lminP, lmax=lmaxP)

if notch is not None:
    print("Reading notch file and multiplying to alm")
    notchalm = hp.read_alm(notch)
    notchalm = hutils.reduce_lmax(notchalm, lmax=hp.Alm.getlmax(len(tlm)))
    wT *= notchalm
    wP *= notchalm

# Apply transfer function
tfblT_2d_nobeam = base_utils.load_tf(
    base_config, lmax=cinvlmax, include_beam=False, verbose=True, freq=150
)["2d"]["150T"]
tfblP_2d_nobeam = base_utils.load_tf(
    base_config, lmax=cinvlmax, include_beam=False, verbose=True, freq=150
)["2d"]["150P"]


almbar = {
    "T": tlm * wT * tfblT_2d_nobeam,
    "E": elm * wP * tfblP_2d_nobeam,
    "B": blm * wP * tfblP_2d_nobeam,
}

# Compute normalization
nrm = mb.shape[0] / np.sum(mb**4 * mr**4)

# Return all qes to compute e.g. PP=[EE,EB,BE]
qes = hutils.get_qes(qetype)


if curl:
    suffix = "curl"
    qes = [qe + "curl" for qe in qes]
else:
    suffix = "grad"

l = np.arange(Lmax + 1)
v = (0.5 * l * (l + 1)) ** 2

Bflip = ["TBEB", "TBBE", "BTEB", "BTBE", "EBTB", "EBBT", "BETB", "BEBT"]
TEflip = ["TTTE", "TTET", "EETE", "EEET", "TETT", "TEEE", "ETTT", "ETEE"]

file_out = dir_out + "clqq_%s_%s_%d.npy" % (qetype, suffix, seed)

if not args.src_yaml or not os.path.isfile(file_out):
    # Compute when not hardening or hardening but file doesnt exist
    clqq = {}
    clqq_sum = 0
    resp_sum = 0
    for qe1 in qes:
        for qe2 in qes:
            qeXY = weights.weights(qe1, cls, Lmax)
            qeZA = weights.weights(qe2, cls, Lmax)

            XZ = hp.alm2cl(almbar[qe1[0]], almbar[qe2[0]]) * nrm
            YA = hp.alm2cl(almbar[qe1[1]], almbar[qe2[1]]) * nrm
            XA = hp.alm2cl(almbar[qe1[0]], almbar[qe2[1]]) * nrm
            YZ = hp.alm2cl(almbar[qe1[1]], almbar[qe2[0]]) * nrm
            ret = np.zeros(Lmax + 1, dtype=np.complex_)
            clqq["qe1+qe2"] = resp.fill_clq1q2_fullsky(qeXY, qeZA, ret, XZ, YA, XA, YZ)

            if qe1[:2] + qe2[:2] in Bflip:
                # -ve 1 needed because phi(EB)*phi(TB) is +ve, but the resp cov_helper
                # returns -ve results
                fac = -1
            elif qe1[:2] + qe2[:2] in TEflip and curl==True:
                fac = -1
            else:
                fac = 1

            if qetype[:3] != "GMV":
                print(qe1 + qe2, fac)
                resp1 = np.load(f"{dir_plm}/{qe1}/respavg{qe1}_nops.npz")["resp"]
                resp2 = np.load(f"{dir_plm}/{qe2}/respavg{qe2}_nops.npz")["resp"]
                resp_sum += resp1 * resp2

            clqq_sum += fac * v * clqq["qe1+qe2"]

    if qetype[:3]=="GMV":
        resp0 = np.load(f"{dir_plm}/{qetype}/respavg{qetype}_nops.npz")["resp"]
        clqq_sum /= resp0**2
    else:
        clqq_sum /= resp_sum

    np.save(file_out, clqq_sum)
    print("Saving: %s" % file_out)

if args.src_yaml is not None:
    print("Loading: %s" % file_out)
    assert qetype == "TT" or qetype == "GMV" or qetype == "GMVTTEETE"
    clqqtt = np.load(file_out)

    print("Computing TT-profhard SAN0")
    config_src = yaml.safe_load(open(args.src_yaml))
    sdir_out = config_src["lensrec"]["dir_out"].format(
        runname=runname,
        rectype=rectype,
        lminT=lminT,
        lminP=lminP,
        lmaxT=lmaxT,
        lmaxP=lmaxP,
        mmin=mmin,
        prftype=prftype,
    )
    qes_h = config_src["qes_h"]
    arespss_fname = (
        sdir_out
        + "%sbh%s/" % (qetype, qes_h)
        + config_src["ss_resp"]["fnamestub"].format(prftype=prftype)
    )
    arespse_fname = (
        sdir_out
        + "%sbh%s/" % (qetype, qes_h)
        + config_src["se_resp"]["fnamestub"].format(prftype=prftype)
    )
    aresp_fname = dir_plm + config["aresp"]["fnamestub"]
    lmaxTP = max(lmaxT, lmaxP, Lmax)

    if qes_h == "TTsrc":
        u = np.ones(lmaxTP)
    elif qes_h == "TTprf":
        if prftype == "1am":
            gauss_fwhm = config_src["gauss_fwhm_arcmin"]
            u = profiles.profileGaussian(gauss_fwhm, lmax=lmaxTP).fourier()
        elif prftype == "tsz":
            u = np.load(config_src["profile_file"])
        else:
            assert 0
    else:
        assert 0, "must be TTsrc or TTprf"

    resp_tot, weight = hutils.get_aresp_tot(
        aresp_fname, arespss_fname, arespse_fname, qetype
    )
    resplmax = len(weight) - 1
    if Lmax > resplmax:
        weight_l = np.zeros(Lmax + 1)
        weight_l[: resplmax + 1] = weight
        weight = weight_l.copy()
        print("resp lmax: %i; src-lm Lmax: %i" % (resplmax, Lmax))
        print("zero-pad weight to match src-lm lmax")

    prefac = {"TTTT": 1, "TTTTprf": weight, "TTprfTT": weight, "TTprfTTprf": weight**2}

    clqq_sum2 = 0
    for qe1 in ["TT", "TTprf"]:
        for qe2 in ["TT", "TTprf"]:
            qeXY = weights.weights(qe1, cls, Lmax, u=u)
            qeZA = weights.weights(qe2, cls, Lmax, u=u)

            XZ = YA = XA = YZ = hp.alm2cl(almbar["T"], almbar["T"]) * nrm

            ret = np.zeros(Lmax + 1, dtype=np.complex_)
            clqq2 = resp.fill_clq1q2_fullsky(qeXY, qeZA, ret, XZ, YA, XA, YZ)
            # import ipdb;ipdb.set_trace()

            if qe1 == qe2 == "TT" and qetype == "TT":
                resp1 = np.load(f"{dir_plm}/{qe1}/respavg{qe1}_nops.npz")["resp"]
                assert np.allclose(v * clqq2.real / resp1**2, clqqtt.real)
            elif qe1 == qe2 == "TT":
                print("use GMV/GMVTTTEEE clqq")
                resp0 = np.load(f"{dir_plm}/{qetype}/respavg{qetype}_nops.npz")["resp"]
                clqq2 = clqqtt.copy()*resp0**2/v  # replace TT-only clqq with GMV/GMVTTTEEE clqq
                #in clqq not-response-corrected, not in clkk units
            # import ipdb;ipdb.set_trace()
            clqq_sum2 += prefac[qe1 + qe2] * clqq2

    resp0 = np.load(f"{sdir_out}/{qetype}bh{qes_h}/respavg{qetype}bh{qes_h}_nops.npz")[
        "resp"
    ]
    clqq_sum2 *= v / resp0**2

    file_out = sdir_out + "/SAN0/clqq_%sBHTTPRF_%s_%d.npy" % (qetype, suffix, seed)
    Path(sdir_out + "/SAN0/").mkdir(parents=True, exist_ok=True)
    np.save(file_out, clqq_sum2)
    print("Saving: %s" % file_out)
