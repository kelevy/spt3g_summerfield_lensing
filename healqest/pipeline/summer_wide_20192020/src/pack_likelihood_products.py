# Examples of usage:
# python src/pack_likelihood_products.py yaml/sqe/config_sqe_081024_autotf_lmaxt3500_mmin100_lmaxp3500.yaml tt

from tqdm import tqdm
from pathlib import Path
import os, sys
import pickle
import numpy as np
import tarfile
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../healqest/src/")
import healqest_utils as hutils

##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument("file_yaml", default=None, type=str, help="main yaml")
parser.add_argument("qe", default="TT", type=str, help="qe")
parser.add_argument("didx", default=0, type=int, help="didx")
parser.add_argument('-src_yaml', default=None, dest='src_yaml', help="srcprf yamlfile")
parser.add_argument('-prftype' , default='tsz',dest='prftype'     , help="tsz or 1am")
parser.add_argument("--blinded", default=False, dest="blinded", action="store_true")
args = parser.parse_args()

file_yaml = args.file_yaml
qe        = args.qe.lower()
didx      = args.didx
prftype   = args.prftype
blinded   = args.blinded

# Load all config from yaml file
config = hutils.parse_yaml(file_yaml)

runname = config["base"]["runname"]

rectype = config["lensrec"]["rectype"]
lminT = config["lensrec"]["lminT"]
lminP = config["lensrec"]["lminP"]
lmaxT = config["lensrec"]["lmaxT"]
lmaxP = config["lensrec"]["lmaxP"]
mmin = config["lensrec"]["mmin"]
psname = config["pspec"]["psname"]


# Directory of output Cls
if args.src_yaml is None:
    dir_base = config["lensrec"]["dir_out"].format(
        runname=runname,
        rectype=rectype,
        lminT=lminT,
        lminP=lminP,
        lmaxT=lmaxT,
        lmaxP=lmaxP,
        mmin=mmin,
    )
else:
    config_src = hutils.load_yaml(args.src_yaml) 
    dir_base   = config_src['lensrec']['dir_out'].format(
            runname=runname,
            rectype=rectype,
            lminT=lminT,
            lminP=lminP,
            lmaxT=lmaxT,
            lmaxP=lmaxP,
            mmin=mmin,
            prftype=prftype
            )

if psname is not None and psname != "":
    dir_cls = dir_base + f"/clkk_polspice_{psname}/"
    dir_like = (
        config["like"]["dir_like"]
        + f"spt3g_20192020_{runname}_k{qe}_"
        + Path(dir_base).name
        + f"_{psname}/"
    )

else:
    dir_cls = dir_base + f"/clkk_polspice_nops/"
    dir_like = (
        config["like"]["dir_like"]
        + f"spt3g_20192020_{runname}_k{qe}_"
        + Path(dir_base).name
        + "/"
    )


Path(dir_like).mkdir(parents=True, exist_ok=True)

##########################################################################

nsims1 = config["like"]["nsims1"]
nsims2 = config["like"]["nsims2"]

nbins = config["like"]["nbins"]
Lmin = config["like"]["Lmin"]
Lmax = config["like"]["Lmax"]

blind_str = config["like"]["blind_str"]

# Load theory
import pickle

#with open(
#    "planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_rawCls.pickle", "rb"
#) as handle:
with open('/sdf/home/w/wlwu/repos/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_rawCls.pickle', 'rb') as handle:
    clsa = pickle.load(handle)
l = np.arange(4101)
t = lambda l: (l * (l + 1)) ** 2 / 4
ell = np.arange(19901)
tlkk = (ell * (ell + 1) / 2) ** 2 * clsa["lens_potential"][:, 0]
tlkk[:2] = np.inf

# Define bin edges
bb = np.round(np.geomspace(Lmin, Lmax, nbins + 1))

# Load various spectra
print("Loading dd:")
dd = hutils.loadcls(
    dir_cls, nsims1, "dd", N0=None, Lmin=Lmin, Lmax=Lmax, qe=qe, curl=False, didx=didx
)
print("Loading xx:")
xx = hutils.loadcls(
    dir_cls, nsims1, "xx", N0=None, Lmin=Lmin, Lmax=Lmax, qe=qe, curl=False, didx=didx
)
print("Loading N0:")
N0 = hutils.loadcls(
    dir_cls, nsims1, "N0", N0=None, Lmin=Lmin, Lmax=Lmax, qe=qe, curl=False, didx=didx
)
print("Loading N1:")
N1 = hutils.loadcls(
    dir_cls, nsims2, "N1", N0=N0, Lmin=Lmin, Lmax=Lmax, qe=qe, curl=False, didx=didx
)
print("Loading RDN0:")
RDN0 = hutils.loadcls(
    dir_cls, nsims1, "RDN0", N0=N0, Lmin=Lmin, Lmax=Lmax, qe=qe, curl=False, didx=didx
)

print("Constructing band power window functions:")
bpwf0 = hutils.get_bpwf(dir_cls, bb, nsims1, N0, N1, ellfac=0, qe=qe, curl=False)
bpwf1 = hutils.get_bpwf(dir_cls, bb, nsims1, N0, N1, ellfac=1, qe=qe, curl=False)

# print("Loading SAN0:")
# SAN0g   = np.load(dir + '/SAN0/SAN0_array_%s.npy'%qe.upper())
# SAN0tf  = N0/np.mean(SAN0g[:len(N0),:],axis=1)
# SAN0arr = SAN0g[:len(N0),:]*SAN0tf[:,np.newaxis]#loadcls(dir_cls, nsims1 ,'SAN0g', N0=N0, Lmin=Lmin, Lmax=Lmax,curl=False,R=1,qe=qe,SAN0tf=SAN0tf)

print("Packing dvec:")
rl0, rdl0, rcl0, err0, arr0, rdl_corr0 = hutils.get_dvec(
    dir_cls,
    bb,
    nsims1,
    N0,
    N1,
    RDN0=RDN0,
    SAN0=None,
    ellfac=0,
    bpwf=bpwf0,
    qe=qe,
    didx=didx,
    theory=tlkk,
)
rl, rdl, rcl, err, arr, rdl_corr = hutils.get_dvec(
    dir_cls,
    bb,
    nsims1,
    N0,
    N1,
    RDN0=RDN0,
    SAN0=None,
    ellfac=1,
    bpwf=bpwf1,
    qe=qe,
    didx=didx,
    theory=tlkk,
)
rlR, rdlR, rclR, errR, arrR, rdl_corrR = hutils.get_dvec(
    dir_cls,
    bb,
    nsims1,
    N0,
    N1,
    RDN0=RDN0,
    SAN0=None,
    ellfac=0,
    bpwf=bpwf0,
    qe=qe,
    didx=didx,
    ratio=True,
    theory=tlkk,
)

if psname is not None:
    print(
        dir_like
        + f"ratio_cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_{psname}.npz"
    )
    np.savez(
        dir_like
        + f"ratio_cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_{psname}.npz",
        rlR=rlR,
        rdlR=rdlR,
        rclR=rclR,
        errR=errR,
        nsims1=nsims1,
        nsims2=nsims2,
        bine=bb,
        N0=N0,
        xx=xx,
        N1=N1,
        RDN0=RDN0,
        dd=dd,
        bpwf=bpwf1,
        bpwf0=bpwf0,
    )

    print(" ")

    print(dir_like + f"cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_{psname}.npz")
    np.savez(
        dir_like + f"cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_{psname}.npz",
        rl=rl,
        rdl=rdl,
        rdl_corr=rdl_corr,
        rcl=rcl,
        err=err,
        rcl_arr=arr,
        rl0=rl0,
        rdl0=rdl0,
        rdl_corr0=rdl_corr0,
        rcl0=rcl0,
        err0=err0,
        rcl_arr0=arr0,
        nsims1=nsims1,
        nsims2=nsims2,
        bine=bb,
        N0=N0,
        xx=xx,
        N1=N1,
        RDN0=RDN0,
        dd=dd,
        bpwf=bpwf1,
        bpwf0=bpwf0,
    )
else:
    print(dir_cls + f"ratio_cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp.npz")
    np.savez(
        dir_cls + f"ratio_cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp.npz",
        rlR=rlR,
        rdlR=rdlR,
        rclR=rclR,
        errR=errR,
        nsims1=nsims1,
        nsims2=nsims2,
        bine=bb,
        N0=N0,
        xx=xx,
        N1=N1,
        RDN0=RDN0,
        dd=dd,
        bpwf=bpwf1,
        bpwf0=bpwf0,
    )

    print(" ")

    print(dir_cls + f"cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp.npz")
    np.savez(
        dir_cls + f"cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp.npz",
        rl=rl,
        rdl=rdl,
        rdl_corr=rdl_corr,
        rcl=rcl,
        err=err,
        rcl_arr=arr,
        rl0=rl0,
        rdl0=rdl0,
        rdl_corr0=rdl_corr0,
        rcl0=rcl0,
        err0=err0,
        rcl_arr0=arr0,
        nsims1=nsims1,
        nsims2=nsims2,
        bine=bb,
        N0=N0,
        xx=xx,
        N1=N1,
        RDN0=RDN0,
        dd=dd,
        bpwf=bpwf1,
        bpwf0=bpwf0,
    )

# ----------------------------------------------------------------------------------
'''
print("-----Packing likelihood-----")
if blinded:
    file_like = dir_like[:-1] + "_blinded.tar.gz"
    with open(f"blinding/blinding_{Lmin}_{Lmax}_{nbins}_{blind_str}", "rb") as f:
        obfuscating_factor = f.read()
    rdl0 = rdl0 * pickle.loads(obfuscating_factor)

else:
    file_like = dir_like[:-1] + "_unblinded.tar.gz"


# Likelihood format
binnum = np.arange(nbins) + 1
Lmin = np.int32(np.round(bb[:-1]))  # Lower end of the bin
Lmax = np.int32(np.round(bb[1:]))  # Upper end of the bin
L_av = 0.5 * (bb[1:] + bb[:-1])  # Mid-point of the bin
clpp_hat = rdl0 * 4 / 2 / np.pi  # In units of [L(L+1)]^2/(2*pi)
err = np.zeros(nbins)  # This isnt needed (reads from cov)
Ahat = np.zeros(nbins)  # This isnt needed (reads from cov)
print("Saving bandpower file:", dir_like + "bandpowers.dat")
np.savetxt(
    dir_like + "bandpowers.dat",
    np.c_[binnum, Lmin, Lmax, L_av, clpp_hat, err, Ahat],
    fmt="%d %d %d %d %.5e %.5e %.3f",
    header="[0]bin [1]L_min [2]L_max [3]L_av [4]PP [5]Error [6]Ahat",
    delimiter="\t",
)

# Bin window functions right now, tophat
Path(dir_like + "windows/").mkdir(parents=True, exist_ok=True)
for i in range(0, nbins):
    print("Saving window file:", dir_like + "windows/window%u.dat" % i)
    # Li = np.arange(Lmin[i],Lmax[i])
    # w  = np.ones_like(Li)/(Li[-1]-Li[0])
    # np.savetxt(dir_out+'windows/window%u.dat'%i,np.c_[Li,w],fmt='%d %.5f')
    l = np.arange(4001)
    bw = bpwf0[:, i][bpwf0[:, i] != 0]
    bl = l[bpwf0[:, i] != 0]
    np.savetxt(dir_like + "windows/window%u.dat" % i, np.c_[bl, bw], fmt="%d %.5f")

# Covariance
print("Saving cov file:", dir_like + "cov.dat")
cov = np.cov(arr0 * 4 / 2 / np.pi)
np.savetxt(dir_like + "cov.dat", cov)

# Foreground template
print("Saving foreground template file:", dir_like + "foreground_template.dat")
np.savetxt(
    dir_like + "foreground_template.dat",
    np.c_[binnum, clpp_hat * 0],
    fmt="%d %.5e",
    header="# binnum PP",
)

# lens_delta_window
Path(dir_like + "lens_delta_windows/").mkdir(parents=True, exist_ok=True)
Path(dir_like + "lens_delta_windows_phicmb/").mkdir(parents=True, exist_ok=True)
Path(dir_like + "lens_delta_windows_phionly/").mkdir(parents=True, exist_ok=True)

with tarfile.open(file_like, "w:gz") as tar:
    tar.add(dir_like, arcname=os.path.basename(dir_like))

print(f"Saving %s" % (os.path.abspath(file_like)))


# python old_lpt.py lpt 1 -1 10 --butterworth 400 12
'''
