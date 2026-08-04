"""
Code to compute the final cmbkappa auto-spectra using PolSpice or alm2cl.
"""

import os, sys, yaml
import pathlib
import numpy as np
import healpy as hp
import subprocess
import argparse
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../healqest/src/")
import healqest_utils as hutils
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("file_yaml", default=None, type=str, help="file_yaml")
parser.add_argument("qeset", default=None, type=str, help="qeset")
parser.add_argument("seed", default=None, type=int, help="index")
parser.add_argument("dseed", default=None, type=int, help="index")
parser.add_argument("-src_yaml", default=None, type=str, dest="src_yaml", help="src/prf yaml")
parser.add_argument('-prftype' , default='tsz',dest='prftype', help="tsz or 1am")
parser.add_argument("--xx", default=False, dest="xx", action="store_true")
parser.add_argument("--N0", default=False, dest="N0", action="store_true")
parser.add_argument("--N0x", default=False, dest="N0x", action="store_true")
parser.add_argument("--RDN0", default=False, dest="RDN0", action="store_true")
parser.add_argument("--curlRDN0", default=False, dest="curlRDN0", action="store_true")
parser.add_argument("--N1", default=False, dest="N1", action="store_true")
parser.add_argument("--N1_sep", default=False, dest='N1_sep', action="store_true")
parser.add_argument("--MF", default=False, dest="MF", action="store_true")
parser.add_argument("--mfsplit", default=False, dest="mfsplit", action="store_true")
parser.add_argument("--mfsub", default=False, dest="mfsub", action="store_true")
parser.add_argument("--curl", default=False, dest="curl", action="store_true")
parser.add_argument("--alm2cl", default=False, dest="alm2cl", action="store_true")
parser.add_argument("--nops", default=False, dest="nops", action="store_true")
parser.add_argument("--combine_mf", default=False, dest="combine_mf", action="store_true")


args         = parser.parse_args()
qeset        = args.qeset
seed         = args.seed
nber_bundles = 1
prftype      = args.prftype
xx           = args.xx
N0           = args.N0
N0x          = args.N0x
N1           = args.N1
N1_sep       = args.N1_sep
RDN0         = args.RDN0
curlRDN0     = args.curlRDN0
MF           = args.MF
dseed        = args.dseed
mfsplit      = args.mfsplit
mfsub        = args.mfsub
curl         = args.curl
alm2cl       = args.alm2cl
nops         = args.nops
mfsubxx      = True
combine_mf   = args.combine_mf



#sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/build/")
#sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/mapspectra/python")
#from spt3g import core, maps
#import curved_sky

def polspice(map1, map2=None, mask=None, nlmax=3500, apodizesigma=30, thetamax=30, subav=True, apodizetype = 1, 
             clfile=''):

    binned_el, binned_cl = curved_sky.spectrum_spice(
                            map1 = map1, 
                            map2 = map2, 
                            mask=mask,
                            nlmax = nlmax, 
                            apodizesigma=apodizesigma,
                            thetamax=thetamax,
                            subav=subav,
                            apodizetype=apodizetype,
                            clfile=clfile,
                            verbose=True)


def compute_cls(config, seed, ktype="xxxx", bundleid1=None, bundleid2=None):
    """
    Compute power spectra using alm2cl or PolSpice.

    Parameters
    ----------
    config : dict
        Configuration based on YAML file.
    seed : int
        Seed number.
    ktype : str
        Type of correlation.

    Returns
    -------
    None
        This function does not return any value.

    """

    print("-----------------------------------------------------")
    
    # grad/curl flag
    gcid = ("kk" if gcmode == "g" else "ww" if gcmode == "c" else sys.exit("unknown gcmode"))
    qid = config["lensrec"]["qesttype"]

    # Assign values based on ktype
    i = seed
    u = i + 1
    t = config["pspec"]["dseed"]

    ktype_map = {
        "xxxx": (f"{i}a", f"{i}a", f"{i}a", f"{i}a"),
        "xyxy": (f"{i}a", f"{u}a", f"{i}a", f"{u}a"),
        "xyyx": (f"{i}a", f"{u}a", f"{u}a", f"{i}a"),
        "xdxd": (f"{i}a", f"{t}a", f"{i}a", f"{t}a"),
        "xddx": (f"{i}a", f"{t}a", f"{t}a", f"{i}a"),
        "dxxd": (f"{t}a", f"{i}a", f"{i}a", f"{t}a"),
        "dxdx": (f"{t}a", f"{i}a", f"{t}a", f"{i}a"),
        "wdwd": (f"{i}b", f"{t}a", f"{i}b", f"{t}a"),
        "wddw": (f"{i}b", f"{t}a", f"{t}a", f"{i}b"),
        "dwwd": (f"{t}a", f"{i}b", f"{i}b", f"{t}a"),
        "dwdw": (f"{t}a", f"{i}b", f"{t}a", f"{i}b"),
        "abab": (f"{i}a", f"{i}b", f"{i}a", f"{i}b"),
        "abba": (f"{i}a", f"{i}b", f"{i}b", f"{i}a"),
        "ayay": (f"{i}a", f"{u}b", f"{i}a", f"{u}b"),
        "ayya": (f"{i}a", f"{u}b", f"{u}b", f"{i}a"),
        "mfmf": ("mf", "mf", "mf", "mf"),
    }
    ii, jj, xx, yy = ktype_map.get(ktype, ("", "", "", ""))

    # Temporary map files
    if bundleid1 is None and bundleid2 is None:
        file_1 = config["pspec"]["dir_kmaps"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"]) + "N1/kmap%s_%s_%d_1.fits" % (ktype[:2], qid, i)
        file_2 = config["pspec"]["dir_kmaps"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"]) + "N1/kmap%s_%s_%d_2.fits" % (ktype[2:], qid, i)
    else:
        file_1 = config["pspec"]["dir_kmaps"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"]) + "kmap%s_%s_%d_1.fits" % (ktype[:2], qid, i)
        file_2 = config["pspec"]["dir_kmaps"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"]) + "kmap%s_%s_%d_2.fits" % (ktype[2:], qid, i)
    file_mask = config["pspec"]["mask_analysis"]

    map1 = hp.read_map(file_1)
    map2 = hp.read_map(file_2)

    mask = hp.read_map(file_mask)
    mask[mask == hp.UNSEEN] = 0.
    fsky = np.mean(mask**2)

    # Compute Cls and remove intermediate files
    if not config["pspec"]["polspice"]:
        print('Using healpy')
        alm1 = hp.map2alm(map1 * mask, lmax=config["pspec"]["nlmax"], use_pixel_weights=True)
        alm2 = hp.map2alm(map2 * mask, lmax=config["pspec"]["nlmax"], use_pixel_weights=True)
        cls = hp.alm2cl(alm1, alm2)/fsky #* mask.shape[0] / np.sum(mask**2)
        tmp = np.c_[np.arange(config["pspec"]["nlmax"] + 1), cls].T

        dir_pspec = config["pspec"]["dir_pspec"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"])+"healpy/"
        if bundleid1 is None and bundleid2 is None: 
            dir_pspec = dir_pspec+'N1/'
        else:
            dir_pspec = dir_pspec
        pathlib.Path(dir_pspec).mkdir(parents=True, exist_ok=True)
        np.savez(dir_pspec+ "/cl%s_%s_%s_%s_%s_%s.npz" % (gcid, qid, ii, jj, xx, yy), cls=tmp.T)

    else:
        print('Using PolSpice')
        os.environ["HEALPIX"] = config["base"]["dir_healpix"]
        spice = config["base"]["dir_spice"]
        polspice(map1, map2, mask=mask,
                nlmax =  config["pspec"]["nlmax"], apodizesigma = config["pspec"]["apodizesigma"], thetamax = config["psec"]["thetamax"], subav = config["psec"]["subav"],
                apodizetype = config["psec"]["apodizetype"], clfile=config["psec"]["dir_cls"]+"cl%s_%s_%s_%s_%s_%s.dat"%(gcid,qid,ii,jj,xx,yy))

        tmp = np.loadtxt(config["pspec"]["dir_cls"]+"cl%s_%s_%s_%s_%s_%s.dat" % (gcid, qid, ii, jj, xx, yy))

        print(f"Applying mask correction factor: {config['pspec']['maskfac']}")
        tmp[:, 1] = tmp[:, 1] * config['pspec']['maskfac']


        dir_pspec = config["pspec"]["dir_pspec"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"])+"polspice/"
        if bundleid1 is None and bundleid2 is None: 
            dir_pspec = dir_pspec+'N1/'
        else:
            dir_pspec = dir_pspec+'bundle%sXbundle%s/'%(bundleid1,bundleid2)
        pathlib.Path(dir_pspec).mkdir(parents=True, exist_ok=True)
    
        np.savez(dir_pspec+"cl%s_%s_%s_%s_%s_%s.npz" % (gcid, qid, ii, jj, xx, yy),cls=tmp,**psconfig)
        os.remove(dir_pspec+"cl%s_%s_%s_%s_%s_%s.dat" % (gcid, qid, ii, jj, xx, yy))

    print("Saved Cls to: %s" %(dir_pspec+"cl%s_%s_%s_%s_%s_%s.npz" % (gcid, qid, ii, jj, xx, yy)))


def process_mf(seed, plm, plmstack, mfsplit, ktype, gcmode):
    """
    Process mean-field.

    Parameters
    ----------
    seed : int
        Seed number.
    plm : ndarray
        Particular plm array.
    plmstack : ndarray
        Summed plm array.
    split : int
        Which split to use.
    ktype : str
        Kappa type.
    gcmode : str
        Grad or curl mode.

    Returns
    -------
    Array
        Mean-field.

    """

    idx0 = np.arange(1, plmstack["nsim"] + 1)
    idx1, idx2 = np.split(idx0, 2)

    if mfsplit == 0:
        print("using split0")
        if seed in idx0:
            mf = (plmstack[f"{gcmode}mf{ktype}"] - plm) / (plmstack["nsim"] - 1.0)
        else:
            mf = (
                (plmstack[f"{gcmode}mf{ktype}"]) / (plmstack["nsim"])
            )  # mainly for data

    elif mfsplit == 1:
        print("using split1")
        if seed in idx1:
            mf = (plmstack[f"{gcmode}mf{ktype}_half1"] - plm) / (
                plmstack["nsim_half"] - 1.0
            )
        else:
            mf = (plmstack[f"{gcmode}mf{ktype}_half1"]) / (plmstack["nsim_half"])

    elif mfsplit == 2:
        print("using split2")
        if seed in idx2:
            mf = (plmstack[f"{gcmode}mf{ktype}_half2"] - plm) / (
                plmstack["nsim_half"] - 1.0
            )
        else:
            mf = (plmstack[f"{gcmode}mf{ktype}_half2"]) / (plmstack["nsim_half"])

    return mf

def get_kmap(config, seed, ktype="xx", mapnum=1, bundleid=None):
    """
    Create kappa map given plms, response and meanfield

    Parameters
    ----------
    config : dict
        Dictionary containg all settings.
    seed: int
        Seed number.
    ktype: str
        kappa type xx/xy/yx/ab/ba.
    mapnum: int
        map identifier 1 or 2.

    Returns
    -------
    None
        This function does not return any value.

    """

    l = np.arange(config["pspec"]["nlmax"] + 1)
    ell, emm = hp.Alm.getlm(config["pspec"]["nlmax"])

    # Assign values based on ktype
    z = seed
    u = seed + 1
    t = config["pspec"]["dseed"]

    ktype_map = {
        "xx": (f"{z}a", f"{z}a", f"{z}a"),
        "xy": (f"{z}a", f"{u}a"),
        "yx": (f"{u}a", f"{z}a"),
        "xd": (f"{z}a", f"{t}a"),
        "dx": (f"{t}a", f"{z}a"),
        "wd": (f"{z}b", f"{t}a"),
        "dw": (f"{t}a", f"{z}b"),
        "ab": (f"{z}a", f"{z}b"),
        "ba": (f"{z}b", f"{z}a"),
        "ay": (f"{z}a", f"{u}b"),
        "ya": (f"{u}b", f"{z}a"),
        "mf": (f"{t}a", f"{t}a"),
    }

    values = ktype_map.get(ktype)
    if values is None:
        sys.exit("Undefined")
    elif ktype == "xx":
        ii, jj, kk = values
    else:
        ii, jj = values

    # For GMV, estimators are combined already.
    if config["lensrec"]["qesttype"] == "GMV":
        qes = ["GMVTTEETE", "GMVTBEB"]
    else:
        qes = [config["lensrec"]["qesttype"]] 

    for qe in qes:
        print("using %s estimator" % qe)
    
    # Loop over all the qes of interest
    kmv = 0
    respmv = 0

    for qe in qes:
        if bundleid is None:
            dir_plm = config["lensrec"]["dir_out_N1"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"]) 
        else:
            dir_plm = config["lensrec"]["dir_out"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"])
        file_resp = dir_plm + f"respavg{qe}{psflag}.npz"
        

        print("Loading resp:", file_resp)
        resp = np.load(file_resp)["resp"]
        resp = np.where(np.isnan(resp) | (resp == 0), 1e30, resp)
        resp = resp[ell]

        # Load plm and plmstack
        file_plm = dir_plm + f"plm{qe}_{ii}_{jj}.npz"

        if ktype == "xd" or ktype == "dx":
            file_plmstack = dir_plm + f"{gcmode}lmstack{qe}_{ktype}_dataidx{t}.npz"
        else:
            file_plmstack = dir_plm + f"{gcmode}lmstack{qe}_{ktype}.npz"
            if config["pspec"]["combine_mf"] and ktype == "xx":
                file_plmstack = dir_plm + f"{gcmode}lmstack{qe}_xx.npz"  #xxyy.npz"

        print(f"Loading plm     : {file_plm}")
        print(f"Loading plmstack: {file_plmstack}")
        print(f"Using gcmode    : {gcmode}")

        plm = np.load(file_plm)[f"{gcmode}lm"]

        if mfsub:
            print("Subtracting MF")
            plmstack = np.load(file_plmstack)
            mf = process_mf(seed, plm, plmstack, config["pspec"][f"mfsplit{mapnum}"], ktype, gcmode)
            klm = 0.5 * ell * (ell + 1) * (plm - mf)

        else:
            print("Not subtracting MF")
            klm = 0.5 * ell * (ell + 1) * (plm)

        if ktype == "mf":
            file_plmstack = dir_plm + f"{gcmode}lmstack{qe}_xx.npz"
            plmstack = np.load(file_plmstack)
            mf = process_mf(
                seed, plm, plmstack, config["pspec"][f"mfsplit{mapnum}"], "xx", gcmode
            )  # using the xx mf
            klm = 0.5 * ell * (ell + 1) * (mf)

        # if qeset == 'qPP', 'qMV', 'qEBBE', 'qTBBT', instead of computing twice, just x2
        fac = 2 if config["lensrec"]["qesttype"][0] == "q" and qe in ["TE", "TB", "EB"] else 1

        kmv += klm * fac
        respmv += resp * fac

    kmv = kmv / respmv
    kmap = hp.alm2map(kmv, config["pspec"]["nside"])
    
    
    kmaps_dir = config["pspec"]["dir_kmaps"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"])
    if bundleid is None: 
        kmaps_dir = kmaps_dir+'N1/'
    else:
        kmaps_dir = kmaps_dir
    pathlib.Path(kmaps_dir).mkdir(parents=True, exist_ok=True)

    hp.write_map(
        kmaps_dir + f"kmap{ktype}_{qe}_{seed}_{mapnum}.fits",
        kmap,
        overwrite=True,
        dtype=np.float32)
    print("Creating:", kmaps_dir + f"/kmap{ktype}_{seed}_{mapnum}.fits")


def get_maskfac(config):
    """
    Calculate mask correction factor for power spectrum estimation.

    This correction factor accounts for the difference between masks used
    during map processing and power spectrum estimation. Since input T/E/B
    maps have a boundary mask applied, but PolSpice only considers the
    analysis mask in its correction, a conversion factor is needed.

    Parameters
    ----------
    config : dict
        Dictionary containg all settings.

    Returns
    -------
    float
        Mask correction factor.

    Notes
    -----
    nrm1 is the normalization constant already applied by PolSpice.
    nrm2 is the normalization constant computed taking into account
    the reconstruction mask.
    """

    psconfig = config["pspec"]

    assert psconfig["mask_analysis"] is not None, "Analysis mask must be specified"
    assert psconfig["mask_boundary"] is not None, "Boundary mask must be specified"

    ma = hp.read_map(psconfig["mask_analysis"])
    mb = hp.read_map(psconfig["mask_boundary"])

    assert hp.npix2nside(len(ma)) == hp.npix2nside(len(mb)), (
        "Masks must have same Nside"
    )

    nside = hp.npix2nside(ma.shape[0])

    if config["pspec"]["mask_analysis"] is not None:
        if "inverse" in config["pspec"]["mask_analysis"]:
            mr = hp.read_map(config["pspec"]["mask_analysis"], partial=True)
            mr[mr == hp.UNSEEN] = 0
            mr = 1 - mr
        else:
            mr = hp.read_map(config["pspec"]["mask_analysis"])

        if hp.npix2nside(len(mr)) != nside:
            mr = hp.ud_grade(mr, nside_out=nside)
    else:
        mr = np.ones_like(mb)

    nrm1 = hp.nside2npix(nside) / np.sum(ma * ma)
    nrm2 = hp.nside2npix(nside) / np.sum(
        (mb * mb * mr * mr * ma) * (mb * mb * mr * mr * ma)
    )
    maskfac = nrm2 / nrm1
    print(f"Normalization fac1: {nrm1:0.4f}")
    print(f"Normalization fac2: {nrm2:0.4f}")
    print(f"Mark correction factor (nrm2/nrm1): {maskfac:0.4f}")
    return maskfac


def cleanup(config, seed, nber_bundles=None):
    """
    Remove all temporary files.

    Parameters
    ----------
    dir_tmp : str
        Directory containing the temporary maps.
    seed : int
        Seed number used in the file naming.

    Returns
    -------
    None
        This function does not return any value.
    """
    print("Deleting the following files:")
    dir_tmp = config["pspec"]["dir_kmaps"].format(runname=config["base"]["runname"], rectype=config["lensrec"]["rectype"])       
    if nber_bundles is None:
        [print(f) for f in Path(dir_tmp+'N1/').glob(f"kmap*_{seed}_1.fits")]
        [print(f) for f in Path(dir_tmp+'N1/').glob(f"kmap*_{seed}_2.fits")]
        [f.unlink() for f in Path(dir_tmp+'N1/').glob(f"kmap*_{seed}_1.fits")]
        [f.unlink() for f in Path(dir_tmp+'N1/').glob(f"kmap*_{seed}_2.fits")]
    else:
        for bundleid in range(nber_bundles):
            [print(f) for f in Path(dir_tmp).glob(f"kmap*_{seed}_1.fits")]
            [print(f) for f in Path(dir_tmp).glob(f"kmap*_{seed}_2.fits")]
            [f.unlink() for f in Path(dir_tmp).glob(f"kmap*_{seed}_1.fits")]
            [f.unlink() for f in Path(dir_tmp).glob(f"kmap*_{seed}_2.fits")]


if __name__ == "__main__":
    print("Reading from yaml file: %s" % args.file_yaml)
    config = hutils.load_yaml(args.file_yaml)

    # Read ell ranges (mainly for naming)
    runname = config["base"]["runname"]
    rectype = config["lensrec"]["rectype"]
    psname = config["pspec"]["psname"]
    lminT = config["lensrec"]["lminT"]
    lminP = config["lensrec"]["lminP"]
    lmaxT = config["lensrec"]["lmaxT"]
    lmaxP = config["lensrec"]["lmaxP"]
    mmin = config["lensrec"]["mmin"]
    mask_lensrec = config["pspec"]["mask_analysis"]
    polspice = config["pspec"]["polspice"]

    # Set various flags
    gcmode = "c" if curl else "g"
    mfsplit1, mfsplit2 = (1, 2) if mfsplit else (0, 0)
    psflag = "_nops" if nops else ""

    # Compute mask conversion factor
    maskfac = get_maskfac(config)

    # Add extra information
    config["pspec"].update({"mfsub": mfsub,
                            "mfsplit1": mfsplit1,
                            "mfsplit2": mfsplit2,
                            "psflag": psflag,
                            "dseed": dseed,
                            "gcmode": gcmode,
                            "maskfac": maskfac,
                            "mask_lensrec": mask_lensrec,
                            "combine_mf": combine_mf})

    
    if args.xx:
        # Compute power spectra required for biased raw spectra (xxxx)
        get_kmap(config, seed, ktype="xx", mapnum=1, bundleid=0)
        get_kmap(config, seed, ktype="xx", mapnum=2, bundleid=0)
        
        compute_cls(config, seed, ktype='xxxx', bundleid1=0, bundleid2=0)

        # keep data and 10 sims for inspection, delete rest
        if seed >= 11:
            cleanup(config, seed, nber_bundles)

    if args.N0:
        # Compute power spectra required for N0 (xyxy + xyyx)
        for bundleid in range(nber_bundles):
            get_kmap(config, seed, ktype="xy", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="xy", mapnum=2, bundleid=bundleid)
            get_kmap(config, seed, ktype="yx", mapnum=2, bundleid=bundleid)

        for bundleid1 in range(nber_bundles):
            for bundleid2 in range(nber_bundles):
                if bundleid1>bundleid2: 
                    continue    
                elif bundleid1==bundleid2 and not config['pspec']['auto']:
                    continue
                elif bundleid1<bundleid2 and not config['pspec']['cross']:
                    continue
                else:
                    compute_cls(config, seed, ktype="xyxy", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="xyyx", bundleid1=bundleid1, bundleid2=bundleid2)

        if seed >= 11:
            cleanup(config, seed, nber_bundles)

    if args.RDN0:
        # Compute power spectra required for RDN0 (xdxd + xddx + dxdx + dxxd)
        for bundleid in range(nber_bundles):
            get_kmap(config, seed, ktype="xd", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="xd", mapnum=2, bundleid=bundleid)
            get_kmap(config, seed, ktype="dx", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="dx", mapnum=2, bundleid=bundleid)
           
        for bundleid1 in range(nber_bundles):
            for bundleid2 in range(nber_bundles):
                if bundleid1>bundleid2: 
                    continue        
                elif bundleid1==bundleid2 and not config['pspec']['auto']:
                    continue
                elif bundleid1<bundleid2 and not config['pspec']['cross']:
                    continue
                else:
                    compute_cls(config, seed, ktype="xdxd", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="xddx", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="dxxd", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="dxdx", bundleid1=bundleid1, bundleid2=bundleid2)

        if seed >= 11:
            cleanup(config, seed, nber_bundles)

    if args.curlRDN0:
        # Compute RDN0 for curl
        for bundleid in range(nber_bundles):
            get_kmap(config, seed, ktype="wd", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="wd", mapnum=2, bundleid=bundleid)
            get_kmap(config, seed, ktype="dw", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="dw", mapnum=2, bundleid=bundleid)
        
        for bundleid1 in range(nber_bundles):
            for bundleid2 in range(nber_bundles):
                if bundleid1>bundleid2: 
                    continue       
                elif bundleid1==bundleid2 and not config['pspec']['auto']:
                    continue
                elif bundleid1<bundleid2 and not config['pspec']['cross']:
                    continue
                else:
                    compute_cls(config, seed, ktype="wdwd", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="wddw", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="dwwd", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="dwdw", bundleid1=bundleid1, bundleid2=bundleid2)

        if seed >= 11:
            cleanup(config, seed, nber_bundles)

    if args.N1:
        for bundleid in range(nber_bundles):
            get_kmap(config, seed, ktype="ab", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="ab", mapnum=2, bundleid=bundleid)
            get_kmap(config, seed, ktype="ba", mapnum=2, bundleid=bundleid)

        for bundleid1 in range(nber_bundles):
            for bundleid2 in range(nber_bundles):
                if bundleid1>bundleid2: 
                    continue       
                elif bundleid1==bundleid2 and not config['pspec']['auto']:
                    continue
                elif bundleid1<bundleid2 and not config['pspec']['cross']:
                    continue
                else:
                    compute_cls(config, seed, ktype="abab", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="abba", bundleid1=bundleid1, bundleid2=bundleid2)
        
        if seed >= 11:
            cleanup(config, seed)


    if args.N1_sep:
        get_kmap(config, seed, ktype='xy', mapnum=1)
        get_kmap(config, seed, ktype='xy', mapnum=2)
        get_kmap(config, seed, ktype='yx', mapnum=2)
        get_kmap(config, seed, ktype="ab", mapnum=1)
        get_kmap(config, seed, ktype="ab", mapnum=2)
        get_kmap(config, seed, ktype="ba", mapnum=2)
            
        compute_cls(config, seed, ktype="xyxy")
        compute_cls(config, seed, ktype="xyyx")
        compute_cls(config, seed, ktype="abab")
        compute_cls(config, seed, ktype="abba")
        
        if seed >= 11:
            cleanup(config, seed)

    if args.N0x:
        # Compute power spectra required for N0 (xyxy + xyyx)
        for bundleid in range(nber_bundles):
            get_kmap(config, seed, ktype="ay", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="ay", mapnum=2, bundleid=bundleid)
            get_kmap(config, seed, ktype="ya", mapnum=2, bundleid=bundleid)
        
        for bundleid1 in range(nber_bundles):
            for bundleid2 in range(nber_bundles):
                if bundleid1>bundleid2: 
                    continue       
                elif bundleid1==bundleid2 and not config['pspec']['auto']:
                    continue
                elif bundleid1<bundleid2 and not config['pspec']['cross']:
                    continue
                else:
                    compute_cls(config, seed, ktype="ayay", bundleid1=bundleid1, bundleid2=bundleid2)
                    compute_cls(config, seed, ktype="ayya", bundleid1=bundleid1, bundleid2=bundleid2)
        
        if seed >= 11:
            cleanup(config, seed, nber_bundles)

    if args.MF:
        # Compute power spectra of meanfield
        for bundleid in range(nber_bundles):
            get_kmap(config, seed, ktype="mf", mapnum=1, bundleid=bundleid)
            get_kmap(config, seed, ktype="mf", mapnum=2, bundleid=bundleid)
        
        for bundleid1 in range(nber_bundles):
            for bundleid2 in range(nber_bundles):
                if bundleid1>bundleid2: 
                    continue  
                elif bundleid1==bundleid2 and not config['pspec']['auto']:
                    continue
                elif bundleid1<bundleid2 and not config['pspec']['cross']:
                    continue
                else:
                    compute_cls(config, seed, ktype="mfmf", bundleid1=bundleid1, bundleid2=bundleid2)

        if seed >= 11:
            cleanup(config, seed, nber_bundles)
