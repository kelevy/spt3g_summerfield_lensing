import os
import sys
import numpy as np
import healpy as hp
import argparse
import pathlib
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../healqest/src/')))
import healqest_utils as utils


############################################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("file_yaml", default=None, type=str, help="file_yaml")
parser.add_argument('--nsims', default=1, type=int, dest="nsims")
parser.add_argument('--nber_bundles', default=1, type=int, dest="nber_bundles")
args = parser.parse_args()

nsims        = args.nsims 
nber_bundles = args.nber_bundles
config       = utils.load_yaml(args.file_yaml)
runname      = config['base']['runname']
rectype      = config['lensrec']['rectype']
nside        = config['lensrec']['nside']
qe           = config['lensrec']['qesttype']
mask_file    = config['pspec']['mask_analysis']
polspice     = config['pspec']['polspice']

mask = hp.read_map(mask_file)
mask[mask == hp.UNSEEN] = 0.
fsky = np.mean(mask**2)

kmaps_dir = config["pspec"]["dir_kmaps"].format(runname=runname, rectype=rectype)

if polspice:
    dir_pspec = config["pspec"]["dir_pspec"].format(runname=runname, rectype=rectype)+"polspice/"
else:
    dir_pspec = config["pspec"]["dir_pspec"].format(runname=runname, rectype=rectype)+"healpy/"

dir_pspec = dir_pspec+'sanity_checks/'
pathlib.Path(dir_pspec).mkdir(parents=True, exist_ok=True)


##############################################################################################################################################


def phi2kappa_alm(ell, phi_alm):
    kappa_alm = hp.almxfl(phi_alm, 0.5*ell*(ell+1))
    kappa_alm[np.isnan(kappa_alm)] = 0
    return kappa_alm


# cross-spectrum inputXoutput
"""
for seedval in range(1, nsims+1):

    klm_in = hp.read_alm(config["inputs"]["kappa_in"].format(runname=runname, rectype=rectype, seed=seedval))
    kmap_in = hp.alm2map(klm_in, nside=nside)

    kmap_out_arr = []

    for bundleid in range(nber_bundles):
        
        kmap_out_1 = hp.read_map(kmaps_dir+'bundle%s/kmapxx_TT_%s_1.fits'%(bundleid, seedval))
        kmap_out_2 = hp.read_map(kmaps_dir+'bundle%s/kmapxx_TT_%s_2.fits'%(bundleid, seedval))
        kmap_out = (kmap_out_1+kmap_out_2)/2
        kmap_out_arr.append(kmap_out)

    kmap_outavg = np.mean(kmap_out_arr, axis=0)
    cls_in_out = hp.anafast(kmap_in, kmap_outavg)/fsky
    np.save(dir_pspec+'cross_spectrum_input%sxoutput%s.npy' %(seedval, seedval), cls_in_out)

"""
# cross-spectrum mean-fieldXoutput

for bundleid in range(nber_bundles):

    kmap_out0_1 = hp.read_map(kmaps_dir+'bundle%s/kmapxx_TT_0_1.fits'%(bundleid))
    kmap_out0_2 = hp.read_map(kmaps_dir+'bundle%s/kmapxx_TT_0_2.fits'%(bundleid))
    kmap_out0 = (kmap_out0_1+kmap_out0_2)/2

    plm_dir = config["lensrec"]["dir_out"].format(runname=runname, rectype=rectype, bundleid=bundleid)
    mf = np.load(plm_dir+'glmstack%s_xx.npz' %(qe))
    plm_mf = mf['gmfxx']
    nsim = mf['nsim']
    plm_mf = plm_mf/nsim
    totresp = np.load(plm_dir+'respavg%s.npz' %(qe))['resp']
    plm_mf_norm = hp.almxfl(plm_mf, 1/totresp) 
    klm_mf_norm = phi2kappa_alm(np.arange(len(plm_mf_norm)), plm_mf_norm)
    kmap_mf_norm = hp.alm2map(klm_mf_norm, nside = nside)

    kmap_out_arr = []

    for seedval in range(1, nsims+1):
        
        kmap_out_1 = hp.read_map(kmaps_dir+'bundle%s/kmapxx_TT_%s_1.fits'%(bundleid, seedval))
        kmap_out_2 = hp.read_map(kmaps_dir+'bundle%s/kmapxx_TT_%s_2.fits'%(bundleid, seedval))
        kmap_out = (kmap_out_1+kmap_out_2)/2
        kmap_out_arr.append(kmap_out)

    kmap_outavg = np.mean(kmap_out_arr, axis=0)
    
    cls_mf_out0 = hp.anafast(kmap_mf_norm, kmap_out0)/fsky
    np.save(dir_pspec+'cross_spectrum_bundle%s_mfxoutput0.npy' %(bundleid), cls_mf_out0)

    cls_mf_outavg = hp.anafast(kmap_mf_norm, kmap_outavg)/fsky
    np.save(dir_pspec+'cross_spectrum_bundle%s_mfxoutputavg.npy' %(bundleid), cls_mf_outavg)



