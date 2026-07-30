# Apply mask to input kappa to be used for response caluclation
 
import os,sys
import numpy as np
import healpy as hp
sys.path.append('/data/gpfs/projects/punim1922/summerfield_lensing/healqest/healqest/src/')
from pathlib import Path
import yaml
import argparse


def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
    lmaxin  = hp.Alm.getlmax(alm.shape[0])
    print( "reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
    ell,emm = hp.Alm.getlm(lmaxin)
    almout  = np.zeros(hp.Alm.getsize(lmax),dtype=np.complex_)
    oldi=0
    oldf=0
    newi=0
    newf=0
    dl = lmaxin-lmax
    for i in range(0,lmax+1):
        oldf=oldi+lmaxin+1-i
        newf=newi+lmax+1-i
        almout[newi:newf]=alm[oldi:oldf-dl]
        oldi=oldf
        newi=newf
    return almout


def tp2rd(tht,phi):
        ra      = phi/np.pi*180.0
        dec     = -1*(tht/np.pi*180.0-90)
        return ra,dec

def rd2tp(ra,dec):
        phi     = ra/180.0*np.pi
        tht     = (-dec+90)/180.0*np.pi
        return tht,phi

parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest = 'seed', type=int)
parser.add_argument('--config_file', dest='config_file', type=str)
args = parser.parse_args()

config = yaml.safe_load(open(args.config_file))


if args.seed>250:
    patch  = 2
else:
    patch  = 1

runname = config['base']['runname']
rectype = config['lensrec']['rectype']
nside = config['inputs']['nside']
mask_file = config['inputs']['mask_file']
lmax = config['inputs']['lmax']
phi_in = config['inputs']['phi_in']
kappa_in = config['inputs']['kappa_in']

# Load mask to apply 
mask = hp.read_map(mask_file)
mask[mask == hp.UNSEEN] = 0.
pix = np.where(mask>0.0)[0]

pixs={}
pixs[1] = pix

idx = pixs[patch]

# Load fullsky input phi
if args.seed>250:
    seed = args.seed-250
else:
    seed = args.seed

plm      = hp.read_alm(phi_in.format(seed=args.seed))

# Convert to fullsky kappa
l       = np.arange(lmax+1)
ell,emm = hp.Alm.getlm(lmax)
plm     = reduce_lmax(plm,lmax=lmax)
klm     = hp.almxfl(plm,0.5*l*(l+1))

# Apply mask to get masked kappa
k       = hp.alm2map(klm,nside)
klm       = hp.map2alm(k*mask*mask,lmax=lmax,use_pixel_weights=True) 

# Deconvolve pixwin function
pixwin = hp.pixwin(nside)
#klm = hp.almxfl(klm, 1/pixwin)
# Reconstructed map are made from [X*mask,X*mask] so mask is multiplied twice
# We apply the same power of mask here as well.

# Save output kappa
dir_out = str(Path(kappa_in.format(runname=runname,rectype=rectype,seed=args.seed)).parent)
Path(dir_out).mkdir(parents=True, exist_ok=True)
hp.write_alm(kappa_in.format(runname=runname,rectype=rectype,seed=args.seed),klm,overwrite=True)
