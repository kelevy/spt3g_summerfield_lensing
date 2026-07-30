import sys, os
import numpy as np
import healpy as hp
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../healqest/src/')))
import healqest_utils as utils


##############################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("file_yaml", default=None, type=str, help="file_yaml")
parser.add_argument('--nsims', default=1, type=int, dest="nsims")
parser.add_argument("--xx", default=False, dest="xx", action="store_true")
parser.add_argument("--N0", default=False, dest="N0", action="store_true")
parser.add_argument("--RDN0", default=False, dest="RDN0", action="store_true")
parser.add_argument("--curl", default=False, dest="curl", action="store_true")
parser.add_argument("--N1", default=False, dest="N1", action="store_true")
parser.add_argument("--N1_sep", default=False, dest='N1_sep', action="store_true")
args = parser.parse_args()

nsims        = args.nsims 
config       = utils.load_yaml(args.file_yaml)
xx           = args.xx
N0           = args.N0
RDN0         = args.RDN0
N1           = args.N1
N1_sep       = args.N1_sep
curl         = args.curl
runname      = config['base']['runname']
rectype      = config['lensrec']['rectype']
qe           = config['lensrec']['qesttype']
nside        = config['lensrec']['nside']
auto         = config['pspec']['auto'] 
cross        = config['pspec']['cross']
dir_kmaps_in = config['inputs']['kappa_in']
dir_kmaps    = config['pspec']['dir_kmaps']
dir_pspec    = config['pspec']['dir_pspec']
polspice     = config['pspec']['polspice']
mask_file    = config['pspec']['mask_analysis']

dir_kmaps_in = dir_kmaps_in.format(runname=runname, rectype=rectype, seed=1)
dir_kmaps = dir_kmaps.format(runname=runname, rectype=rectype)

if polspice:
    dir_pspec = dir_pspec.format(runname=runname, rectype=rectype)+'polspice/'
else:
    dir_pspec = dir_pspec.format(runname=runname, rectype=rectype)+'healpy/'


################################################################################################################


# getting stacked convergence maps
arcmin = (1/60)*(np.pi/180)
bl_1deg = hp.gauss_beam(fwhm=60*arcmin)

mask = hp.read_map(mask_file)

# input kappa
klm_in = hp.read_alm(dir_kmaps_in)
kmap_in = hp.alm2map(klm_in, nside=nside)
kmap_in_smoothed = hp.smoothing(kmap_in, beam_window=bl_1deg)
hp.write_map('/data/gpfs/projects/punim1922/summerfield_lensing/healqest/lensrec/%s/sqe/kappa_in/kmap_in_seed1.fits'%(runname), kmap_in, overwrite=True)
hp.write_map('/data/gpfs/projects/punim1922/summerfield_lensing/healqest/lensrec/%s/sqe/kappa_in/kmap_in_seed1_smoothed1deg.fits'%(runname), kmap_in_smoothed, overwrite=True)


# data
kmap_idx_1 = hp.read_map(dir_kmaps+'/kmapxx_TT_%s_1.fits'%(0))
kmap_idx_2 = hp.read_map(dir_kmaps+'/kmapxx_TT_%s_2.fits'%(0))
kmap_idx = (kmap_idx_1+kmap_idx_2)/2
kmap_idx_smoothed = hp.smoothing(kmap_idx, beam_window=bl_1deg)
hp.write_map(dir_kmaps+'kmapxx_TT_%s.fits' %(0), kmap_idx, overwrite=True)
hp.write_map(dir_kmaps+'kmapxx_TT_%s_smoothed1deg.fits' %(0), kmap_idx_smoothed, overwrite=True)

# sims
kmap_idx_1 = hp.read_map(dir_kmaps+'kmapxx_TT_%s_1.fits'%(1))
kmap_idx_2 = hp.read_map(dir_kmaps+'kmapxx_TT_%s_2.fits'%(1))
kmap_idx = (kmap_idx_1+kmap_idx_2)/2
kmap_idx_smoothed = hp.smoothing(kmap_idx, beam_window=bl_1deg)
hp.write_map(dir_kmaps+'kmapxx_TT_%s.fits' %(1), kmap_idx, overwrite=True)
hp.write_map(dir_kmaps+'kmapxx_TT_%s_smoothed1deg.fits' %(1), kmap_idx_smoothed, overwrite=True)


#################################################################################################################


# gett stacked power spectra
fkk   = []
fN0   = []
fRDN0 = []
fN1   = []
fww = []
fN0ww = []
fRDN0ww = []
fN1ww = []

for i in range(1,nsims+1): 

    if xx:
        xxxx=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,i,i,i,i))['cls'].T[1]
        fkk.append(xxxx)

    if N0:
        xyxy=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i,i+1))['cls'].T[1]
        xyyx=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i+1,i))['cls'].T[1]
        fN0.append(xyxy+xyyx)

    if RDN0:
        dxdx=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,0,i,0,i))['cls'].T[1]
        dxxd=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,0,i,i,0))['cls'].T[1]
        xdxd=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,i,0,i,0))['cls'].T[1]
        xddx=np.load(dir_pspec+'clkk_%s_%da_%da_%da_%da.npz'%(qe,i,0,0,i))['cls'].T[1]
        fRDN0.append((xdxd+dxdx+xddx+dxxd) - (xyxy+xyyx))
                
    if N1:
        abab=np.load(dir_pspec+'clkk_%s_%da_%db_%da_%db.npz'%(qe,i,i,i,i))['cls'].T[1]
        abba=np.load(dir_pspec+'clkk_%s_%da_%db_%db_%da.npz'%(qe,i,i,i,i))['cls'].T[1]
        fN1.append(abab+abba-(xyxy+xyyx))
               
    if curl:
        xxxx = np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,i,i,i,i))['cls'].T[1]
        fww.append(xxxx)

        xyxy=np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i,i+1))['cls'].T[1]
        xyyx=np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i+1,i))['cls'].T[1]
        fN0ww.append(xyxy+xyyx)

        dxdx=np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,0,i,0,i))['cls'].T[1]
        dxxd=np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,0,i,i,0))['cls'].T[1]
        xdxd=np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,i,0,i,0))['cls'].T[1]
        xddx=np.load(dir_pspec+'clww_%s_%da_%da_%da_%da.npz'%(qe,i,0,0,i))['cls'].T[1]
        fRDN0ww.append((xdxd+dxdx+xddx+dxxd) - (xyxy+xyyx))
                    
        if N1:
            abab=np.load(dir_pspec+'clww_%s_%da_%db_%da_%db.npz'%(qe,i,i,i,i))['cls'].T[1]
            abba=np.load(dir_pspec+'clww_%s_%da_%db_%db_%da.npz'%(qe,i,i,i,i))['cls'].T[1]
            fN1ww.append(abab+abba-(xyxy+xyyx))

if xx:
    navgfkk = np.mean(fkk, axis=0)
    np.save(dir_pspec+'fkk_avg.npy', navgfkk)

if N0:
    navgfN0 = np.mean(fN0, axis=0)
    np.save(dir_pspec+'N0_avg.npy', navgfN0)

if RDN0:
    navgfRDN0 = np.mean(fRDN0, axis=0)
    np.save(dir_pspec+'RDN0_avg.npy', navgfRDN0)

if N1:
    navgfN1 = np.mean(fN1, axis=0)
    np.save(dir_pspec+'N1_avg.npy', navgfN1)

if curl:
    navgfww = np.mean(fww,axis=0)
    np.save(dir_pspec+'fww_avg.npy', navgfww)

    navgfN0ww = np.mean(fN0ww, axis=0)
    np.save(dir_pspec+'N0ww_avg.npy', navgfN0ww)

    navgfRDN0ww = np.mean(fRDN0ww, axis=0)
    np.save(dir_pspec+'RDN0ww_avg.npy', navgfRDN0ww)

    if N1:
        navgfN1ww = np.mean(fN1ww, axis=0)
        np.save(dir_pspec+'N1ww_avg.npy', navgfN1ww)



if N1_sep:
    N1_arr = []
    for i in range(1,nsims+1): 
        N1xyxy=np.load(dir_pspec+'N1/clkk_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i,i+1))['cls'].T[1]
        N1xyyx=np.load(dir_pspec+'N1/clkk_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i+1,i))['cls'].T[1]
        N1abab=np.load(dir_pspec+'N1/clkk_%s_%da_%db_%da_%db.npz'%(qe,i,i,i,i))['cls'].T[1]
        N1abba=np.load(dir_pspec+'N1/clkk_%s_%da_%db_%db_%da.npz'%(qe,i,i,i,i))['cls'].T[1]
        N1_arr.append(N1abab+N1abba-(N1xyxy+N1xyyx))

    navgfN1 = np.mean(N1_arr, axis=0)
    np.save(dir_pspec+'N1_avg.npy', navgfN1)
    
    if curl:
        N1ww_arr = []
        for i in range(1,nsims+1): 
            N1xyxy=np.load(dir_pspec+'N1/clww_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i,i+1))['cls'].T[1]
            N1xyyx=np.load(dir_pspec+'N1/clww_%s_%da_%da_%da_%da.npz'%(qe,i,i+1,i+1,i))['cls'].T[1]
            N1abab=np.load(dir_pspec+'N1/clww_%s_%da_%db_%da_%db.npz'%(qe,i,i,i,i))['cls'].T[1]
            N1abba=np.load(dir_pspec+'N1/clww_%s_%da_%db_%db_%da.npz'%(qe,i,i,i,i))['cls'].T[1]
            N1ww_arr.append(N1abab+N1abba-(N1xyxy+N1xyyx))

        navgfN1ww = np.mean(N1ww_arr, axis=0)
        np.save(dir_pspec+'N1ww_avg.npy', navgfN1ww)
