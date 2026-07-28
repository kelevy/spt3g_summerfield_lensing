import sys
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/build/")
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/mapspectra/python")
import os
from spt3g import core, maps
import numpy as np
import healpy as hp
import misc


########################################################################################################################################

freq = 150
nval_T = 12 #*np.sqrt(5)
nside = 2048
lmax = nside*3-1
seed = 251
el = np.arange(lmax)

nl_onedT = misc.get_nl(nval_T, el, beamval=0., make_2d = 0, mapparams = None)
nl_onedQ = misc.get_nl(np.sqrt(2) * nval_T, el, beamval=0., make_2d = 0, mapparams = None)
nl_onedU = misc.get_nl(np.sqrt(2) * nval_T, el, beamval=0., make_2d = 0, mapparams = None)


for i in range(201, seed+1):
    print(i)
    noise_tmap = hp.synfast(nl_onedT, nside=nside)
    #noise_qmap = hp.synfast(nl_onedQ, nside=nside)
    #noise_umap = hp.synfast(nl_onedU, nside=nside)

    hmaps = [noise_tmap] #, noise_qmap, noise_umap]

    hp.write_map('/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/maps/noise_maps_'+str(freq)+'ghz_'+str(nval_T)+'uKarcmin_seed'+str(i)+'.fits', hmaps, overwrite=True)


#for i in range(0, seed+1):
#    print(i)
#    hmaps = hp.read_map('/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/maps/noise_maps_'+str(freq)+'ghz_'+str(nval_T)+'uKarcmin_seed'+str(i)+'_nside'+str(nside)+'.fits', field = (0,1,2))
#    cls = hp.anafast(hmaps) 
#    np.savetxt('/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/spectra/cls_noise_'+str(freq)+'ghz_'+str(nval_T)+'uKarcmin_seed'+str(i)+'_nside'+str(nside)+'.dat', cls)


#cls_avg = 0
#for i in range(1, seed+1):
#    cls = np.loadtxt('/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/spectra/cls_noise_'+str(freq)+'ghz_'+str(nval_T)+'uKarcmin_seed'+str(i)+'_nside'+str(nside)+'.dat', unpack = True)
#    cls_avg += cls

#cls_avg /= seed
#ell = np.arange(len(cls_avg[0]))
#cls_avg = np.append([ell], cls_avg, axis=0)
#np.savetxt('/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/spectra/cls_noise_'+str(freq)+'ghz_'+str(nval_T)+'uKarcmin_avg'+str(seed)+'_nside'+str(nside)+'.dat', cls_avg)

