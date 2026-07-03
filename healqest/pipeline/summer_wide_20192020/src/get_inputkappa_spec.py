import numpy as np
import healpy as hp

clkk_in_arr = []
for i in range(1,251):
    klm_in = hp.read_alm("/data/gpfs/projects/punim1922/summerfield_lensing/healqest/lensrec/030425_summer-c/sqe/kappa_in/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_kappa1_summer-c_seed"+str(i)+".alm")
    clkk_in = hp.alm2cl(klm_in)
    clkk_in_arr.append(clkk_in) 
cl_in_avg = np.mean(clkk_in_arr, axis=0)
np.save("/data/gpfs/projects/punim1922/summerfield_lensing/healqest/lensrec/030425_summer-c/sqe/kappa_in/cl_in_avg_summer-c.npy", cl_in_avg)
