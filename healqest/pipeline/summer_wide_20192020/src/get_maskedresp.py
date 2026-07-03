import os,sys 
import healpy as hp
import numpy as np
import subprocess
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='yaml file for mask loc')
parser.add_argument('dir_p'    , default=1   , type=str, help='dir_p')
parser.add_argument('dir_out'  , default=1   , type=str, help='dir_out')
parser.add_argument('qe'       , default=1   , type=str, help='qe')
parser.add_argument('nsims'    , default=1   , type=int, help='nsims')
parser.add_argument('seed'     , default=1   , type=int, help='seed')
parser.add_argument('--usespice', default=False, dest='usespice', action='store_true')
parser.add_argument('--merge'   , default=False, dest='merge' , action='store_true')
args = parser.parse_args()

dir_p    = args.dir_p
dir_out  = args.dir_out
qe       = args.qe
nsims    = args.nsims
seed     = args.seed
usespice = args.usespice
merge    = args.merge

config  = yaml.safe_load(open(args.file_yaml))

os.environ['HEALPIX'] = config['base']['dir_healpix']
spice=config['base']['dir_spice']

# Initial response
resp0  = np.load(dir_p+'/respavg%s.npz'%(qe))['resp']
resp0_lmax = len(resp0) - 1
if resp0_lmax > config['lensrec']['Lmax']:
    resp0[config['lensrec']['Lmax']-resp0_lmax:]=1e30
resp0[resp0==0]=1e30
resp_dum = np.zeros(4101)+1e30
resp_dum[:resp0_lmax+1] = resp0
resp0 = resp_dum.copy()

if merge:
   c=0; cl1 = 0; cl2 = 0
   for i in tqdm(range(1,nsims+1) ):
      cl1+=np.load(dir_out+'/inin_%d.npy'%i)[:,1]
      cl2+=np.load(dir_out+'/inout_%d.npy'%i)[:,1]
      c+=1
   respps=cl2/cl1

   np.savez(dir_p+'/respavg%s_masked_spice.npz'%(qe),resp=respps*resp0,nsim=c)
   print("Saving: ",dir_p+'/respavg%s_masked_spice.npz'%(qe))

else:

   Path(dir_out).mkdir(parents=True, exist_ok=True)

   ##############################################################################

   if usespice:
         #dir_kap = "/sdf/group/kipac/users/wlwu_alt1/"
         dir_kap = os.path.expandvars("$LSCRATCH/")
         plmstack  = np.load(dir_p+'/glmstack%s_xx.npz'%(qe))

         # This file is squared already
         #'/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense_mb2.fits'
         file_mask2 = config['spiceresp']['bmask2']
         mask2   = hp.read_map(file_mask2)
         mask2[mask2==0]=np.inf

         inkapfname = config['resp']['input_kappa']
         ilm    = hp.read_alm( inkapfname.format( seed=seed ) )
         imap   = hp.alm2map(ilm,2048)
         hp.write_map(dir_kap+'inkappa_%d.fits'%seed,imap/mask2,dtype=np.float32,overwrite=True)

         olm    = np.load(dir_p+'/plm%s_%da_%da.npz'%(qe,seed,seed))['glm']
         mfi    = ((plmstack['gmfxx']-olm)/(plmstack['nsim']-1.0))
         omap   = hp.alm2map(hp.almxfl(olm-mfi,1/resp0),2048)
         hp.write_map(dir_kap+'outkappa_%d.fits'%seed,omap/mask2,dtype=np.float32,overwrite=True)

         #'/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/masks/mask2048_border_apod_mask_threshold0.1_allghz_dense_1500d_eetelensing-1920_mapmaking_mask_list_fluxcut6mJy_clusterSNcut10_dense_mb2_mp1.fits'
         file_mask = config['spiceresp']['fmask']


         subprocess.call([spice,'-mapfile'      , dir_kap+'inkappa_%d.fits'%seed,
                                 '-weightfile'  , file_mask,
                                 '-mapfile2'    , dir_kap+'inkappa_%d.fits'%seed,
                                 '-weightfile2' , file_mask,
                                 '-clfile'      , dir_out+'/inin_%d.txt'%seed,
                                 '-nlmax'       ,'4100',
                                 '-apodizesigma','20',
                                 '-thetamax'    ,'30',
                                 '-subav'       ,'YES',
                                 '-verbosity'   , 'NO',
                                 '-apodizetype' , '1'
                              ])

         subprocess.call([spice,'-mapfile'      , dir_kap+'inkappa_%d.fits'%seed,
                                 '-weightfile'  , file_mask,
                                 '-mapfile2'    , dir_kap+'outkappa_%d.fits'%seed,
                                 '-weightfile2' , file_mask,
                                 '-clfile'      , dir_out+'/inout_%d.txt'%seed,
                                 '-nlmax'       ,'4100',
                                 '-apodizesigma','20',
                                 '-thetamax'    ,'30',
                                 '-subav'       ,'YES',
                                 '-verbosity'   , 'NO',
                                 '-apodizetype' , '1'
                              ])
         os.remove(dir_kap+'inkappa_%d.fits'%seed)
         os.remove(dir_kap+'outkappa_%d.fits'%seed)
         ii=np.loadtxt(dir_out+'/inin_%d.txt'%seed,unpack=True)
         io=np.loadtxt(dir_out+'/inout_%d.txt'%seed,unpack=True)
         np.save(dir_out+'/inin_%d.npy'%seed,ii.T)
         np.save(dir_out+'/inout_%d.npy'%seed,io.T)
         os.remove(dir_out+'/inin_%d.txt'%seed)
         os.remove(dir_out+'/inout_%d.txt'%seed)

   else:
      plmstack  = np.load(dir_p+'/glmstack%s_xx.npz'%(qe))
      '''
      ilm    = hp.read_alm('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/sims/inputkappa/inkappa_seed%d.alm'%seed)

      olm    = np.load(dir_p+'/plm%s_%da_%da.npz'%(qe,seed,seed))['glm']
      mfi    = ((plmstack['gmfxx']-olm)/(plmstack['nsim']-1.0))
      olm    = hp.almxfl(olm-mfi,1/resp0)

      ii=hp.alm2map(ilm,2048)
      oo=hp.alm2map(olm,2048)

      ilm=hp.map2alm(ii/mask2**2*

      l      = np.arange(4101)
      clii   = hp.alm2cl(ilm,ilm)
      clio   = hp.alm2cl(ilm,olm)
      
      np.save(dir_out+'/inin_%d.npy'%seed,np.c_[l,clii])
      np.save(dir_out+'/inout_%d.npy'%seed,np.c_[l,clio])

      '''
