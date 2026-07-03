import os,sys,argparse,yaml,shutil,datetime
import healpy as hp
import numpy as np
import logging as lg
from pathlib import Path
sys.path.insert(0,'/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/utils/')
import utils as base_utils
sys.path.insert(0,'/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/')
sys.path.insert(0,'/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/cinv/')
import maps
from cinv import cinv_hp as cinv
import healqest_utils as utils
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='main yaml')
parser.add_argument('seed'     , default=1   , type=int, help='seed')
parser.add_argument('cmbset'   , default=1   , type=int, help='cmbset')
parser.add_argument('--nops'   , default=False, dest='nops' , action='store_true')
args = parser.parse_args()

file_yaml    = args.file_yaml
seed         = args.seed
cmbset       = args.cmbset
nops         = args.nops

config       = yaml.safe_load(open(file_yaml))
nside        = config['cinv']['nside']
lmax         = config['cinv']['lmax']
lmin         = config['cinv']['lmin']
mmin         = config['cinv']['mmin']
eps_min      = config['cinv']['eps_min']
dir_tmp      = config['cinv']['dir_tmp'].format(seed=seed,cmbset=cmbset)
file_mask    = config['cinv']['file_mask']
file_alm     = config['cinv']['file_alm']
file_noisefg = config['cinv']['file_noisefg']
file_out     = config['cinv']['file_out_sepTP']
file_cambcls = config['cls']['file_lcmb'] 
file_bconfig = config['base']['config'] 
runname      = config['base']['runname']
notch        = config['cinv']['notch']
nlev_t       = config['cinv']['nlev_t']
nlev_p       = config['cinv']['nlev_p']
scal_res_t   = config['cinv']['scal_res_t']
scal_res_p   = config['cinv']['scal_res_p']
file_ninv_t  = config['cinv']['ninv_t']
file_ninv_p  = config['cinv']['ninv_p']
ninv_fac_t   = config['cinv']['ninv_fac_t']
ninv_fac_p   = config['cinv']['ninv_fac_p']

file_pmask_t  = config['cinv']['pmask_t']
file_pmask_p  = config['cinv']['pmask_p']
eps_t  = config['cinv']['eps_t']
eps_p  = config['cinv']['eps_p']


suffix       = ''

print('Removing dir_tmp:',dir_tmp)
shutil.rmtree(dir_tmp, ignore_errors=True)

print('nside--',nside)
pixarea_sqrad = hp.nside2pixarea(nside)

print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix))
p = pathlib.Path(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix))
Path(p.parent).mkdir(parents=True, exist_ok=True)
p = pathlib.Path(dir_tmp)
Path(p).mkdir(parents=True, exist_ok=True)

print('--------------------------------------------------------------------------')
print('Loading base config file')
print('- %s'%file_bconfig)
base_config   = yaml.safe_load(open(file_bconfig))

print("Setting scratch outputs")
print("- %s"%dir_tmp+'/outputs/')
shutil.rmtree(dir_tmp+'/outputs/', ignore_errors=True)

# Conversion between raw sims and lensing sim
# cmbset1: 1-500
# cmbset2: 1001-1500
if cmbset==2  : seedR = seed+1000
elif cmbset==3: seedR = seed+7000
elif cmbset==5: seedR = seed+7500
elif cmbset==4: seedR = seed+8000
else        : seedR = seed 
print(' - %s'%file_alm.format(seed=seedR))
#sys.exit()

print('Loading camb Cls')
print('- %s'%file_cambcls)
cl_len = utils.load_cambcls(file_cambcls,lmax=lmax,dict=True)

print(' - %s'%file_mask)
mask   = hp.read_map(file_mask)
if nside!=2048:
    mask=hp.ud_grade(mask,nside)

d = '/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/midell/sims/foregrounds/mcmc/emulator/emmtest2_map2alm/'
mcorr = np.load(d+'mcorr_map2alm.npz')


# Load tf
print('Loading transfer function')
#tfbl   = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,mode='auto')['2d'][150]
tfblT_2d = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d']['150T']
tfblP_2d = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d']['150P']

tfblT_2d_nobeam = base_utils.load_tf(base_config,lmax=lmax,include_beam=False,verbose=True,freq=150)['2d']['150T']
tfblP_2d_nobeam = base_utils.load_tf(base_config,lmax=lmax,include_beam=False,verbose=True,freq=150)['2d']['150P']

tfblT_2d_nobeam[tfblT_2d_nobeam<0.05]=0.05
tfblP_2d_nobeam[tfblP_2d_nobeam<0.05]=0.05

#tfblT_2d[tfblT_2d<0.05]=0.05
#tfblP_2d[tfblP_2d<0.05]=0.05



tfblT_1d = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf_withbeam.npz')['tt'][:lmax+1];tfblT_1d[np.isnan(tfblT_1d)]=1e-30;tfblT_1d[tfblT_1d<0.01]=1e20
tfblP_1d = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf_withbeam.npz')['ee'][:lmax+1];tfblP_1d[np.isnan(tfblP_1d)]=1e-30;tfblP_1d[tfblP_1d<0.01]=1e20
tfblT_1d[:lmin]=1e20
tfblP_1d[:lmin]=1e20


'''
tfblT_1d = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf_withbeam.npz')['tt'][:lmax+1];tfblT_1d[np.isnan(tfblT_1d)]=1e-30;tfblT_1d[tfblT_1d<0.1]=0.1
tfblP_1d = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf_withbeam.npz')['ee'][:lmax+1];tfblP_1d[np.isnan(tfblP_1d)]=1e-30;tfblP_1d[tfblP_1d<0.1]=0.1

#import pdb;pdb.set_trace()
tfblT_2d[tfblT_2d<0.1]=0.1
tfblP_2d[tfblP_2d<0.1]=0.1
'''

lmaxs=16000
ell,emm=hp.Alm.getlm(lmaxs)
ws=np.ones(hp.Alm.getsize(lmaxs),dtype=np.complex128)
ws[emm<220]=0

_tfblT_2d = base_utils.load_tf(base_config,lmax=lmaxs,include_beam=True,verbose=True,freq=150)['2d']['150T']
_tfblP_2d = base_utils.load_tf(base_config,lmax=lmaxs,include_beam=True,verbose=True,freq=150)['2d']['150P']

nfg = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/almstack/ilc/noisefg2_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_0001_0500_withsignflipnoise_2dilc_crosstf_full_021125.npz')
#cltt_nfg = hp.alm2cl( (nfg['almTT']/nfg['nsims'])**0.5*ws*_tfblT_2d)[:16001]/mcorr['TT'][:16001]*nfg['nrm']
#clee_nfg = hp.alm2cl( (nfg['almEE']/nfg['nsims'])**0.5*ws*_tfblP_2d)[:16001]/mcorr['TT'][:16001]*nfg['nrm']
#clbb_nfg = hp.alm2cl( (nfg['almBB']/nfg['nsims'])**0.5*ws*_tfblP_2d)[:16001]/mcorr['TT'][:16001]*nfg['nrm']
#np.save('cltt_nfg.npy',cltt_nfg)
#np.save('clee_nfg.npy',clee_nfg)
#sys.exit()
cltt_nfg = hp.alm2cl( (nfg['almTT']/nfg['nsims'])**0.5*ws*_tfblT_2d)[:lmax+1]/mcorr['TT'][:lmax+1]*nfg['nrm']
clee_nfg = hp.alm2cl( (nfg['almEE']/nfg['nsims'])**0.5*ws*_tfblP_2d)[:lmax+1]/mcorr['TT'][:lmax+1]*nfg['nrm']
clbb_nfg = hp.alm2cl( (nfg['almBB']/nfg['nsims'])**0.5*ws*_tfblP_2d)[:lmax+1]/mcorr['TT'][:lmax+1]*nfg['nrm']


# alm-space mask
print('Setting almspace masking')
print('-lmin: %d'%lmin )
print('-mmin: %d'%mmin )

ell,emm=hp.Alm.getlm(lmax)
ww=np.ones_like(tfblT_2d,dtype=np.complex_)
ww[emm<mmin]=0
ww[ell<lmin]=0


#cltt_nfg[:lmin]=0
#alm = hp.synalm(cltt_nfg[:lmax+1]*0.08)
beam = base_utils.load_bl(base_config,lmax=lmax)

alm = hp.synalm((np.pi/180./60.*11.0)**2* np.ones(lmax+1)*beam['150T']**16 )
_n  = hp.alm2map(alm,nside)
nlm = hp.map2alm(_n*mask,lmax=lmax)
print(_n)
#np.save('clttnfg.npy',cltt_nfg)
#sys.exit()

# Converting from npz format to healpix alm format
print('Loading alm file')
print(' - %s'%file_alm.format(seed=seedR))
almin = np.load(file_alm.format(seed=seedR))
#almin = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/sims/3.3.1.3.1/mockdata/healpix/ilc/cmb/data/inpainted//data_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_seed0001_withsignflipnoise_2dilc_crosstf_full_012425_scalfg1.000_nullgaussconstinp_teb.npz')
#from scipy.ndimage import gaussian_filter1d as gf1
#xx=np.ones(4096)
#xx[:160]=0.2
#y=gf1(xx,20)
#wm=np.interp(emm,xx,y)

tlm   = base_utils.reduce_lmax(almin['almT'],lmax=lmax)#*wm#*tfblT_2d
elm   = base_utils.reduce_lmax(almin['almE'],lmax=lmax)#*wm#*tfblP_2d
blm   = base_utils.reduce_lmax(almin['almB'],lmax=lmax)#*wm#*tfblP_2d

_aa = hp.alm2map([tlm,elm,blm],nside)
_aa[0][mask==0]=0
_aa[1][mask==0]=0
_aa[2][mask==0]=0
tlm,elm,blm = hp.map2alm(_aa,lmax=lmax,use_pixel_weights=True)
'''
if seed==0:
    hp.write_alm(dir_tmp+'input_%d'%(seedR),[tfblT_2d_nobeam * (tlm+nlm)*ww,
                                             tfblP_2d_nobeam * elm*ww,
                                             tfblP_2d_nobeam * blm*ww], overwrite=True )
else:
    hp.write_alm(dir_tmp+'input_%d'%(seedR),[tfblT_2d_nobeam * (tlm)*ww,
                                             tfblP_2d_nobeam * elm*ww,
                                             tfblP_2d_nobeam * blm*ww], overwrite=True )
'''

if seed==0:
    hp.write_alm(dir_tmp+'input_%d'%(seedR),[ ww*(tlm+nlm),
                                              ww*elm,
                                              ww*blm], overwrite=True )
else:
    hp.write_alm(dir_tmp+'input_%d'%(seedR),[ ww*(tlm),
                                              ww*elm,
                                              ww*blm], overwrite=True )


# alm here is convolved with 150 GHz transfer function (without beam)
# This means that what comed out is a TF convolevd WF map.

# Simulation dictionary
sim_dict = {}
sim_dict['nside']       = nside
sim_dict['ivf_lrange']  = [lmin,lmax]
sim_dict['dir_output']  = dir_tmp+'/output/'
sim_dict['dir_cinvT']   = dir_tmp+'/T/'
sim_dict['dir_cinvP']   = dir_tmp+'/P/'
sim_dict['file_mask']   = file_mask
sim_dict['file_signal'] = dir_tmp+'input_%d'%(seedR)
sim_dict['file_noise']  = None #dir_tmp+'input_%d'%(seedR)
sim_dict['eps_min']     = eps_min
sim_dict['tf2d']        = None
sims                    = maps.maps(sim_dict)

print('Loading mask')
print(' - %s'%file_mask)

############## crazyninv *2
binmask = np.copy(mask)
binmask[binmask>0]=1
ninv_t = binmask*( pixarea_sqrad/(np.pi/180./60.*nlev_t)**2 )*ninv_fac_t# uK^-2.pix^-1
ninv_p = binmask*( pixarea_sqrad/(np.pi/180./60.*nlev_p)**2 )*ninv_fac_p# uK^-2.pix^-1  
# NOTE: just because you use the exact same value for this it doesnt mean that the 

'''
if file_ninv_t is not None:
    print('file_ninv_t is not None')
    ninv_t = hp.read_map(file_ninv_t)*1.1 #mask*( pixarea_sqrad/(np.pi/180./60.*nlev_t)**2 )# uK^-2.pix^-1
if file_ninv_p is not None:
    print('file_ninv_p is not None')
    ninv_p = hp.read_map(file_ninv_p)*1.03 #mask*( pixarea_sqrad/(np.pi/180./60.*nlev_p)**2 )# uK^-2.pix^-1  
'''

if nops:
    pass
else:
    #mt=hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/masks/mask4096_inverse_1500d_eetelensing-1920_mapmaking_mask_list_fluxcut6mJy_clusterSNcut10_badclusremoved.fits',partial=True)
    #mt=hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/masks/mask4096_inverse_1500d_eetelensing-1920_mapmaking_mask_list_fluxcut6mJy_clusterSNcut10_badclusremoved_optimized_plus0.25arcmin.fits',partial=True)
    print('Loading pmask_t: ',file_pmask_t)
    mt=hp.read_map(file_pmask_t,partial=True)
    mt[mt==hp.UNSEEN]=0
    mt=1-mt
    mt=hp.ud_grade(mt,nside_out=nside)
    mt[mt<1]=0

    #mp=hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/masks/mask4096_inverse_1500d_eetelensing-1920_mapmaking_mask_list_fluxcut6mJy_clusterSNcut10_poldetected.fits',partial=True)
    mp=hp.read_map(file_pmask_p,partial=True)
    mp[mp==hp.UNSEEN]=0
    mp=1-mp
    mp=hp.ud_grade(mp,nside_out=nside)
    mp[mp<1]=0
    
    ninv_t *= mt
    ninv_p *= mp
    
    #np.save('ninv_t.npy',ninv_t)
    #sys.exit()
print('nlev_t--',nlev_t)
print('nlev_p--',nlev_p)

# Treat the residual part in harmonic space.
# This is known not to be perfect so some scaling 
# might be needed (this is true even in an 
# idealized set up).
#scal_res_t = 1.00
#scal_res_p = 1.00

_tt=(cltt_nfg - (np.pi/180./60.*nlev_t)**2)*scal_res_t
_ee=(clee_nfg - (np.pi/180./60.*nlev_p)**2)*scal_res_p
_bb=(clbb_nfg - (np.pi/180./60.*nlev_p)**2)*scal_res_p
_tt[_tt<0]=0; _tt[np.isnan(_tt)]=0
_ee[_ee<0]=0; _ee[np.isnan(_ee)]=0
_bb[_bb<0]=0; _bb[np.isnan(_bb)]=0


nl_res = {'tt': _tt ,
          'ee': _ee ,
          'bb': _bb }


cinv_t  = cinv.cinv_t(dir_tmp+"/T/", lmax, nside, cl_len, nl_res, [ninv_t], tfblT_1d, ww*tfblT_2d,
                       eps_min = eps_t,#eps_min,5e-7
                      
                      )


cinv_p  = cinv.cinv_p(dir_tmp+"/P/", lmax, nside, cl_len, nl_res, [ninv_p], tfblP_1d, ww*tfblP_2d,
                       eps_min = eps_p,#eps_min,5e-6
                      
                      )

#import pdb;pdb.set_trace()

print('Apply a lrange cut: [%d<ell<%d]'%(lmin,lmax) )
lfilt = np.ones(lmax + 1)+0.0
lfilt[:lmin]=0
lfilt[lmax:]=0

print('               ')
print('-------------------------------- Running C^-1 --------------------------------')
#sys.exit()
ivfs   = cinv.library_cinv_sepTP(dir_tmp+ 'outputs', sims, cinv_t, cinv_p, cl_len, lfilt = lfilt)
ivf_t       = ivfs.get_sim_tlmivf(seed)

ivf_e,ivf_b = ivfs.get_sim_eblmivf(seed)
#ivf_e,ivf_b=ivf_t,ivf_t

print('Saving output file')
if nops:
    print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix)[:-4]+'_nops.npz')
    np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix)[:-4]+'_nops.npz', tlm=ivf_t, elm=ivf_e, blm=ivf_b)
else:
    print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix))
    np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix), tlm=ivf_t, elm=ivf_e, blm=ivf_b)

# V7: 7.42 - 7.42 - 1.00
# V8: 5.00 - 5.00 - 1.00 matched in CInv spectra but N0 != 1/resp, sim mean is bad 
# V9: 5.00 - 3.00 - 1.00
# V10: 7.42 - 7.42 - 0.9
shutil.rmtree(dir_tmp, ignore_errors=True)
