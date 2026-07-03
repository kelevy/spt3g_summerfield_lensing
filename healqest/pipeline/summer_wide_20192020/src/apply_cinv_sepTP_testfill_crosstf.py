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
parser.add_argument('--Tpar'      , nargs=2, type=float, help='Third set of three numbers')
parser.add_argument('--Ppar'      , nargs=2, type=float, help='Third set of three numbers')
parser.add_argument('--noinpaint', default=False, dest='noinpaint' , action='store_true')
args = parser.parse_args()

file_yaml    = args.file_yaml
seed         = args.seed
cmbset       = args.cmbset
Tpar         = args.Tpar
Ppar         = args.Ppar
noinpaint    = args.noinpaint

config       = yaml.safe_load(open(file_yaml))
nside        = config['cinv']['nside']
lmax         = config['cinv']['lmax']
lmin         = config['cinv']['lmin']
mmin         = config['cinv']['mmin']
nlev_t       = Tpar[0] #config['cinv']['nlev_t']
nlev_p       = Ppar[0] #config['cinv']['nlev_p']
eps_min      = config['cinv']['eps_min']
scal_t       = Tpar[1] #config['cinv']['scal_t']
scal_p       = Ppar[1] #config['cinv']['scal_p']
dir_tmp      = config['cinv']['dir_tmp'].format(seed=seed,cmbset=cmbset)
file_mask    = config['cinv']['file_mask']
file_alm     = config['cinv']['file_alm']
file_noisefg = config['cinv']['file_noisefg']
#file_noise   = None #config['cinv']['file_noise']
file_out     = config['cinv']['file_out_sepTP']
file_cambcls = config['cls']['file_lcmb'] 
file_bconfig = config['base']['config'] 
runname      = config['base']['runname']
notch        = config['cinv']['notch']
file_ninv_t  = config['cinv']['ninv_t']
file_ninv_p  = config['cinv']['ninv_p']

suffix       = ''

if noinpaint:
    file_alm = config['cinv']['file_alm_noinp']
    suffix   ='_noinpaint'
    file_out  = config['cinv']['file_out_sepTP_noinp']
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
#mask=hp.ud_grade(mask,4096)##################################################
d = '/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/midell/sims/foregrounds/mcmc/emulator/emmtest2_map2alm/'
mcorr = np.load(d+'mcorr_map2alm.npz')


# Load tf
print('Loading transfer function')
#tfbl   = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,mode='auto')['2d'][150]
tfblT_2d = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d']['150T']
tfblP_2d = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d']['150P']
tfblT_1d = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf.npz')['tt'][:lmax+1];tfblT_1d[np.isnan(tfblT_1d)]=1e-30;tfblT_1d[tfblT_1d<0.001]=1e-20
tfblP_1d = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/tf1d_150ghz_20sims_crosstf.npz')['ee'][:lmax+1];tfblP_1d[np.isnan(tfblP_1d)]=1e-30;tfblP_1d[tfblP_1d<0.001]=1e-20

#import pdb;pdb.set_trace()
tfblT_2d[tfblT_2d<0.001]=1e-20
tfblP_2d[tfblP_2d<0.001]=1e-20


lmaxs=16000
ell,emm=hp.Alm.getlm(lmaxs)
ws=np.ones(hp.Alm.getsize(lmaxs),dtype=np.complex128)
ws[emm<220]=0

_tfblT_2d = base_utils.load_tf(base_config,lmax=lmaxs,include_beam=True,verbose=True,freq=150)['2d']['150T']
_tfblP_2d = base_utils.load_tf(base_config,lmax=lmaxs,include_beam=True,verbose=True,freq=150)['2d']['150P']

nfg = np.load('/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/almstack/ilc/noisefg2_tqu1_agora0.7_datamatched_mcmccal_0707231033_Coadd_allfields_cmbmv_0001_0500_withsignflipnoise_2dilc_crosstf_full_012425.npz')
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

#_aa = hp.alm2map([tlm,elm,blm],2048)
#tlm,elm,blm = hp.map2alm(_aa*mask,lmax=4096,use_pixel_weights=True)

if seed==0:
    hp.write_alm(dir_tmp+'input_%d'%(seedR),[(tlm+nlm)*ww,elm*ww,blm*ww], overwrite=True )
else:
    hp.write_alm(dir_tmp+'input_%d'%(seedR),[(tlm)*ww,elm*ww,blm*ww], overwrite=True )
# alm here is convolved with 150 GHz transfer function (with beam)

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

#noise=np.load('/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/midell/inpaint/nls/nl_cmbmv_150beamconv_avg.npz')
#nlev_t = noise['nlev_t']
#nlev_p = noise['nlev_p']
#whitett = noise['whitett'] #(nlev_t*np.pi/180./60.)**2
#whiteee = noise['whiteee'] #(nlev_p*np.pi/180./60.)**2
#nltt = noise['nltt']
#nlee = noise['nlee']


nlev_t=7.0 # This should be considered the flat part of noise+foreground
nlev_p=5.0 # This should be considered the flat part of noise+foreground

print('Loading mask')
print(' - %s'%file_mask)
#mask   = hp.read_map(file_mask)

ninv_t_7 = mask*( pixarea_sqrad/(np.pi/180./60.*7)**2 )# uK^-2.pix^-1
ninv_p_5 = mask*( pixarea_sqrad/(np.pi/180./60.*5)**2 )# uK^-2.pix^-1  

ninv_t = mask*( pixarea_sqrad/(np.pi/180./60.*nlev_t)**2 )# uK^-2.pix^-1
ninv_p = mask*( pixarea_sqrad/(np.pi/180./60.*nlev_p)**2 )# uK^-2.pix^-1  

#ninv_t = hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/scratch/ninvt_7mukscaled_psmasked.fits')*np.max(ninv_t)/np.max(ninv_t_7) #mask*( pixarea_sqrad/(np.pi/180./60.*nlev_t)**2 )# uK^-2.pix^-1
#ninv_p = hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/scratch/ninvp_5mukscaled_psmasked.fits')*np.max(ninv_p)/np.max(ninv_p_5) #mask*( pixarea_sqrad/(np.pi/180./60.*nlev_p)**2 )# uK^-2.pix^-1  
if ninv_t is not None:
    ninv_t = hp.read_map(file_ninv_t)*np.max(ninv_t)/np.max(ninv_t_7) #mask*( pixarea_sqrad/(np.pi/180./60.*nlev_t)**2 )# uK^-2.pix^-1
if ninv_p is not None:
    ninv_p = hp.read_map(file_ninv_p)*np.max(ninv_p)/np.max(ninv_p_5) #mask*( pixarea_sqrad/(np.pi/180./60.*nlev_p)**2 )# uK^-2.pix^-1  

#mm=hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/scratch/mask2048_binary_inner_smoothmask_apod_v2.fits')
#ninv_t*=mm
#ninv_p*=mm


print('nlev_t--',nlev_t)
print('nlev_p--',nlev_p)



# Load noise+foreground stack the raw files are TF decnovolved since they go through ILC
#print('Loading noise+fg stack')
#print(' - %s'%file_noisefg)
#nfg = np.load(file_noisefg)
#nltt2d = base_utils.reduce_lmax(nfg['almTT'],lmax=lmax)/nfg['nsims']*nfg['nrm']*tfblT**2
#nlee2d = base_utils.reduce_lmax(nfg['almEE'],lmax=lmax)/nfg['nsims']*nfg['nrm']*tfblP**2
'''
if notch is not None:
    print("Reading notch file and setting noise in hole to infinity")
    notchalm = hp.read_alm(notch)
    notchalm = base_utils.reduce_lmax(notchalm,lmax=lmax)
    notchalm[notchalm==0] = 1e30
    nltt2d*=notchalm
    nlee2d*=notchalm
''' 

# nlm^2
#nl2dt = ((nltt2d.astype(np.complex_) - (nlev_t*np.pi/180./60.)**2))/scal_t
#nl2dt[nl2dt < 0]=1e-10
#nl2dp = ((nlee2d.astype(np.complex_) - (nlev_p*np.pi/180./60.)**2))/scal_p
#nl2dp[nl2dp < 0]=1e-10


#dict_nl2d = {'tt': nl2dt,
#             'ee': nl2dp,
#             'bb': nl2dp}
'''
cinv_tp = cinv.cinv_tp(dir_tmp+"/TP/", lmax, nside, cl_len, np.ones_like(bl1d), [ninv_t, ninv_p],
                       eps_min = eps_min,
                       nl      = dict_nl2d,
                       tf2d    = tfbl*ww,
                       #cmb2d   = None,
                      )
''';

# Treat the residual part in harmonic space.
# This is known not to be perfect so some scaling 
# might be needed (this is true even in an 
# idealized set up).
scal_res_t = 1.
scal_res_p = 1.

_tt=(cltt_nfg - (np.pi/180./60.*nlev_t)**2)*scal_res_t
_ee=(clee_nfg - (np.pi/180./60.*nlev_p)**2)*scal_res_p
_bb=(clbb_nfg - (np.pi/180./60.*nlev_p)**2)*scal_res_p
_tt[_tt<0]=0; _tt[np.isnan(_tt)]=0
_ee[_ee<0]=0; _ee[np.isnan(_ee)]=0
_bb[_bb<0]=0; _bb[np.isnan(_bb)]=0

#assert all(v > 0 for v in _tt), "Not all nl_res['tt'] are positive"
#assert all(v > 0 for v in _ee), "Not all nl_res['ee'] are positive"
#assert all(v > 0 for v in _bb), "Not all nl_res['bb'] are positive"

nl_res = {'tt': _tt ,
          'ee': _ee ,
          'bb': _bb }

#import pdb;pdb.set_trace()

cinv_t  = cinv.cinv_t(dir_tmp+"/T/", lmax, nside, cl_len, nl_res, [ninv_t], tfblT_1d, tfblT_2d*ww,
                       eps_min = 1e-7,#eps_min,
                      
                      )


cinv_p  = cinv.cinv_p(dir_tmp+"/P/", lmax, nside, cl_len, nl_res, [ninv_p], tfblT_1d, tfblP_2d*ww,
                       eps_min = 1e-7,#eps_min,
                      
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

ivf_e,ivf_b = ivfs.get_sim_eblmivf(seed)
#np.savez('cinv_p_%d.npz'%seed,ivf_e=ivf_e,ivf_b=ivf_b)
#sys.exit()


ivf_t       = ivfs.get_sim_tlmivf(seed)
#np.savez('cinv_t_%d_crosstf.npz'%seed,tlm=ivf_t)
#sys.exit()
#ivf_e=ivf_t
#ivf_b=ivf_t


print('Saving output file')
print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix))
#np.savez('cinv_tp_%d.npz'%seed, tlm=ivf_t, elm=ivf_e, blm=ivf_b)
np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix), tlm=ivf_t, elm=ivf_e, blm=ivf_b)
#print('Cleaning intermediate files')
#os.remove(dir_tmp+'input_%d'%(seedR))
