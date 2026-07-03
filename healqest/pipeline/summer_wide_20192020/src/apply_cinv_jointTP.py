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
file_out     = config['cinv']['file_out_jointTP']
file_cambcls = config['cls']['file_lcmb'] 
file_bconfig = config['base']['config'] 
runname      = config['base']['runname']
notch        = config['cinv']['notch']
suffix       = ''

if noinpaint:
    file_alm = config['cinv']['file_alm_noinpaint']
    suffix   ='_noinpaint'

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

print('Loading camb Cls')
print('- %s'%file_cambcls)
cl_len = utils.load_cambcls(file_cambcls,lmax=lmax,dict=True)

# Load tf
print('Loading transfer function')
#tfbl   = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,mode='auto')['2d'][150]
tfblT   = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d']['150T']
tfblP   = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d']['150P']
#psfilt = np.load('/lcrc/project/SPT3G/users/ac.yomori/scratch/ptsrc_filt.npz')#,T=Txx/Tdd,E=Exx/Edd,B=Bxx/Bdd)
#tfblT *= base_utils.reduce_lmax(psfilt['B'],lmax=lmax).astype(np.float64)
#tfblP *= base_utils.reduce_lmax(psfilt['B'],lmax=lmax).astype(np.float64)

# alm-space mask
print('Setting almspace masking')
print('-lmin: %d'%lmin )
print('-mmin: %d'%mmin )
ell,emm=hp.Alm.getlm(lmax)
ww=np.ones_like(tfblT,dtype=np.complex_)
ww[emm<mmin]=0
ww[ell<lmin]=0

bl1dT   = (hp.alm2cl(tfblT*np.ones_like(tfblT,dtype=np.complex_)))**0.5#*np.nan
bl1dT[:220] = 0

bl1dP   = (hp.alm2cl(tfblP*np.ones_like(tfblP,dtype=np.complex_)))**0.5#*np.nan
bl1dP[:220] = 0

# Converting from npz format to healpix alm format
print('Loading alm file')
print(' - %s'%file_alm.format(seed=seedR))
almin = np.load(file_alm.format(seed=seedR))
tlm   = base_utils.reduce_lmax(almin['almT'],lmax=lmax)
elm   = base_utils.reduce_lmax(almin['almE'],lmax=lmax)
blm   = base_utils.reduce_lmax(almin['almB'],lmax=lmax)
hp.write_alm(dir_tmp+'input_%d'%(seedR),[tlm*ww,elm*ww,blm*ww], overwrite=True )

# Simulation dictionary
sim_dict = {}
sim_dict['nside']       = nside
sim_dict['ivf_lrange']  = [lmin,lmax]
sim_dict['dir_output']  = dir_tmp+'/output/'
sim_dict['dir_cinvTP']   = dir_tmp+'/TP/'
sim_dict['file_mask']   = file_mask
sim_dict['file_signal'] = dir_tmp+'input_%d'%(seedR)
sim_dict['eps_min']     = eps_min
sims                    = maps.maps(sim_dict)

print('Loading mask')
print(' - %s'%file_mask)
mask   = hp.read_map(file_mask)
ninv_t = mask*( (180.*60./np.pi)**2 * pixarea_sqrad / nlev_t**2 )
ninv_p = mask*( (180.*60./np.pi)**2 * pixarea_sqrad / nlev_p**2 )

# Load noise+foreground stack the raw files are TF decnovolved since they go through ILC
print('Loading noise+fg stack')
print(' - %s'%file_noisefg)
nfg = np.load(file_noisefg)
nltt2d = base_utils.reduce_lmax(nfg['almTT'],lmax=lmax)/nfg['nsims']*nfg['nrm']*tfblT**2
nlee2d = base_utils.reduce_lmax(nfg['almEE'],lmax=lmax)/nfg['nsims']*nfg['nrm']*tfblP**2
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
nl2dt = ((nltt2d.astype(np.complex_) - (nlev_t*np.pi/180./60.)**2))/scal_t
nl2dt[nl2dt < 0]=1e-10
nl2dp = ((nlee2d.astype(np.complex_) - (nlev_p*np.pi/180./60.)**2))/scal_p
nl2dp[nl2dp < 0]=1e-10


dict_nl2d = {'tt': nl2dt,
             'ee': nl2dp,
             'bb': nl2dp}

cinv_tp = cinv.cinv_tp(dir_tmp+"/TP/", lmax, nside, cl_len, np.ones_like(bl1dT), [ninv_t, ninv_p],
                       eps_min = eps_min,
                       nl      = dict_nl2d,
                       tf2d    = tfblT*ww, 
                       tf2d_eb = tfblP*ww,
                       #cmb2d   = None,
                      )

'''
cinv_t  = cinv.cinv_t(dir_tmp+"/T/", lmax, nside, cl_len, np.ones_like(bl1dT), [ninv_t],
                       eps_min = eps_min,
                       nl      = dict_nl2d,
                       tf2d    = tfblT*ww,
                       #cmb2d   = None,
                      )

cinv_p  = cinv.cinv_p(dir_tmp+"/P/", lmax, nside, cl_len, np.ones_like(bl1dP), [ninv_p],
                       eps_min = eps_min,
                       nl      = dict_nl2d,
                       tf2d    = tfblP*ww,
                       #cmb2d   = None,
                      )

'''
print('Apply a lrange cut: [%d<ell<%d]'%(lmin,lmax) )
lfilt  = np.ones(sim_dict['ivf_lrange'][1] + 1, dtype=float) * (np.arange(sim_dict['ivf_lrange'][1] + 1) >= sim_dict['ivf_lrange'][0])

print('Running C^-1...')
jivfs  = cinv.library_cinv_jointTP(dir_tmp + 'outputs', sims, cinv_tp, cl_len, lfilt = lfilt)
ivf_t,ivf_e,ivf_b  = jivfs.get_sim_teblmivf(seed)

print('Saving output file')
print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix))
np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix), tlm=ivf_t, elm=ivf_e, blm=ivf_b)

'''
print('Running C^-1...')
#jivfs  = cinv.library_cinv_jointTP(dir_tmp + 'outputs', sims, cinv_tp, cl_len, lfilt = lfilt)
#ivf_t,ivf_e,ivf_b  = jivfs.get_sim_teblmivf(seed)
ivfs   = cinv.library_cinv_sepTP(dir_tmp+ 'outputs', sims, cinv_t, cinv_p, cl_len, lfilt = lfilt)
ivf_t       = ivfs.get_sim_tlmivf(seed)
#np.savez('cinv_t_%.2f_%.2f_newscaling.npz'%(Tpar[0],Tpar[1]),ivf_t=ivf_t)
ivf_e,ivf_b = ivfs.get_sim_eblmivf(seed)
#np.savez('cinv_p_%.2f_%.2f_newscaling.npz'%(Ppar[0],Ppar[1]),ivf_e=ivf_e,ivf_b=ivf_b)

print('Saving output file')
print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix))
np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname,suffix=suffix), tlm=ivf_t, elm=ivf_e, blm=ivf_b)
'''
#print('Cleaning intermediate files')
#os.remove(dir_tmp+'input_%d'%(seedR))
