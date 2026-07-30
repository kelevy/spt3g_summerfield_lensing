import os,sys,argparse,yaml,shutil,datetime,git
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

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='main yaml')
parser.add_argument('seed'     , default=1   , type=int, help='seed')
parser.add_argument('cmbset'   , default=1   , type=int, help='cmbset')
parser.add_argument('--sepTP'  , default=False, dest='sepTP' , action='store_true')
parser.add_argument('--log'   , default=False, dest='savelog' , action='store_true')
args = parser.parse_args()

file_yaml    = args.file_yaml
seed         = args.seed
cmbset       = args.cmbset
sepTP        = args.sepTP
savelog      = args.savelog


config       = yaml.safe_load(open(file_yaml))
nside        = config['cinv']['nside']
lmax         = config['cinv']['lmax']
lmin         = config['cinv']['lmin']
mmin         = config['cinv']['mmin']
nlev_t       = config['cinv']['nlev_t']
nlev_p       = config['cinv']['nlev_p']
eps_min      = config['cinv']['eps_min']
scal_t       = config['cinv']['scal_t']
scal_p       = config['cinv']['scal_p']
dir_tmp      = config['cinv']['dir_tmp']
file_mask    = config['cinv']['file_mask']
file_alm     = config['cinv']['file_alm']
file_noisefg = config['cinv']['file_noisefg']
file_cambcls = config['cls']['file_lcmb'] 
file_bconfig = config['base']['config'] 


print('Loading base config file')
print('- %s'%file_bconfig)
base_config   = yaml.safe_load(open(file_bconfig))
runname       = base_config['base']['runname']

# Save githash
repo    = git.Repo('./',search_parent_directories=True)
sha     = repo.head.object.hexsha

# Setup logger if we want
base_utils.setup_logger(savelog,file_log=f'logs/{runname}/log_cinv_jtp_{seed}.txt')
lg.warning(f'githash: {sha}')

pixarea_sqrad = hp.nside2pixarea(nside)

print("Setting scratch outputs")
print("- %s"%dir_tmp+'/outputs/')
shutil.rmtree(dir_tmp+'/outputs/', ignore_errors=True)

# Conversion between raw sims and lensing sim
# cmbset1: 1-500
# cmbset2: 1001-1500
if cmbset==2: seedR = seed+1000
else        : seedR = seed 

if cmbset==1 and seed>1000 and seed<2000:
    sys.exit("Need to set cmbset=2 and seed<1000")

print('Loading camb Cls')
print('- %s'%file_cambcls)
cl_len = utils.load_cambcls(file_cambcls,lmax=lmax,dict=True)

# Load tf
print('Loading transfer function')
tfbl   = base_utils.load_tf(base_config,lmax=lmax,include_beam=True,verbose=True,freq=150)['2d'][150]

# alm-space mask
print('Setting almspace masking')
print(f'lmin: {lmin}')
print(f'mmin: {mmin}')
ell,emm=hp.Alm.getlm(lmax)
ww=np.ones_like(tfbl,dtype=np.complex_)
ww[emm<mmin]=0
ww[ell<lmin]=0

bl1d   = (hp.alm2cl(tfbl*np.ones_like(tfbl,dtype=np.complex_)))**0.5
bl1d[:220] = 0

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
sim_dict['dir_cinvT']   = dir_tmp+'/T/'
sim_dict['dir_cinvP']   = dir_tmp+'/P/'
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
nfg    = np.load(file_noisefg)
nltt2d = base_utils.reduce_lmax(nfg['almTT'],lmax=lmax)/nfg['nsims']*tfbl**2#*nfg['nrm']
nlee2d = base_utils.reduce_lmax(nfg['almEE'],lmax=lmax)/nfg['nsims']*tfbl**2#*nfg['nrm']

# nlm^2 We need to compare the in-patch power so we divide the flat-noise part with nrm
nl2dt = ((nltt2d.astype(np.complex_)*scal_t - (nlev_t*np.pi/180./60.)**2/nfg['nrm'])); nl2dt[nl2dt < 0]=1e-10
nl2dp = ((nlee2d.astype(np.complex_)*scal_p - (nlev_p*np.pi/180./60.)**2/nfg['nrm'])); nl2dp[nl2dp < 0]=1e-10

dict_nl2d = {'tt': nl2dt,
             'ee': nl2dp,
             'bb': nl2dp}
'''
np.save('nl2dt',nl2dt)
np.save('nl2dp',nl2dp)
np.save('nltt2d',nltt2d)
np.save('nlee2d',nlee2d)
np.save('nt',(nlev_t*np.pi/180./60.)**2/nfg['nrm'])
np.save('np',(nlev_t*np.pi/180./60.)**2/nfg['nrm'])
#sys.exit()
''';

if sepTP:
    file_out     = config['cinv']['file_out_sepTP']
    cinv_t  = cinv.cinv_t(dir_tmp+"/T/", lmax, nside, cl_len, np.ones_like(bl1d), [ninv_t],
                       eps_min = eps_min,
                       nl      = dict_nl2d,
                       tf2d    = tfbl*ww,
                       #cmb2d   = None,
                      )

    cinv_p  = cinv.cinv_p(dir_tmp+"/P/", lmax, nside, cl_len, np.ones_like(bl1d), [ninv_p],
                        eps_min = eps_min,
                        nl      = dict_nl2d,
                        tf2d    = tfbl*ww,
                        #cmb2d   = None,
                        )



    print('Apply a lrange cut: [%d<ell<%d]'%(lmin,lmax) )
    lfilt  = np.ones(sim_dict['ivf_lrange'][1] + 1, dtype=float) * (np.arange(sim_dict['ivf_lrange'][1] + 1) >= sim_dict['ivf_lrange'][0])

    print('Running C^-1...')
    #jivfs  = cinv.library_cinv_jointTP(dir_tmp + 'outputs', sims, cinv_tp, cl_len, lfilt = lfilt)
    #ivf_t,ivf_e,ivf_b  = jivfs.get_sim_teblmivf(seed)
    ivfs        = cinv.library_cinv_sepTP(dir_tmp+ 'outputs', sims, cinv_t, cinv_p, cl_len, lfilt = lfilt)
    ivf_t       = ivfs.get_sim_tlmivf(seed)
    ivf_e,ivf_b = ivfs.get_sim_eblmivf(seed)

    print('Saving output file')
    print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed))
    np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed), tlm=ivf_t, elm=ivf_e, blm=ivf_b)

    print('Cleaning intermediate files')
    os.remove(dir_tmp+'input_%d'%(seedR))

else:
    file_out     = config['cinv']['file_out_jointTP']
    cinv_tp = cinv.cinv_tp(dir_tmp+"/TP/", lmax, nside, cl_len, np.ones_like(bl1d), [ninv_t, ninv_p],
                        eps_min = eps_min,
                        nl      = dict_nl2d,
                        tf2d    = tfbl*ww,
                        cmb2d   = None,
                        )

    print('Apply a lrange cut: [%d<ell<%d]'%(lmin,lmax) )
    lfilt  = np.ones(sim_dict['ivf_lrange'][1] + 1, dtype=float) * (np.arange(sim_dict['ivf_lrange'][1] + 1) >= sim_dict['ivf_lrange'][0])

    print('Running C^-1...')
    jivfs  = cinv.library_cinv_jointTP(dir_tmp + 'outputs', sims, cinv_tp, cl_len, lfilt = lfilt)
    ivf_t,ivf_e,ivf_b  = jivfs.get_sim_teblmivf(seed)

    print('Saving output file')
    print(' - %s'%file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname))
    np.savez(file_out.format(cmbset=cmbset,nside=nside,lmin=lmin,lmax=lmax,mmin=mmin,seed=seed,runname=runname), tlm=ivf_t, elm=ivf_e, blm=ivf_b)

    print('Cleaning intermediate files')
    os.remove(dir_tmp+'input_%d'%(seedR))
