'''
Compute lensing with and without emm removed
It has been tested where adding noise changed anything.
In the end, the dominant noise comes from teh cmb and noise
doesnt add much.
'''

import os,sys,camb,yaml
import numpy as np
import healpy as hp
sys.path.insert(0,'/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/utils/')
import utils
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/')
import weights, qest
import healqest_utils as hutils
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter1d as gf1

parser  = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str , help='yaml')
parser.add_argument('seed'     , default=None, type=int , help='seed')
args    = parser.parse_args()

file_yaml = args.file_yaml
seed      = args.seed

print("Loading lensing config")
config = hutils.parse_yaml(file_yaml) 
cls    = config['cls']
lminT  = config['lensrec']['lminT']
lmaxT  = config['lensrec']['lmaxT']
lminP  = config['lensrec']['lminP']
lmaxP  = config['lensrec']['lmaxP']
lmax   = 6000 # Some temporary lmax 
print('Using lranget: [%d,%d]'%(lminT,lmaxT))
print('Using lrangep: [%d,%d]'%(lminP,lmaxP))

print("Loading baseline config")
file_yaml_base = config['base']['config_base']
config_base    = yaml.safe_load(open(file_yaml_base))
sys.path.insert(0,config['base']['dir_sptsoft']+'/scratch/yomori/utils/')
import utils

dir_out = config['resp']['tmpdir_emmcut'].format(lminT=lminT,lmaxT=lmaxT,lminP=lminP,lmaxP=lmaxP)
print("Setting output to: ", dir_out)
Path(dir_out).mkdir(parents=True, exist_ok=True)

file_mcorr = config['inputs']['file_mcorr']
print('Loading mcorr file: %s'%file_mcorr )
mcorr = np.load(file_mcorr)

print('Computing emm mask')
mm    = hutils.get_mmask(lmax,config['lensrec']['mmin'])

print('Loading tfbl')
tfbl  = utils.load_tf(config_base,lmax=lmax,include_beam=True,silent=True)

print("Loading boundary mask")
mask  = hp.read_map(config['inputs']['bmask'])
mask[mask==hp.UNSEEN]=0
nrm   = mask.shape[0]/np.sum(mask**2)
print("normalization %.3f"%nrm)

print("Extracting pixel index")
if seed>250:
    patch   = 2
    seedi   = seed-250
    polconv = -1
else:
    patch   = 1
    seedi   = seed
    polconv = 1

pix0,pidx = hutils.extract_patch(mask,patch)

print('Loading noise+forground stacks')
nfg      = np.load(config['inputs']['nfgstack'])
cltt_nfg = hp.alm2cl((nfg['almTT']/nfg['nsims'])**0.5*mm)/mcorr['TT'][:lmax+1]
clee_nfg = hp.alm2cl((nfg['almEE']/nfg['nsims'])**0.5*mm)/mcorr['TT'][:lmax+1]
clbb_nfg = hp.alm2cl((nfg['almBB']/nfg['nsims'])**0.5*mm)/mcorr['TT'][:lmax+1]

print('Load CMB cls')
cambcls    = config['cls']['file_lcmb']
cls_dict   = utils.load_cambcls(cambcls,lmax=6000,dict=True,dls=False)

print('Computing 1D filter')
ftt  = 1/(cls_dict['tt']+cltt_nfg)
fee  = 1/(cls_dict['ee']+clee_nfg)
fbb  = 1/(cls_dict['bb']+clbb_nfg)

print('Setting lranges in the 1D filter')
ftt[:lminT]=0; ftt[lmaxT:]=0
fee[:lminP]=0; fee[lmaxP:]=0
fbb[:lminP]=0; fbb[lmaxP:]=0

print("Loading CMB file: %s"%config['inputs']['rawcmb'].format(seed=seed))
cmb = hp.read_map(config['inputs']['rawcmb'].format(seed=seed),field=[0,1,2])
cmb[0][pix0]  = cmb[0][pidx]
cmb[1][pix0]  = cmb[1][pidx]
cmb[2][pix0]  = cmb[2][pidx]*polconv
ilm = hp.map2alm(cmb*mask,lmax=lmax,use_pixel_weights=True)

print('Computing CMB alms without filtering')
xlmt   = hutils.reduce_lmax(hp.almxfl(ilm[0],ftt),config['lensrec']['lmax'])
xlme   = hutils.reduce_lmax(hp.almxfl(ilm[1],fee),config['lensrec']['lmax'])
xlmb   = hutils.reduce_lmax(hp.almxfl(ilm[2],fbb),config['lensrec']['lmax'])
xlm    = {'T': xlmt, 'E': xlme, 'B': xlmb}

print('Computing CMB alms with filtering')
zlmt   = hutils.reduce_lmax(hp.almxfl(ilm[0]*mm,ftt),config['lensrec']['lmax'])
zlme   = hutils.reduce_lmax(hp.almxfl(ilm[1]*mm,fee),config['lensrec']['lmax'])
zlmb   = hutils.reduce_lmax(hp.almxfl(ilm[2]*mm,fbb),config['lensrec']['lmax'])
zlm    = {'T': zlmt, 'E': zlme, 'B': zlmb}

rec={}
#for qe in (['TT']):
#for qe in (['EE','TB','EB']):
for qe in (['TT','EE','TB','TE','EB']):
    print('Reconstructing %s'%qe )
    Q1    = qest.qest(config,cls)
    Q1.eval(qe,xlm[qe[0]],xlm[qe[1]])
    x_pglm = Q1.glm[qe]
    np.savez(dir_out+'plm%s_%da_%da_noemmcut.npz'%(qe,seed,seed),glm=x_pglm)

    Q2    = qest.qest(config,cls)
    Q2.eval(qe,zlm[qe[0]],zlm[qe[1]])
    z_pglm = Q2.glm[qe]
    np.savez(dir_out+'plm%s_%da_%da_emmcut220.npz'%(qe,seed,seed),glm=z_pglm)

print('Now compute the meanfield and then run resp_emmcuttest.py')