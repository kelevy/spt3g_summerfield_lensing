# Main lensing reconstruction code

import sys,argparse,git,yaml
import numpy as np
import healpy as hp
import os
p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../healqest/src/"))
sys.path.insert(0, p)
import weights, qest
import healqest_utils as hutils
from pathlib import Path
import logging as lg
#from utils import utils as base_utils


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
parser.add_argument('seed1'    , default=1, type=int, help='seed1')
parser.add_argument('cmbset1'  , default=1, type=int, help='cmbset1')
parser.add_argument('seed2'    , default=1, type=int, help='seed2')
parser.add_argument('cmbset2'  , default=1, type=int, help='cmbset2')
parser.add_argument('bundleid',  default=0, type=int, help='bundleid')
parser.add_argument('--curl'   , default=False, dest='curl' , action='store_true')
parser.add_argument('--nops'   , default=False, dest='nops' , action='store_true')
parser.add_argument('--log'      , default=False, dest='savelog'   , action='store_true')
args = parser.parse_args()

file_yaml = args.file_yaml
bundleid  = args.bundleid
seed1     = args.seed1
seed2     = args.seed2
cmbset1   = args.cmbset1
cmbset2   = args.cmbset2
savelog   = args.savelog
nops      = args.nops
curl      = args.curl

# Naming alias for cmb set
cmbset = {1: 'a', 2: 'b', 3: 'a', 4: 'b'} 

print(f"Reading from yaml file: {file_yaml}")
config       = hutils.parse_yaml(file_yaml)
runname      = config['base']['runname']
#file_bconfig = config['base']['config']
rectype      = config['lensrec']['rectype']
qesttype     = config['lensrec']['qesttype']
cls          = config['cls']
rectype      = config['lensrec']['rectype']
lminB        = config['lensrec']['lminB']
lminT        = config['lensrec']['lminT']
lminP        = config['lensrec']['lminP']
lmaxT        = config['lensrec']['lmaxT']
lmaxP        = config['lensrec']['lmaxP']
#lminB        = config['lensrec']['lminB']
mmin         = config['lensrec']['mmin']
notch        = config['lensrec']['notch']
file_mask    = config['lensrec']['mask']
apply_emmtf  = config['lensrec']['apply_emmtf']

nside        = config['lensrec']['nside']
#cinvlmax     = config['cinv']['lmax']
#cinvlmin     = config['cinv']['lmin']
#cinvmmin     = config['cinv']['mmin']

#assert mmin>=cinvmmin, 'mmin for lensrec must be greater than cinv mmin'
#assert lminT>=cinvlmin, 'lminT for lensrec must be greater than or eq cinv lmin'
#assert lminP>=cinvlmin, 'lminP for lensrec must be greater than or eq cinv lmin'


# Save githash
repo    = git.Repo('./',search_parent_directories=True)
sha     = repo.head.object.hexsha

# Setup logger if we want
hutils.setup_logger(savelog,file_log=f'logs/{runname}/log_rec_lens_{rectype}_{cmbset1}_{seed1}_{cmbset2}_{seed2}.txt')
lg.warning(f'githash: {sha}')

if rectype=='sqe':
    if nops: file_cinv    = config['cinv']['file_out_sepTP'][:-4]+'_nops.npz'
    else   : file_cinv    = config['cinv_approx']['file_out'] #config['cinv']['file_out_sepTP']
elif 'gmv' in rectype:
    if nops: file_cinv    = config['cinv']['file_out_jTP'][:-4]+'_nops.npz'
    else   : file_cinv    = config['cinv']['file_out_jTP']
else:
    sys.exit('Undefined rectype')

#lg.warning(f'Loading base config file {file_bconfig}')
#base_config   = yaml.safe_load(open(file_bconfig))

lg.warning(f"Reconstruction type: {rectype}")
file_almbar1 = file_cinv.format(lminT=lminT,lmaxT=lmaxT,seed=seed1,runname=runname,rectype=rectype,bundleid=bundleid)
file_almbar2 = file_cinv.format(lminT=lminT,lmaxT=lmaxT,seed=seed2,runname=runname,rectype=rectype,bundleid=bundleid)
print('file_almbar1:'+file_almbar1)
print('file_almbar2:'+file_almbar2)


d1=np.load(file_almbar1, allow_pickle=True)
d2=np.load(file_almbar2, allow_pickle=True)
if qesttype == 'TT':
    tlm1,elm1,blm1 = d1['tlm'],d1['tlm'],d1['tlm']
    tlm2,elm2,blm2 = d2['tlm'],d2['tlm'],d2['tlm']
else:
    tlm1,elm1,blm1 = d1['tlm'],d1['elm'],d1['blm']
    tlm2,elm2,blm2 = d2['tlm'],d2['elm'],d2['blm']


if apply_emmtf:
    tlm1 *= tfblT_2d_nobeam
    elm1 *= tfblP_2d_nobeam
    blm1 *= tfblP_2d_nobeam
    tlm2 *= tfblT_2d_nobeam
    elm2 *= tfblP_2d_nobeam
    blm2 *= tfblP_2d_nobeam

###### Apply additional mask before reconstructing
nside=2048

if file_mask is not None:
    if 'inverse' in file_mask:
        mb=hp.read_map(file_mask,partial=True)
        mb[mb==hp.UNSEEN]=0
        mb=1-mb
    else: 
        mb=hp.read_map(file_mask)

    a=hp.alm2map([tlm1,elm1,blm1],hp.npix2nside(mb.shape[0]))
    tlm1,elm1,blm1=hp.map2alm(a*mb,lmax=4096,use_pixel_weights=True)

    a=hp.alm2map([tlm2,elm2,blm2],hp.npix2nside(mb.shape[0]))
    tlm2,elm2,blm2=hp.map2alm(a*mb,lmax=4096,use_pixel_weights=True)

# ell filtering
ell,emm  = hp.Alm.getlm(config['lensrec']['lmax'])
lmaskt   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
lmaskt[ell>config['lensrec']['lmaxT']]=0 
lmaskt[ell<config['lensrec']['lminT']]=0 
lmaskp   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
lmaskp[ell>config['lensrec']['lmaxP']]=0 
lmaskp[ell<config['lensrec']['lminP']]=0 

if lminB is not None:
    lmaskb = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
    lmaskb[ell>config['lensrec']['lmaxP']]=0
    lmaskb[ell<lminB]=0
else:
    lmaskb = 1

mmask   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']),dtype=np.complex128)
mmask[(emm<config['lensrec']['mmin']) ]=0

if notch is not None:
    print("Reading notch file and setting noise in hole to infinity")
    notchalm = hp.read_alm(notch)
    notchalm = hutils.reduce_lmax(notchalm,lmax=config['lensrec']['lmax'])
    #notchalm[notchalm==0] = 1e30
    mmask*=notchalm

# mask to do T-only or P-only
tpmask = {'TP':(1,1), 'T':(1,0), 'P':(0,1)}

almbar1={}
almbar1['T'] = hutils.reduce_lmax(tlm1,config['lensrec']['lmaxT'])*lmaskt*mmask
almbar1['E'] = hutils.reduce_lmax(elm1,config['lensrec']['lmaxP'])*lmaskp*mmask
almbar1['B'] = hutils.reduce_lmax(blm1,config['lensrec']['lmaxP'])*lmaskp*lmaskb*mmask
almbar2={}
almbar2['T'] = hutils.reduce_lmax(tlm2,config['lensrec']['lmaxT'])*lmaskt*mmask
almbar2['E'] = hutils.reduce_lmax(elm2,config['lensrec']['lmaxP'])*lmaskp*mmask
almbar2['B'] = hutils.reduce_lmax(blm2,config['lensrec']['lmaxP'])*lmaskp*lmaskb*mmask


lg.warning("Starting QE calculations")
qess=(' '.join(config['qes']))
lg.warning(f"Using the following QEs: {qess}" )
glmmv=0; clmmv=0

l=np.arange(config['lensrec']['Lmax']+1)
v=0.5*l*(l+1)

if curl==True:
    config['qes'] = [j+'curl' for j in (config['qes'])]
    prefix        = 'clm'
else:
    prefix        = 'plm'


if rectype=='gmvjtp' or rectype=='gmvjtp_tteete' or rectype=='gmvjtp_tbeb' or rectype=='gmvjtp_sep':
    if rectype=='gmvjtp'       : gmvlist = ['GMV']
    if rectype=='gmvjtp_sep'       : gmvlist = ['GMVTTEETE','GMVTBEB']
    if rectype=='gmvjtp_tteete': gmvlist = ['GMVTTEETE']
    if rectype=='gmvjtp_tbeb'  : gmvlist = ['GMVTBEB']
     
    for gmvname in gmvlist:
        # Save the sum
        kglm =0 
        kclm =0 
        for qe in (hutils.get_qes(gmvname)):
            Q       = qest.qest(config,cls)
            print(qe)
            Q.eval(qe,almbar1[qe[0]],almbar2[qe[1]])
            kglm += Q.glm[qe]
            kclm += Q.clm[qe]
    
        dir_out = config['lensrec']['dir_out'].format(runname=runname,rectype=rectype,bundleid=bundleid,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)+'/%s/'%gmvname

        Path(dir_out).mkdir(parents=True, exist_ok=True)
        print('Saving outputs to %s:'%dir_out )
        if nops:
            np.savez(dir_out+'%s%s_%d%s_%d%s_nops.npz'%(prefix,gmvname,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2]),glm=kglm,clm=kclm)
            lg.warning(f'Saving: {dir_out}plm{gmvname}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}_nops.npz')
            print(f'Saving: {dir_out}plm{gmvname}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}_nops.npz')
        else:
            np.savez(dir_out+'%s%s_%d%s_%d%s.npz'%(prefix,gmvname,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2]),glm=kglm,clm=kclm)
            lg.warning(f'Saving: {dir_out}plm{gmvname}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')
            print(f'Saving: {dir_out}plm{gmvname}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')
            

elif rectype=='sqe':
    # Save each estimator separately
    for qe in ['TT']: # (['TT','EB','EE','TB','TE']):#(config['qes']):
        Q       = qest.qest(config,cls)
        Q.eval(qe,almbar1[qe[0]],almbar2[qe[1]])
        glm = Q.glm[qe]
        clm = Q.clm[qe]

        dir_out = config['lensrec']['dir_out'].format(runname=runname,rectype=rectype,bundleid=bundleid,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)  #+'/%s/'%qe

        Path(dir_out).mkdir(parents=True, exist_ok=True)
        print('Saving outputs to %s:'%dir_out )
        if nops:
            np.savez(dir_out+'%s%s_%d%s_%d%s_nops.npz'%(prefix,qe,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2]),glm=glm,clm=clm)
            lg.warning(f'Saving: {dir_out}plm{qe}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}_nops.npz')
            print(f'Saving: {dir_out}plm{qe}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}_nops.npz')
        else:
            np.savez(dir_out+'%s%s_%d%s_%d%s.npz'%(prefix,qe,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2]),glm=glm,clm=clm)
            lg.warning(f'Saving: {dir_out}plm{qe}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')
            print(f'Saving: {dir_out}plm{qe}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')
            

    
 
