# Main lensing reconstruction code

import sys,argparse,git,yaml
import numpy as np
import healpy as hp
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/')
import weights, qest
import healqest_utils as hutils
from pathlib import Path
import logging as lg


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
parser.add_argument('seed1'    , default=1   , type=int, help='seed1')
parser.add_argument('cmbset1'  , default=1   , type=int, help='cmbset1')
parser.add_argument('seed2'    , default=1   , type=int, help='seed2')
parser.add_argument('cmbset2'  , default=1   , type=int, help='cmbset2')
parser.add_argument('--curl'   , default=False, dest='curl' , action='store_true')
parser.add_argument('--noinpaint', default=False, dest='noinpaint' , action='store_true')
parser.add_argument('--log'      , default=False, dest='savelog'   , action='store_true')
args = parser.parse_args()

file_yaml = args.file_yaml
seed1   = args.seed1
seed2   = args.seed2
cmbset1 = args.cmbset1
cmbset2 = args.cmbset2
savelog = args.savelog
noinpaint = args.noinpaint
curl      = args.curl

# Naming alias for cmb set
cmbset = {1: 'a', 2: 'b'} 

print(f"Reading from yaml file: {file_yaml}")
config  = hutils.parse_yaml(file_yaml)
runname = config['base']['runname']
file_bconfig = config['base']['config']
rectype = config['lensrec']['rectype']
cls     = config['cls']
rectype = config['lensrec']['rectype']
lminT   = config['lensrec']['lminT']
lminP   = config['lensrec']['lminP']
lmaxT   = config['lensrec']['lmaxT']
lmaxP   = config['lensrec']['lmaxP']
#tpflag  = config['lensrec']['TPflag']
mmin    = config['lensrec']['mmin']
notch   = config['lensrec']['notch']

nside       = config['cinv']['nside']
cinvlmax    = config['cinv']['lmax']
cinvlmin    = config['cinv']['lmin']
cinvmmin    = config['cinv']['mmin']

# Save githash
repo    = git.Repo('./',search_parent_directories=True)
sha     = repo.head.object.hexsha

# Setup logger if we want
hutils.setup_logger(savelog,file_log=f'logs/{runname}/log_rec_lens_{rectype}_{cmbset1}_{seed1}_{cmbset2}_{seed2}.txt')
lg.warning(f'githash: {sha}')

if rectype=='sqe':
    file_cinv    = config['cinv']['file_out_sepTP']
else:
    file_cinv    = config['cinv']['file_out_jointTP']

if noinpaint:
    suffix = '_noinpaint'
else:
    suffix = ''

lg.warning(f'Loading base config file {file_bconfig}')
base_config   = yaml.safe_load(open(file_bconfig))
#runname       = base_config['base']['runname']

lg.warning(f"Reconstruction type: {rectype}")

file_almbar1 = file_cinv.format(cmbset=cmbset1,nside=nside,lmin=cinvlmin,lmax=cinvlmax,mmin=cinvmmin,seed=seed1,runname=runname,suffix=suffix)
file_almbar2 = file_cinv.format(cmbset=cmbset2,nside=nside,lmin=cinvlmin,lmax=cinvlmax,mmin=cinvmmin,seed=seed2,runname=runname,suffix=suffix)
print('file_almbar1:'+file_almbar1)
print('file_almbar2:'+file_almbar2)

#sys.exit()
d1=np.load(file_almbar1)
d2=np.load(file_almbar2)
tlm1,elm1,blm1 = d1['tlm'],d1['elm'],d1['blm']
tlm2,elm2,blm2 = d2['tlm'],d2['elm'],d2['blm']

# ell filtering
ell,emm  = hp.Alm.getlm(config['lensrec']['lmax'])
lmaskt   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
lmaskt[ell>config['lensrec']['lmaxT']]=0 
lmaskt[ell<config['lensrec']['lminT']]=0 
lmaskp   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
lmaskp[ell>config['lensrec']['lmaxP']]=0 
lmaskp[ell<config['lensrec']['lminP']]=0 

mmask   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']),dtype=np.complex128)
mmask[(emm<config['lensrec']['mmin']) ]=0

if notch is not None:
    print("Reading notch file and setting noise in hole to infinity")
    notchalm = hp.read_alm(notch)
    notchalm = hutils.reduce_lmax(notchalm,lmax=config['lensrec']['lmax'])
    #notchalm[notchalm==0] = 1e30
    mmask*=notchalm
#mmaskp   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
#mmaskp[(emm<config['lensrec']['mminP']) ]=0


# mask to do T-only or P-only
tpmask = {'TP':(1,1), 'T':(1,0), 'P':(0,1)}

almbar1={}
almbar1['T'] = hutils.reduce_lmax(tlm1,config['lensrec']['lmax'])*lmaskt*mmask#*tpmask[tpflag][0]
almbar1['E'] = hutils.reduce_lmax(elm1,config['lensrec']['lmax'])*lmaskp*mmask#*tpmask[tpflag][1]
almbar1['B'] = hutils.reduce_lmax(blm1,config['lensrec']['lmax'])*lmaskp*mmask#*tpmask[tpflag][1]
almbar2={}
almbar2['T'] = hutils.reduce_lmax(tlm2,config['lensrec']['lmax'])*lmaskt*mmask#*tpmask[tpflag][0]
almbar2['E'] = hutils.reduce_lmax(elm2,config['lensrec']['lmax'])*lmaskp*mmask#*tpmask[tpflag][1]
almbar2['B'] = hutils.reduce_lmax(blm2,config['lensrec']['lmax'])*lmaskp*mmask#*tpmask[tpflag][1]

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


if rectype=='gmvjtp' or rectype=='gmvjtp_tteete' or rectype=='gmvjtp_tbeb':
    if rectype=='gmvjtp'       : gmvname = 'GMV'
    if rectype=='gmvjtp_tteete': gmvname = 'GMVTTEETE'
    if rectype=='gmvjtp_tbeb'  : gmvname = 'GMVTBEB'
     
    # Save the sum
    kglm =0 
    kclm =0 
    for qe in (config['qes']):
        Q       = qest.qest(config,cls)
        print(qe)
        Q.eval(qe,almbar1[qe[0]],almbar2[qe[1]])
        kglm += hp.almxfl(Q.glm[qe],v)
        kclm += hp.almxfl(Q.clm[qe],v)
   
    if noinpaint:
        dir_out = config['outputs']['dir_out_noinpaint'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)+'/%s/'%gmvname
    else:
        dir_out = config['outputs']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)+'/%s/'%gmvname

    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.savez(dir_out + '%s%s_%d%s_%d%s.npz'%(prefix,gmvname,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2]), glm=kglm, clm=kclm)
    lg.warning(f'Saving: {dir_out}plm{gmvname}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')
    print(f'Saving: {dir_out}plm{gmvname}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')

elif rectype=='sqe':
    # Save each estimator separately
    for qe in (config['qes']):
        Q       = qest.qest(config,cls)
        Q.eval(qe,almbar1[qe[0]],almbar2[qe[1]])
        kglm = hp.almxfl(Q.glm[qe],v)
        kclm = hp.almxfl(Q.clm[qe],v)
        
        if noinpaint:
            dir_out = config['outputs']['dir_out_noinpaint'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)+'/%s/'%qe
        else:
            dir_out = config['outputs']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)+'/%s/'%qe

        Path(dir_out).mkdir(parents=True, exist_ok=True)
        print('Saving outputs to %s:'%dir_out )
        np.savez(dir_out+'%s%s_%d%s_%d%s.npz'%(prefix,qe,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2]),glm=kglm,clm=kclm)
        lg.warning(f'Saving: {dir_out}plm{qe}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')
        print(f'Saving: {dir_out}plm{qe}_{seed1}{cmbset[cmbset1]}_{seed2}{cmbset[cmbset2]}.npz')

    
 
