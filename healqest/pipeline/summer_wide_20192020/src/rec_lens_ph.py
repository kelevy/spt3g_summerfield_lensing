# Main lensing reconstruction code for profile hardening
# (following rec_lens_masked.py)
#
# call: 
#   python src/rec_lens_ph.py s3df_yaml/config_gmv.yaml 1 1 1 1 -src_yaml s3df_yaml/config_gmvbhttprf.yaml -prftype tsz [--nops --log]


import os,sys,argparse,git,yaml
import numpy as np
import healpy as hp
p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../healqest/src/"))
sys.path.insert(0, p)
import weights, qest, profiles
import healqest_utils as hutils
from pathlib import Path
import logging as lg
sys.path.insert(0,'//sdf/home/w/wlwu/repos/spt3g_software/scratch/yomori/utils/')
#import utils as base_utils


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
parser.add_argument('seed1'    , default=1   , type=int, help='seed1')
parser.add_argument('cmbset1'  , default=1   , type=int, help='cmbset1')
parser.add_argument('seed2'    , default=1   , type=int, help='seed2')
parser.add_argument('cmbset2'  , default=1   , type=int, help='cmbset2')
parser.add_argument('bundleid1', default=0, type=int, help='bundleid1')
parser.add_argument('--bundleid2', default=None, help='bundleid2')
parser.add_argument('-src_yaml', default=None, dest='src_yaml', help="srcprf yamlfile")
parser.add_argument('-prftype' , default='tsz',dest='prftype'     , help="tsz or 1am")
parser.add_argument('--curl'   , default=False, dest='curl' , action='store_true')
parser.add_argument('--nops'   , default=False, dest='nops' , action='store_true')
parser.add_argument('--log'      , default=False, dest='savelog'   , action='store_true')
args = parser.parse_args()

file_yaml = args.file_yaml
bundleid1 = args.bundleid1
bundleid2 = args.bundleid2
if bundleid2 == None:
    bundleid2=bundleid1
seed1   = args.seed1
seed2   = args.seed2
cmbset1 = args.cmbset1
cmbset2 = args.cmbset2
savelog = args.savelog
prftype = args.prftype
nops    = args.nops
curl    = args.curl


# Naming alias for cmb set
cmbset = {1: 'a', 2: 'b'} 

print(f"Reading from yaml file: {file_yaml}")
config       = hutils.load_yaml(args.file_yaml)
runname      = config['base']['runname']
#file_bconfig = config['base']['config']
rectype      = config['lensrec']['rectype']
qesttype     = config['lensrec']['qesttype']
cls          = config['cls']
lminT        = config['lensrec']['lminT']
lminP        = config['lensrec']['lminP']
lmaxT        = config['lensrec']['lmaxT']
lmaxP        = config['lensrec']['lmaxP']
mmin         = config['lensrec']['mmin']
lmaxTP       = max(lmaxT, lmaxP)
notch        = config['lensrec']['notch']
file_mask    = config['lensrec']['mask']
apply_emmtf  = config['lensrec']['apply_emmtf']
nside        = config['lensrec']['nside']


# Save githash
repo    = git.Repo('./',search_parent_directories=True)
sha     = repo.head.object.hexsha

# Setup logger if we want
hutils.setup_logger(savelog,file_log=f'logs/{runname}/log_rec_lens_{rectype}_{cmbset1}_{seed1}_{cmbset2}_{seed2}.txt')
lg.warning(f'githash: {sha}')

if rectype=='sqe':
    if nops: file_cinv    = config['cinv']['file_out_sepTP'][:-4]+'_nops.npz'
    else   : file_cinv    = config['cinv']['file_out_sepTP']
else:
    if nops: file_cinv    = config['cinv']['file_out_jointTP'][:-4]+'_nops.npz'
    else   : file_cinv    = config['cinv']['file_out_jointTP']

#lg.warning(f'Loading base config file {file_bconfig}')
#base_config   = yaml.safe_load(open(file_bconfig))
#base_config['data']['tf']['file'] = "/sdf/home/w/wlwu/data/spt3glens1920_gmvph/tf/tf2d_{freq}ghz_750sims.npz" 

lg.warning(f"Reconstruction type: {rectype}")

file_almbar1 = file_cinv.format(lminT=lminT,lmaxT=lmaxT,seed=seed1,runname=runname,rectype=rectype,bundleid=bundleid1)
file_almbar2 = file_cinv.format(lminT=lminT,lmaxT=lmaxT,seed=seed2,runname=runname,rectype=rectype,bundleid=bundleid2)
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
###### Not used in baseline
nside=2048

if file_mask is not None:
    if 'inverse' in file_mask:
        mb=hp.read_map(file_mask,partial=True)
        mb[mb==hp.UNSEEN]=0
        mb=1-mb
    else:
        mb=hp.read_map(file_mask)


# ell filtering
ell,emm  = hp.Alm.getlm(config['lensrec']['lmax'])
lmaskt   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
lmaskt[ell>config['lensrec']['lmaxT']]=0 
lmaskt[ell<config['lensrec']['lminT']]=0 
lmaskp   = np.ones(hp.Alm.getsize(config['lensrec']['lmax']))
lmaskp[ell>config['lensrec']['lmaxP']]=0 
lmaskp[ell<config['lensrec']['lminP']]=0 

mmask   = np.ones(hp.Alm.getsize(lmaxTP),dtype=np.complex128)
mmask[(emm<config['lensrec']['mmin']) ]=0

if notch is not None:
    print("Reading notch file and masking hole")
    notchalm = hp.read_alm(notch)
    notchalm = hutils.reduce_lmax(notchalm,lmax=lmaxTP)
    #notchalm[notchalm==0] = 1e30
    mmask*=notchalm


# mask to do T-only or P-only
tpmask = {'TP':(1,1), 'T':(1,0), 'P':(0,1)}

almbar1={}
almbar1['T'] = hutils.reduce_lmax(tlm1,lmaxTP)*lmaskt*mmask#*tpmask[tpflag][0]
almbar1['E'] = hutils.reduce_lmax(elm1,lmaxTP)*lmaskp*mmask#*tpmask[tpflag][1]
almbar1['B'] = hutils.reduce_lmax(blm1,lmaxTP)*lmaskp*mmask#*tpmask[tpflag][1]
almbar2={}
almbar2['T'] = hutils.reduce_lmax(tlm2,lmaxTP)*lmaskt*mmask#*tpmask[tpflag][0]
almbar2['E'] = hutils.reduce_lmax(elm2,lmaxTP)*lmaskp*mmask#*tpmask[tpflag][1]
almbar2['B'] = hutils.reduce_lmax(blm2,lmaxTP)*lmaskp*mmask#*tpmask[tpflag][1]


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

if "lcmb" not in cls:
    ### FIXME
    config = hutils.load_cambfiles_dict(config)
    cls     = config['cls']
    #print(config)
    if "qes" not in config:
        if "includes" in config:
            yamlfile = config['includes'][-1] if type(config['includes']) is list else config['includes']
            config_base = hutils.parse_yaml("s3df_yaml/"+yamlfile)
        else:
            config_base = hutils.parse_yaml(file_yaml)
        config['qes'] = config_base['qes']

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


Q       = qest.qest(config,cls)
dir_out0 = config['lensrec']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)


if rectype=='sqe':
    print('11111')
    # Save each estimator separately
    for qe in ['TT']:
        assert qe=='TT', "only hardening TT"
        dir_out = dir_out0+'/%s/'%qe
        fnameout = '%s%s_%d%s_%d%s.npz'%(prefix,qe,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2])
        if nops: fnameout = fnameout[:-4]+'_nops.npz'
        if not os.path.isfile(dir_out+fnameout):    
            Q.eval(qe,almbar1[qe[0]],almbar2[qe[1]])
            kglm = Q.glm[qe]
            kclm = Q.clm[qe]


            Path(dir_out).mkdir(parents=True, exist_ok=True)
            print('Saving outputs to %s:'%dir_out )
            np.savez(dir_out+fnameout,glm=kglm,clm=kclm)
            lg.warning(f'Saving: {dir_out}{fnameout}')
            print(f'Saving: {dir_out}{fnameout}')
        else:
            #load the TT kappa (to be hardened)
            print(f'Loading: {dir_out}{fnameout}')
            kglm = np.load(dir_out+fnameout)['glm']


if args.src_yaml is not None:
    print('222222')
    config_src = hutils.load_yaml(args.src_yaml)
    estname = 'TT' if rectype=='sqe' else gmvname
    qes_h         = config_src['qes_h']
    sdir_out      = config_src['lensrec']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin,prftype=prftype)+"%sbh%s/"%(estname,qes_h)
    arespss_fname = sdir_out+config_src['ss_resp']['fnamestub'].format(prftype=prftype)
    arespse_fname = sdir_out+config_src['se_resp']['fnamestub'].format(prftype=prftype)
    Path(sdir_out).mkdir(parents=True, exist_ok=True)
    aresp_fname   = dir_out0+config['aresp']['fnamestub']

    #compute src/prf est
    srcfname = sdir_out +'plm%s_%d%s_%d%s.npz'%(qes_h,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2])
    if nops: srcfname = srcfname[:-4]+'_nops.npz'
    if qes_h == "TTsrc":
        u = np.ones(lmaxTP)
    elif qes_h == "TTprf":
        if prftype == '1am': 
            gauss_fwhm = config_src['gauss_fwhm_arcmin']
            u    = profiles.profileGaussian(gauss_fwhm, lmax=lmaxTP).fourier()
        elif prftype == 'tsz':
            u    = np.load(config_src['profile_file'])
        else:
            assert 0
    else:
        assert 0, "must be TTsrc or TTprf"

    if not os.path.isfile(srcfname):
        #use 'TTprf'; but if u = 1 (effectively TTsrc)
        Q.eval('TTprf', almbar1[qes_h[0]], almbar2[qes_h[1]], u=u)
        kslm = Q.glm['TTprf']
        np.savez(srcfname,glm=kslm)
        print(f'Saving: {srcfname}')
    else:
        print(f'Loading: {srcfname}')
        kslm = np.load(srcfname)['glm']

    #compute src/prf-harden GMV/SQE
    hdnfname = sdir_out +'plm%sbh%s_%d%s_%d%s.npz'%(estname,qes_h,seed1,cmbset[cmbset1],seed2,cmbset[cmbset2])
    if nops: hdnfname = hdnfname[:-4]+'_nops.npz'
    resp_tot, weight = hutils.get_aresp_tot(aresp_fname, arespss_fname, arespse_fname, estname)
    plm_bh = hutils.harden_est(kglm, kslm, weight)
    np.savez( hdnfname ,glm=plm_bh, analytic_resp=resp_tot)
    print(f'Saving: {hdnfname}')



