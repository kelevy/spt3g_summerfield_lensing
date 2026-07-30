'''
pre-compute gmv and sqe analytic response for profile hardening 

Call:
    python src/compute_aresp.py yaml/gmv_gauss.yaml -src_yaml yaml/prf1am.yaml -prftype tsz

'''
import os, sys
import numpy as np
import yaml,argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='yamlfile')
parser.add_argument('-src_yaml', default=None, dest='src_yaml', help="srcprf yamlfile")
parser.add_argument('-prftype' , default='tsz',dest='prftype'     , help="tsz or 1am")
args = parser.parse_args()
prftype = args.prftype

#config       = yaml.safe_load(open(args.file_yaml))
#dir_healqest = config['base']['dir_healqest']
#sys.path.insert(0, dir_healqest+'/healqest/src/')
sys.path.insert(0, '/sdf/home/w/wlwu/repos/healqest/healqest/src/')
import healqest_utils as hutils
import gmv_resp, profiles, weights, resp

config  = hutils.load_yaml(args.file_yaml)
runname = config['base']['runname']
file_bconfig = config['base']['config']
cls     = config['cls']
rectype = config['lensrec']['rectype']
Lmax    = config['lensrec']['Lmax']
lminT   = config['lensrec']['lminT']
lminP   = config['lensrec']['lminP']
lmaxT   = config['lensrec']['lmaxT']
lmaxP   = config['lensrec']['lmaxP']
mmin    = config['lensrec']['mmin']
lmaxTP  = max(lmaxT, lmaxP)

dir_out = config['lensrec']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)
Path(dir_out).mkdir(parents=True, exist_ok=True)
aresp_fname = dir_out+config['aresp']['fnamestub']

#import pdb; pdb.set_trace()
#noise+fg
file_clnoisefg = config['aresp']['cl_nfg']
if "lcmb" not in cls:
    config = hutils.load_cambfiles_dict(config)
    cls     = config['cls']
res  = np.load(file_clnoisefg)
nltt = np.nan_to_num(res['cltt'])
nlee = nlbb = np.nan_to_num(res['clee'])
cls = hutils.add_clsdict(cls,'res',nltt,nlee,nlbb, clte=np.zeros_like(nltt))

tcls = hutils.get_totalcls(cls, lmaxT,lmaxP,lmaxTP, lminT,lminP)
totalcls = np.zeros([lmaxTP+1, 4])
totalcls[:,0] = tcls['tt']
totalcls[:,1] = tcls['ee']
totalcls[:,2] = tcls['bb']
totalcls[:,3] = tcls['te']
cltype  = config['lensrec']['cltype']

if not os.path.isfile(aresp_fname):
    if 'gmv' in rectype:
        r = gmv_resp.gmv_resp(config, cltype, totalcls, u=None, save_path=aresp_fname)
        r.calc_tvar()
        print("GMV analytic resp: Done at %s"%aresp_fname)
    else:
        est   = "TT"
        flTT  = 1/tcls['tt']; flTT[:lminT]=0; flTT[lmaxT:]=0
        qeTT  = weights.weights(est, cls[cltype], lmaxTP)
        aresp = resp.fill_resp( qeTT, np.zeros(lmaxTP+1, dtype=np.complex_), flTT, flTT)
        np.save(aresp_fname, aresp) #Note lmaxTP can be < Lmax 
        print("%s analytic resp: Done at %s"%(est, aresp_fname))


if args.src_yaml is not None:
    config_src    = hutils.load_yaml(args.src_yaml)
    if rectype=='gmvjtp' or rectype=='gmvjtp_tteete' or rectype=='gmvjtp_tbeb':
        if rectype=='gmvjtp'       : gmvname = 'GMV'
        if rectype=='gmvjtp_tteete': gmvname = 'GMVTTEETE'
        if rectype=='gmvjtp_tbeb'  : gmvname = 'GMVTBEB'
    estname = 'TT' if rectype=='sqe' else gmvname
    qes_h         = config_src['qes_h']

    sdir_out      = config_src['lensrec']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin,prftype=prftype)+"%sbh%s/"%(estname,qes_h)
    arespss_fname = sdir_out+config_src['ss_resp']['fnamestub'].format(prftype=prftype)
    arespse_fname = sdir_out+config_src['se_resp']['fnamestub'].format(prftype=prftype)
    Path(sdir_out).mkdir(parents=True, exist_ok=True)

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

    if not os.path.isfile(arespss_fname):
        if 'gmv' in rectype:
            r = gmv_resp.gmv_resp(config, cltype, totalcls, u=u, save_path=arespss_fname)
            r.calc_tvar_PRF(cross=False)
            print("GMV: src-src analytic resp: Done at %s"%arespss_fname)
        else:
            flTT  = 1/tcls['tt']; flTT[:lminT]=0; flTT[lmaxT:]=0
            qeTTp = weights.weights("TTprf", cls[cltype], lmaxTP, u=u)
            aresp = resp.fill_resp( qeTTp, np.zeros(lmaxTP+1, dtype=np.complex_), flTT, flTT)
            np.save(arespss_fname, aresp)
            print("TTprf analytic resp: Done at %s"%arespss_fname)


    if not os.path.isfile(arespse_fname):
        if 'gmv' in rectype:
            r = gmv_resp.gmv_resp(config, cltype, totalcls, u=u, save_path=arespse_fname)
            r.calc_tvar_PRF(cross=True)
            print("GMV src-phi analytic resp: Done at %s"%arespse_fname)
        else:
            flTT  = 1/tcls['tt']; flTT[:lminT]=0; flTT[lmaxT:]=0
            qeTTp = weights.weights("TTprf", cls[cltype], lmaxTP, u=u)
            qeTT  = weights.weights("TT", cls[cltype], lmaxTP)
            aresp = resp.fill_resp(qeTT,
                             np.zeros(lmaxTP+1, dtype=np.complex_), flTT, flTT,
                             qeZA=qeTTp)
            np.save(arespse_fname, aresp)
            print("TTprf-TT analytic resp: Done at %s"%arespse_fname)


