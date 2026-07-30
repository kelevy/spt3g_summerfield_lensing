import os,sys,yaml,argparse
import numpy as np
import healpy as hp
from pathlib import Path
sys.path.append('/data/gpfs/projects/punim1922/summerfield_lensing/healqest/healqest/src/')
import weights,qest,resp
from pathlib import Path
import pathlib




print("Loading config")
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str)
args = parser.parse_args()
config       = yaml.safe_load(open(args.config_file)) 
runname      = config['base']['runname']
file_cambcls = config['cinv_approx']['file_cambcls']
file_clnoise = config['cinv_approx']['file_clnoise']
file_clfg    = config['cinv_approx']['file_clfg'] 
lminT        = config['cinv_approx']['lminT']
lmaxT        = config['cinv_approx']['lmaxT']
rectype      = config['lensrec']['rectype']
aresp_file   = config['lensrec']['aresp']
print('Using lranget: [%d,%d]'%(lminT,lmaxT))


def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl

def get_lensedcls(file,lmax=4000,dict=False):
    ell,sltt,slee,slbb,slte=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi; slee=zeropad(slee)
    slte=slte/ell/(ell+1)*2*np.pi; slte=zeropad(slte)
    slbb=slbb/ell/(ell+1)*2*np.pi; slbb=zeropad(slbb)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    if dict==False:
        return ell,sltt,slee,slbb,slte
    else:
        d={}
        d['tt']=sltt
        d['ee']=slee
        d['bb']=slbb
        d['te']=slte
        return d


print('Loading noise+forground stacks')
cls_noise = np.loadtxt(file_clnoise, unpack=True)[:lmaxT+1]
cls_totfg = np.loadtxt(file_clfg, unpack=True)[:lmaxT+1]
res = cls_totfg + cls_noise

print('Load CMB cls')
cls_dict = {}
ell,sltt,_,_,_ = get_lensedcls(file_cambcls,lmax=lmaxT)
cls_dict['tt'] = sltt

print('Computing 1D filter')
ftt  = 1/(cls_dict['tt']+res)

print('Setting lranges in the 1D filter')
ftt[:lminT]=0; ftt[lmaxT:]=0

print('Computing analytical resp')
arespTT = resp.fill_resp(weights.weights('TT',cls_dict,lmaxT,u=None),np.zeros(lmaxT+1, dtype=np.complex_),ftt, ftt)


p = pathlib.Path(aresp_file.format(runname=runname,rectype=rectype,lminT=lminT,lmaxT=lmaxT))
Path(p.parent).mkdir(parents=True, exist_ok=True)
file_out = aresp_file.format(runname=runname,rectype=rectype,lminT=lminT,lmaxT=lmaxT)
print('Saving file: %s'%file_out)
np.savez(file_out, TT=arespTT)


