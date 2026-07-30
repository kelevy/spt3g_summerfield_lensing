import os,sys,argparse,yaml
import healpy as hp
import numpy as np
import logging as lg
from pathlib import Path
import pathlib


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

def add_clsdict(d,key,cltt):
    d[key]  = {}
    d[key]['tt'] = cltt

    return d

def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
    lmaxin  = hp.Alm.getlmax(alm.shape[0])
    print( "reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
    ell,emm = hp.Alm.getlm(lmaxin)
    almout  = np.zeros(hp.Alm.getsize(lmax),dtype=np.complex_)
    oldi=0
    oldf=0
    newi=0
    newf=0
    dl = lmaxin-lmax

    for i in range(0,lmax+1):
        oldf=oldi+lmaxin+1-i
        newf=newi+lmax+1-i
        almout[newi:newf]=alm[oldi:oldf-dl]
        oldi=oldf
        newi=newf

    return almout

def get_fl(cls, dict_lrange):
    lminT   = dict_lrange['lminT']
    lmaxT  = dict_lrange['lmaxT']
    flT = 1.0/(cls['cmb']['tt']+cls['res']['tt'][:lmaxT+1])
    flT[lmaxT+1:] = 0
    flT[:lminT] = 0
    
    return flT

def get_almbar(alm, cls, dict_lrange):
    flT   = get_fl(cls, dict_lrange)
    lmaxT = dict_lrange['lmaxT']

    tlm = reduce_lmax(alm,lmax=lmaxT)
    tlmbar = hp.almxfl(tlm,flT)

    return tlmbar


parser = argparse.ArgumentParser()
parser.add_argument('--bundleid', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

seed          = args.seed 
file_yaml     = args.config_file

config        = yaml.safe_load(open(file_yaml))
runname       = config['base']['runname']
rectype       = config['lensrec']['rectype']
lminT         = config['cinv_approx']['lminT']
lmaxT         = config['cinv_approx']['lmaxT']
file_cambcls  = config['cinv_approx']['file_cambcls']
file_clnoise  = config['cinv_approx']['file_clnoise']
file_clfg     = config['cinv_approx']['file_clfg'] 
file_in       = config['cinv_approx']['file_in']
file_in_N1    = config['cinv_approx']['file_in_N1']
file_out      = config['cinv_approx']['file_out']
file_out_N1   = config['cinv_approx']['file_out_N1']

p = pathlib.Path(file_out.format(runname=runname,rectype=rectype,lminT=lminT,lmaxT=lmaxT,seed=seed))
Path(p.parent).mkdir(parents=True, exist_ok=True)
pN1 = pathlib.Path(file_out_N1.format(cmbset=1, runname=runname,rectype=rectype,lminT=lminT,lmaxT=lmaxT,seed=seed))
Path(pN1.parent).mkdir(parents=True, exist_ok=True)

print('Loading alms')
file_tlm = file_in.format(runname=runname,rectype=rectype,lminT=lminT,lmaxT=lmaxT,seed=seed)
file_N1_tlm1 = file_in_N1.format(runname=runname,rectype=rectype,cmbset=1,lminT=lminT,lmaxT=lmaxT,seed=seed)
file_N1_tlm2 = file_in_N1.format(runname=runname,rectype=rectype,cmbset=2,lminT=lminT,lmaxT=lmaxT,seed=seed)

tlm = hp.read_alm(file_tlm)
N1_tlm1 = hp.read_alm(file_N1_tlm1)
N1_tlm2 = hp.read_alm(file_N1_tlm2)

print('Filtering alms')
dict_lrange  = {}
dict_lrange['lminT']  = lminT
dict_lrange['lmaxT']  = lmaxT

dict_cls = {}
ell,sltt,_,_,_ = get_lensedcls(file_cambcls,lmax=dict_lrange['lmaxT'])
dict_cls = add_clsdict(dict_cls,'cmb',sltt)
cls_noise = np.loadtxt(file_clnoise, unpack=True)[:dict_lrange['lmaxT']+1]
cls_totfg = np.loadtxt(file_clfg, unpack=True)[:dict_lrange['lmaxT']+1]
res       = cls_totfg + cls_noise
dict_cls  = add_clsdict(dict_cls,'res',res)

tlmbar = get_almbar(tlm,dict_cls,dict_lrange)
N1_tlmbar1 = get_almbar(N1_tlm1,dict_cls,dict_lrange)
N1_tlmbar2 = get_almbar(N1_tlm2,dict_cls,dict_lrange)

print('Saving output file')
np.savez(file_out.format(lminT=lminT,lmaxT=lmaxT,seed=seed,runname=runname,rectype=rectype), tlm=tlmbar)
np.savez(file_out_N1.format(cmbset=1,lminT=lminT,lmaxT=lmaxT,seed=seed,runname=runname,rectype=rectype), tlm=N1_tlmbar1)
np.savez(file_out_N1.format(cmbset=2,lminT=lminT,lmaxT=lmaxT,seed=seed,runname=runname,rectype=rectype), tlm=N1_tlmbar2)
