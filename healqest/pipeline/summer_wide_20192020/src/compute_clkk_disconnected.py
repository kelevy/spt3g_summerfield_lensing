import os,sys,yaml
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/')
import healqest_utils as hutils
import pathlib
import numpy as np
import healpy as hp
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='file_Yaml')
parser.add_argument('qe'       , default=None, type=str, help='qe')
parser.add_argument('i'        , default=None, type=int, help='index')
parser.add_argument('--N0'     , default=False, dest='N0'  ,action='store_true')
parser.add_argument('--RDN0'   , default=False, dest='RDN0',action='store_true')
parser.add_argument('--N1'     , default=False, dest='N1'  ,action='store_true')
args = parser.parse_args()

qe   = args.qe
i    = args.i
N0   = args.N0
N1   = args.N1
RDN0 = args.RDN0

#dir_base  = args.dir_base
#file_mask = args.file_mask

def spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx',comp='totfg',fgseed=0,gcmode='grad'):
    os.environ['HEALPIX'] = "/lcrc/project/SPT3G/users/ac.yomori/envs/analysis/Healpix_3.80/"
    spice='/lcrc/project/SPT3G/users/ac.yomori/packages/PolSpice_v03-07-03/bin/spice'

    u=i+1
    if   ktype=='xxxx': ii='%da'%i; jj='%da'%i; xx='%da'%i; yy='%da'%i
    elif ktype=='xyxy': ii='%da'%i; jj='%da'%u; xx='%da'%i; yy='%da'%u
    elif ktype=='xyyx': ii='%da'%i; jj='%da'%u; xx='%da'%u; yy='%da'%i
    elif ktype=='x0x0': ii='%da'%i; jj='%da'%0; xx='%da'%i; yy='%da'%0
    elif ktype=='x00x': ii='%da'%i; jj='%da'%0; xx='%da'%0; yy='%da'%i
    elif ktype=='0xx0': ii='%da'%0; jj='%da'%i; xx='%da'%i; yy='%da'%0
    elif ktype=='0x0x': ii='%da'%0; jj='%da'%i; xx='%da'%0; yy='%da'%i
    elif ktype=='abab': ii='%da'%i; jj='%db'%i; xx='%da'%i; yy='%db'%i
    elif ktype=='abba': ii='%da'%i; jj='%db'%i; xx='%db'%i; yy='%da'%i

    file_1 = dir_p+'/kmap%s_%d.fits'%(ktype[:2],i)
    file_2 = dir_p+'/kmap%s_%d.fits'%(ktype[2:],i)

    if gcmode=='grad'  : gcid='kk'
    elif gcmode=='curl': gcid='ww'
    else: sys

    subprocess.call([spice,'-mapfile'     , file_1,
                           '-weightfile'  , file_mask,
                           '-mapfile2'    , file_2,
                           '-weightfile2' , file_mask,
                           '-clfile'      , dir_cls+'cl%s_%s_%s_%s_%s_%s.dat'%(gcid,qid,ii,jj,xx,yy),
                           '-nlmax'       ,'4100',
                           '-apodizesigma','20',
                           '-thetamax'    ,'90',
                           '-subav'       ,'YES',
                           '-verbosity'   , 'NO',
                        ])

def get_klm(dir_base0,i,ktype='xx',qetype='TT',compname='all',fgseed=0,curl=True):

    l=np.arange(4101)

    print("qetype:%s"%qetype)

    if qetype=='MV':
        qes = ['TT','EE','EB','TE','TB','EB','TE','TB']

    elif qetype=='PP':
        qes = ['EE','EB']

    elif qetype=='TT' or qetype=='TE' or qetype=='TB' or qetype=='EB' or qetype=='EE':
        qes = [qetype]


    for qe in qes:
        print('using %s estimator'%qe)

     
    kmv    = 0
    respmv = 0

    for qe in qes:

        u=i+1
        if   ktype=='xx': ii='%da'%i; jj='%da'%i
        elif ktype=='xy': ii='%da'%u; jj='%da'%i
        elif ktype=='yx': ii='%da'%i; jj='%da'%u
        elif ktype=='x0': ii='%da'%i; jj='%da'%0
        elif ktype=='0x': ii='%da'%0; jj='%da'%i
        elif ktype=='ab': ii='%da'%i; jj='%db'%i
        elif ktype=='ba': ii='%db'%i; jj='%da'%i
        else: sys.exit('Undefined')
        i+=1

        dir_base=dir_base0+'/%s/'%qe
        print('------',qe)
        # Choose response
        print('N1' in dir_base)
        if 'N1' in dir_base:
            dir_resp=dir_base[:-4]
        else:
            dir_resp=dir_base
        print(dir_resp)
        resp = np.load(dir_resp+'/respavg%s.npz'%qe)['resp']
        resp[-100:]=np.inf

        if curl:
            gc='clm'
            mf = np.load(dir_base+'clmstack%s.npz'%(qe))
            mfname = 'cmf%s'%ktype
        else:
            gc='glm'
            mf = np.load(dir_base+'glmstack%s.npz'%(qe))
            mfname = 'gmf%s'%ktype


        # Load plm
        if ktype=='xx':
            if i==0:
                plm = np.load(dir_base+'plm%s_%s_%s.npz'%(qe,ii,jj))[gc]
                klm = hp.almxfl(plm-(mf[mfname])/(mf['nsim']),1/resp)
            else:
                plm = np.load(dir_base+'plm%s_%s_%s.npz'%(qe,ii,jj))[gc]
                klm  = hp.almxfl(plm-(mf[mfname]-plm)/(mf['nsim']-1.0),1/resp)

        elif ktype=='xy' or ktype=='yx' or ktype=='x0' or ktype =='0x' or  ktype=='ab' or ktype =='ba':
            if i>0:
                plm = np.load(dir_base+'plm%s_%s_%s.npz'%(qe,ii,jj))[gc]
                klm  = hp.almxfl(plm-(mf[mfname]-plm)/(mf['nsim']-1.0),1/resp)
            else:
                continue
        else: 
            print("specified ktype=%s"%ktype)
            sys.exit('aborting')


        kmv    += hp.almxfl(klm,resp)
        respmv += resp

    respmv = 1/(respmv)
    respmv[-100:]=np.inf

    kmv     = hp.almxfl(kmv,respmv)
    ell,emm = hp.Alm.getlm(4100)
    kmv[ell>4000]=0

    return kmv

print("Reading from yaml file: %s"%args.file_yaml)
config   = hutils.parse_yaml(args.file_yaml)

runname   = config['base']['runname']
rectype   = config['rec']['rectype']
dir_p     = config['pspec']['dir_tmp'] # temporary location to save maps
dir_cls   = config['pspec']['dir_cls'].format(runname=runname,rectype=rectype)
file_mask = config['pspec']['file_mask']
dir_base  = config['outputs']['dir_out'].format(runname=runname,rectype=rectype)
qid       = 'k%s'%qe.lower()

pathlib.Path(dir_cls).mkdir(parents=True, exist_ok=True)

if i==0:
    kxx = get_klm(dir_base,i,ktype='xx',qetype=qe); kmapxx = hp.alm2map(kxx,2048); hp.write_map(dir_p+'/kmapxx_%d.fits'%i,kmapxx,overwrite=True,dtype=np.float32)
    spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx')
    os.remove(dir_p+'/kmapxx_%d.fits'%i)
else:
    kxx = get_klm(dir_base,i,ktype='xx',qetype=qe); kmapxx = hp.alm2map(kxx,2048); hp.write_map(dir_p+'/kmapxx_%d.fits'%i,kmapxx,overwrite=True,dtype=np.float32)
    spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx')
    os.remove(dir_p+'/kmapxx_%d.fits'%i)

    if args.N0:
        kxy = get_klm(dir_base,i,ktype='xy',qetype=qe); kmapxy = hp.alm2map(kxy,2048); hp.write_map(dir_p+'/kmapxy_%d.fits'%i,kmapxy,overwrite=True,dtype=np.float32)
        kyx = get_klm(dir_base,i,ktype='yx',qetype=qe); kmapyx = hp.alm2map(kyx,2048); hp.write_map(dir_p+'/kmapyx_%d.fits'%i,kmapyx,overwrite=True,dtype=np.float32)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xyxy')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xyyx')
        os.remove(dir_p+'/kmapxy_%d.fits'%i)
        os.remove(dir_p+'/kmapyx_%d.fits'%i)
    
    if args.RDN0:
        kx0 = get_klm(dir_base,i,ktype='x0',qetype=qe); kmapx0 = hp.alm2map(kx0,2048); hp.write_map(dir_p+'/kmapx0_%d.fits'%i,kmapx0,overwrite=True,dtype=np.float32)
        k0x = get_klm(dir_base,i,ktype='0x',qetype=qe); kmap0x = hp.alm2map(k0x,2048); hp.write_map(dir_p+'/kmap0x_%d.fits'%i,kmap0x,overwrite=True,dtype=np.float32)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='x0x0')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='x00x')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='0xx0')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='0x0x')
        os.remove(dir_p+'/kmapx0_%d.fits'%i)
        os.remove(dir_p+'/kmap0x_%d.fits'%i)

    if args.N1:
        kxy = get_klm(dir_base,i,ktype='xy',qetype=qe); kmapxy = hp.alm2map(kxy,2048); hp.write_map(dir_p+'/kmapxy_%d.fits'%i,kmapxy,overwrite=True,dtype=np.float32)
        kyx = get_klm(dir_base,i,ktype='yx',qetype=qe); kmapyx = hp.alm2map(kyx,2048); hp.write_map(dir_p+'/kmapyx_%d.fits'%i,kmapyx,overwrite=True,dtype=np.float32)
        kab = get_klm(dir_base,i,ktype='ab',qetype=qe); kmapab = hp.alm2map(kab,2048); hp.write_map(dir_p+'/kmapab_%d.fits'%i,kmapab,overwrite=True,dtype=np.float32)
        kba = get_klm(dir_base,i,ktype='ba',qetype=qe); kmapba = hp.alm2map(kba,2048); hp.write_map(dir_p+'/kmapba_%d.fits'%i,kmapba,overwrite=True,dtype=np.float32)

        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xyxy')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xyyx')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='abab')
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='abba')
        os.remove(dir_p+'/kmapxy_%d.fits'%i)
        os.remove(dir_p+'/kmapyx_%d.fits'%i)
        os.remove(dir_p+'/kmapab_%d.fits'%i)
        os.remove(dir_p+'/kmapba_%d.fits'%i)
