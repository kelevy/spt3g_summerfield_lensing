'''
Compute cmbkappa auto-spectra using PolSpice or alm2cl.
'''
import os,sys,yaml
import pathlib
import numpy as np
import healpy as hp
import subprocess
import argparse
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../../healqest/src/')
import healqest_utils as hutils

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml' , default=None, type=str, help='file_Yaml')
parser.add_argument('qe'        , default=None, type=str, help='qe')
parser.add_argument('i'         , default=None, type=int, help='index')
parser.add_argument('dseed'     , default=None, type=int, help='index')
parser.add_argument('-src_yaml' , default=None, type=str, dest="src_yaml", help="src/prf yaml")
parser.add_argument('--xx'      , default=False, dest='xx'      ,action='store_true')
parser.add_argument('--N0'      , default=False, dest='N0'      ,action='store_true')
parser.add_argument('--RDN0'    , default=False, dest='RDN0'    ,action='store_true')
parser.add_argument('--curlRDN0', default=False, dest='curlRDN0',action='store_true')
parser.add_argument('--N1'      , default=False, dest='N1'      ,action='store_true')
parser.add_argument('--MF'      , default=False, dest='MF'      ,action='store_true')
parser.add_argument('--mfsplit' , default=False, dest='mfsplit' ,action='store_true')
parser.add_argument('--mfsubxy' , default=False, dest='mfsubxy' ,action='store_true')
parser.add_argument('--mfsubxd' , default=False, dest='mfsubxd' ,action='store_true')
parser.add_argument('--mfsubab' , default=False, dest='mfsubab' ,action='store_true')
parser.add_argument('--curl'    , default=False, dest='curl'    ,action='store_true')
parser.add_argument('--alm2cl'  , default=False, dest='alm2cl'  ,action='store_true')


args     = parser.parse_args()
qe       = args.qe
i        = args.i
xx       = args.xx
N0       = args.N0
N1       = args.N1
RDN0     = args.RDN0
curlRDN0 = args.curlRDN0
MF       = args.MF
dseed    = args.dseed
mfsplit  = args.mfsplit
mfsubxy  = args.mfsubxy
mfsubxd  = args.mfsubxd
mfsubab  = args.mfsubab
curl     = args.curl
alm2cl   = args.alm2cl
mfsubxx  = False # always true

config0      = hutils.load_yaml(args.file_yaml) 
dir_healqest = config0['base']['dir_healqest']


def spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='xxxx',comp='totfg',fgseed=0,gcmode='g',alm2cl=False,nlmax=4100,apodizesigma=30,thetamax=31,apodizetype=0,subav='YES',maskfac=1):

    print('-----------------------------------------------------')
    os.environ['HEALPIX'] = config['base']['dir_healpix']
    spice = config['base']['dir_spice']
    
    t=dseed
    u=i+1
    if   ktype=='xxxx': ii='%da'%i; jj='%da'%i; xx='%da'%i; yy='%da'%i
    elif ktype=='xyxy': ii='%da'%i; jj='%da'%u; xx='%da'%i; yy='%da'%u
    elif ktype=='xyyx': ii='%da'%i; jj='%da'%u; xx='%da'%u; yy='%da'%i
    elif ktype=='xdxd': ii='%da'%i; jj='%da'%t; xx='%da'%i; yy='%da'%t
    elif ktype=='xddx': ii='%da'%i; jj='%da'%t; xx='%da'%t; yy='%da'%i
    elif ktype=='dxxd': ii='%da'%t; jj='%da'%i; xx='%da'%i; yy='%da'%t
    elif ktype=='dxdx': ii='%da'%t; jj='%da'%i; xx='%da'%t; yy='%da'%i
    elif ktype=='wdwd': ii='%db'%i; jj='%da'%t; xx='%db'%i; yy='%da'%t
    elif ktype=='wddw': ii='%db'%i; jj='%da'%t; xx='%da'%t; yy='%db'%i
    elif ktype=='dwwd': ii='%da'%t; jj='%db'%i; xx='%db'%i; yy='%da'%t
    elif ktype=='dwdw': ii='%da'%t; jj='%db'%i; xx='%da'%t; yy='%db'%i
    elif ktype=='abab': ii='%da'%i; jj='%db'%i; xx='%da'%i; yy='%db'%i
    elif ktype=='abba': ii='%da'%i; jj='%db'%i; xx='%db'%i; yy='%da'%i
    elif ktype=='mfmf': ii='mf'; jj='mf'; xx='mf'; yy='mf'
    
    file_1 = dir_p+'/kmap%s_%d_1.fits'%(ktype[:2],i)
    file_2 = dir_p+'/kmap%s_%d_2.fits'%(ktype[2:],i)

    if gcmode=='g'  : gcid='kk'
    elif gcmode=='c': gcid='ww'
    else: sys.exit('unknown gcmode')

    print('Applying mask: %s'%file_mask)

    if alm2cl:
        map1=hp.read_map(file_1)
        map2=hp.read_map(file_2)
        mask=hp.read_map(file_mask)
        alm1=hp.map2alm(map1*mask,lmax=4100,use_pixel_weights=True)
        alm2=hp.map2alm(map2*mask,lmax=4100,use_pixel_weights=True)
        cls =hp.alm2cl(alm1,alm2)*mask.shape[0]/np.sum(mask**2)
        tmp = np.c_[np.arange(4101),cls].T
        np.save(dir_cls+'cl%s_%s_%s_%s_%s_%s.npy'%(gcid,qid,ii,jj,xx,yy),tmp.T)
        
    else:
        subprocess.call([spice,'-mapfile'     , file_1,
                               '-weightfile'  , file_mask,
                               '-mapfile2'    , file_2,
                               '-weightfile2' , file_mask,
                               '-clfile'      , dir_cls+'cl%s_%s_%s_%s_%s_%s.dat'%(gcid,qid,ii,jj,xx,yy),
                               '-nlmax'       , str(nlmax),
                               '-apodizesigma', str(apodizesigma),
                               '-thetamax'    , str(thetamax),
                               '-subav'       , str(subav),
                               '-apodizetype' , str(apodizetype),
                               '-verbosity'   , 'NO',
                        ])
    
        tmp = np.loadtxt(dir_cls+'cl%s_%s_%s_%s_%s_%s.dat'%(gcid,qid,ii,jj,xx,yy),unpack=True)
        print("Applying mask correction factor: {maskfac}")
        tmp[:,1] = tmp[:,1]*maskfac  
        np.save(dir_cls+'cl%s_%s_%s_%s_%s_%s.npy'%(gcid,qid,ii,jj,xx,yy),tmp.T)
        os.remove(dir_cls+'cl%s_%s_%s_%s_%s_%s.dat'%(gcid,qid,ii,jj,xx,yy))

    print('Saved Cls to: %s'%(dir_cls+'cl%s_%s_%s_%s_%s_%s.npy'%(gcid,qid,ii,jj,xx,yy))  )


def process_mf(seed,plm,plmstack,split,ktype,gcmode):
    idx0      = np.arange(1,plmstack['nsim']+1)
    idx1,idx2 = np.split(idx0, 2)
    
    if split==0:
        print('using split0')
        if seed in idx0: 
            mf= (plmstack[f'{gcmode}mf{ktype}']-plm)/(plmstack['nsim']-1.0)
        else:
            mf= (plmstack[f'{gcmode}mf{ktype}'])/(plmstack['nsim']) # mainly for data
        
    elif split==1:
        print('using split1')
        if seed in idx1: 
            mf= (plmstack[f'{gcmode}mf{ktype}_half1']-plm)/(plmstack['nsim_half']-1.0)
        else:
            mf= (plmstack[f'{gcmode}mf{ktype}_half1'])/(plmstack['nsim_half'])
        
    elif split==2:
        print('using split2')
        if seed in idx2: 
            mf= (plmstack[f'{gcmode}mf{ktype}_half2']-plm)/(plmstack['nsim_half']-1.0)
        else:
            mf= (plmstack[f'{gcmode}mf{ktype}_half2'])/(plmstack['nsim_half'])
              
    return mf


def get_klm(dir_base0,seed,maskb,ktype='xx',qetype='TT',dseed=0,dir_p='./',mfsplit=0,mfsub=True,gcmode='g',mapnum=1):

    
    lmax=config['lensrec']['Lmax'] #4100
    ell,emm = hp.Alm.getlm(lmax)

    l=np.arange(lmax+1)

    if ktype=='mf' and seed!=0:
        sys.exit('ktype cannot be mf and i!=0')

    z=seed
    u=seed+1
    t=dseed
    
    if   ktype=='xx': ii='%da'%z; jj='%da'%z; kk='%da'%z; 
    elif ktype=='xy': ii='%da'%z; jj='%da'%u
    elif ktype=='yx': ii='%da'%u; jj='%da'%z
    elif ktype=='xd': ii='%da'%z; jj='%da'%t
    elif ktype=='dx': ii='%da'%t; jj='%da'%z
    elif ktype=='wd': ii='%db'%z; jj='%da'%t
    elif ktype=='dw': ii='%da'%t; jj='%db'%z
    elif ktype=='ab': ii='%da'%z; jj='%db'%z
    elif ktype=='ba': ii='%db'%z; jj='%da'%z
    elif ktype=='mf': ii='%da'%t; jj='%da'%t
    else: sys.exit('Undefined')
    print('-----------------------------------------------------')
    print("qetype: %s"%qetype)
    print("ktype : %s"%ktype)

    if (qetype=='GMV' or qetype=='GMVTTEETE' or  qetype=='GMVTBEB' or
        qetype=='GMVbhTTprf' or qetype=='GMVTTEETEbhTTprf'):
        qes = [qetype]
      
    elif qetype=='MV':
        qes = ['TT','EE','EB','TE','TB','EB','TE','TB']

    elif qetype=='PP':
        qes = ['EE','EB','BE']

    elif qetype=='TT' or qetype=='EE':
        qes = [qetype]

    elif qetype=='TEET':
        qes = ['TE', 'ET']
    
    elif qetype=='EBBE':
        qes = ['EB', 'BE']

    elif qetype=='TBBT':
        qes = ['TB', 'BT']
        
    for qe in qes:
        print('using %s estimator'%qe)

     
    kmv    = 0
    respmv = 0

    for qe in qes:

        dir_base=dir_base0+'/%s/'%qe if args.src_yaml is None else dir_base0

        #dir_base=dir_base0[:-1]+'_noinpaint'+'/%s/'%qe if args.src_yaml is None else dir_base0


        #file_resp = dir_base+"/"+"respavg%s_masked_spice.npz"%qe
        file_resp = dir_base0[:-1]+'_noinpaint/%s/respavg%s.npz'%(qe,qe)
        #file_resp = dir_base+"/"+"resp2d%s.npz"%qe
        #file_resp = '/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/lensrec/sqe081024/sqe/autotf_v2_lmint350_lminp350_lmaxt3500_lmaxp4000_mmin100_aggcinv_noinpaint/%s/respavg%s.npz'%(qe,qe)
        #file_resp  = '/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/lensrec/sqe081024/sqe/autotf_v2_lmint350_lminp350_lmaxt3500_lmaxp3500_mmin220_maskedprerecinnermask/%s/respavg%s.npz'%(qe,qe)
        print("respfile:%s"%file_resp)
        
        print("Loading resp:",file_resp)
        resp        = np.load(file_resp)['resp']
        resp[-100:] = np.inf
        resp[:10]   = np.inf
        resp[resp==0]=np.inf
        resp        = resp[ell] # make it a 2d resp
        
        if gcmode=='c':
            print('Using curl response')
            #resp = np.load(dir_base0+f"/SAN0/resp_{qe}_avg.npy").real
            resp = np.load(dir_base0+f"/SAN0/aresp_{qe}.npy").real
            resp = resp[ell]


        # Load plm
        file_plm      = dir_base+'plm%s_%s_%s.npz'%(qe,ii,jj)

        if ktype=='xd' or ktype=='dx':
            file_plmstack = dir_base+f'{gcmode}lmstack{qe}_{ktype}_dataidx{dseed}_unl.npz'
        else:
            file_plmstack = dir_base+f'{gcmode}lmstack{qe}_{ktype}_unl.npz'

        print(f"Loading plm     : {file_plm}" )
        print(f"Loading plmstack: {file_plmstack}" )
        print(f"Using gcmode    : {gcmode}" )
        
        plm      = np.load(dir_base+'plm%s_%s_%s.npz'%(qe,ii,jj))[f'{gcmode}lm']
            
        if mfsub:
            print('Subtracting MF: ',file_plmstack )
            plmstack = np.load(file_plmstack)
            mf       = process_mf(seed,plm,plmstack,mfsplit,ktype,gcmode)
            klm = (plm-mf)/resp
            #np.save('klm.npy',klm)
            #np.save('plm.npy',plm)
            #np.save('mf.npy',mf)
            #np.save('resp.npy',resp)
            
        else:
            print('Not subtracting MF')
            klm = (plm)/resp

        if ktype=='mf':
            plmstack = np.load(dir_base+f'{gcmode}lmstack{qe}_xx.npz')
            mf       = process_mf(seed,plm,plmstack,mfsplit,'xx',gcmode) # using the xx mf
            klm = (mf)/resp

        resp0   = np.copy(resp)
        resp0[resp==np.inf]=0

        kmv     += klm*resp0
        respmv  += resp
        

    #print('------',respmv)

    respmv = 1/(respmv)
    kmv    = kmv*respmv
    
    ell,emm = hp.Alm.getlm(lmax)
    kmv[ell>4000]=0
    kmv[ell<5]=0
    
    kmap  = hp.alm2map(kmv,2048)
    
    hp.write_map(dir_p+f'/kmap{ktype}_{seed}_{mapnum}.fits',kmap,overwrite=True,dtype=np.float32)
    
    print("Creating:",dir_p+f'/kmap{ktype}_{seed}_{mapnum}.fits')



print("Reading from yaml file: %s"%args.file_yaml)
config    = hutils.load_yaml(args.file_yaml)

#specname  = config['base']['specname']
runname    = config['base']['runname']
rectype    = config['lensrec']['rectype']
dir_p      = config['pspec']['dir_tmp'] 
file_mask  = config['pspec']['file_mask']
file_bmask = config['pspec']['bmask']

apodizetype  = config['pspec']['apodizetype']
apodizesigma = config['pspec']['apodizesigma'] 
thetamax     = config['pspec']['thetamax']
subav        = config['pspec']['subav']
nlmax        = config['pspec']['nlmax']
qid          = 'k%s'%qe.lower()

# Read ell ranges (mainly for naming)
lminT      = config['lensrec']['lminT']
lminP      = config['lensrec']['lminP']
lmaxT      = config['lensrec']['lmaxT']
lmaxP      = config['lensrec']['lmaxP']
mmin       = config['lensrec']['mmin']

# Naming
suffix    = 'clkk_lranget_%d_%d_lrangep_%d_%d'%(lminT,lmaxT,lminP,lmaxP)

if mfsplit  : suffix+='_mfsplit'
else        : suffix+='_nomfsplit'
if mfsubxy  : suffix+='_mfsubxy'
else        : suffix+='_nomfsubxy'
if mfsubab  : suffix+='_mfsubab'
else        : suffix+='_nomfsubab'
if mfsubxd  : suffix+='_mfsubxd'
else        : suffix+='_nomfsubxd'

if mfsplit  :
    split1 = 1 # use half1
    split2 = 2 # use half2
else:
    split1 = 0 # use full sample
    split2 = 0 # use full sample
    

if curl:
    gcmode='c'
else:
    gcmode='g'

print(f'Using mask: {file_mask}')


if args.src_yaml is None:
    dir_base  = config['outputs']['dir_out'].format(rectype=rectype,runname=runname,lmaxT=lmaxT,lminT=lminT,lminP=lminP,lmaxP=lmaxP,mmin=mmin)
    #dir_base  = config['outputs']['dir_out'].format(rectype=rectype,runname=runname,lmaxT=lmaxT,lmin=lminT,lmaxP=lmaxP,mmin=mmin)
    algo='polspice'
    if alm2cl:
        algo='alm2cl'
    dir_cls   = dir_base+f'/resp{algo}_unmask_innermask_nomfsubxx/'
else:
    config_src = yaml.safe_load(open(args.src_yaml))
    dir_base  = config_src['outputs']['dir_out']
    dir_cls   = dir_base+"clkk/"
    dir_p     = config_src['pspec']['dir_tmp'] 

maskb = hp.read_map(file_bmask)
maskb[maskb==0]=np.inf

# Mask correction factor --- this comes from the fact that input T/E/B maps have a boundary mask applied
# but when we compute the spectra from PolSpice, only the analysis mask (boundary+ptsrc) is considered in the
# correction.
mb      = hp.read_map(file_bmask)
ma      = hp.read_map(file_mask)
nrm1    = hp.nside2npix(2048)/np.sum(mb**2)
nrm2    = hp.nside2npix(2048)/np.sum(ma**4*mb**2)
maskfac = nrm2/nrm1
print("Mark correction factor: {maskfac}")

pathlib.Path(dir_cls).mkdir(parents=True, exist_ok=True)
pathlib.Path(dir_p).mkdir(parents=True, exist_ok=True)

if i>=0:
    if args.xx: 
        get_klm(dir_base,i,maskb,ktype='xx',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubxx,gcmode=gcmode,mapnum=1)
        get_klm(dir_base,i,maskb,ktype='xx',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxx,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='xxxx',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac) # xx1 xx2
        
        if i<6:
            ktype='xx'
            print('Copying:', dir_p+'/kmap%s_%d_1.fits'%(ktype,i), ' to ',dir_base+'/kmap%s_%d_1.fits'%(ktype,i))
            if curl:
                shutil.copyfile(dir_p+'/kmap%s_%d_1.fits'%(ktype,i), dir_base+'/wmapxx_k%s_%d_1.fits'%(qe,i))
                shutil.copyfile(dir_p+'/kmap%s_%d_2.fits'%(ktype,i), dir_base+'/wmapxx_k%s_%d_2.fits'%(qe,i))
            else:
                dir_base='/lcrc/globalscratch/ac.yomori/'
                shutil.copyfile(dir_p+'/kmap%s_%d_1.fits'%(ktype,i), dir_base+'/kmapxx_k%s_%d_1.fits'%(qe,i))
                #shutil.copyfile(dir_p+'/kmap%s_%d_2.fits'%(ktype,i), dir_base+'/kmapxx_k%s_%d_2.fits'%(qe,i))

        os.remove(dir_p+'/kmapxx_%d_1.fits'%i)
        os.remove(dir_p+'/kmapxx_%d_2.fits'%i)

    if args.N0:#
        get_klm(dir_base,i,maskb,ktype='xy',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubxy,gcmode=gcmode,mapnum=1)
        get_klm(dir_base,i,maskb,ktype='xy',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxy,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='xyxy',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# xy1 xy2
        
        get_klm(dir_base,i,maskb,ktype='yx',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxy,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='xyyx',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# xy1 yx2
        os.remove(dir_p+'/kmapxy_%d_1.fits'%i)
        os.remove(dir_p+'/kmapxy_%d_2.fits'%i)
        os.remove(dir_p+'/kmapyx_%d_2.fits'%i)
    
    
    if args.RDN0:
        # Compute power spectra required for RDN0 (xdxd + xddx + dxdx + dxxd)
        get_klm(dir_base,i,maskb,ktype='xd',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubxd,gcmode=gcmode,mapnum=1)
        get_klm(dir_base,i,maskb,ktype='xd',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxd,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='xdxd',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# xd1 xd2
        
        get_klm(dir_base,i,maskb,ktype='dx',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxd,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='xddx',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# xd1 dx2
        
        get_klm(dir_base,i,maskb,ktype='dx',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubxd,gcmode=gcmode,mapnum=1)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='dxxd',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# dx1 xd2
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='dxdx',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# dx1 dx2

        os.remove(dir_p+'/kmapxd_%d_1.fits'%i)
        os.remove(dir_p+'/kmapxd_%d_2.fits'%i)
        os.remove(dir_p+'/kmapdx_%d_1.fits'%i)
        os.remove(dir_p+'/kmapdx_%d_2.fits'%i)

    if args.curlRDN0:
        # Compute RDN0 for curl
        get_klm(dir_base,i,maskb,ktype='wd',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubxd,gcmode=gcmode,mapnum=1)
        get_klm(dir_base,i,maskb,ktype='wd',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxd,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='wdwd',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# xd1 xd2
        
        get_klm(dir_base,i,maskb,ktype='dw',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubxd,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='wddw',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# xd1 dx2
        
        get_klm(dir_base,i,maskb,ktype='dw',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubxd,gcmode=gcmode,mapnum=1)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='dwwd',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# dx1 xd2
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='dwdw',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# dx1 dx2

        os.remove(dir_p+'/kmapwd_%d_1.fits'%i)
        os.remove(dir_p+'/kmapwd_%d_2.fits'%i)
        os.remove(dir_p+'/kmapdw_%d_1.fits'%i)
        os.remove(dir_p+'/kmapdw_%d_2.fits'%i)

    if args.N1:
        # Compute power spectra required for N1 (abab + abba)
        get_klm(dir_base,i,maskb,ktype='ab',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubab,gcmode=gcmode,mapnum=1)
        get_klm(dir_base,i,maskb,ktype='ab',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubab,gcmode=gcmode,mapnum=2)
        get_klm(dir_base,i,maskb,ktype='ba',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubab,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='abab',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# ab1 ab2
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='abba',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)# ab1 ba2
        os.remove(dir_p+'/kmapab_%d_1.fits'%i)
        os.remove(dir_p+'/kmapab_%d_2.fits'%i)
        os.remove(dir_p+'/kmapba_%d_2.fits'%i)

    if args.MF:
        # Compute power spectra of meanfield
        get_klm(dir_base,i,maskb,ktype='mf',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split1,mfsub=mfsubab,gcmode=gcmode,mapnum=1)
        get_klm(dir_base,i,maskb,ktype='mf',qetype=qe,dseed=dseed,dir_p=dir_p,mfsplit=split2,mfsub=mfsubab,gcmode=gcmode,mapnum=2)
        spice(qid,dir_cls,file_mask,dir_p,i,dseed,ktype='mfmf',gcmode=gcmode,alm2cl=alm2cl,nlmax=nlmax,apodizesigma=apodizesigma,thetamax=thetamax,apodizetype=apodizetype,subav=subav,maskfac=maskfac)
