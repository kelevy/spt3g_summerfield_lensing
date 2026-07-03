import os,sys
import numpy as np
import healpy as hp
import logging as lg
from pathlib import Path

class RelativeSeconds(lg.Formatter):
    def format(self, record):
        nhrs  = record.relativeCreated//(1000*60*60)
        nmins = record.relativeCreated//(1000*60)-nhrs*60
        nsecs = record.relativeCreated//(1000)-nmins*60
        record.relativeCreated = "%02d:%02d:%02d"%(nhrs,nmins,nsecs)#, record.relativeCreated//(1000) )
        #print( dtype(record.relativeCreated//(1000)) )
        return super(RelativeSeconds, self).format(record)

def setup_logger(nolog=False,file_log='test.log'):

    if nolog==True:
        print("printing to stdout")
        lg.basicConfig(level = lg.WARNING)
        formatter = RelativeSeconds("[%(relativeCreated)s]  %(message)s")
        lg.root.handlers[0].setFormatter(formatter)

    else:
        dir_log = str(Path(file_log).parent)
        Path(dir_log).mkdir(parents=True, exist_ok=True)
        print('saving log to: %s'%dir_log)
        lg.basicConfig(filename=file_log, filemode = 'w+', level = lg.WARNING)
        formatter = RelativeSeconds("[%(relativeCreated)s]  %(message)s")
        lg.root.handlers[0].setFormatter(formatter)



def reduce_lmax(alm, lmax=4000):
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

def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl

def load_cambcls(file,lmax=2000,dict=False,dls=False):
    d = np.loadtxt(file)
    ell,sltt,slee,slbb,slte = d[:,(0,1,2,3,4)].T

    if dls==False:
        # Removing the ell factors and padding with zeros (since the file starts with l=2)
        sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)
        slee=slee/ell/(ell+1)*2*np.pi; slee=zeropad(slee)
        slte=slte/ell/(ell+1)*2*np.pi; slte=zeropad(slte)
        slbb=slbb/ell/(ell+1)*2*np.pi; slbb=zeropad(slbb)
        ell  = np.insert(ell,0,1); ell=np.insert(ell,0,0)
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

def load_bl(config,lmax=-1):
    file_bl = config['data']['beam']
    lg.warning('Loading beam file: %s'%file_bl)
    tmp  = np.loadtxt(file_bl)
    bl   = {}
    if lmax!=-1:
        bl[90],bl[150],bl[220] = tmp[:lmax+1,1], tmp[:lmax+1,2],  tmp[:lmax+1,3]
    else:
        bl[90],bl[150],bl[220] = tmp[:,1], tmp[:,2],  tmp[:,3]
    return bl

def load_tf(config,fill_value=0,lmax=4000,include_beam=True,freq=None):

    file_tf = config['data']['tf2d']

    lg.warning('Loading tf file: %s'%file_tf)

    if include_beam:
        print('Including beam in the transfer function')
        bl = load_bl(config,lmax=lmax)

    tf = {}
    tf['1d'] = {}
    tf['2d'] = {}

    if freq==None:
        freqs=[90,150,220]
    else:
        freqs=[freq]

    for freqi in (freqs):
        y = np.load(file_tf.format(freq=freqi))['tf2d'].real
        y[np.isnan(y)] = fill_value
        y    = reduce_lmax(y,lmax=lmax)

        if include_beam:
            y = hp.almxfl(y,bl[freqi])

        tf1d = np.sqrt(hp.alm2cl(y))
        tf2d = y.real
        tf['1d'][freqi] = tf1d
        tf['2d'][freqi] = tf2d
    return tf


def get_mmask(mmin,lmax=6000):
    '''Generate a 2d almspace mask'''
    ell,emm = hp.Alm.getlm(lmax)
    w       = np.ones_like(ell,dtype=np.complex_)
    w[emm<mmin]=0
    return w

def load_mask(nsides):
    mask_dict = {}
    
    for ns in nsides:
        file_mask='/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/masks/mask%d_border_apod_mask_threshold0.1_allghz_dense.fits'%ns
        lg.warning('Loading mask: %s'%file_mask )    
        mask = hp.read_map(file_mask)
        mask_dict[ns]=mask
    
    return mask_dict

'''
def load_nstack(config,freq,key,lmax,deconvtf=False):
    if type(freq)==int:
        file_stack = config['data']['nlmstack'].format(freq=freq)
        lg.warning('loading nlmstack: %s'%file_stack)
        n   = np.load(file_stack)
        nlm = reduce_lmax(n[key]/n['nsims']*n['nrm'],lmax=lmax) 
    
    elif type(freq)==str:
        dir = str(Path(config['simulations']['gauss']['mockdata']).parent)+'/ilc/'
        file_stack = dir+'nlm2_'+config['ilc']['cmbmv'].format(seed=0,depth='full')
        lg.warning('loading nlmstack: %s'%file_stack)
        n   = hp.read_alm(file_stack,hdu=[1,2,3])
        if key == 'nlmTT': idx=1
        if key == 'nlmEE': idx=2
        nlm = reduce_lmax(n[idx],lmax=lmax) 

    if deconvtf:
        tf = load_tf(config,fill_value=np.inf,lmax=lmax,include_beam=True,freq=freq)
    else:
        tf       = {}
        tf['2d'] = {freq: np.ones_like(nlm)}

    return nlm/tf['2d'][freq]**2


def load_fgstack(config,freq1,freq2,key,lmax,deconvtf=False):

    if type(freq1)==int and type(freq2)==int:
        file_stack = config['data']['fgstack'].format(freq1=freq1,freq2=freq2)
        lg.warning('loading fgstack: %s'%file_stack)
        d    = np.load(file_stack)
        alm  = reduce_lmax(d[key]/1000,lmax=lmax) * d['nrm']

    elif type(freq1)==str and type(freq2)==str:
        dir = str(Path(config['simulations']['gauss']['mockdata']).parent)+'/ilc/'
        file_stack = dir+'fglm2_'+config['ilc']['cmbmv'].format(seed=0,depth='full')
        lg.warning('loading fglmstack: %s'%file_stack)
        n   = hp.read_alm(file_stack,hdu=[1,2,3])
        if key == 'almTT': idx=1
        if key == 'almEE': idx=2
        alm = reduce_lmax(n[idx],lmax=lmax) 
        
    else:
        sys.exit()

    if deconvtf:
        tf       = {}
        tf['2d'] = {freq: np.ones_like(alm) for freq in [90,150,220]}
    else:
        tf = load_tf(config,fill_value=0,lmax=lmax,include_beam=True)

    return alm*tf['2d'][freq1]*tf['2d'][freq2]


def load_cmbfgstack(config,freq1,freq2,key,lmax,deconvtf=False):

    if type(freq)==int:
        file_stack = config['data']['cmbfgstack'].format(freq1=freq1,freq2=freq2)
        lg.warning('loading cmbfgstack: %s'%file_stack)
        d    = np.load(file_stack)
        alm  = reduce_lmax(d[key]/1000,lmax=lmax) * d['nrm']
    
        # single frequency stacks are tf+beam convolved  
        if deconvtf:
            tf       = {}
            tf['2d'] = {freq: np.ones_like(alm) for freq in [90,150,220]}
        else:
            tf = load_tf(config,fill_value=0,lmax=lmax,include_beam=True,freq=freq)

        alm = alm * tf['2d'][freq1]*tf['2d'][freq2]
        

    elif type(freq1)==str and type(freq2)==str:
        dir = str(Path(config['simulations']['gauss']['mockdata']).parent)+'/ilc/'
        file_stack = dir+'cmbfglm2_'+config['ilc']['cmbmv'].format(seed=0,depth='full')
        lg.warning('loading cmbfglmstack: %s'%file_stack)
        n   = hp.read_alm(file_stack,hdu=[1,2,3])
        if key == 'almTT': idx=1
        if key == 'almEE': idx=2
        alm = reduce_lmax(n[idx],lmax=lmax) 

        if deconvtf:
            tf       = {}
            tf['2d'] = {freq: np.ones_like(alm) for freq in [90,150,220]}
        else:
            tf = load_tf(config,fill_value=0,lmax=lmax,include_beam=True,freq=freq)

        alm = alm * tf['2d'][freq1]*tf['2d'][freq2]
        
    else:
        sys.exit()


    return alm
'''


def load_nlmstack(config,freq,key,lmax,deconvtf=False):
    file_stack = config['data']['nlmstack'].format(freq=freq)
    lg.warning('loading nlmstack: %s'%file_stack)
    n    = np.load(file_stack)
    nlm  = reduce_lmax(n[key]/n['nsims'],lmax=lmax) * n['nrm']

    # nlmstacks are naturally tbfl convolved
    if deconvtf:
        tf = load_tf(config,fill_value=np.inf,lmax=lmax,include_beam=True,freq=freq)
    else:
        tf       = {}
        tf['2d'] = {freq: np.ones_like(nlm)}

    return nlm/tf['2d'][freq]**2