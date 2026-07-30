from tqdm import tqdm 
import numpy as np

def rebincl(ell,cl, bb):
    #bb   = np.linspace(minell,maxell,Nbins+1)
    Nbins=len(bb)-1
    ll   = (bb[:-1]).astype(np.int_)
    uu   = (bb[1:]).astype(np.int_)
    ret  = np.zeros(Nbins)
    retl = np.zeros(Nbins)
    err  = np.zeros(Nbins)
    for i in range(0,Nbins):
        idx     = np.where((ell>ll[i]) & (ell<=uu[i]))[0]
        ret[i]  = np.mean(cl[idx])
        retl[i] = np.mean(ell[idx])
        err[i]  = np.std(cl[idx])
    return retl,ret
    
def loadcls(dir,nsims,cltype,N0=None,lmin=0,lmax=4000,curl=False,R=1,qe='gmv'):

    if curl:
        spec='ww'
    else:
        spec='kk'

    lmin=np.int32(lmin)
    lmax=np.int32(lmax)
    
    lmask = np.ones(4101)
    lmask[:lmin]=np.nan
    lmask[lmax+1:]=np.nan
    

    if cltype=='xx':
        
        xx=0;c=0
        for i in tqdm(range(1,nsims+1)):
            #i=500-i
            x=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,i,i,i))[:,1]*R
            xx+=x
            #rl,rcl      = rebincl(np.arange(4101),x-N0-N1,bb)
            #arr[:,i-1]  = rcl
            #arr2[:,i-1] = (x-N0-N1)/tlkk[:4101]
            c+=1
        xx/=c
        return xx#*lmask


    if cltype=='dd':
        
        dd=0;c=0
        for i in tqdm(range(0,1)):
            #i=500-i
            x=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,i,i,i))[:,1]*R
            dd+=x
            #rl,rcl      = rebincl(np.arange(4101),x-N0-N1,bb)
            #arr[:,i-1]  = rcl
            #arr2[:,i-1] = (x-N0-N1)/tlkk[:4101]
            c+=1
        dd/=c
        return dd#*lmask

    

    if cltype=='N0':
        N0=0;c=0
        for i in tqdm(range(1,nsims+1)):
            #i=500-i
            a=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,i+1,i,i+1))[:,1]*R
            b=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,i+1,i+1,i))[:,1]*R
            N0+=(a+b)
            c+=1
        N0/=c
        return N0#*lmask
        
    if cltype=='N1':
        assert N0 is not None
        N1=0;c=0
        for i in tqdm(range(1,nsims+1)):
            #i=250-i
            abab = np.load(dir+'cl%s_k%s_%da_%db_%da_%db.npy'%(spec,qe,i,i,i,i))[:,1]*R
            abba = np.load(dir+'cl%s_k%s_%da_%db_%db_%da.npy'%(spec,qe,i,i,i,i))[:,1]*R
            N1+=(abab+abba)-N0
            c+=1
        N1/=c
        return N1#*lmask

    if cltype=='RDN0':
        assert N0 is not None
        RDN0=0;c=0
        for i in tqdm(range(1,nsims+1)):
            #i=500-i
            xdxd=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,didx,i,didx))[:,1]*R
            xddx=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,didx,didx,i))[:,1]*R
            dxdx=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,didx,i,didx,i))[:,1]*R
            dxxd=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,didx,i,i,didx))[:,1]*R
            RDN0+=(xdxd+xddx+dxdx+dxxd)-N0
            c+=1
        RDN0/=c
        return RDN0#*lmask

def get_bpwf(bine,nsims,N0,N1,RDN0=None,ellfac=1,ratio=False,curl=False,R=1,qe='gmv'):

    if curl:
        spec='ww'
    else:
        spec='kk'
        
    import pickle
    l  =np.arange(4001)
    arr=np.zeros((4001,nsims))
    xx=0; c=0

    for i in tqdm(range(1,nsims+1)):
        x=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,i,i,i))[:4001,1]*R
        arr[:,i-1]  = l**(ellfac)*( x-N0[:4001]-N1[:4001])
        
    v    = np.var(arr,axis=1)
    bpwf = np.zeros((4001,len(bine)-1))
    
    for i in range(len(bine)-1):
        #print(i)
        bi=np.int32(bine[i])
        bf=np.int32(bine[i+1])
        
        sumv=np.sum(1/v[bi:bf])    
        bpwf[bi:bf,i]=(1/v[bi:bf])/sumv
        #print((1/v[bi:bf])/sumv)
    return bpwf



def get_dvec(bine,nsims,N0,N1,RDN0=None,ellfac=1,ratio=False,curl=False,R=1,bpwf=None,qe='gmv'):

    if curl:
        spec='ww'
    else:
        spec='kk'
        
    import pickle
    l  =np.arange(4001)
    arr=np.zeros((len(bine)-1,nsims))
    xx=0; c=0

    #with open('/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/midell/sims/lensed_cmb/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_rawCls.pickle', 'rb') as handle:
    with open('/sdf/home/w/wlwu/repos/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_rawCls.pickle', 'rb') as handle:
        clsa = pickle.load(handle)

    ell  = np.arange(4001)
    tlkk = (ell*(ell+1)/2)**2* clsa['lens_potential'][:4001,0]
    tlkk[:2]=np.inf
    
    l = np.arange(4001)
    t = lambda l: (l*(l+1))**2/4

    rl,_      = rebincl(l,l,bine)

    for i in tqdm(range(1,nsims+1)):
        #i=500-i
        x=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,i,i,i,i))[:4001,1]*R
        if ratio:
            #import pdb;pdb.set_trace()
            if bpwf is None:
                rl,rcl = rebincl(l[:4001],(x[:4001]-N0[:4001]-N1[:4001])/tlkk[:4001],bine)
            else:
                rcl      = ((x[:4001]-N0[:4001]-N1[:4001])/tlkk[:4001])@bpwf#rebincl(l,(x-N0-N1)/tlkk,bine)
            
        else:
            if bpwf is None:
                rl,rcl       = rebincl(l[:4001],l[:4001]**(ellfac)*(x[:4001]-N0[:4001]-N1[:4001]),bine)
            else:
                rcl      = (l[:4001]**(ellfac)*(x[:4001]-N0[:4001]-N1[:4001]))[:4001]@bpwf #rebincl(l,l**(ellfac)*(x-N0-N1),bine)
            #import pdb;pdb.set_trace()
        arr[:,c]  = rcl
        c+=1

    if RDN0 is None:
        RDN0=N0

    if didx is not None:
        x=np.load(dir+'cl%s_k%s_%da_%da_%da_%da.npy'%(spec,qe,didx,didx,didx,didx))[:,1]*R
        if ratio:
            if bpwf is None:
                rl,rdl = rebincl(l[:4001],(x[:4001]-RDN0[:4001]-N1[:4001])/tlkk[:4001],bine)
            else:
                rdl    = ((x[:4001]-RDN0[:4001]-N1[:4001])/tlkk[:4001])@bpwf#rebincl(l,(x-RDN0-N1)/tlkk,bine)
            
        else:
            if bpwf is None:
                rl,rdl = rebincl(l[:4001],l[:4001]**(ellfac)*(x[:4001]-RDN0[:4001]-N1[:4001]),bine)
            else:
                rdl    = (l[:4001]**(ellfac)*(x[:4001]-RDN0[:4001]-N1[:4001]))[:4001]@bpwf#rebincl(l,l**(ellfac)*(x-RDN0-N1),bine)

        #import pdb;pdb.set_trace()
        return rl,rdl,np.mean(arr,axis=1),np.std(arr,axis=1),arr
    else:
        return rl, np.mean(arr,axis=1),np.std(arr,axis=1),arr


##########################################################################
#dir     = '/sdf/home/w/wlwu/data/spt3glens1920_gmvph/lensrec/081024/gmvjtp/lmin350_lmaxt3500_lmaxp4000_mmin130_crosstf_badclusrm_prf1am/clkk/'
#dir     = '/sdf/home/w/wlwu/data/spt3glens1920_gmvph/lensrec/081024/gmvjtp/lmin350_lmaxt3500_lmaxp4000_mmin130_crosstf_badclusrm/resppolspice_unmask_mcresp/'
#dir = "/sdf/group/kipac/users/wlwu/spt3g/lensrec/sqe081024/sqe/autotf_v2_lmint350_lminp350_lmaxt3500_lmaxp4000_mmin130_prftsz/clkk/"
dir = "/sdf/group/kipac/users/wlwu/spt3g/lensrec/sqe081024/sqe/autotf_v2_lmint350_lminp350_lmaxt3500_lmaxp3500_mmin100_aggcinv_prf1am/clkk/"
nsims1 = 498 # xx,N0,RDN0 
nsims2 = 250 # N1
qe     = 'ttbhttprf' #'gmvbhttprf'
R      =1
didx   =None  #0
#########################################################################
bb    = np.round(np.geomspace(24,3500,18))

N0    = loadcls(dir,nsims1 ,'N0'  ,N0=None,lmin=bb[0],lmax=bb[-1],R=R,qe=qe)
N1    = loadcls(dir,nsims2 ,'N1'  ,N0=N0  ,lmin=bb[0],lmax=bb[-1],R=R,qe=qe)
if didx is not None: RDN0  = loadcls(dir,nsims1 ,'RDN0',N0=N0  ,lmin=bb[0],lmax=bb[-1],R=R,qe=qe)
xx    = loadcls(dir,nsims1,'xx',N0=None,lmin=bb[0],lmax=bb[-1],R=R,qe=qe)

bpwf  = get_bpwf(bb,nsims1,N0,N1,RDN0=None,ellfac=0,ratio=False,curl=False,R=1,qe=qe)
bpwf2 = get_bpwf(bb,nsims1,N0,N1,RDN0=None,ellfac=1,ratio=False,curl=False,R=1,qe=qe)

if didx is not None:
    rl0,rdl0,rcl0,err0,arr0 = get_dvec(bb,nsims1,N0,N1,RDN0=RDN0,ellfac=0,R=R,bpwf=bpwf,qe=qe)
    rl,rdl,rcl,err,arr      = get_dvec(bb,nsims1,N0,N1,RDN0=RDN0,ellfac=1,R=R,bpwf=bpwf2,qe=qe)
    rlR,rdlR,rclR,errR,_    = get_dvec(bb,nsims1,N0,N1,RDN0=RDN0,ellfac=0,ratio=True,R=R,bpwf=bpwf,qe=qe)

    np.savez(dir+f'ratio_cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_dseed{didx}.npz',rlR=rlR,rdlR=rdlR,rclR=rclR,errR=errR,nsims1=nsims1,nsims2=nsims2, didx=didx)
    print(dir+f'ratio_cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_dseed{didx}.npz')
    print(' ')
    np.savez(dir+f'cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_dseed{didx}.npz',
            rl0=rl0,rdl0=rdl0,rcl0=rcl0,err0=err0, rcl_arr0=arr0,
            rl =rl, rdl = rdl,rcl = rcl,err = err, rcl_arr =arr)
    print(dir+f'cls_k{qe}_nsims1_{nsims1}_nsims2_{nsims2}_mcresp_dseed{didx}.npz')
else:
    rl0,rcl0,err0,arr = get_dvec(bb,nsims1,N0,N1,RDN0=None,ellfac=0,R=R,bpwf=None,qe=qe)
    rl, rcl,err,_       = get_dvec(bb,nsims1,N0,N1,RDN0=None,ellfac=1,R=R,bpwf=None,qe=qe)
    rlR,rclR,errR,_   = get_dvec(bb,nsims1,N0,N1,RDN0=None,ellfac=0,ratio=True,R=R,bpwf=None,qe=qe)
    if 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.errorbar(rlR, rclR, yerr=errR/np.sqrt(nsims1), capsize=1, capthick=1,
fmt='.', mfc="C0", mec="C0", ms=5)
        plt.axhline(1, color="k", ls="--")
        plt.xscale("log")
        plt.ylim([0.9, 1.1])
        plt.show()
