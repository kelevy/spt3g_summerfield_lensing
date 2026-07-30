import os,sys,yaml
import healpy as hp
import numpy as np
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../healqest/src/')))
import healqest_utils as utils
import weights
import qest
import wignerd,resp

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
parser.add_argument('qeidx', default=0, type=int, help='qeidx')
parser.add_argument('nsims', default=10, type=int, help='comp')
parser.add_argument('dataidx', default=0, type=int, help='data index')
parser.add_argument('--bundleid', default=0, type=int, help='bundleid')
parser.add_argument('-src_yaml', default=None , type=str, dest='src_yaml', help='src/prf yaml')
parser.add_argument('-prftype', default='tsz',dest='prftype'     , help="tsz or 1am")
parser.add_argument('--getxx', default=False, dest='getxx' ,action='store_true')
parser.add_argument('--getxxyy', default=False, dest='getxxyy' ,action='store_true')
parser.add_argument('--getxy', default=False, dest='getxy' ,action='store_true')
parser.add_argument('--getxd', default=False, dest='getxd' ,action='store_true')
parser.add_argument('--getab', default=False, dest='getab' ,action='store_true')
parser.add_argument('--getay', default=False, dest='getay' ,action='store_true')
parser.add_argument('--getxx_N1', default=False, dest='getxx_N1', action='store_true')
parser.add_argument('--getxy_N1', default=False, dest='getxy_N1', action='store_true')
parser.add_argument('--getab_N1', default=False, dest='getab_N1', action='store_true')
parser.add_argument('--getresp' , default=False, dest='getresp' ,action='store_true')
parser.add_argument('--getresp_N1_sep', default=False, dest='getresp_N1_sep', action='store_true')
parser.add_argument('--getresp2', default=False, dest='getresp2' ,action='store_true')
parser.add_argument('--nops', default=False, dest='nops' ,action='store_true')
parser.add_argument('--N1', default=False, dest='N1' ,action='store_true')
parser.add_argument('--noinpaint', default=False, dest='noinpaint', action='store_true')
parser.add_argument('--unl', default=False, dest='unl' , action='store_true')

args = parser.parse_args()

nsims          = args.nsims
bundleid       = args.bundleid
getxx          = args.getxx
getxxyy        = args.getxxyy
getxy          = args.getxy
getxd          = args.getxd
getab          = args.getab
getay          = args.getay
getxx_N1       = args.getxx_N1
getxy_N1       = args.getxy_N1
getab_N1       = args.getab_N1
getresp        = args.getresp
getresp_N1_sep = args.getresp_N1_sep
nops           = args.nops
getresp2       = args.getresp2
qeidx          = args.qeidx
noinpaint      = args.noinpaint
unl            = args.unl
prftype        = args.prftype

N1    = args.N1

d     = args.dataidx

if nsims%2!=0:
     sys.exit("want even number of sims")

config  = utils.load_yaml(args.file_yaml)

qedict  = {0:'TT', 1:'EE', 2:'TE', 3:'TB', 4:'EB', 5:'ET', 6:'BT', 7:'BE', 
           8:'GMV', 9:'GMVTTEETE', 10:'GMVTBEB',
           11: 'GMVbhTTprf', 12: 'GMVTTEETEbhTTprf',
           13: 'TTbhTTprf'}
qe      = qedict[qeidx]
#-----------------------------------------------------------------------

runname = config['base']['runname']
rectype = config['lensrec']['rectype']
lmaxT   = config['lensrec']['lmaxT']
lmaxP   = config['lensrec']['lmaxP']
lminT   = config['lensrec']['lminT']
lminP   = config['lensrec']['lminP']
Lmax    = config['lensrec']['Lmax']

mmin    = config['lensrec']['mmin']

if args.src_yaml is None:
    #save stacks in SQE/GMV plm directories
    dir_p    = config['lensrec']['dir_out'].format(runname=runname,rectype=rectype,bundleid=bundleid,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)
    dir_p_N1 = config['lensrec']['dir_out_N1'].format(runname=runname,rectype=rectype,bundleid=bundleid,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)
else:
    #save stacks in hardened GMV plm directories
    sconfig  = utils.load_yaml(args.src_yaml)
    dir_p    = sconfig['lensrec']['dir_out'].format(runname=runname,rectype=rectype,bundleid=bundleid,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin,prftype=prftype)
    dir_p_N1 = sconfig['lensrec']['dir_out_N1'].format(runname=runname,rectype=rectype,bundleid=bundleid,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin,prftype=prftype)
    
suffix = ''
if nops:
    suffix = '_nops'

mf={}

print('----- Computing plmstack ------')
    
print("Processing %s"%qe)
gmfxx=0; gmfyy=0; gmfxy=0; gmfyx=0; gmfab=0; gmfba=0; gmfdx=0; gmfxd=0; gmfay=0; gmfya=0; gmfxx_N1=0; gmfxy_N1=0; gmfyx_N1=0; gmfab_N1=0; gmfba_N1=0
cmfxx=0; cmfyy=0; cmfxy=0; cmfyx=0; cmfab=0; cmfba=0; cmfdx=0; cmfxd=0; cmfay=0; cmfya=0; cmfxx_N1=0; cmfxy_N1=0; cmfyx_N1=0; cmfab_N1=0; cmfba_N1=0

#half1
gmfxx1=0; gmfyy1=0; gmfxy1=0; gmfyx1=0; gmfab1=0; gmfba1=0; gmfdx1=0; gmfxd1=0; gmfay1=0; gmfya1=0; gmfxx1_N1=0; gmfxy1_N1=0; gmfyx1_N1=0; gmfab1_N1=0; gmfba1_N1=0
cmfxx1=0; cmfyy1=0; cmfxy1=0; cmfyx1=0; cmfab1=0; cmfba1=0; cmfdx1=0; cmfxd1=0; cmfay1=0; cmfya1=0; cmfxx1_N1=0; cmfxy1_N1=0; cmfyx1_N1=0; cmfab1_N1=0; cmfba1_N1=0

#half2
gmfxx2=0; gmfyy2=0; gmfxy2=0; gmfyx2=0; gmfab2=0; gmfba2=0; gmfdx2=0; gmfxd2=0; gmfay2=0; gmfya2=0; gmfxx2_N1=0; gmfxy2_N1=0; gmfyx2_N1=0; gmfab2_N1=0; gmfba2_N1=0
cmfxx2=0; cmfyy2=0; cmfxy2=0; cmfyx2=0; cmfab2=0; cmfba2=0; cmfdx2=0; cmfxd2=0; cmfay2=0; cmfya2=0; cmfxx2_N1=0; cmfxy2_N1=0; cmfyx2_N1=0; cmfab2_N1=0; cmfba2_N1=0

mf[qe]={}

print(f'Loading from: {dir_p}')

print("Processing half1")
cc1=0    
for i in range(1,int(nsims/2)+1):
    if getxx:
        plmxx=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,i,suffix));  gmfxx1+=plmxx['glm']; 
        if 'clm' in plmxx: cmfxx1+=plmxx['clm']
    if getxxyy:
        plmyy=np.load(dir_p+'/plm%s_%db_%db%s.npz'%(qe,i,i,suffix));  gmfyy1+=plmyy['glm']; 
        if 'clm' in plmyy: cmfyy1+=plmyy['clm']
    if getxy:
        plmxy=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,i+1,suffix));  gmfxy1+=plmxy['glm']; 
        if 'clm' in plmxy: cmfxy1+=plmxy['clm']
        plmyx=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i+1,i,suffix));  gmfyx1+=plmyx['glm']; 
        if 'clm' in plmyx: cmfyx1+=plmyx['clm']
    if getxd:
        plmxd=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,d,suffix));  gmfxd1+=plmxd['glm']; 
        if 'clm' in plmxd: cmfxd1+=plmxd['clm']
        plmdx=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,d,i,suffix));  gmfdx1+=plmdx['glm']; 
        if 'clm' in plmdx: cmfdx1+=plmdx['clm']
    if getab:
        plmab=np.load(dir_p+'/plm%s_%da_%db%s.npz'%(qe,i,i,suffix));  gmfab1+=plmab['glm']; 
        if 'clm' in plmab: cmfab1+=plmab['clm']
        plmba=np.load(dir_p+'/plm%s_%db_%da%s.npz'%(qe,i,i,suffix));  gmfba1+=plmba['glm']; 
        if 'clm' in plmba: cmfba1+=plmba['clm']
    if getay:
        plmay=np.load(dir_p+'/plm%s_%da_%db%s.npz'%(qe,i,i+1,suffix));  gmfay1+=plmay['glm']; 
        if 'clm' in plmay: cmfay1+=plmay['clm']
        plmya=np.load(dir_p+'/plm%s_%db_%da%s.npz'%(qe,i+1,i,suffix));  gmfya1+=plmya['glm']; 
        if 'clm' in plmya: cmfya1+=plmya['clm']
    if getxx_N1:
        plmxx_N1=np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i,i,suffix));  gmfxx1_N1+=plmxx_N1['glm']; 
        if 'clm' in plmxx_N1: cmfxx1_N1+=plmxx_N1['clm']
    if getxy_N1:
        plmxy_N1=np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i,i+1,suffix));  gmfxy1_N1+=plmxy_N1['glm']; 
        if 'clm' in plmxy_N1: cmfxy1_N1+=plmxy_N1['clm']
        plmyx_N1=np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i+1,i,suffix));  gmfyx1_N1+=plmyx_N1['glm']; 
        if 'clm' in plmyx_N1: cmfyx1_N1+=plmyx_N1['clm']
    if getab_N1:
        plmab_N1=np.load(dir_p_N1+'/plm%s_%da_%db%s.npz'%(qe,i,i,suffix));  gmfab1_N1+=plmab_N1['glm']; 
        if 'clm' in plmab_N1: cmfab1_N1+=plmab_N1['clm']
        plmba_N1=np.load(dir_p_N1+'/plm%s_%db_%da%s.npz'%(qe,i,i,suffix));  gmfba1_N1+=plmba_N1['glm']; 
        if 'clm' in plmba_N1: cmfba1_N1+=plmba_N1['clm'] 
        
    cc1+=1


print("Processing half2")
cc2=0
for i in range(int(nsims/2)+1,nsims+1):
    if getxx:
        plmxx=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,i,suffix));    gmfxx2+=plmxx['glm']; 
        if 'clm' in plmxx: cmfxx2+=plmxx['clm']
    if getxxyy:
        plmyy=np.load(dir_p+'/plm%s_%db_%db%s.npz'%(qe,i,i,suffix));    gmfyy2+=plmyy['glm']; 
        if 'clm' in plmyy: cmfyy2+=plmyy['clm']
    if getxy:
        plmxy=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,i+1,suffix));  gmfxy2+=plmxy['glm']; 
        if 'clm' in plmxy: cmfxy2+=plmxy['clm']
        plmyx=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i+1,i,suffix));  gmfyx2+=plmyx['glm']; 
        if 'clm' in plmyx: cmfyx2+=plmyx['clm']
    if getxd:
        plmxd=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,d,suffix));  gmfxd2+=plmxd['glm']; 
        if 'clm' in plmxd: cmfxd2+=plmxd['clm']
        plmdx=np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,d,i,suffix));  gmfdx2+=plmdx['glm']; 
        if 'clm' in plmdx: cmfdx2+=plmdx['clm']
    if getab:
        plmab=np.load(dir_p+'/plm%s_%da_%db%s.npz'%(qe,i,i,suffix));  gmfab2+=plmab['glm']; 
        if 'clm' in plmab: cmfab2+=plmab['clm']
        plmba=np.load(dir_p+'/plm%s_%db_%da%s.npz'%(qe,i,i,suffix));  gmfba2+=plmba['glm']; 
        if 'clm' in plmba: cmfba2+=plmba['clm']
    if getay:
        plmay=np.load(dir_p+'/plm%s_%da_%db%s.npz'%(qe,i,i+1,suffix));  gmfay2+=plmay['glm']; 
        if 'clm' in plmay: cmfab2+=plmay['clm']
        plmya=np.load(dir_p+'/plm%s_%db_%da%s.npz'%(qe,i+1,i,suffix));  gmfya2+=plmya['glm']; 
        if 'clm' in plmya: cmfba2+=plmya['clm']
    if getxx_N1:
        plmxx_N1=np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i,i,suffix));  gmfxx2_N1+=plmxx_N1['glm']; 
        if 'clm' in plmxx_N1: cmfxx2_N1+=plmxx_N1['clm']
    if getxy_N1:
        plmxy_N1=np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i,i+1,suffix));  gmfxy2_N1+=plmxy_N1['glm']; 
        if 'clm' in plmxy_N1: cmfxy2_N1+=plmxy_N1['clm']
        plmyx_N1=np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i+1,i,suffix));  gmfyx2_N1+=plmyx_N1['glm']; 
        if 'clm' in plmyx_N1: cmfyx2_N1+=plmyx_N1['clm']
    if getab_N1:
        plmab_N1=np.load(dir_p_N1+'/plm%s_%da_%db%s.npz'%(qe,i,i,suffix));  gmfab2_N1+=plmab_N1['glm']; 
        if 'clm' in plmab_N1: cmfab2_N1+=plmab_N1['clm']
        plmba_N1=np.load(dir_p_N1+'/plm%s_%db_%da%s.npz'%(qe,i,i,suffix));  gmfba2_N1+=plmba_N1['glm']; 
        if 'clm' in plmba_N1: cmfba2_N1+=plmba_N1['clm'] 

    cc2+=1

cc=cc1+cc2

if getxx:
    gmfxx=gmfxx1+gmfxx2
    cmfxx=cmfxx1+cmfxx2
    np.savez(dir_p+'/glmstack%s_xx%s.npz'%(qe,suffix),gmfxx=gmfxx, gmfxx_half1=gmfxx1, gmfxx_half2=gmfxx2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_xx%s.npz'%(qe,suffix),cmfxx=cmfxx, cmfxx_half1=cmfxx1, cmfxx_half2=cmfxx2, nsim=cc, nsim_half=cc1)
    print("Saving: ",dir_p+'/glmstack%s_xx%s.npz'%(qe,suffix))
    print("Saving: ",dir_p+'/clmstack%s_xx%s.npz'%(qe,suffix))

if getxxyy:
    gs = np.load(dir_p+'/glmstack%s_xx%s.npz'%(qe,suffix))# ,gmfxx=gmfxx, gmfxx_half1=gmfxx1, gmfxx_half2=gmfxx2, nsim=cc, nsim_half=cc1)
    cs = np.load(dir_p+'/clmstack%s_xx%s.npz'%(qe,suffix))# ,cmfxx=cmfxx, cmfxx_half1=cmfxx1, cmfxx_half2=cmfxx2, nsim=cc, nsim_half=cc1)
    gmfxx1 = gs['gmfxx_half1']
    gmfxx2 = gs['gmfxx_half2']
    cmfxx1 = cs['cmfxx_half1']
    cmfxx2 = cs['cmfxx_half2']
    gmfxx  = gs['gmfxx']
    cmfxx  = cs['cmfxx']
    ccxx   = gs['nsim']
    ccxx1  = gs['nsim_half']

    gmfyy = gmfyy1 + gmfyy2 + gs['gmfxx_half1'] + gs['gmfxx_half2']
    cmfyy = cmfyy1 + cmfyy2 + cs['cmfxx_half1'] + cs['cmfxx_half2'] 
    np.savez(dir_p+'/glmstack%s_xxyy%s.npz'%(qe,suffix), gmfxx=gmfyy+gmfxx, gmfxx_half1=gmfxx1+gmfyy1, gmfxx_half2=gmfxx2+gmfyy2, nsim=cc+ccxx, nsim_half=cc1+ccxx1)
    np.savez(dir_p+'/clmstack%s_xxyy%s.npz'%(qe,suffix), cmfxx=cmfyy+cmfxx, cmfxx_half1=cmfxx1+cmfyy1, cmfxx_half2=cmfxx2+cmfyy2, nsim=cc+ccxx, nsim_half=cc1+ccxx1)
    print("Saving: ",dir_p+'/glmstack%s_xxyy%s.npz'%(qe,suffix))
    print("Saving: ",dir_p+'/clmstack%s_xxyy%s.npz'%(qe,suffix))
       
if getxy:
    gmfxy=gmfxy1+gmfxy2; cmfxy=cmfxy1+cmfxy2
    gmfyx=gmfyx1+gmfyx2; cmfyx=cmfyx1+cmfyx2
    np.savez(dir_p+'/glmstack%s_xy.npz'%(qe),gmfxy=gmfxy, gmfxy_half1=gmfxy1, gmfxy_half2=gmfxy2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_xy.npz'%(qe),cmfxy=cmfxy, cmfxy_half1=cmfxy1, cmfxy_half2=cmfxy2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/glmstack%s_yx.npz'%(qe),gmfyx=gmfyx, gmfyx_half1=gmfyx1, gmfyx_half2=gmfyx2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_yx.npz'%(qe),cmfyx=cmfyx, cmfyx_half1=cmfyx1, cmfyx_half2=cmfyx2, nsim=cc, nsim_half=cc1)

if getxd:
    gmfxd=gmfxd1+gmfxd2; cmfxd=cmfxd1+cmfxd2
    gmfdx=gmfdx1+gmfdx2; cmfdx=cmfdx1+cmfdx2
    np.savez(dir_p+'/glmstack%s_xd_dataidx%d.npz'%(qe,d),gmfxd=gmfxd, gmfxd_half1=gmfxd1, gmfxd_half2=gmfxd2, nsim=cc, nsim_half=cc1, dataidx=d)
    np.savez(dir_p+'/clmstack%s_xd_dataidx%d.npz'%(qe,d),cmfxd=cmfxd, cmfxd_half1=cmfxd1, cmfxd_half2=cmfxd2, nsim=cc, nsim_half=cc1, dataidx=d)
    np.savez(dir_p+'/glmstack%s_dx_dataidx%d.npz'%(qe,d),gmfdx=gmfdx, gmfdx_half1=gmfdx1, gmfdx_half2=gmfdx2, nsim=cc, nsim_half=cc1, dataidx=d)
    np.savez(dir_p+'/clmstack%s_dx_dataidx%d.npz'%(qe,d),cmfdx=cmfdx, cmfdx_half1=cmfdx1, cmfdx_half2=cmfdx2, nsim=cc, nsim_half=cc1, dataidx=d)

if getab:
    gmfab=gmfab1+gmfab2; cmfab=cmfab1+cmfab2
    gmfba=gmfba1+gmfba2; cmfba=cmfba1+cmfba2
    np.savez(dir_p+'/glmstack%s_ab.npz'%(qe),gmfab=gmfab, gmfab_half1=gmfab1, gmfab_half2=gmfab2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_ab.npz'%(qe),cmfab=cmfab, cmfab_half1=cmfab1, cmfab_half2=cmfab2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/glmstack%s_ba.npz'%(qe),gmfba=gmfba, gmfba_half1=gmfba1, gmfba_half2=gmfba2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_ba.npz'%(qe),cmfba=cmfba, cmfba_half1=cmfba1, cmfba_half2=cmfba2, nsim=cc, nsim_half=cc1)

if getay:
    gmfay=gmfay1+gmfay2; cmfab=cmfay1+cmfay2
    gmfya=gmfya1+gmfya2; cmfba=cmfya1+cmfya2
    np.savez(dir_p+'/glmstack%s_ay.npz'%(qe),gmfab=gmfay, gmfab_half1=gmfay1, gmfab_half2=gmfay2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_ay.npz'%(qe),cmfab=cmfay, cmfab_half1=cmfay1, cmfab_half2=cmfay2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/glmstack%s_ya.npz'%(qe),gmfba=gmfya, gmfba_half1=gmfya1, gmfba_half2=gmfya2, nsim=cc, nsim_half=cc1)
    np.savez(dir_p+'/clmstack%s_ya.npz'%(qe),cmfba=cmfya, cmfba_half1=cmfya1, cmfba_half2=cmfya2, nsim=cc, nsim_half=cc1)

if getxx_N1:
    gmfxx_N1=gmfxx1_N1+gmfxx2_N1
    cmfxx_N1=cmfxx1_N1+cmfxx2_N1
    np.savez(dir_p_N1+'/glmstack%s_xx%s.npz'%(qe,suffix),gmfxx=gmfxx_N1, gmfxx_half1=gmfxx1_N1, gmfxx_half2=gmfxx2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/clmstack%s_xx%s.npz'%(qe,suffix),cmfxx=cmfxx_N1, cmfxx_half1=cmfxx1_N1, cmfxx_half2=cmfxx2_N1, nsim=cc, nsim_half=cc1)

if getxy_N1:
    gmfxy_N1=gmfxy1_N1+gmfxy2_N1; cmfxy_N1=cmfxy1_N1+cmfxy2_N1
    gmfyx_N1=gmfyx1_N1+gmfyx2_N1; cmfyx_N1=cmfyx1_N1+cmfyx2_N1
    np.savez(dir_p_N1+'/glmstack%s_xy.npz'%(qe),gmfxy=gmfxy_N1, gmfxy_half1=gmfxy1_N1, gmfxy_half2=gmfxy2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/clmstack%s_xy.npz'%(qe),cmfxy=cmfxy_N1, cmfxy_half1=cmfxy1_N1, cmfxy_half2=cmfxy2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/glmstack%s_yx.npz'%(qe),gmfyx=gmfyx_N1, gmfyx_half1=gmfyx1_N1, gmfyx_half2=gmfyx2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/clmstack%s_yx.npz'%(qe),cmfyx=cmfyx_N1, cmfyx_half1=cmfyx1_N1, cmfyx_half2=cmfyx2_N1, nsim=cc, nsim_half=cc1)

if getab_N1:
    gmfab_N1=gmfab1_N1+gmfab2_N1; cmfab_N1=cmfab1_N1+cmfab2_N1
    gmfba_N1=gmfba1_N1+gmfba2_N1; cmfba_N1=cmfba1_N1+cmfba2_N1
    np.savez(dir_p_N1+'/glmstack%s_ab.npz'%(qe),gmfab=gmfab_N1, gmfab_half1=gmfab1_N1, gmfab_half2=gmfab2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/clmstack%s_ab.npz'%(qe),cmfab=cmfab_N1, cmfab_half1=cmfab1_N1, cmfab_half2=cmfab2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/glmstack%s_ba.npz'%(qe),gmfba=gmfba_N1, gmfba_half1=gmfba1_N1, gmfba_half2=gmfba2_N1, nsim=cc, nsim_half=cc1)
    np.savez(dir_p_N1+'/clmstack%s_ba.npz'%(qe),cmfba=cmfba_N1, cmfba_half1=cmfba1_N1, cmfba_half2=cmfba2_N1, nsim=cc, nsim_half=cc1)




if getresp:
    print('Getting resp')
    inkapfname = config['inputs']['kappa_in']

    # Load plmstack
    plmstack  = np.load(dir_p+'/glmstack%s_xx%s.npz'%(qe,suffix))
             
    c   = 0
    cl1 = 0
    cl2 = 0
    for i in range(1,nsims+1):
        print(inkapfname.format(runname=runname, rectype=rectype, seed = i))
        
        ilm = hp.read_alm(inkapfname.format(runname=runname, rectype=rectype, seed=i))
        ilm = utils.reduce_lmax(ilm,lmax=Lmax)
        olm = np.load(dir_p+'/plm%s_%da_%da%s.npz'%(qe,i,i,suffix))['glm'] 
        
        mfi = ((plmstack['gmfxx']-olm)/(plmstack['nsim']-1.0))
        
        cl1+=hp.alm2cl(ilm,ilm)
        l=np.arange(len(cl1))
        cl2+=hp.alm2cl(olm-mfi,ilm)*0.5*(l*(l+1))
        c+=1
    
    respavg=cl2/cl1

    respavg[:4]=1e30
    respavg[-1]=1e30

    np.savez(dir_p+'/respavg%s%s.npz'%(qe,suffix),resp=respavg,nsim=c)
    print("Saving to: ", dir_p+'/respavg%s%s.npz'%(qe,suffix))

if getresp_N1_sep:
    print('Getting seperate resp for N1')
    inkapfname = config['inputs']['kappa_in']

    # Load plmstack
    plmstack  = np.load(dir_p_N1+'/glmstack%s_xx%s.npz'%(qe,suffix))
             
    c   = 0
    cl1 = 0
    cl2 = 0
    for i in range(1,nsims+1):
        print(inkapfname.format(runname=runname, rectype=rectype, seed = i))
        
        ilm = hp.read_alm(inkapfname.format(runname=runname, rectype=rectype, seed=i))
        ilm = utils.reduce_lmax(ilm,lmax=Lmax)
        olm = np.load(dir_p_N1+'/plm%s_%da_%da%s.npz'%(qe,i,i,suffix))['glm'] 
        
        mfi = ((plmstack['gmfxx']-olm)/(plmstack['nsim']-1.0))
        
        cl1+=hp.alm2cl(ilm,ilm)
        l=np.arange(len(cl1))
        cl2+=hp.alm2cl(olm-mfi,ilm)*0.5*(l*(l+1))
        c+=1
    
    respavg=cl2/cl1

    respavg[:4]=1e30
    respavg[-1]=1e30

    np.savez(dir_p_N1+'/respavg%s%s.npz'%(qe,suffix),resp=respavg,nsim=c)
    print("Saving to: ", dir_p_N1+'/respavg%s%s.npz'%(qe,suffix))

if getresp2:
    print('Getting resp')
    inkapfname = config['inputs']['kappa_in']

    # Load plmstack
    plmstack  = np.load(dir_p+'/glmstack%s_xx.npz'%(qe))

    c   = 0
    cl1 = 0
    cl2 = 0
    for i in range(1,nsims+1):
        print(inkapfname.format( seed = i ) )

        ilm = hp.read_alm( inkapfname.format( seed = i ) )
        ilm = utils.reduce_lmax(ilm,lmax=Lmax)
        ilm2=hp.read_alm(f'/lcrc/project/SPT3G/users/ac.yomori/scratch/inputkappa/inputkappa_extrawide_{i}.alm')
        olm = np.load(dir_p+'/plm%s_%da_%da.npz'%(qe,i,i))['glm']

        mfi = ((plmstack['gmfxx']-olm)/(plmstack['nsim']-1.0))

        cl1+=hp.alm2cl(ilm,ilm2)
        cl2+=hp.alm2cl(olm-mfi,ilm2)#*0.5*(l*(l+1))
        c+=1

    respavg=cl2/cl1

    respavg[:4]=1e30
    respavg[-1]=1e30

    np.savez(dir_p+'/respavg%s2.npz'%(qe),resp=respavg,nsim=c)
