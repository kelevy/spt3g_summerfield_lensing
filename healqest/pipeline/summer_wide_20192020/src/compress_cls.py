# compress all cls into one file to save ionode
# this is meant to be done once 
import numpy as np
from tqdm import tqdm
import os,sys
from pathlib import Path

def clean(file_list):
    for file in file_list:
        Path(file).unlink(missing_ok=True)

dir = str(sys.argv[1])
qe = str(sys.argv[2])
spec = str(sys.argv[3])
dtype = str(sys.argv[4])

didx = 0
lmax = 4000

# Load the data vector first to determine length
if dtype=='dddd':
    file_list = []
    fname = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, 0, 0, 0, 0 )
    dddd = np.load(fname)["cls"][:lmax+1,1]
    file_list.append(fname)
    np.save(dir + f'all_cl{spec}_{qe}_dddd_didx{didx}.npy',dddd)
    print(dir + f'all_cl{spec}_{qe}_dddd_didx{didx}.npy')
    #clean(file_list)

if dtype=='xxxx':
    file_list = []
    xxxx = np.zeros((lmax+1,500)) 
    for i in range(1,501):
        fname = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i, i, i )
        xxxx[:,i-1] = np.load(fname)["cls"][:lmax+1,1]    
        file_list.append(fname)
    np.save(dir + f'all_cl{spec}_{qe}_xxxx.npy',xxxx)
    print('Saved:',dir + f'all_cl{spec}_{qe}_xxxx.npy')
    #clean(file_list)
    
if dtype=='xyxy':
    file_list = []
    xyxy = np.zeros((lmax+1,499))
    xyyx = np.zeros((lmax+1,499))

    for i in range(1,500):
        fname1 = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i, i + 1)
        fname2 = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i + 1, i)
        xyxy[:,i-1] = np.load(fname1)["cls"][:lmax+1,1]
        xyyx[:,i-1] = np.load(fname2)["cls"][:lmax+1,1]
        file_list.append(fname1)
        file_list.append(fname2)
    
    np.save(dir + f'all_cl{spec}_{qe}_xyxy.npy',xyxy)
    np.save(dir + f'all_cl{spec}_{qe}_xyyx.npy',xyyx)
    print('Saved:',dir + f'all_cl{spec}_{qe}_xyxy.npy')
    print('Saved:',dir + f'all_cl{spec}_{qe}_xyyx.npy')
    #clean(file_list)
    
if dtype=='abab':
    file_list = []
    abab = np.zeros((lmax+1,250))
    abba = np.zeros((lmax+1,250))

    for i in range(1,251):
        fname1 = dir + "cl%s_k%s_%da_%db_%da_%db.npz" % (spec, qe, i, i, i, i)
        fname2 = dir + "cl%s_k%s_%da_%db_%db_%da.npz" % (spec, qe, i, i, i, i)
        abab[:,i-1] = np.load(fname1)["cls"][:lmax+1,1]
        abba[:,i-1] = np.load(fname2)["cls"][:lmax+1,1]
        file_list.append(fname1)
        file_list.append(fname2)
    
    np.save(dir + f'all_cl{spec}_{qe}_abab.npy',abab)
    np.save(dir + f'all_cl{spec}_{qe}_abba.npy',abba)
    print('Saved:',dir + f'all_cl{spec}_{qe}_abab.npy')
    print('Saved:',dir + f'all_cl{spec}_{qe}_abba.npy')
    #clean(file_list)

if dtype=='xdxd':
    file_list = []
    xdxd = np.zeros((lmax+1,500))
    xddx = np.zeros((lmax+1,500))
    dxxd = np.zeros((lmax+1,500))
    dxdx = np.zeros((lmax+1,500))

    for i in range(1,501):
        fname1 = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, didx, i, didx)
        fname2 = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, didx, didx, i)
        fname3 = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, i, didx, i)
        fname4 = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, i, i, didx)
        xdxd[:,i-1] = np.load(fname1)["cls"][:lmax+1,1]
        xddx[:,i-1] = np.load(fname2)["cls"][:lmax+1,1]
        dxdx[:,i-1] = np.load(fname3)["cls"][:lmax+1,1]
        dxxd[:,i-1] = np.load(fname4)["cls"][:lmax+1,1]
        file_list.append(fname1)
        file_list.append(fname2)
        file_list.append(fname3)
        file_list.append(fname4)

    np.save(dir + f'all_cl{spec}_{qe}_xdxd_didx{didx}.npy',xdxd)
    np.save(dir + f'all_cl{spec}_{qe}_xddx_didx{didx}.npy',xddx)
    np.save(dir + f'all_cl{spec}_{qe}_dxdx_didx{didx}.npy',dxdx)
    np.save(dir + f'all_cl{spec}_{qe}_dxxd_didx{didx}.npy',dxxd)
    print('Saved:',dir + f'all_cl{spec}_{qe}_xdxd_didx{didx}.npy')
    print('Saved:',dir + f'all_cl{spec}_{qe}_xddx_didx{didx}.npy')
    print('Saved:',dir + f'all_cl{spec}_{qe}_dxdx_didx{didx}.npy')
    print('Saved:',dir + f'all_cl{spec}_{qe}_dxxd_didx{didx}.npy')
    #clean(file_list)
    

if dtype=='uuuu':
    file_list = []
    uuuu = np.zeros((lmax+1,50)) 
    for i in range(3001,3051):
        fname = dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i, i, i)
        uuuu[:,i-3001] = np.load(fname)["cls"][:lmax+1,1]   
        file_list.append(fname) 
    np.save(dir + f'all_cl{spec}_{qe}_uuuu.npy',uuuu)
    print('Saved:',dir + f'all_cl{spec}_{qe}_uuuu.npy')
    #clean(file_list)
