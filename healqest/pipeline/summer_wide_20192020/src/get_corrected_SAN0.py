from pathlib import Path
import pickle,sys,yaml,argparse
import numpy as np
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/')
#sys.path.append(Path(__file__).resolve()+'../../../healqest/src/')
#sys.path.append(Path(__file__).resolve().parent.parent.parent.joinpath("/../../healqest/src/") )
import weights, qest, resp
import healqest_utils as hutils

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
parser.add_argument('qe', default=None, type=str, help='qe')
args   = parser.parse_args()

file_yaml = args.file_yaml
qe        = args.qe

qe        = qe.upper()

config  = hutils.load_yaml(file_yaml)
lminT   = config['lensrec']['lminT']
lmaxT   = config['lensrec']['lmaxT']
lminP   = config['lensrec']['lminP']
lmaxP   = config['lensrec']['lmaxP']
mmin    = config['lensrec']['mmin']
runname = config['base']['runname']
rectype = config['lensrec']['rectype']
dir_out = config['lensrec']['dir_out'].format(runname=runname,rectype=rectype,lminT=lminT,lminP=lminP,lmaxT=lmaxT,lmaxP=lmaxP,mmin=mmin)
dir_resp= dir_out

print(f"Loading from: {dir_out}")

###################################################################################################################

l = np.arange(4001)
v = (0.5*l*(l+1))**2

if qe=='TT':
    aresp = np.load(dir_resp+'/TT/respavgTT_nops.npz')['resp']
    #aresp[:10]      = np.inf
    #aresp[-100:]    = np.inf
    #aresp[aresp==0] = np.inf
    aresp = np.where(np.isnan(aresp) | (aresp == 0), 1e30, aresp)

elif qe=='PP':
    arespEE = np.load(dir_resp+'/EE/respavgEE_nops.npz')['resp']
    arespBE = arespEB = np.load(dir_resp+'/EB/respavgEB_nops.npz')['resp']
    #arespBE = np.load(dir_resp+'/BE/respavgBE_nops.npz')['resp']
    aresp   = arespEB+arespBE+arespEE

    aresp = np.where(np.isnan(aresp) | (aresp == 0), 1e30, aresp)

    #aresp[:10]      = np.inf
    #aresp[-100:]    = np.inf
    #aresp[aresp==0] = np.inf

if qe=='GMV':
    aresp = np.load(dir_resp+'/GMV/respavgGMV.npz')['resp']
    aresp[:10]      = np.inf
    aresp[-100:]    = np.inf
    aresp[aresp==0] = np.inf


# SAN0 Calculation
clqq = np.zeros((4001,500))
c=0
for i in range(1,501):
    tmp = v*np.load(f'{dir_out}/SAN0/clqq_{qe}_grad_{i}.npy')/(aresp)**2 
    tmp[tmp==0] = 1e30
    clqq[:,i-1] = tmp

np.save(f'{dir_out}/SAN0/SAN0_array_{qe}.npy', clqq)
