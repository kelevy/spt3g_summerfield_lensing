import sys
import os
import copy
import glob
import argparse
import yaml
import numpy as np
import healpy as hp
import copy


##################################################################################################################################################################


# These settings are specified via the command line only.
parser = argparse.ArgumentParser(description='Maps for a CMB field')
cli = parser.add_argument_group('Command Line Inputs')
cli.add_argument('--config_file', dest = 'config_file', type = str)

# These settings are specified in the config file.
config = parser.add_argument_group('Config File Inputs')
config.add_argument('--fd_python_files', dest='fd_python_files', type = str)
config.add_argument('--fd_spt3g_software', dest='fd_spt3g_software', type = str)
config.add_argument('--which_field', dest='which_field', type=str)
config.add_argument('--fd_for_weights', dest='fd_for_weights', type = str, default=None)
config.add_argument('--maprun_date_iden', dest='maprun_date_iden', type = str)
config.add_argument('--mock_obs_fd_pref', dest='mock_obs_fd_pref', type = str)
config.add_argument('--result_dir', dest='result_dir', type = str)
config.add_argument('--result_file', dest='result_file', type = str)


# parse all the arguments now
args = parser.parse_args()

# If configuration yaml is specified, load it and pull parameters from there.
if args.config_file is not None:
    settings = yaml.safe_load(open(args.config_file, 'r'))
    for k, v in settings.items():
        setattr(args, k, v)

# replace args.name by dest
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

sys.path.append(fd_python_files)
sys.path.append(fd_spt3g_software)
import coadd
from spt3g import core, maps


##################################################################################################################################################################

field = 'summer_b'

if not os.path.exists(result_dir):
    os.mkdir(result_dir)


for seed in range(0, 21):
    coadd_alm = None
    for bundleid in range(10):
        ilc_dir = mock_obs_fd_pref.replace('bundleid', str(bundleid))
        ilc_alm = hp.read_alm(ilc_dir+'cmbmv_tqu1_lmax4500_%s_seed%s.alm'%(field, seed))
        if coadd_alm is None:
            coadd_alm = copy.deepcopy(ilc_alm)
        else:
            coadd_alm += copy.deepcopy(ilc_alm)
    hp.write_alm(result_dir+'cmbmv_tqu1_lmax4500_%s_seed%s.alm'%(field, seed), coadd_alm/10, overwrite=True)
    if seed==0 or seed==1:
        ilc_map = hp.alm2map(coadd_alm/10,nside=2048)
        hp.write_map(result_dir+'cmbmv_tqu1_lmax4500_%s_seed%s.fits'%(field, seed), ilc_map, overwrite=True)
