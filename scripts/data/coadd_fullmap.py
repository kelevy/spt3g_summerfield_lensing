import sys
import os
import copy
import glob
import argparse
import yaml
import numpy as np
import healpy as hp


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


if not os.path.exists(result_dir.replace('which_field', which_field)):
    os.mkdir(result_dir.replace('which_field', which_field))


curr_mock_obs_fd_pref = mock_obs_fd_pref.replace('which_field', which_field)
mock_obs_searchstr = '%sno_signflip_bundle_00*.g3.gz' %(curr_mock_obs_fd_pref)
mock_obs_flist = sorted(glob.glob(mock_obs_searchstr))[:]
print(mock_obs_flist)

for band in ['90GHz', '150GHz', '220GHz']:
    coadd_map = coadd.coadd_maps(mock_obs_flist, fd_for_weights = None, maprun_date_iden = maprun_date_iden, band = band)
    
    #maps.RemoveWeights(field_coadd, zero_nans = True)
    result_path = (result_dir+band+'_'+result_file).replace('which_field', which_field)
    wr = core.G3Writer(result_path)
    wr(coadd_map)
    wr(core.G3Frame(core.G3FrameType.EndProcessing))
