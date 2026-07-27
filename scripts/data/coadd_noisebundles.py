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
cli.add_argument('--freq', dest='freq', type=int)
cli.add_argument('--bundleid', dest='bundleid', type=int)
cli.add_argument('--config_file', dest = 'config_file', type = str)

# These settings are specified in the config file.
config = parser.add_argument_group('Config File Inputs')
config.add_argument('--fd_python_files', dest='fd_python_files', type = str)
config.add_argument('--fd_spt3g_software', dest='fd_spt3g_software', type = str)
config.add_argument('--which_field', dest='which_field', type=str)
config.add_argument('--subfield_arr', dest='subfield_arr', type = str)
config.add_argument('--fd_for_weights', dest='fd_for_weights', type = str, default=None)
config.add_argument('--maprun_date_iden', dest='maprun_date_iden', type = str)
config.add_argument('--beam_file', dest='beam_file', type = str)
config.add_argument('--mask_file', dest='mask_file', type = str)
config.add_argument('--mock_obs_fd_pref', dest='mock_obs_fd_pref', type = str)
config.add_argument('--nber_obs', dest='nber_obs', type = str)
config.add_argument('--result_dir', dest='result_dir', type = str)
config.add_argument('--result_file', dest='result_file', type = str)
config.add_argument('--sanity_check', dest='sanity_check', type = int)
config.add_argument('--lmax', dest='lmax', type = int)


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


if not os.path.exists(result_dir):
    os.mkdir(result_dir)

if which_field == 'summera':
    field_arr = subfield_arr[0]
elif which_field == 'summerb':
    field_arr = subfield_arr[1]
elif which_field == 'summerc':
    field_arr = subfield_arr[2]   
elif which_field == 'summer':
    field_arr = ['ra5hdec-29.75', 'ra5hdec-33.25','ra5hdec-38.5', 'ra5hdec-45.5', 'ra5hdec-52.5', 'ra5hdec-59.5', 'ra1h40dec-29.75', 'ra1h40dec-33.25', 'ra1h40dec-36.75', 'ra1h40dec-40.25', 
                 'ra12h30dec-29.75', 'ra12h30dec-33.25', 'ra12h30dec-36.75', 'ra12h30dec-40.25']

#loop over sub-fields now
coadd_map_dic_left = {}
coadd_map_dic_right = {}
field_coadd_left = None
field_coadd_right = None

for i, field_val in enumerate(field_arr):
    curr_mock_obs_fd_pref = mock_obs_fd_pref.replace('field_val', field_val)
    mock_obs_searchstr_left = '%sleft/no_signflip_bundle_00%s.g3.gz' %(curr_mock_obs_fd_pref, bundleid)
    mock_obs_searchstr_right = '%sright/no_signflip_bundle_00%s.g3.gz' %(curr_mock_obs_fd_pref, bundleid)
    mock_obs_flist_left = sorted(glob.glob(mock_obs_searchstr_left))[:] 
    mock_obs_flist_right = sorted(glob.glob(mock_obs_searchstr_right))[:]

    coadd_map_left = coadd.coadd_maps(mock_obs_flist_left, fd_for_weights = None, maprun_date_iden = maprun_date_iden, band = str(freq)+'GHz')
    coadd_map_right = coadd.coadd_maps(mock_obs_flist_right, fd_for_weights = None, maprun_date_iden = maprun_date_iden, band = str(freq)+'GHz')

    coadd_map_dic_left[field_val] = copy.deepcopy(coadd_map_left)
    coadd_map_dic_right[field_val] = copy.deepcopy(coadd_map_right)    

    #perform field coadd
    if field_coadd_left is None:
        field_coadd_left = copy.deepcopy(coadd_map_left)
        field_coadd_right = copy.deepcopy(coadd_map_right)
    else:
        field_coadd_left = coadd.add_frames(copy.deepcopy(coadd_map_left), field_coadd_left)
        field_coadd_right = coadd.add_frames(copy.deepcopy(coadd_map_right), field_coadd_right)
        
maps.RemoveWeights(field_coadd_left, zero_nans = True)
maps.RemoveWeights(field_coadd_right, zero_nans = True)


frame = core.G3Frame('M')
frame['Id'] = "SPT3G_sim"
frame['Q'] =  maps.HealpixSkyMap((np.asarray(field_coadd_left['Q'])-np.asarray(field_coadd_right['Q']))/2) 
frame['T'] =  maps.HealpixSkyMap((np.asarray(field_coadd_left['T'])-np.asarray(field_coadd_right['T']))/2)
frame['U'] =  maps.HealpixSkyMap((np.asarray(field_coadd_left['U'])-np.asarray(field_coadd_right['U']))/2) 


result_path = (result_dir+result_file).replace('which_field', which_field).replace('freq', str(freq)).replace('bundleid', str(bundleid))
wr = core.G3Writer(result_path)
wr(frame)
wr(core.G3Frame(core.G3FrameType.EndProcessing))
