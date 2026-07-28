import os
import argparse
import yaml
import numpy as np
import healpy as hp
import glob


##############################################################################################################################################################


# These settings are specified via the command line only.'
parser = argparse.ArgumentParser(description='Maps for a CMB field')
cli = parser.add_argument_group('Command Line Inputs')
cli.add_argument('--freq', dest='freq', type=int)
cli.add_argument('--seed', dest='seed', type=int)
cli.add_argument('--config_file', dest = 'config_file', type = str)

# These settings are specified in the config file.'
config = parser.add_argument_group('Config File Inputs')
config.add_argument('-loc_dic_fg_argonne', dest='loc_dic_fg_argonne', type = str)
config.add_argument('-loc_dic_cmb_argonne', dest='loc_dic_cmb_argonne', type = str)
config.add_argument('-op_fd', dest='op_fd', type = str)
config.add_argument('-cmb_sim_fd_suff', dest='cmb_sim_fd_suff', type = str)
config.add_argument('-fname_dic_cmb', dest='fname_dic_cmb', type = str)
config.add_argument('-fg_sim_fd_suff', dest='fg_sim_fd_suff', type = str)
config.add_argument('-fname_dic_fg', dest='fname_dic_fg', type = str)
config.add_argument('-output_loc', dest='output_loc', type = str)
config.add_argument('-opfname_pref', dest='opfname_pref', type = str)
config.add_argument('-beam_file', dest='beam_file', type = str)
config.add_argument('-mask_file', dest='beam_file', type = str)
config.add_argument('--sanity_check', dest='sanity_check', type = int)
config.add_argument('--lmax', dest='lmax', type = int)
config.add_argument('--cl_sim_file', dest='cl_sim_file', type = int)


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


##############################################################################################################################################################


#location on argonne xover
loc_dic = {}
loc_dic['fg'] = '%s%s' %(loc_dic_fg_argonne, fg_sim_fd_suff)
loc_dic['cmb'] = '%s%s' %(loc_dic_cmb_argonne, cmb_sim_fd_suff)

#file names
fname_dic = {}
fname_dic['fg'] = fname_dic_fg 
fname_dic['cmb'] = fname_dic_cmb 

#output folders and output file name prefix
op_fd_dic = {}
op_fd_dic['cmb'] = '%s%s' %(op_fd, cmb_sim_fd_suff)
op_fd_dic['fg'] = '%s%s' %(op_fd, fg_sim_fd_suff)
for comp in op_fd_dic:
    if not os.path.exists(op_fd_dic[comp]): os.system('mkdir -p %s' %(op_fd_dic[comp]))    
output_fd = '%s%s' %(op_fd, output_loc)

if not os.path.exists(output_fd+str(freq)+'ghz/'): 
    os.system('mkdir -p %s' %(output_fd+str(freq)+'ghz/'))

# beams
bl_dic = {}
bl_dic['ell'], bl_dic[95], bl_dic[150], bl_dic[220] = np.loadtxt(beam_file, unpack=True)

# boundary mask
apod_mask = hp.read_map(mask_file)
apod_mask = hp.ud_grade(apod_mask, 8192)
masked_inds = np.where(apod_mask == 0.)[0]

# Create simulation
print('\tBand = %s GHz' %(freq))
hmap = None     

print('Adding CMB and foregrounds ...')   
for comp in loc_dic:
    print('Composition:', comp)
    curr_fname_xover = '%s%s' %(loc_dic[comp], fname_dic[comp])
    curr_fname_xover = curr_fname_xover.replace('seedval', str(seed)).replace('nuval', str(freq))

    # copy relevant files from argonne to spartan
    curr_fname_local = op_fd_dic[comp]
    #cmd = 'rsync -trvz --progress anl:%s %s' %(curr_fname_xover, curr_fname_local)
    #os.system(cmd)

    curr_map = (curr_fname_local+fname_dic[comp]).replace('seedval', str(seed)).replace('nuval', str(freq))
    # read CMB map
    if hmap is None:
        hmap = np.copy(hp.read_map(curr_map, field = (0,1,2)))
    # add foregrounds
    else:
       hmap += np.copy(hp.read_map(curr_map, field = (0,1,2)))

# smooth simulation with the beam
print('Applying beam ...')
hmap_smoothed = hp.smoothing(hmap,beam_window=bl_dic[freq])

# applying boundary mask
print('Applying boundary mask ...')
hmap_smoothed = hmap_smoothed * apod_mask 
# mask the UNSEEN pixels
hmap_smoothed[:, masked_inds] = hp.UNSEEN

# save (CMB+foreground) X beam simulation
print('Saving simulation ...')
opfname = (output_fd+str(freq)+'ghz/'+opfname_pref).replace('nuval', str(freq)).replace('seedval', str(seed-1))
hp.write_map(opfname, hmap_smoothed, overwrite=True, partial = True)
print('\t\tcheck %s\n' %(opfname))
