import sys
import os
import yaml
import argparse
import numpy as np 


##############################################################################################################################################################


# These settings are specified via the command line only.'
parser = argparse.ArgumentParser(description='Maps for a CMB field')
cli = parser.add_argument_group('Command Line Inputs')
cli.add_argument('--config_file', dest = 'config_file', type = str)

# These settings are specified in the config file.'
config = parser.add_argument_group('Config File Inputs')
config.add_argument('-hh', dest='hh', action='store', help='hh', type=int, default=5)
config.add_argument('-mm', dest='mm', action='store', help='mm', type=int, default=00)
config.add_argument('--memory', dest='memory', action='store', help='memory', type=int, default=1)
config.add_argument('--nodes', dest='nodes', action='store', help='nodes', type=int, default=1)

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


batch_jobs_folder = ('batch_jobs_coadds/')
if not os.path.exists(batch_jobs_folder): os.system('mkdir -p %s' %(batch_jobs_folder))

batch_fname = batch_jobs_folder+'coadd.sh'
batchf = open(batch_fname, 'w')
linearr = ['#!/bin/bash', '#SBATCH --partition=sapphire', '#SBATCH --account="punim1922"', '#SBATCH --output='+batch_jobs_folder+'job.o%j', '#SBATCH --cpus-per-task=1']
for lines in linearr:
	opline = lines	
	batchf.writelines('%s\n' %(opline))

# nodes	
opline = '#SBATCH --nodes=%s' %(args.nodes)
batchf.writelines('%s\n' %(opline))
				
# time 
opline = '#SBATCH --time=%d:%d:00' %(hh, mm)
batchf.writelines('%s\n' %(opline))

# memory
opline = '#SBATCH --mem=%dG' %(memory)
batchf.writelines('%s\n' %(opline))

# other
opline = 'export SHELL=bash'
batchf.writelines('%s\n' %(opline))

opline = '\n\n python coadd_fullmap.py --config_file %s' %(config_file)
batchf.writelines('%s\n' %(opline))

opline = 'my-job-stats -a -n -s'
batchf.writelines('%s\n' %(opline))

batchf.close()

# submit job
cmd = 'sbatch %s' %(batch_fname)
print(cmd)
os.system(cmd)
