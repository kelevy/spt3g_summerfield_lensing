import sys
import os
import argparse
import yaml
import numpy as np 


##############################################################################################################################################################


# Create a parser.
# These settings are specified via the command line only.'
parser = argparse.ArgumentParser(description='')
cli = parser.add_argument_group('Command Line Inputs')
cli.add_argument('--config_file', dest = 'config_file', type=str)

#  These settings are specified in the config file.'
config = parser.add_argument_group('Config File Inputs')
config.add_argument('--hh', dest='hh', type=int)
config.add_argument('--mm', dest='mm', type=int)
config.add_argument('--nodes', dest='nodes', type=int)
config.add_argument('--ntasks', dest='nodes', type=int)
config.add_argument('--cpus_per_task', dest='nodes', type=int)
config.add_argument('--memory', dest='memory', type=int)
config.add_argument('--batch_jobs_folder', dest='batch_jobs_folder', type=int)
config.add_argument("--script_loc", dest = 'script_loc', type=str)
config.add_argument("--script_file", dest = 'script_file', type=str)
config.add_argument('--freq_arr', dest='freq_arr', type = int)
config.add_argument("--start", dest = 'start', type=int)
config.add_argument("--end", dest = 'total', type=int)

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


# script path
script_path = script_loc+script_file

# seeds
seed_arr = np.arange(start, end+1)

if not os.path.exists(batch_jobs_folder): 
    os.system('mkdir -p '+batch_jobs_folder)


##############################################################################################################################################################


count = 0 
for seed in seed_arr:

    for freq in freq_arr:

        batch_fname = batch_jobs_folder+'sim_seed%s_freq%s.sh' %(seed, freq)
        batchf = open(batch_fname, 'w' )

        linearr = ['#!/bin/bash', '#SBATCH --partition=cascade', '#SBATCH --account="punim1922"', '#SBATCH --output='+batch_jobs_folder+'/job.o%j']
        for lines in linearr:
            opline = lines  
            batchf.writelines('%s\n' %(opline))
                
        # time 
        opline = '#SBATCH --time=%d:%d:00' %(hh, mm)
        batchf.writelines('%s\n' %(opline))

        # nodes 
        opline = '#SBATCH --nodes=%s' %(args.nodes)
        batchf.writelines('%s\n' %(opline))

        # ntasks 
        opline = '#SBATCH --ntasks=%s' %(args.ntasks)
        batchf.writelines('%s\n' %(opline))

        # cpus per task
        opline = '#SBATCH --cpus-per-task=%s' %(args.cpus_per_task)
        batchf.writelines('%s\n' %(opline))

        # memory
        opline = '#SBATCH --mem=%dG' %(memory)
        batchf.writelines('%s\n' %(opline))

        # more stuff
        opline = 'export SHELL=bash'
        batchf.writelines('%s\n' %(opline))

        opline = '\n\n python %s --freq %s --seed %s --config_file %s' %(script_path, freq, seed, config_file)
        batchf.writelines('%s\n' %(opline))
        
        opline = 'my-job-stats -a -n -s'
        batchf.writelines('%s\n' %(opline))
    
        batchf.close()
        
        # submit job
        cmd = 'sbatch %s' %(batch_fname)
        print(cmd)
        os.system(cmd)
        count += 1

print('Total jobs submitted = %s\n' %(count))
sys.exit()
