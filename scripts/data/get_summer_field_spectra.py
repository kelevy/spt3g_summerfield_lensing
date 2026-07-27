import numpy as np, glob, healpy as hp, os, sys, argparse
from pylab import *

import warnings
warnings.filterwarnings("ignore")

spt3g_folder = '/sptlocal/user/sri/spt3g_software_collection/wide_fields/spt3g_software/'
datafolder = '%s/simulations/python/data/' %(spt3g_folder)
cambfolder = 'camb/planck18_TTEEEE_lowl_lowE_lensing_highacc'
builddir = '%sbuild' %(spt3g_folder)
sys.path.append(builddir)

from spt3g import core
from spt3g.mapspectra import map_analysis, curved_sky
from spt3g import maps

parser = argparse.ArgumentParser()
cli = parser.add_argument_group('Command Line Inputs')
cli.add_argument('--summer_field_arr', dest='summer_field_arr', action="append")
cli.add_argument('--null_test_name_arr', dest='null_test_name_arr', action="append")
args = parser.parse_args()

# replace args.name by dest
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

#specs
fd_pref = '/sptgrid/user/guidi/maprun_4apr2024/4apr2024_summer12_coadd_wafers_null_config_dir/'
output_dir = '/sptlocal/analysis/summer_fields_2y/lensing/noise_spectra/'
band_keyname_arr = [('90GHz', '90GHz'), ('90GHz', '150GHz'), ('90GHz', '220GHz'), 
                    ('150GHz', '150GHz'), ('150GHz', '220GHz'), ('220GHz', '220GHz')]
nber_bundles = 10
scale_fac = 1e6

#power spectrum specs for spice
bin_width = 1
nside = 2048
lmax = 2 * nside 
lmin = 0
verbose = True

def add_frames(frame1, frame2, check_weights = True, final_frame_id = 'coadd'): 
    import copy
    frame1 = copy.deepcopy( frame1 )
    frame2 = copy.deepcopy( frame2 )
    for key in ['T', 'Q', 'U', 'Wpol', 'Wunpol']:
        if key not in frame1 and key not in frame2: continue        
        if check_weights:
            assert frame1[key].weighted and frame2[key].weighted
        frame1_val, frame2_val = frame1[key], frame2[key]
        del frame1[key]
        frame1[key] = frame1_val + frame2_val
    del frame1['Id']
    frame1['Id'] = final_frame_id
    return frame1

def read_map(f, debug = False, band = '150GHz'): #read map frames
    m = core.G3File(f)
    map_frame = None
    for frame in m:
        if frame.type != core.G3FrameType.Map: continue
        ##print(frame['Id'])
        if frame['Id'] != band: continue
        if debug: print(f, frame['Id'])
        if map_frame is None:
            map_frame = frame
        else:
            map_frame = add_frames(frame, map_frame)
    return map_frame

def get_cl_from_polspice(m1, m2 = None, mask = None, bl1 = None, bl2 = None, 
                         lmin = 100, lmax = 3000, bin_width = 50, 
                         return_mask = False, return_error = False, return_kernel = False,
                         pixwin = 'NO', pixwin_lab = 'False', lfac = 0, 
                         verbose = False, symmetric_cl = 0, apodizetype = 1, 
                         apodizesigma = 30, thetamax = 30, tolerance = 1.e-08):

    
    binned_el, binned_cl = curved_sky.spectrum_spice(
            map1 = m1, 
            map2 = m2, 
            lmin=lmin, 
            lmax=lmax, 
            bin_width=bin_width,
            return_mask=return_mask,
            mask=apod_mask,
            return_error=return_error,
            return_kernel=return_kernel, 
            lfac=lfac, 
            verbose=verbose,
            beam=bl1,
            pixwin=pixwin,
            beam2=bl2,
            pixwin2=pixwin,
            symmetric_cl=symmetric_cl,
            apodizetype=apodizetype,
            apodizesigma=apodizesigma,
            thetamax=thetamax,
            tolerance=tolerance)

    return binned_el, binned_cl

# get masks
mask_fname_pref = '/sptlocal/user/kevinlevy/summerfield_lensing/data/masks/point_source_mask_6.0-5000.0mJy_nside2048_apodised_withborderapodmask_fieldname.fits'
apod_mask_dic = {}
for cntr, curr_field_name in enumerate( summer_field_arr ):
    curr_mask_fname = mask_fname_pref.replace('fieldname', curr_field_name)
    apod_mask = H.read_map( curr_mask_fname )
    apod_mask[apod_mask == H.UNSEEN] = 0.
    apod_mask_dic[curr_field_name] = apod_mask

for curr_field_name in summer_field_arr:
    for curr_null_test_name in null_test_name_arr:
        if curr_null_test_name=='full':
            nl_dic_opfname_bundles = output_dir+'%s_signal_spectra_bundles.npy'%(curr_field_name)
        else:
            nl_dic_opfname_bundles = output_dir+'%s_noise_spectra_bundles.npy'%(curr_field_name)
            
        if os.path.exists(nl_dic_opfname_bundles):
            nl_dic = np.load(nl_dic_opfname_bundles, allow_pickle = True ).item()
        else:
            nl_dic = {}
        if curr_field_name not in nl_dic:
            nl_dic[curr_field_name] = {}
    
        searchstr = '%s/%s/coadd/%s/no_*.g3.gz' %(fd_pref, curr_field_name, curr_null_test_name)
        curr_flist = sorted( glob.glob( searchstr ) )[0:nber_bundles]
        print( curr_field_name, curr_null_test_name, len( curr_flist ) )
        if curr_null_test_name not in nl_dic[curr_field_name]:
            nl_dic[curr_field_name][curr_null_test_name] = {}

        for band_keyname in band_keyname_arr:
            print('\tBands = %sx%s' %(band_keyname[0], band_keyname[1]))
            if band_keyname not in nl_dic[curr_field_name][curr_null_test_name]:
                nl_dic[curr_field_name][curr_null_test_name][band_keyname] = {}
    
            #read maps and compute spectra
            for fcntr1, curr_fname1 in enumerate( curr_flist):
                for fcntr2, curr_fname2 in enumerate( curr_flist):
                    if (fcntr1,fcntr2) in nl_dic[curr_field_name][curr_null_test_name][band_keyname]: continue
                        
                    if fcntr2<fcntr1: continue
                        
                    print('\t\t%s of %s' %(fcntr1+1, len( curr_flist )) )
                    print('\t\t%s of %s' %(fcntr2+1, len( curr_flist )) )
                    map_frame1 = read_map(curr_fname1, band = band_keyname[0] )
                    map_frame2 = read_map(curr_fname2, band = band_keyname[1] )
                    maps.RemoveWeights(map_frame1, zero_nans=True)
                    maps.RemoveWeights(map_frame2, zero_nans=True)
                    tmap1 = np.asarray( map_frame1['T'] )
                    tmap2 = np.asarray( map_frame2['T'] )
    
                    print( tmap1, tmap2 )
    
                    #mask for ps 
                    apod_mask = apod_mask_dic[curr_field_name]

                    binned_el, binned_cl = get_cl_from_polspice(tmap1, m2 = tmap2, mask = apod_mask, 
                                             lmin = lmin, lmax = lmax, bin_width = bin_width)
        
                    nl_dic[curr_field_name][curr_null_test_name][band_keyname][(fcntr1,fcntr2)] = [binned_el, binned_cl*scale_fac]
                    np.save( nl_dic_opfname_bundles, nl_dic )
    
    # get average spectra for each field  
    avg_nl_dic = {}
    avg_nl_dic[curr_field_name] = {}
    for curr_band_keyname in band_keyname_arr:
        avg_nl_dic[curr_field_name][curr_band_keyname] = {}
        nl_auto_arr = []
        nl_cross_arr = []
        for curr_null_test_name in null_test_name_arr:  
            if curr_null_test_name=='full':
                nl_dic_opfname_bundles = output_dir+'%s_signal_spectra_bundles.npy'%(curr_field_name)
                nl_dic = np.load(nl_dic_opfname_bundles, allow_pickle = True ).item()
                for fcntr in nl_dic[curr_field_name][curr_null_test_name][curr_band_keyname]:
                    if fcntr[0]==fcntr[1]:
                        nl_auto_arr.append(nl_dic[curr_field_name][curr_null_test_name][curr_band_keyname][fcntr])
                    else:
                        nl_cross_arr.append(nl_dic[curr_field_name][curr_null_test_name][curr_band_keyname][fcntr])
                nl_auto_mean = np.mean(nl_auto_arr, axis = 0)
                nl_cross_mean = np.mean(nl_cross_arr, axis = 0)
                nl_auto_arr = []
                nl_cross_arr = []
            else:
                nl_dic_opfname_bundles = output_dir+'%s_noise_spectra_bundles.npy'%(curr_field_name)
                nl_dic = np.load(nl_dic_opfname_bundles, allow_pickle = True ).item()
                for fcntr in nl_dic[curr_field_name][curr_null_test_name][curr_band_keyname]:
                    if fcntr[0]==fcntr[1]:
                        nl_auto_arr.append(nl_dic[curr_field_name][curr_null_test_name][curr_band_keyname][fcntr])
                    else:
                        nl_cross_arr.append(nl_dic[curr_field_name][curr_null_test_name][curr_band_keyname][fcntr])
                nl_auto_mean = np.mean(nl_auto_arr, axis = 0)
                nl_cross_mean = np.mean(nl_cross_arr, axis = 0)
            avg_nl_dic[curr_field_name][curr_band_keyname]['auto'] = nl_auto_mean
            avg_nl_dic[curr_field_name][curr_band_keyname]['cross'] = nl_cross_mean
            if curr_null_test_name=='full':
                np.save(output_dir+'%s_signal_spectra_avg.npy'%(curr_field_name), avg_nl_dic )
            else:
                np.save(output_dir+'%s_noise_spectra_avg.npy'%(curr_field_name), avg_nl_dic )
    
    print('%s done'%(curr_field_name))