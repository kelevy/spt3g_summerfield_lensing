#folders
spt3g_folder = '/data/gpfs/projects/punim1922/spt3g_software/'
datafolder = '%s/simulations/python/data/' %(spt3g_folder)
cambfolder = 'camb/planck18_TTEEEE_lowl_lowE_lensing_highacc' 
import sys
builddir = '%sbuild' %(spt3g_folder)
sys.path.append(builddir)
ilcdir = '%s/ilc/python' %(spt3g_folder)
sys.path.append(ilcdir)

from spt3g import core
from spt3g.simulations import foregrounds as fg
from spt3g.mapspectra import basicmaputils as utils
from spt3g.beams import beam_analysis as beam_stuff
from spt3g.mapspectra import map_analysis
from spt3g import maps
from spt3g.simulations import cmb

import ilc
import inpaint

import numpy as np, sys, os, scipy as sc, warnings, healpy as H
import pandas as pd
from copy import deepcopy
from astropy.io import fits


##########################################################################################################################################################################


fieldname = 'summer_c'
nsims = 250
nber_bundles = 1

fieldsize_dic = {'summer_a':1500, 'summer_b':600, 'summer_c':900}
nuarr = [90, 150, 220]
freqcomb_arr = ['90GHzX90GHz', '90GHzX150GHz', '90GHzX220GHz', '150GHzX150GHz', '220GHzX150GHz','220GHzX220GHz']
band_keyname_arr = [('90GHz', '90GHz'), ('90GHz', '150GHz'), ('90GHz', '220GHz'), ('150GHz', '150GHz'), ('150GHz', '220GHz'), ('220GHz', '220GHz')]
nu1nu2_arr = [(90, 90), (90, 150), (90, 220), (150, 150), (150, 220), (220, 220)] #ordered for synfast "new = False"
lmax = 4000
nside = 2048
mk_to_uk = 1e3

final_comp_for_ilc = 'cmb'
reqd_ilc_keynames = ['cmbmv']
source_mask_str = '20.0-5000.0mJy'
source_template_str = '6.0to5000.0mJy'
use_sim_as_data = True
deconvolve_tf = True
mask_ilc_map = True
perform_inpainting = True
fullsky_or_flatsky_inp = 'flat'

which_analytical_cl = 'mdpl2'
which_sim = 'mdpl2'
mdpl2_version='v0.7'
source_masking_thresh = 3.0
tf_threshold_for_lmin = 0.1
possible_rounded_lmin_arr = np.asarray( [300, 350, 400, 450, 500])

els = np.arange(lmax)
expname = 'spt3g_%s_2y' %(fieldname)
exparr = np.tile(expname, len(nuarr))

if os.path.exists('/sptlocal/analysis/'):
    scott = True
    spartan, campuscluster = False, False
elif os.path.exists('/u/srinirag/projects_caps/'):
    campuscluster = True
    scott, spartan = False, False
elif os.path.exists('/data/gpfs/projects/punim1922/'):
    spartan = True
    scott, campuscluster = False, False

# TF files
if expname == 'spt3g_summer_a_2y':
    if scott:
        tf_file = '/sptlocal/analysis/summer_fields_2y/lensing/tfs/summer-a/tf1d_%sghz_250sims.npz'
        tf2d_file = '/sptlocal/analysis/summer_fields_2y/lensing/tfs/summer-a/tf2d_%sghz_250sims.npz'
    elif spartan:
        tf_file = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/tfs/summer-a/tf1d_%sghz_100sims.npy'
        tf2d_file = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/tfs/summer-a/tf2d_freqghz_250sims.npz'
elif expname == 'spt3g_summer_b_2y':
    if scott:
        tf_file = '/sptlocal/analysis/summer_fields_2y/lensing/tfs/summer-b/tf1d_%sghz_250sims.npz'
        tf2d_file = '/sptlocal/analysis/summer_fields_2y/lensing/tfs/summer-b/tf2d_%sghz_250sims.npz'
    elif spartan:
        tf_file = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/tfs/summer-b/tf1d_%sghz_100sims.npy'
        tf2d_file = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/tfs/summer-b/tf2d_freqghz_250sims.npz'
elif expname == 'spt3g_summer_c_2y':
    if scott:
        tf_file = '/sptlocal/analysis/summer_fields_2y/lensing/tfs/summer-c/tf1d_%sghz_250sims.npz'
        tf2d_file = '/sptlocal/analysis/summer_fields_2y/lensing/tfs/summer-c/tf2d_%sghz_250sims.npz'
    elif spartan:
        tf_file = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/tfs/summer-c/tf1d_%sghz_100sims.npy'
        tf2d_file = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/tfs/summer-c/tf2d_freqghz_250sims.npz'

# beam files
if scott:
    beam_file_rc4 = '/home/sri/analysis/spt3g_summer_field_lensing/healqest/data/beams/rc4/B_ell.npz'
    beam_file_v2 = '/home/sri/analysis/spt3g_summer_field_lensing/healqest/data/beams/compiled_2020_beams.txt'
elif spartan:
    beam_file_rc4 = '/data/gpfs/projects/punim1922/summerfield_lensing/data/beams/B_ell.npz'
    beam_file_v2 = '/data/gpfs/projects/punim1922/summerfield_lensing/data/beams/compiled_2020_beams.txt'

# noise files
if scott:
    nl_dir = '/sptlocal/user/guidi/summer_spectra/10oct2022_summer12/signflips_noise_curves_and_cov/recalibrated_interfrequency_TPleakage_Pcal_EEpla/'
elif spartan:
    nl_dir = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/spectra/'

if expname == 'spt3g_summer_a_2y':
    nl_file = 'summer-a/full/cls_freq1ghz_freq2ghz_avg100.npy'
elif expname == 'spt3g_summer_b_2y':
    nl_file = 'summer-b/full/cls_freq1ghz_freq2ghz_avg100.npy'
elif expname == 'spt3g_summer_c_2y':
    nl_file = 'summer-c/full/cls_freq1ghz_freq2ghz_avg100.npy'

# foreground directory
if scott:
    fg_dir = '/home/sri/analysis/spt3g_summer_field_lensing/healqest/data/foregrounds/spectra/'
elif spartan:
    fg_dir = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/foregrounds/spectra/'

# ILC directory
if scott:
    ilc_dir = '/sptlocal/analysis/summer_fields_2y/lensing/ilc/'
elif spartan:
    ilc_dir = '/data/gpfs/projects/punim1922/summerfield_lensing/ilc/'


##########################################################################################################################################################################


def combine_signal_noise_cl(nuarr, cl_dic, nl_dic = None):
    cl_nl_dic = {}
    for (b1, b2) in cl_dic.keys():
        if b1 not in nuarr or b2 not in nuarr: continue
        keyname = ('%sGHz' %(b1), '%sGHz' %(b2))
        if nl_dic is not None:
            cl_nl_dic[keyname] = cl_dic[(b1, b2)] + nl_dic[(b1, b2)]
        else:
            cl_nl_dic[keyname] = cl_dic[(b1, b2)]
    return cl_nl_dic

def get_fg_spectra(fd, els, nuarr, nsims = 10):

    cl_dict = {}
    for nu1 in nuarr:
        for nu2 in nuarr:
            if nu1 == 90:
                nu1_mod = 95
            else:
                nu1_mod = nu1
            if nu2 == 90:
                nu2_mod = 95
            else:
                nu2_mod = nu2
            fname = '%s/cls_allfg%s_allfg%s_%ssims.dat' %(fd, nu1_mod, nu2_mod, nsims)
            if not os.path.exists( fname ):
                fname = '%s/cls_allfg%s_allfg%s_%ssims.dat' %(fd, nu2_mod, nu1_mod, nsims)

            curr_el, curr_cl = np.loadtxt( fname, usecols = [0,1], unpack = True )
            cl_dict[(nu1, nu2)] = np.interp(els, curr_el, curr_cl)

    return cl_dict

def read_map(f, band, debug = False):
    m = core.G3File(f)
    for frame in m:
        if frame.type != core.G3FrameType.Map: continue
        if frame['Id'] != band: continue
        if debug: print(frame)
        return frame

def read_map_frame(f, debug = False):
    m = core.G3File(f)
    for frame in m:
        if frame.type != core.G3FrameType.Map: continue
        if debug: print(frame)
        return frame

def get_map_locations(fieldname, nuarr, data_or_sim, bundle_index, sim_index, source_template_str = '6.0to5000.0mJy', scott=True, mask_ilc_map = True):
    
    if scott:
        sim_fd = '/sptlocal/analysis/summer_fields_2y/sims/mocks/results/coadds/' 
        psource_template_fd = '/sptlocal/analysis/summer_fields_2y/lensing/point_source_sims/coadds/' 
        op_fd = '/sptlocal/analysis/summer_fields_2y/lensing/ilc/'
        if not mask_ilc_map:
            op_fd = '%s/unmasked/' %(op_fd)
        if fieldname == 'summer_a':
            data_fd = '/sptgrid/user/guidi/10oct2022_summer12_20bundles_null_tests_recalibrated_interfrequency_TPleakage_Pcal_EEpla/summer/coadd/full/' 
            point_source_apod_mask_fname = '/sptlocal/analysis/summer_fields_2y/lensing/masks/apodized_mask_from_weights_summer-a_dsmooth0d25_width0d166_ptsrcmask_summer-a_sources_dsmooth0d0_width0d03.fits'
            noise_fd = ''
        if fieldname == 'summer_b':
            data_fd = '/sptgrid/user/guidi/10oct2022_summer12_20bundles_null_tests_recalibrated_interfrequency_TPleakage_Pcal_EEpla/summer-b/coadd/full/' 
            point_source_apod_mask_fname = '/sptlocal/analysis/summer_fields_2y/lensing/masks/apodized_mask_from_weights_summer-b_dsmooth0d25_width0d166_ptsrcmask_summer-b_sources_dsmooth0d0_width0d03.fits' 
            noise_fd = ''
        if fieldname == 'summer_c':
            data_fd = '/sptgrid/user/guidi/10oct2022_summer12_20bundles_null_tests_recalibrated_interfrequency_TPleakage_Pcal_EEpla/summer-c/coadd/full/' 
            point_source_apod_mask_fname = '/sptlocal/analysis/summer_fields_2y/lensing/masks/apodized_mask_from_weights_summer-c_dsmooth0d25_width0d166_ptsrcmask_summer-c_sources_dsmooth0d0_width0d03.fits'  
            noise_fd = ''

    else: #campuscluster/spartan
        sim_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/coadds/maps/unl/' 
        psource_template_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/mocks/coadds/maps/'  
        op_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/ilc/maps/leftright/'
               
        if fieldname == 'summer_a':
            data_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/data/coadds/full/maps/summer-a/full/'  
            point_source_apod_mask_fname = '/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/apod_mask_45arcmin_15arcmin_summer-a_100mJy.fits' 
            noise_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/maps/summer-a/full/'
        if fieldname == 'summer_b':
            data_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/data/coadds/full/maps/summer-b/full/'  
            point_source_apod_mask_fname = '/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/apod_mask_45arcmin_15arcmin_summer-b_100mJy.fits'
            noise_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/maps/summer-b/full/'
        if fieldname == 'summer_c':
            data_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/data/coadds/full/maps/summer-c/full/'  
            point_source_apod_mask_fname = '/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/apod_mask_45arcmin_15arcmin_summer-c_100mJy.fits' 
            noise_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/noise/maps/summer-c/full/'
 
    mapfname_dic = {}
    psource_template_fname_dic = {}
    noise_maps_dic = {}
    for nu in nuarr:
        if data_or_sim == 'data': #bundle index matters
            mapfname_dic[nu] = '%s/full_coadd.g3.gz' %(data_fd)
        elif data_or_sim == 'sims':  #sim index matter       
            mapfname_dic[nu] = '%s/coadd_tqu1_agora0.7_datamatched_mcmccal_0707231033_summerall_%sghz_seed%s.g3' %(sim_fd, nu, sim_index)
        psource_template_fname_dic[nu] = '%s/coadd_point_source_map_%s_%sghz.g3' %(psource_template_fd, source_template_str, nu)
        noise_maps_dic[nu] = '%s/%sghz_%s.fits' %(noise_fd, nu, sim_index) 
    
    return mapfname_dic, psource_template_fname_dic, point_source_apod_mask_fname, noise_maps_dic, op_fd

def make_noise_realisation(nl_dic, nside, lmax = 6143, total_bundles = 1, alm_or_map = 'map', nu1nu2_arr=nu1nu2_arr):
    assert alm_or_map in ['map', 'alm']
    
    nl_arr = []
    for nu1nu2 in nu1nu2_arr:
        curr_nl = nl_dic[nu1nu2] * 10 #bundle scaling
        nl_arr.append( curr_nl )
   
    noise_alm_dic = {}
    if len(nl_arr)==1:
        noise_alm_dic[150] = H.synalm(nl_arr, lmax=lmax, new=False)[0]
    else:
        noise_alm_dic[90], noise_alm_dic[150], noise_alm_dic[220] = H.synalm( nl_arr, lmax=lmax, new = False )

    if alm_or_map == 'alm':
        return noise_alm_dic
    
    noise_sim_dic = {}
    for nu in noise_alm_dic:
        noise_sim_dic[nu] = H.alm2map( noise_alm_dic[nu], nside = nside )
    
    return noise_sim_dic

def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
    lmaxin  = H.Alm.getlmax(alm.shape[0])
    print( "reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
    ell,emm = H.Alm.getlm(lmaxin)
    almout  = np.zeros(H.Alm.getsize(lmax),dtype=np.complex_)
    oldi=0
    oldf=0
    newi=0
    newf=0
    dl = lmaxin-lmax
    for i in range(0,lmax+1):
        oldf=oldi+lmaxin+1-i
        newf=newi+lmax+1-i
        almout[newi:newf]=alm[oldi:oldf-dl]
        oldi=oldf
        newi=newf
    return almout

def round_inp_rad_am(inp_rad_am, possible_inp_rad_am_arr = [6., 8., 10., 15., 20.], rad_am_for_nans = 10.):
    if np.isnan(inp_rad_am): return rad_am_for_nans
    possible_inp_rad_am_arr = np.asarray( possible_inp_rad_am_arr )
    inp_rad_am = int(inp_rad_am) + 1
    closest_ind = np.argmin( abs( possible_inp_rad_am_arr - inp_rad_am ) )
    return possible_inp_rad_am_arr[closest_ind]

def get_mask_radius_outer(inp_rad):
        if inp_rad <= 6:
            return 25.
        elif inp_rad <= 10:
            return 25.
        elif inp_rad <= 20:
            return 50.
        else:
            return 50.

def inpaiting_wrapper(curr_hmap, inp_ra_list, inp_dec_list, inp_rad_am_list, parent_sigma_dic, perform_inp = True, perform_intrp = True):
    for inpcntr, (r, d, rad) in enumerate( zip(inp_ra_list, inp_dec_list, inp_rad_am_list) ):
        mask_radius_inner_am = rad
        mask_radius_outer_am = get_mask_radius_outer(mask_radius_inner_am)
        sigma_dic = parent_sigma_dic[(mask_radius_inner_am, mask_radius_outer_am)]

        #get flatsky
        map_to_inpaint = H.gnomview(curr_hmap, rot = [r, d], reso=dx, xsize=nx, return_projected_map=True, no_plot=True)

        if perform_inp: #do inpainting
            cmb_inpainted_map, map_inpainted, map_to_inpaint = inpaint.perform_inpainting(r, d, ra_grid, dec_grid, mapparams, map_to_inpaint, sigma_dic, mask_radius_inner_am, mask_radius_outer_am, nside = nside)
        else:
            map_inpainted = np.copy( map_to_inpaint )

        #interpolate from flatsky to fullsky now
        curr_hmap = inpaint.interpolate_from_flat_to_fullsky(curr_hmap, r, d, ra_grid, dec_grid, map_inpainted, mask_radius_inner_am)

    return curr_hmap


##########################################################################################################################################


# get TF (power units)
tf1d_pow_units = {}
for nu in nuarr:
    tf1d_pow_units[nu] = np.load(tf_file %(nu)) #np.load(tf_file %(nu))['tf1d'][0].real
    tf1d_pow_units[nu][np.isnan(tf1d_pow_units[nu]) | np.isinf(tf1d_pow_units[nu])] = 0.
    curr_el = np.arange( len( tf1d_pow_units[nu]) )
    tf1d_pow_units[nu] = np.interp( els, curr_el, tf1d_pow_units[nu])

closest_lmin = np.argmin( abs(tf1d_pow_units[150]-tf_threshold_for_lmin))
lmin_ind = np.argmin( abs(possible_rounded_lmin_arr-closest_lmin) )
lmin = possible_rounded_lmin_arr[lmin_ind]
els_inds_to_null = np.where( els< lmin)[0]


# get beams: 
## (a) beams for ILC weights: rc4
## (b) beams for deconvol: rc4 for data and v2 for sims.
## (c) effective beams: v2 150 (same as the one used for sims).

# rc4
beam_dic_rc4 = np.load( beam_file_rc4 )
bl_dic_rc4 = {}
for nu in nuarr:
    #if blkeyname == 'ell': continue
    curr_el, curr_bl = beam_dic_rc4['ell'], beam_dic_rc4[str(nu)]
    curr_bl = np.interp(els, curr_el, curr_bl)
    bl_dic_rc4[nu] = curr_bl

# v2
beam_dic_v2 = {}
beam_dic_v2['ell'] = np.loadtxt(beam_file_v2, unpack=True)[0]
for i, nu in enumerate(nuarr):
    beam_dic_v2[nu] = np.loadtxt(beam_file_v2, unpack=True)[i+1]

bl_dic_v2 = {}
for nu in nuarr:
    #if blkeyname == 'ell': continue
    curr_el, curr_bl = beam_dic_v2['ell'], beam_dic_v2[nu]
    curr_bl = np.interp(els, curr_el, curr_bl)
    bl_dic_v2[nu] = curr_bl

# effective beams
bl_dic = deepcopy( bl_dic_rc4 )
bl_dic['effective'] = bl_dic_v2[150]


noise_dic = {}
for nu1nu2 in nu1nu2_arr:
    noise_dic[nu1nu2] = np.load(nl_dir+nl_file.replace('freq1', str(nu1nu2[0])).replace('freq2', str(nu1nu2[1])))


nl_dic = {}
nl_dic_no_beam_deconv = {}
for band1band2 in noise_dic:

    nu1, nu2 = band1band2

    # get nl
    curr_nl = noise_dic[band1band2]
    curr_nl = np.interp(els, np.arange(len(curr_nl)), curr_nl)
    nl_dic_no_beam_deconv[(nu1, nu2)] = nl_dic[(nu2, nu1)]= curr_nl

    # deconvolve the beam and TF
    curr_nl /= (bl_dic[nu1] * bl_dic[nu2]) 
    curr_nl /= (tf1d_pow_units[nu1]*tf1d_pow_units[nu2])**0.5 
    curr_nl[np.isnan(curr_nl) | np.isinf(curr_nl)] = 0.
    
    nl_dic[(nu1, nu2)] = nl_dic[(nu2, nu1)] = curr_nl

# get cl_dict
print('\nget Cl and combine with Nl\n')
cl_dict_analytic = get_fg_spectra(fg_dir, els, nuarr)
cl_mod_dic_tt = combine_signal_noise_cl(nuarr, cl_dict_analytic, nl_dic = nl_dic)
cl_1d_dict = {}
cl_1d_dict['TT'] = cl_mod_dic_tt

# define the cib null comp now
if expname in ['spt3g_summer_a_2y', 'spt3g_summer_b_2y', 'spt3g_summer_c_2y']: #20250323 - to be changed later.
    if final_comp_for_ilc == 'cmb':
        cib_null_comp=['misc_cib_tcib10.00_beta2.60', 'misc_cib_tcib20.00_beta1.80'] #ell = 500, 3500
    elif final_comp_for_ilc == 'y':
        cib_null_comp=['misc_cib_tcib30.00_beta2.60', 'misc_cib_tcib5.00_beta1.80'] #ell = 3000, 5000
    
print(cib_null_comp)

print('\n\tdefine ILCs')
perform_1d_ilc = True 
comp_dic_for_ilc = {}

if final_comp_for_ilc == 'cmb': #20230805 - cmb
    #standard ILC
    comp_dic_for_ilc['cmbmv'] = ['CMB', None, ['CMB', 'kSZ'], 'CMB: MV'] #final_comp required from ILC (CMB in this case), components to be nulled (nothing in this case), and ignore_fg (cmb, ksz) list.

    comp_dic_for_ilc['cmbtszfree'] = ['CMB', ['y'], ['CMB', 'kSZ'], 'CMB: tSZ-free'] #final_comp required from ILC (CMB in this case), components to be nulled (nothing in this case), and ignore_fg (cmb, ksz) list.
    comp_dic_for_ilc['cmbcibfree'] = ['CMB', cib_null_comp, ['CMB', 'kSZ'], 'CMB: CIB-free'] #final_comp required from ILC (CMB in this case), components to be nulled (nothing in this case), and ignore_fg (cmb, ksz) list.
    
elif final_comp_for_ilc == 'y': ##20230805 - Compton-y
    #mv
    comp_dic_for_ilc['ymv'] = ['y', None, ['y'], 'y: MV'] #final_comp required from ILC (CMB in this case), components to be nulled (nothing in this case), and ignore_fg (cmb, ksz) list.
    #cmb-free
    comp_dic_for_ilc['ycmbfree'] = ['y', 'CMB', ['y'], 'y: CMB-free']
    #cib-free
    comp_dic_for_ilc['ycibfree'] = ['y', cib_null_comp, ['y'], 'y: CIB-free']

# perform ilc and get weights now
print('\n\tperform ilc and get weights now.')
ilc_op_dic = {}
shape = None 

reqd_bands = ['%sGHz' %(nu) for nu in nuarr]
explist = exparr

for keyname in comp_dic_for_ilc:
    final_comp, null_comp, ignore_fg, tit = comp_dic_for_ilc[keyname]
    if keyname == 'cmbtszfree_noisereduced':
        print('\t\t\t%s with reduced noise' %(keyname))
        if perform_1d_ilc:
            cl_dic_to_use = cl_1d_dict_reduced_noise
        else:
            cl_dic_to_use = cl_dict_reduced_noise  
    elif keyname == 'cmbtszfree_mintszcibonly':
        print('\t\t\t%s for min tsz/cib only' %(keyname))
        if perform_1d_ilc:
            cl_dic_to_use = cl_1d_dict_mintszcib
        else:
            cl_dic_to_use = cl_dict_mintszcib 
    elif keyname == 'cmbcibfree_mincibnoiseonly':
        print('\t\t\t%s for min cib/noise only' %(keyname))
        if perform_1d_ilc:
            cl_dic_to_use = cl_1d_dict_mincibnoise
        else:
            cl_dic_to_use = cl_dict_mincibnoise
    elif keyname == 'cmbcibfree_mintszcibonly':
        print('\t\t\t%s for min cib/noise only' %(keyname))
        if perform_1d_ilc:
            cl_dic_to_use = cl_1d_dict_mintszcib
        else:
            cl_dic_to_use = cl_dict_mintszcib 
    else:
        if perform_1d_ilc:
            cl_dic_to_use = cl_1d_dict
        else:
            cl_dic_to_use = cl_dict    
    
    curr_weights_arr, curr_cl_residual_arr = ilc.perform_ilc(final_comp, reqd_bands, explist, lmax, lmin = lmin, ell=els, cl_dict = cl_dic_to_use, nl_dict = None, null_components = null_comp, ignore_fg = ignore_fg, res = None, map_dict = None, bl_dict = bl_dic)
    curr_weights_arr[:, :, els_inds_to_null] = 0.
    ilc_op_dic[keyname] = [curr_weights_arr, curr_cl_residual_arr]


# store weights, residuals, and maps
print('\n\t Store weights, residuals, and maps')
weights_1d_dic = {}
for kcntr, keyname in enumerate( ilc_op_dic ):
    print(keyname)
    weights_arr, cl_residual_arr = ilc_op_dic[keyname]
    if not os.path.exists(ilc_dir+'weights/'):
        os.mkdir(ilc_dir+'weights/')
    if not os.path.exists(ilc_dir+'spectra/'):
        os.mkdir(ilc_dir+'spectra/')
    np.save(ilc_dir+'weights/weights_arr_%s_%s.npy'%(keyname, fieldname), weights_arr)
    np.savetxt(ilc_dir+'spectra/cl_residual_arr_%s_%s.dat'%(keyname, fieldname), cl_residual_arr.T)
    final_comp, null_comp, ign_comp, tit = comp_dic_for_ilc[keyname]
    tit = comp_dic_for_ilc[keyname][-1]
    acap = ilc.get_freq_response(reqd_bands, explist, component=final_comp).T
    acap = np.asarray(acap)[0]
    if keyname.find('cmbtszfree')>-1:
        bcap = ilc.get_freq_response(reqd_bands, explist, component='y').T
        bcap = np.asarray(bcap)[0]
    elif keyname.find('cmbcibfree')>-1:
        bcap_arr = []
        for curr_cib_null_comp in cib_null_comp:
            bcap = ilc.get_freq_response(reqd_bands, explist, component=curr_cib_null_comp).T
            bcap = np.asarray(bcap)[0]
            bcap_arr.append( bcap )

    weightsarr_for_sum = []
    if keyname.find('cmbtszfree')>-1:
        weightsarr_for_sum_for_null_comp = []
    elif keyname.find('cmbcibfree')>-1:
        weightsarr_for_sum_for_null_comp_1 = []
        if len( bcap_arr )>1:
            weightsarr_for_sum_for_null_comp_2 = []
    
    nc = len(reqd_bands)
    weights_1d_dic[keyname]={}
    for frqcntr, freq in enumerate( reqd_bands ):
        nu = int(freq.replace('GHz',''))
        curr_weights = np.asarray( weights_arr[frqcntr][0] )
        els = np.asarray(els)
        weights_1d_dic[keyname][nu] = [els, curr_weights]

        weightsarr_for_sum.append( curr_weights * acap[frqcntr] )
        if keyname.find('cmbtszfree')>-1:
            weightsarr_for_sum_for_null_comp.append( curr_weights * bcap[frqcntr] )
        elif keyname.find('cmbcibfree')>-1:
            weightsarr_for_sum_for_null_comp_1.append( curr_weights * bcap_arr[0][frqcntr] )
            if len( bcap_arr )>1:
                weightsarr_for_sum_for_null_comp_2.append( curr_weights * bcap_arr[1][frqcntr] )

    weightsarr_for_sum = np.asarray(weightsarr_for_sum)
    if keyname.find('cmbtszfree')>-1:
        tmp = np.sum(weightsarr_for_sum_for_null_comp, axis = 0)
        print(np.min(tmp), np.max(tmp))




###########################################################################################################################################
#                                                         Create ILC maps/alms                                                            # 
###########################################################################################################################################

print('\nCreate ILC maps/alms')

pixwin = H.pixwin(nside)[:lmax]
pixwin_8192 = H.pixwin(8192)[:lmax]
bundle_arr = np.arange(nber_bundles)


if perform_inpainting: 
    list_for_inp_fname = '/data/gpfs/projects/punim1922/spt3g_software/sources/mask_lists/%sd_ptsrc_list_%s_fg_4apr2024_PLAclusters_SPTclusters_20mJy.txt'%(fieldsize_dic[fieldname], fieldname.replace('_', '-')) 
    list_for_inp = pd.read_csv(list_for_inp_fname, delimiter = '\t', skiprows = 1)
    inp_ra_list, inp_dec_list, inp_rad_ori_deg_list = list_for_inp['RA(deg)'], list_for_inp['DEC(deg)'], list_for_inp['Radius(deg)']
    inp_rad_ori_am_list = inp_rad_ori_deg_list * 60.
    inp_rad_am_list = [round_inp_rad_am(rad) for rad in inp_rad_ori_am_list]

    #modifying the radius for source 2
    #if fieldname=='summer_a':
    #    inp_rad_am_list[1] = 30.
    
    mask_radius_inner_arr = np.unique( inp_rad_am_list )
    mask_radius_outer_arr = [get_mask_radius_outer(rad) for rad in mask_radius_inner_arr]
    mask_radius_inner_outer_arr = [(r1, r2) for (r1, r2) in zip( mask_radius_inner_arr, mask_radius_outer_arr )]

    print( mask_radius_inner_outer_arr )
        
    if fullsky_or_flatsky_inp == 'flat': #flatsky stuff
        field_size_deg = 3.
        dx = 2.
        boxsize_am = field_size_deg * 60. #boxsize in arcmins
        nx = int(boxsize_am/dx)
        mapparams = [nx, nx, dx, dx]
        x1, x2 = -nx/2. * dx, nx/2. * dx
        ##print( mapparams ); sys.exit()
        #get ra, dec or map-pixel grid
        ra = np.linspace(x1,x2, nx) #arcmins
        dec = np.linspace(x1,x2, nx) #arcmins
        ra_grid, dec_grid = np.meshgrid(ra,dec)
        noofsims = 20000
    else:
        field_size_deg = 3.
        noofsims = 20

    #CAMB
    camb_fd = '/data/gpfs/projects/punim1922/summerfield_lensing/healqest/healqest/camb/'
    camb_fname = '%s/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat' %(camb_fd)
    camb_el, camb_dl = np.loadtxt(camb_fname, usecols = [0,1], unpack = True)
    camb_dl_fac = camb_el * (camb_el+1)/2/np.pi
    camb_cl = camb_dl / camb_dl_fac
    camb_cl = np.interp(els, camb_el, camb_cl )
    
    band = '%sGHz' %(nu)
    cl_dic_for_inpainting = {'TT': cl_1d_dict['TT'][(band, band)] + camb_cl}

    parent_sigma_dic_fname = '/data/gpfs/projects/punim1922/summerfield_lensing/ilc/inpainting/%s_parent_sigma_dic_%ssims_%ssky_lmax%s.npy' %(fieldname.replace('_', '-'), noofsims, fullsky_or_flatsky_inp, lmax)
    print('get covariances for inpainting and store in %s' %(parent_sigma_dic_fname))
    if os.path.exists(parent_sigma_dic_fname):
        parent_sigma_dic = np.load(parent_sigma_dic_fname, allow_pickle=True).item()
    else:
        parent_sigma_dic = {}
    for mask_radius_inner_outer in mask_radius_inner_outer_arr:
        mask_radius_inner_am, mask_radius_outer_am = mask_radius_inner_outer
        if (mask_radius_inner_am, mask_radius_outer_am) in parent_sigma_dic:
            print('\talready complete for (%s, %s)' %(mask_radius_inner_am, mask_radius_outer_am))
            continue
        if fullsky_or_flatsky_inp == 'full':
            ra_grid, dec_grid, mapparams = None, None, None
        sigma_dic = inpaint.get_covariance(ra_grid, dec_grid, mapparams, els, cl_dic_for_inpainting, bl_dic[nu], noofsims, mask_radius_inner_am, mask_radius_outer_am, nside = nside, field_size_deg = field_size_deg)
    
        parent_sigma_dic[(mask_radius_inner_am, mask_radius_outer_am)] = sigma_dic
        np.save(parent_sigma_dic_fname, parent_sigma_dic)

    print( 'Covariance for inpainting obtained. Keys are', parent_sigma_dic.keys() )


if use_sim_as_data:
    data_or_sim_arr = ['sims']
else:
    data_or_sim_arr = ['data', 'sims']

for data_or_sim in data_or_sim_arr:

    print( 'data_or_sim = %s' %(data_or_sim) )
    
    # beams for deconvolution
    if data_or_sim == 'data':
        bl_dic_for_deconv = deepcopy( bl_dic_rc4 )
        sim_arr = [0]
    elif data_or_sim == 'sims':
        bl_dic_for_deconv = deepcopy( bl_dic_v2 )
        if use_sim_as_data:
            sim_arr = np.arange(0, nsims+1)
        else:
            sim_arr = np.arange(1, nsims+1)

    # rebeamed weights
    weights_1d_dic_rebeamed = {}
    for keyname in reqd_ilc_keynames:
        weights_1d_dic_rebeamed[keyname] = {}
        for nu in bl_dic:
            if nu == 'effective': continue
            bl_eff = bl_dic['effective']
            els, wl = weights_1d_dic[keyname][nu]
            weights_1d_dic_rebeamed[keyname][nu] = [els, wl*bl_eff]

    # Read maps, source template, and mask
    for sim_index in sim_arr:

        print( '\tsim_index = %s' %(sim_index) )

        for bundle_index in bundle_arr:

            print( '\t\tbundle_index = %s' %(bundle_index) )

            #get map locations        
            mapfname_dic, psource_template_fname_dic, point_source_apod_mask_fname, noise_map_dic, op_fd = get_map_locations(fieldname, nuarr, data_or_sim, bundle_index, sim_index, source_template_str = source_template_str, scott=scott, mask_ilc_map = mask_ilc_map)
            
            op_fd_curr = op_fd+'%s/%s/bundle%s/' %(keyname, data_or_sim, bundle_index)
            if not os.path.exists( op_fd_curr ): os.system('mkdir -p %s' %(op_fd_curr))   

            opfname = '%s/%s_tqu1_lmax%s_%s_seed%s.alm' %(op_fd_curr, keyname, lmax, fieldname, sim_index) 
            opfname_map = '%s/%s_tqu1_lmax%s_%s_seed%s.fits' %(op_fd_curr, keyname, lmax, fieldname, sim_index)
            if os.path.exists(opfname):
                print('\t\t\talready done. check %s' %(opfname))
                continue
                
            # source + border mask -- apply it after creating the ILC map.
            point_source_apod_mask = H.read_map( point_source_apod_mask_fname )
            point_source_apod_mask[point_source_apod_mask == H.UNSEEN] = 0.
            
            #read maps and apply weights now
            for keyname in reqd_ilc_keynames: #loop over the ILCs
                print('\t\tCreating the %s ILC map' %(keyname))
                ilc_map_alm = None
                for nu in mapfname_dic:
                    if sim_index == 0: 
                        print('%s GHz sim (seed %s)'%(nu, sim_index))
                        #curr_hmap = np.asarray(read_map_frame( mapfname_dic[nu])['T'])
                        curr_noise_map = H.read_map('/data/gpfs/projects/punim1922/summerfield_lensing/data/coadds/scandir/%s/maps/full_coadd_%sghz.fits' %(fieldname.replace('_', '-'), nu))
                        curr_noise_map *= point_source_apod_mask
                        curr_hmap = curr_noise_map*1e3
                    else:
                        print('%s GHz sim (seed %s)'%(nu, sim_index))
                        #curr_hmap = np.asarray(read_map_frame( mapfname_dic[nu])['T'])
                        curr_noise_map = H.read_map(noise_map_dic[nu])
                        curr_noise_map *= point_source_apod_mask
                        curr_hmap = curr_noise_map*1e3 

                    if perform_inpainting: 
                        curr_hmap = inpaiting_wrapper(curr_hmap, inp_ra_list, inp_dec_list, inp_rad_am_list, parent_sigma_dic)

                    curr_map_alm = H.map2alm(curr_hmap, lmax=lmax)
               

                    # deconvolve tf
                    tf1d = np.load(tf_file %(nu))
                    curr_map_alm = H.almxfl(curr_map_alm, 1/tf1d[:lmax]**0.5)
                        
                    # deconvolve beam
                    curr_map_alm = H.almxfl(curr_map_alm, 1/bl_dic_for_deconv[nu])

                    #apply weights
                    els_, curr_wl_rebeamed = weights_1d_dic_rebeamed[keyname][nu]
                    curr_map_alm = H.almxfl(curr_map_alm, curr_wl_rebeamed)
                    curr_map_alm[np.isnan(curr_map_alm) | np.isinf(curr_map_alm)] = 0.
                
                    #ilc
                    if ilc_map_alm is None:
                        ilc_map_alm = np.copy( curr_map_alm )
                    else:
                        ilc_map_alm += np.copy( curr_map_alm )



                if mask_ilc_map:
                    #convert to map
                    ilc_map_alm[np.isnan(ilc_map_alm) | np.isinf(ilc_map_alm)] = 0.
                    ilc_map = H.alm2map( ilc_map_alm, nside = nside )
                
                    #mask
                    ilc_map = ilc_map * point_source_apod_mask
                    
                    #save
                    if sim_index == 0 or sim_index==1:
                        H.write_map( opfname_map, ilc_map, overwrite=True )
                    
                    #convert back to alm
                    ilc_map_alm = H.map2alm(ilc_map, lmax = lmax) 
                
                ilc_map_alm[np.isnan(ilc_map_alm) | np.isinf(ilc_map_alm)] = 0.    
                               
                H.write_alm( opfname, ilc_map_alm, overwrite=True )
                print('\t\t check %s' %(opfname))      
        
print('\nDone.')
