    import sys
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/build/")
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/mapspectra/python")
import os
from spt3g import core, maps
import numpy as np
import healpy as hp
import curved_sky


################################################################################################################################################################################################

#field = 'summera'
freq1_arr = [150]
freq2_arr = [150]
seed_arr = np.arange(1,11)

mock_dir = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/skies/lensed/maps/' #'/sptlocal/analysis/summer_fields_2y/sims/mocks/results/coadds/no_ps_masking/'
mock_file = 'mockobsinput_tqu1_agora0.7_datamatched_mcmccal_0707231033_freqghz_seedseedval.fits' #'coadd_tqu1_agora0.7_datamatched_mcmccal_0707231033_main_field_freqghz_seedseedval.g3'
mask_file = '/data/gpfs/projects/punim1922/summerfield_lensing/files/exp/masks/10oct2022_widemask_summer-all.fits' #'/sptlocal/analysis/summer_fields_2y/masks/4apr2024_summer12/threshold_frac0.3/apodized_mask_from_weights_main_field_dsmooth0d25_width0d166_ptsrcmask_main_field_sources_dsmooth0d0_width0d03.fits'

ps_dir = '/data/gpfs/projects/punim1922/summerfield_lensing/sims/skies/lensed/spectra/' #'/sptlocal/analysis/summer_fields_2y/sims/mocks/results/spectra/no_ps_masking/'
ps_file = 'ps_mockobsinput_tqu1_agora0.7_datamatched_mcmccal_0707231033_freq1ghz_freq2ghz_seedseedval.npy' #'ps_tqu1_agora0.7_datamatched_mcmccal_0707231033_main_field_freq1ghz_freq2ghz_seedseedval.npy'

#Cl params
binning = 1
nside = 8192
lmax = nside*3-1
lmin = 0
return_mask = False
return_error = False
return_kernel = False
lfac = 0
verbose = True
pixwin = 'NO'
pixwin_lab = 'False'
beam1 = None
beam2 = None
symmetric_cl = 0
apodizetype = 1
apodizesigma = 30
thetamax = 30
tolerance = 1.e-08


###################################################################################################################################################################################################


def read_map_frame(path, id=None):
    """
    Return first map frame in .g3/.fits file located at `path`.
    """
    if not os.path.exists(path):
        raise OSError("File {} does not exist".format(path))

    if path.split(".")[-1] == 'g3': 
        for frame in core.G3File(path): 
            if frame.type == core.G3FrameType.Map:
                return frame
        raise RuntimeError("Map frame {} not found G3File.".format(id))
    elif path.split(".")[-1] == 'fits':
        return maps.fitsio.load_skymap_fits(path)

    raise OSError("G3File {} does not contain a map frame .".format(path))


#if field == 'summera':
#    mask_in_file = mask_file.replace('main_field', 'summer-a')
#if field == 'summerb':
#    mask_in_file = mask_file.replace('main_field', 'summer-b')
#if field == 'summerc':
#    mask_in_file = mask_file.replace('main_field', 'summer-c')
#print(mask_in_file)
mask_in_file = mask_file
mask_in = hp.read_map(mask_in_file)
mask_in = hp.ud_grade(mask_in, 8192)



for seedval in seed_arr:
    print(seedval)
    for freq1 in freq1_arr:
        for freq2 in freq2_arr:
            print(freq1, freq2)
            mock_path1 = (mock_dir+mock_file).replace('freq', str(freq1)).replace('seedval', str(seedval)) #.replace('main_field', field)
            mock_path2 = (mock_dir+mock_file).replace('freq', str(freq2)).replace('seedval', str(seedval)) #.replace('main_field', field)
            print(mock_path1)
            print(mock_path2)
    
            hmap1 = read_map_frame(mock_path1)
            hmap2 = read_map_frame(mock_path2)
    
            ellb, spec = curved_sky.spectrum_spice(
                    map1 = hmap1, 
                    map2 = hmap2, 
                    lmin=lmin, 
                    lmax=lmax, 
                    bin_width=binning,
                    return_mask=return_mask,
                    mask=mask_in,
                    return_error=return_error,
                    return_kernel=return_kernel, 
                    lfac=lfac, 
                    verbose=verbose,
                    beam=beam1,
                    pixwin=pixwin,
                    beam2=beam2,
                    pixwin2=pixwin,
                    symmetric_cl=symmetric_cl,
                    apodizetype=apodizetype,
                    apodizesigma=apodizesigma,
                    thetamax=thetamax,
                    tolerance=tolerance)
        
            ps_path = (ps_dir+ps_file).replace('freq1', str(freq1)).replace('freq2', str(freq2)).replace('seedval', str(seedval)) #.replace('main_field', field)
            print(ps_path)
            np.save(ps_path, spec)



