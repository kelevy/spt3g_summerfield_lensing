import numpy as np
import healpy as hp

import sys, os
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/build/")
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/mapspectra/python")
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/sources/python")
from spt3g import core, maps
import curved_sky
import lists


field='summer-b'
threshold_frac= 0.3
map_id = ['90GHz', '150GHz', '220GHz']
nside=2048
source_list = '/data/gpfs/projects/punim1922/spt3g_software/sources/mask_lists/600d_ptsrc_list_%s_fg_4apr2024_PLAclusters_SPTclusters.txt'%(field)
border_apod_angle=0.75*core.G3Units.degree
source_apod_angle=15*core.G3Units.arcmin
dist_smooth_angle=5.0*core.G3Units.arcmin


#################################################################################################################################################


#border_mask = np.ones(hp.nside2npix(nside))
#map_name_in = '/data/gpfs/projects/punim1922/summerfield_lensing/data/coadds/full/maps/%s/full/full_coadd.g3.gz'%(field)        
#g3file = core.G3File(map_name_in)
#for frame in g3file:
#    if frame.type == core.G3FrameType.Map:
#        ids = frame["Id"]
#        if(ids == map_id[0] or ids == map_id[1] or ids == map_id[2]):
#            mask_wei = np.asarray(curved_sky.make_binary_border_mask(frame["Wpol"].TT, threshold_frac=threshold_frac).to_map())
#            mask_wei = hp.ud_grade(mask_wei, nside_out = nside)
#            border_mask = border_mask*mask_wei
        


border_mask = hp.read_map('/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/border_mask_%s.fits'%(field))
border_mask[border_mask==hp.UNSEEN]=0
#hp.write_map('/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/border_mask_%s.fits'%(field), border_mask, overwrite=True)

apd_brdr = curved_sky.apodize_binary_mask_conv(border_mask, apod_angle=border_apod_angle)
hp.write_map('/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/apod_mask_45arcmin_%s.fits'%(field), apd_brdr, overwrite=True)


radii = lists.read_point_source_mask_file(source_list)[3]
unique_radii = np.unique(radii)
if len(unique_radii) == 1:
    fill_disk = False
    apod_start_dist = unique_radii[0] / core.G3Units.rad
else:
    fill_disk = True
    apod_start_dist = 0.0
source_apod_angle /= core.G3Units.rad
apod_end_dist = apod_start_dist + source_apod_angle

# Make the mask
ps_mask = curved_sky.make_binary_point_source_mask(nside, field=field, analysis='cosmo_v4', fill_disk=fill_disk)
apd_src = curved_sky.apodize_binary_mask_prof(ps_mask, dist_smooth_angle, apod_start_dist, apod_end_dist)

final_mask = apd_brdr * apd_src
final_mask /= np.max(final_mask)

hp.write_map('/data/gpfs/projects/punim1922/summerfield_lensing/data/masks/apod_mask_45arcmin_15arcmin_%s_100mJy.fits'%(field), final_mask, overwrite=True)
