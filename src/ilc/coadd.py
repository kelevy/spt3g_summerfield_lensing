import sys
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/build/")
import numpy as np
import copy
from spt3g import core, maps


##################################################################################################################################################################


def add_frames(frame1, frame2, check_weights = True, final_frame_id = 'coadd'): 
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
        if frame['Id'].find(band)==-1: continue
        if debug: print(f, frame['Id'])
        if map_frame is None:
            map_frame = frame
        else:
            map_frame = add_frames(frame, map_frame)
    return map_frame


def coadd_maps(flist, fd_for_weights = None, maprun_date_iden = '', band = '150GHz'): #read map frames
    coadd_map = None
    for fname in flist:
        curr_map_frame = read_map(fname, band = band)
        print('Current frame: ', curr_map_frame)
        if 'Wpol' not in curr_map_frame: ##fd_for_weights is not None: #add weights now
            print('Adding Wpol to current frame!')
            tmpfname = fname.split('/')[-1]
            tmpfname = '%s/%s%s' %(fd_for_weights, maprun_date_iden, tmpfname)
            tmpmap_frame = read_map(tmpfname, band = band)
            curr_map_frame['Wpol'] = tmpmap_frame['Wpol']
        else:
            print('Wpol already in current frame!')
        if coadd_map is None:
            coadd_map = copy.deepcopy(curr_map_frame)
        else:
            coadd_map = add_frames(copy.deepcopy(curr_map_frame), coadd_map)
    return coadd_map


def get_apod_mask(weight_map, weight_threshold = 0.1, do_apod = False, hanning_rad = 60.):
    weight_map = weight_map / np.max(weight_map)
    apod_mask = np.ones( weight_map.shape )
    apod_mask[weight_map<weight_threshold] = 0.
    
    if do_apod:
        hanning=np.hanning(hanning_rad)
        hanning=np.sqrt(np.outer(hanning,hanning))

        apod_mask=ndimage.convolve(apod_mask, hanning)
        apod_mask=apod_mask/np.max(apod_mask)

    return apod_mask 
