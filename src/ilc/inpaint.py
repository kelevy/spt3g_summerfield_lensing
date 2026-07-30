import numpy as np, os, sys, flatsky
import healpy as H
import scipy as sc
from scipy import interpolate as intrp
from pylab import *

#################################################################################
#################################################################################
#################################################################################

def calccov(sim_mat, noofsims, npixels):
    
    m = sim_mat.flatten().reshape(noofsims,npixels)
    m = np.mat( m ).T
    mt = m.T

    cov = (m * mt) / (noofsims)# - 1)
    return cov

#################################################################################
def get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer, square = 0, in_arcmins = 1):

    if not in_arcmins:
        ra_grid = ra_grid * 60.
        dec_grid = dec_grid * 60.

    if not square:
        radius = np.sqrt( (ra_grid**2. + dec_grid**2.) )
        inds_inner = np.where((radius<=mask_radius_inner))
        inds_outer = np.where((radius>mask_radius_inner) & (radius<=mask_radius_outer) )
    else:
        inds_inner = np.where( (abs(ra_grid)<=mask_radius_inner) & (abs(dec_grid)<=mask_radius_inner) )
        inds_outer = np.where( (abs(ra_grid)<=mask_radius_outer) & (abs(dec_grid)<=mask_radius_outer) & ( (abs(ra_grid)>mask_radius_inner) | (abs(dec_grid)>mask_radius_inner) ) )

    return inds_inner, inds_outer

#################################################################################
def get_mask_indices_fullsky(nside, radius_inner_am, radius_outer_am, ra=0., dec=0.):
    pixel = H.ang2pix(nside, np.radians(dec), np.radians(ra))
    ivec = H.pix2vec(nside, pixel)
    disc_inner = H.query_disc(nside, ivec, np.deg2rad(radius_inner_am/60.))
    disc_outer_tmp = H.query_disc(nside, ivec, np.deg2rad(radius_outer_am/60.))
    disc_outer = disc_outer_tmp[np.in1d(disc_outer_tmp, disc_inner) == False]
    ##print( ra, dec, pixel, ivec, disc_inner, disc_outer ); sys.exit()
    return disc_inner, disc_outer

#################################################################################
def ang_dist(x1, y1, x2, y2, in_arcmins = 0): #all numpy arrays
    """
    all input in radians
    """
    ang_dist_rad = np.arccos(np.sin(y1)*np.sin(y2)+(np.cos(y1)*np.cos(y2)*np.cos(x1 - x2)))
    ##from IPython import embed; embed()
    ang_dist_rad[np.isnan(ang_dist_rad)] = 0.
    ang_dist_deg = np.degrees(ang_dist_rad) # Converting radians to degrees

    if in_arcmins:
        return ang_dist_deg * 60.
    else:   
        return ang_dist_deg

#################################################################################
def get_points_on_sky(nside, howmany = 200, ang_sep_tol_deg = 5.):
    """
    #pick "howmany" pixels from the sky that are "ang_sep_tol_deg" apart or greater from each other.
    #helps in getting "multiple sky sims" from a single sim,
    """
    npix = H.nside2npix(nside)
    #pixels = np.arange( npix )

    sel_pixels = []
    while len(sel_pixels) <= howmany:
        ppp = np.random.randint(0, npix)
        if len(sel_pixels) == 0:
            sel_pixels.append( ppp )
            #print(ppp, len(sel_pixels))
        else:
            curr_dec_rad, curr_ra_rad = H.pix2ang(nside, ppp)
            dec_rad_arr, ra_rad_arr = H.pix2ang(nside, sel_pixels)
            reclen = len(ra_rad_arr)
            ang_dist_deg_arr = ang_dist(ra_rad_arr, dec_rad_arr, np.tile(curr_ra_rad, reclen), np.tile(curr_dec_rad, reclen))
            if np.min(ang_dist_deg_arr)>ang_sep_tol_deg:
                sel_pixels.append( ppp )
    
    return sel_pixels

#################################################################################
def make_multiple_sims_from_single_sim(hmap, field_size_deg = 2., howmany = 5):

    nside = H.get_nside(hmap)
    npix = H.nside2npix(nside)
    sel_pixels = get_points_on_sky(nside, howmany = howmany, ang_sep_tol_deg = field_size_deg * 2.)
    sel_dec_rad_arr, sel_ra_rad_arr = H.pix2ang(nside, sel_pixels)
    sel_dec_arr, sel_ra_arr = np.degrees(sel_dec_rad_arr), np.degrees(sel_ra_rad_arr)

    hmap_arr = []
    for ppp in sel_pixels:
        ivec = H.pix2vec(nside, ppp)
        disc = H.query_disc(nside, ivec, np.deg2rad(field_size_deg), inclusive=True)
        
        curr_hmap = np.zeros_like(hmap)
        if np.ndim(curr_hmap)>1:
            curr_hmap[:, disc] = np.copy(hmap[:, disc])
        else:
            curr_hmap[disc] = np.copy(hmap[disc])
        hmap_arr.append( curr_hmap )

    return hmap_arr, sel_ra_arr, sel_dec_arr

def get_multiple_radec_from_fullsky(nside, field_size_deg = 2., random_or_rashifted_or_decshifted = 'rashifted', howmany = -1):

    npix = H.nside2npix(nside)
    if random_or_rashifted_or_decshifted == 'random':
        assert howmany != -1
        sel_pixels = get_points_on_sky(nside, howmany = howmany, ang_sep_tol_deg = field_size_deg * 2.)
        sel_dec_rad_arr, sel_ra_rad_arr = H.pix2ang(nside, sel_pixels)
        sel_dec_arr, sel_ra_arr = np.degrees(sel_dec_rad_arr), np.degrees(sel_ra_rad_arr)
    elif random_or_rashifted_or_decshifted == 'rashifted':
        sel_ra_arr = np.arange(0., 360., field_size_deg * 2.)
        sel_dec_arr = np.tile( 0., len(sel_ra_arr) )
    elif random_or_rashifted_or_decshifted == 'decshifted':
        sel_dec_arr = np.arange(0., 180., field_size_deg * 2.)
        sel_ra_arr = np.tile( 0., len(sel_dec_arr) )

    return sel_ra_arr, sel_dec_arr

#################################################################################
def get_covariance(ra_grid, dec_grid, mapparams, el, cl_dic, bl, noofsims, mask_radius_inner, mask_radius_outer, nl_dic = None, nside = 2048, beamval = None, lmax = None, field_size_deg = 5.):##, howmanypatchesfromsinglesky = 1):

    print('\n\tcalculating the covariance from simulations for inpainting')
    ############################################################
    #determine full or flatsky first
    fullsky = False
    if ra_grid is None or dec_grid is None:
        fullsky = True
        assert nside is not None
        npix = H.nside2npix(nside)
        if lmax is None:
            lmax = 3 * nside - 1
        ell = np.arange(lmax)
    ############################################################
    #get the sims for covariance calculation
    print('\n\t\tgenerating %s sims' %(noofsims))
    if fullsky:
        t1_for_cov, t2_for_cov = [], []    
        for n in range(noofsims):

            print(n, end = ' ')

            #cmb sim and beam, for CMB include the transfer function and beam
            sim_map_alm = H.synalm(cl_dic['TT'])

            #apply beam
            sim_map_alm = H.almxfl( sim_map_alm, bl )
            
            if nl_dic is not None: #noise map, do not include the beam or transfer function
                noise_map_alm = H.synalm(nl_dic['T'])
                sim_map_alm = sim_map_alm + noise_map

            #convert to map
            sim_map = H.alm2map( sim_map_alm, nside = nside )

            #pick multiple regions from this sim
            print('\tpick regions from this sim=%s' %(n))
            curr_sel_ra_arr, curr_sel_dec_arr = get_multiple_radec_from_fullsky(nside, field_size_deg = field_size_deg)
            for (r, d) in zip(curr_sel_ra_arr, curr_sel_dec_arr):
                curr_inds_inner, curr_inds_outer = get_mask_indices_fullsky(nside, mask_radius_inner, mask_radius_outer, ra = r, dec = d)

                curr_t1_for_cov = np.asarray( sim_map[curr_inds_inner] )          
                curr_t2_for_cov = np.asarray( sim_map[curr_inds_outer] )
                ##print(curr_t1_for_cov.shape, curr_t2_for_cov.shape)

                t1_for_cov.append( curr_t1_for_cov )
                t2_for_cov.append( curr_t2_for_cov )
            

        t1_for_cov = np.asarray( t1_for_cov )
        t2_for_cov = np.asarray( t2_for_cov )
    else:
        sims_for_covariance = []
        for n in range(noofsims):

            #cmb sim and beam, for CMB include the transfer function and beam
            sim_map = flatsky.make_gaussian_realisation(mapparams, el, cl_dic['TT'], bl = bl)

            if nl_dic is not None: #noise map, do not include the beam or transfer function
                noise_map = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])
                sim_map = sim_map + noise_map
            
            sims_for_covariance.append( sim_map )    
        sims_for_covariance = np.asarray( sims_for_covariance)

        inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)
        
        t1_for_cov = sims_for_covariance[:,inds_inner[0], inds_inner[1]]
        t2_for_cov = sims_for_covariance[:,inds_outer[0], inds_outer[1]]        

    ############################################################
    #get the covariance now
    npixels_t1 = t1_for_cov.shape[1]
    npixels_t2 = t2_for_cov.shape[1]

    t1t2_for_cov = np.concatenate( (t1_for_cov,t2_for_cov), axis = 1 )
    noofsimspluspatchs_for_cov, npixels_t1t2 = t1t2_for_cov.shape
    t1t2_cov = calccov(t1t2_for_cov, noofsimspluspatchs_for_cov, npixels_t1t2)
    
    ############################################################
    #https://arxiv.org/pdf/1301.4145.pdf
    ##Eq. 32
    sigma_22 = t1t2_cov[npixels_t1:,npixels_t1:]
    sigma_12 = t1t2_cov[:npixels_t1,npixels_t1:]

    print('\n\t\t\tinvert sigma_22 matrix (%s,%s) now' %(sigma_22.shape[0], sigma_22.shape[1]))
    sigma_22_inv = sc.linalg.pinv(sigma_22)
    sigma_dic = {}
    sigma_dic['sigma_22_inv'] = sigma_22_inv
    sigma_dic['sigma_12'] = sigma_12

    print('\n\t\tcovariance obtained')
    return sigma_dic

#################################################################################

def perform_inpainting(ra, dec, ra_grid, dec_grid, mapparams, map_to_inpaint, sigma_dic, mask_radius_inner, mask_radius_outer, nside = 2048):

    #print('\n\tperform inpainting')
    """
    mask_inner = 1: The inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.
    """

    fullsky = False
    if ra_grid is None or dec_grid is None:
        fullsky = True

    original_map = map_to_inpaint.copy()
    sigma_12 = sigma_dic['sigma_12']
    sigma_22_inv = sigma_dic['sigma_22_inv']

    ############################################################
    #get the inner and outer pixel indices
    if fullsky:
        curr_inds_inner, curr_inds_outer = get_mask_indices_fullsky(nside, mask_radius_inner, mask_radius_outer, ra = ra, dec = dec)
    else:
        curr_inds_inner, curr_inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    ############################################################
    #get the pixel values in the inner and outer regions 
    t1_data = map_to_inpaint[curr_inds_inner]
    t2_data = map_to_inpaint[curr_inds_outer]
    #print(t1_data.shape, t2_data.shape); sys.exit()
    ############################################################
    #get the pixel values in the inner and outer regions from the constrained realisation
    t1_tilde = np.zeros( len(t1_data) )
    t2_tilde = np.zeros( len(t2_data) )
    
    inpainted_t1 = np.asarray( t1_tilde + np.dot(sigma_12, np.dot(sigma_22_inv, ( t2_data - t2_tilde) ) ) )  ##Eq. 36
    #print(inpainted_t1.shape)
    ############################################################
    #create a new inpainted map: copy the old map and replace the t1 region
    inpainted_map = np.copy(map_to_inpaint)
    cmb_inpainted_map = np.copy(map_to_inpaint)*0.
    inpainted_map[curr_inds_inner] = inpainted_t1
    cmb_inpainted_map[curr_inds_inner] = inpainted_t1

    return cmb_inpainted_map, inpainted_map, map_to_inpaint

#################################################################################

def masking_for_filtering(ra_grid, dec_grid, mask_radius = 2., taper_radius = 6., in_arcmins = 1):

    import scipy as sc
    import scipy.ndimage as ndimage

    if not in_arcmins:
        ra_grid_arcmins = ra_grid * 60.
        dec_grid_arcmins = dec_grid * 60.

    radius = np.sqrt( (ra_grid_arcmins**2. + dec_grid_arcmins**2.) )

    mask = np.ones( ra_grid_arcmins.shape )
    if (1): #20180118
        ##print '\n\n\t\t fixing masking radius to %s\n\n' %mask_ra_gridDIUS_ARCMINS
        inds_to_mask = np.where((radius<=mask_radius)) #2arcmins - fix this for now
        mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    ker=np.hanning(taper_radius)
    ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )

    mask=ndimage.convolve(mask, ker2d)
    mask/=mask.max()

    return mask

#################################################################################

def interpolate_from_flat_to_fullsky(hmap, source_ra, source_dec, ra_am_grid, dec_am_grid, flstskymap, radius_am, poly_deg = 3):

    nside = H.get_nside(hmap)

    #fill the full sky map using the inpainted map now

    source_centred_ra_deg_grid  =  ra_am_grid/60. + source_ra
    source_centred_dec_deg_grid  =  dec_am_grid/60. + source_dec

    #use query disc to obtain the full sky pixels within the inpainting radius and then interpolate
    ipix = H.ang2pix(nside, np.radians( 90. - source_dec ), np.radians( source_ra ))
    ivec = H.pix2vec(nside, ipix)

    #ra/dec for interpolation
    source_inds_for_inp = H.query_disc(nside, ivec, np.deg2rad(radius_am/60.))
    tmp_dec_for_inp, tmp_ra_for_inp = np.degrees( H.pix2ang(nside, source_inds_for_inp) )
    tmp_dec_for_inp = 90 - tmp_dec_for_inp

    intrp_values = intrp.RectBivariateSpline(source_centred_ra_deg_grid[0,:], 
                                           source_centred_dec_deg_grid[:,0], 
                                           flstskymap, 
                                           kx = poly_deg, 
                                           ky = poly_deg).ev(tmp_ra_for_inp, tmp_dec_for_inp)


    hmap_mod = np.copy( hmap )
    hmap_mod[source_inds_for_inp] = intrp_values
    return hmap_mod

#################################################################################
