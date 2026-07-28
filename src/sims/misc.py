import numpy as np, sys, os, scipy as sc#, flatsky#, healpy as H
sys.path.append("/data/gpfs/projects/punim1922/spt3g_software/build/")
from spt3g import core, maps
from spt3g.mapspectra import basicmaputils as utils

################################################################################################################

def coth(x):
    return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

################################################################################################################

def compton_y_to_delta_tcmb(freq):

    """
    c.f:  table 1, sec. 3 of arXiv: 1303.5081; 
    table 8 of http://arxiv.org/pdf/1303.5070.pdf
    no relativistic corrections included.
    """

    h=6.62607004e-34 #Planck constant in m2 kg / s
    k_B=1.38064852e-23 #Boltzmann constant in m2 kg s-2 / K-1
    tcmb=2.73 #Kelvin
    delta_nu = 1e9 #Hz

    if freq<1e3: freq = freq * 1e9

    x = h * freq / (k_B * tcmb)
    g_nu = x * coth(x/2.) - 4.

    return tcmb * np.mean(g_nu)

################################################################################################################
def gauss_beam(fwhm, lmax=512, pol=False):
    """Gaussian beam window function

    Computes the spherical transform of an axisimmetric gaussian beam

    For a sky of underlying power spectrum C(l) observed with beam of
    given FWHM, the measured power spectrum will be
    C(l)_meas = C(l) B(l)^2
    where B(l) is given by gaussbeam(Fwhm,Lmax).
    The polarization beam is also provided (when pol = True ) assuming
    a perfectly co-polarized beam
    (e.g., Challinor et al 2000, astro-ph/0008228)

    Parameters
    ----------
    fwhm : float
        full width half max in radians
    lmax : integer
        ell max
    pol : bool
        if False, output has size (lmax+1) and is temperature beam
        if True output has size (lmax+1, 4) with components:
        * temperature beam
        * grad/electric polarization beam
        * curl/magnetic polarization beam
        * temperature * grad beam

    Returns
    -------
    beam : array
        beam window function [0, lmax] if dim not specified
        otherwise (lmax+1, 4) contains polarized beam
    """

    sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1)
    sigma2 = sigma ** 2
    g = np.exp(-0.5 * ell * (ell + 1) * sigma2)

    if not pol:  # temperature-only beam
        return g
    else:  # polarization beam
        # polarization factors [1, 2 sigma^2, 2 sigma^2, sigma^2]
        pol_factor = np.exp([0.0, 2 * sigma2, 2 * sigma2, sigma2])
        return g[:, np.newaxis] * pol_factor

def get_beam_dic(freqs, beam_noise_dic, lmax, opbeam = None, make_2d = 0, mapparams = None):
    bl_dic =  {}
    for freq in freqs:
        beamval, noiseval = beam_noise_dic[freq]
        #bl_dic[freq] = H.gauss_beam(np.radians(beamval/60.), lmax=lmax-1)
        bl_dic[freq] = gauss_beam(np.radians(beamval/60.), lmax=lmax-1)

        if make_2d:
            assert mapparams is not None
            el = np.arange(len(bl_dic[freq]))
            #bl_dic[freq] = flatsky.cl_to_cl2d(el, bl_dic[freq], mapparams)
            nx, ny, dx = mapparams
            shape = [ny, nx]
            res = np.radians(dx/60.)
            bl_dic[freq] = utils.interp_cl_2d(bl_dic[freq], res, shape, ell = el, real = False)

    if opbeam is not None:
        #bl_dic['effective'] = H.gauss_beam(np.radians(opbeam/60.), lmax=lmax-1)
        bl_dic['effective'] = gauss_beam(np.radians(opbeam/60.), lmax=lmax-1)

        if make_2d:
            assert mapparams is not None
            #bl_dic['effective'] = flatsky.cl_to_cl2d(el, bl_dic['effective'], mapparams) 
            nx, ny, dx = mapparams
            shape = [ny, nx]
            res = np.radians(dx/60.)
            bl_dic['effective'] = utils.interp_cl_2d(bl_dic['effective'], res, shape, ell = el, real = False)

    return bl_dic
################################################################################################################

def rebeam(bl_dic, threshold = 1000.):
    #freqarr = sorted( list(bl_dic.keys()) )
    freqarr = []
    for nu in list(bl_dic.keys()): 
        if isinstance(nu, int):
            freqarr.append(nu)
    freqarr = sorted(freqarr)

    bl_eff = bl_dic['effective']
    rebeamarr = []
    for freq in freqarr:
        if freq is 'effective': continue
        currinvbeamval = 1./bl_dic[freq]
        currinvbeamval[currinvbeamval>threshold] = threshold
        rebeamval = bl_eff * currinvbeamval
        rebeamarr.append( rebeamval )

    return np.asarray( rebeamarr )

################################################################################################################

def healpix_rotate_coords(hmap, coord):
    """
    coord = ['C', 'G'] to convert a map in RADEC to Gal.    
    """

    #get map pixel
    pixel = np.arange(len(hmap))

    #get angles in this map first
    nside = H.get_nside(hmap)
    angles = H.pix2ang(nside, pixel)

    #roate the angles to the desired new coordinate
    rotated_angles = H.Rotator(coord=coord)(*angles)

    #get the rotated pixel values
    rotated_pixel = H.ang2pix(nside, *rotated_angles)

    #initialise new map
    rot_hmap = np.zeros(len(pixel))

    #push the original map pixel to the new map (in the rotated pixel positions)
    rot_hmap[rotated_pixel] = hmap[pixel]

    return rot_hmap

################################################################################################################

def get_bl(beamval, el):

    fwhm_radians = np.radians(beamval/60.)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    sigma2 = sigma ** 2
    bl = np.exp(el * (el+1) * sigma2)

    return bl

################################################################################################################

def get_nl(noiseval, el, beamval, use_beam_window = 1, uk_to_K = 0, elknee_t = -1, alpha_knee = 0, make_2d = 0, mapparams = None):

    if uk_to_K: noiseval = noiseval/1e6

    if use_beam_window:
        fwhm_radians = np.radians(beamval/60.)
        sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
        sigma2 = sigma ** 2
        bl = np.exp(el * (el+1) * sigma2)

    delta_T_radians = noiseval * np.radians(1./60.)
    nl = np.tile(delta_T_radians**2., int(max(el)) + 1 )

    nl = np.asarray( [nl[int(l)] for l in el] )

    if use_beam_window: nl *= bl

    if elknee_t != -1.:
        nl = np.copy(nl) * (1. + (elknee_t * 1./el)**alpha_knee )

    if make_2d:
        assert mapparams is not None
        #nl = flatsky.cl_to_cl2d(el, nl, mapparams) 
        nx, ny, dx = mapparams
        shape = [ny, nx]
        res = np.radians(dx/60.)
        nl = utils.interp_cl_2d(nl, res, shape, ell = el, real = False)

    nl[np.isinf(nl)] = 0.
    nl[np.isnan(nl)] = 0.

    return nl

################################################################################################################

