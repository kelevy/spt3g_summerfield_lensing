import numpy as np, sys, os, scipy as sc#, healpy as H

################################################################################################################
#flat-sky routines
################################################################################################################

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d

################################################################################################################

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

################################################################################################################

def get_lxly_az_angle(lx,ly):

    """
    azimuthal angle from lx, ly

    inputs:
    lx, ly = 2d lx and ly arrays

    output:
    azimuthal angle
    """
    return 2*np.arctan2(lx, -ly)

################################################################################################################
def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu = 1):

    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx,ly)

    map1_fft, map2_fft = np.fft.fft2(map1),np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft - np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real
    else:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft + np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( -np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real

    return map1_mod, map2_mod

################################################################################################################
def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):
    """
    filter_type = 0 - low pass filter
    filter_type = 1 - high pass filter
    filter_type = 2 - band pass
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter
################################################################################################################

def wiener_filter(mapparams, cl_signal, cl_noise, el = None):

    if el is None:
        el = np.arange(len(cl_signal))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl_signal2d = cl_to_cl2d(el, cl_signal, flatskymapparams) 
    cl_noise2d = cl_to_cl2d(el, cl_noise, flatskymapparams) 

    wiener_filter = cl_signal2d / (cl_signal2d + cl_noise2d)

    return wiener_filter

################################################################################################################
def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, minbin = 0, maxbin = 10000, mask = None, filter_2d = None, return_2d = False):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    nx, ny, dx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    if filter_2d is not None:
        flatskymap_psd = flatskymap_psd / filter_2d
        flatskymap_psd[np.isinf(flatskymap_psd) | np.isnan(flatskymap_psd)] = 0.
        flatskymap_psd[abs(flatskymap_psd)>1e300] = 0.

    if return_2d:
        el = None
        cl = flatskymap_psd
    else:
        #rad_prf = radial_profile_v1(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
        #el, cl = rad_prf[:,0], rad_prf[:,1]
        el, cl = radial_profile(flatskymap_psd, binsize, maxbin, minbin=minbin, xy=(lx,ly), return_errors=False)

    if mask is not None:
        #fsky = np.mean(mask)
        fsky = np.mean(mask**2.) #20240912
        cl /= fsky

    '''
    if filter_2d is not None:
        if not return_2d:
            #rad_prf_filter_2d = radial_profile_v1(filter_2d, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
            #el, fl = rad_prf_filter_2d[:,0], rad_prf_filter_2d[:,1]
            el, fl = radial_profile(filter_2d, binsize, maxbin, minbin=minbin, xy=(lx,ly), return_errors=False)
        else:
            fl = filter_2d
        cl /= fl
    '''

    return el, cl

################################################################################################################

def radial_profile_v1(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 1):

    """
    get the radial profile of an image (both real and fourier space)
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.

    binarr=np.arange(minbin,maxbin,bin_size)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval

    return radprf

################################################################################################################

def make_gaussian_realisation(mapparams, el, cl, cl2 = None, cl12 = None, bl = None, qu_or_eb = 'eb'):

    nx, ny, dx, dy = mapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx * dy))
    ################################################

    #1d to 2d now
    cltwod = cl_to_cl2d(el, cl, mapparams)
    
    ################################################
    if cl2 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, mapparams)
        cltwod2 = cl_to_cl2d(el, cl2, mapparams)

    ################################################
    if cl2 is None:

        cltwod = cltwod**0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([ny,nx])
        SIM = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        gauss_reals_1 = np.random.standard_normal([ny,nx])
        gauss_reals_2 = np.random.standard_normal([ny,nx])

        '''
        gauss_reals_1 = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2 = np.fft.fft2( gauss_reals_2 )

        t1 = gauss_reals_1 * cltwod12 / cltwod2**0.5
        t2 = gauss_reals_2 * ( cltwod - (cltwod12**2. /cltwod2) )**0.5

        SIM_FFT = (t1 + t2) * norm
        SIM_FFT[np.isnan(SIM_FFT)] = 0.
        SIM = np.fft.ifft2( SIM_FFT ).real
        '''

        gauss_reals_1_fft = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2_fft = np.fft.fft2( gauss_reals_2 )

        #field_1
        cltwod_tmp = np.copy( cltwod )**0.5 * norm
        SIM_FIELD_1 = np.fft.ifft2( cltwod_tmp *  gauss_reals_1_fft ).real
        #SIM_FIELD_1 = np.zeros( (ny, nx) )

        #field 2 - has correlation with field_1
        t1 = np.copy( gauss_reals_1_fft ) * cltwod12 / np.copy(cltwod)**0.5
        t2 = np.copy( gauss_reals_2_fft ) * ( cltwod2 - (cltwod12**2. /np.copy(cltwod)) )**0.5
        SIM_FIELD_2_FFT = (t1 + t2) * norm
        SIM_FIELD_2_FFT[np.isnan(SIM_FIELD_2_FFT)] = 0.
        SIM_FIELD_2 = np.fft.ifft2( SIM_FIELD_2_FFT ).real

        #T and E generated. B will simply be zeroes.
        SIM_FIELD_3 = np.zeros( SIM_FIELD_2.shape )
        if qu_or_eb == 'qu': #T, Q, U: convert E/B to Q/U.
            SIM_FIELD_2, SIM_FIELD_3 = convert_eb_qu(SIM_FIELD_2, SIM_FIELD_3, mapparams, eb_to_qu = 1)
        else: #T, E, B: B will simply be zeroes
            pass

        SIM = np.asarray( [SIM_FIELD_1, SIM_FIELD_2, SIM_FIELD_3] )


    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        SIM = np.fft.ifft2( np.fft.fft2(SIM) * bl).real

    SIM = SIM - np.mean(SIM)

    return SIM

################################################################################################################

def radial_profile(image, binsize, maxbin, minbin=0.0, xy=None, return_errors=False):
    """
    Get the radial profile of an image (both real and fourier space)

    Parameters
    ----------
    image : array
        Image/array that must be radially averaged.
    binsize : float
        Size of radial bins.  In real space, this is
        radians/arcminutes/degrees/pixels.  In Fourier space, this is
        \Delta\ell.
    maxbin : float
        Maximum bin value for radial bins.
    minbin : float
        Minimum bin value for radial bins.
    xy : 2D array
        x and y grid points.  Default is None in which case the code will simply
        use pixels indices as grid points.
    return_errors : bool
        If True, return standard error.

    Returns
    -------
    bins : array
        Radial bin positions.
    vals : array
        Radially binned values.
    errors : array
        Standard error on the radially binned values if ``return_errors`` is
        True.
    """

    image = np.asarray(image)
    if xy is None:
        y, x = np.indices(image.shape)
    else:
        y, x = xy

    radius = np.hypot(y, x)
    radial_bins = np.arange(minbin, maxbin, binsize)

    hits = np.zeros(len(radial_bins), dtype=float)
    vals = np.zeros_like(hits)
    errors = np.zeros_like(hits)

    for ib, b in enumerate(radial_bins):
        inds = np.where((radius >= b) & (radius < b + binsize))
        imrad = image[inds]
        total = np.sum(imrad != 0.0)
        hits[ib] = total

        if total > 0:
            ###print(ib, b, total, np.sum(imrad), imrad)
            # mean value in each radial bin
            vals[ib] = np.sum(imrad) / total
            errors[ib] = np.std(imrad)

    bins = radial_bins + binsize / 2.0

    std_mean = np.sum(errors * hits) / np.sum(hits)
    
    if return_errors:
        errors = std_mean / hits ** 0.5
        return bins, vals, errors
    else:
        return bins, vals

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
