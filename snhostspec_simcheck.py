#!/usr/bin/env python
# coding: utf-8

# ### Part II :  (long version, generating simulated SEDs for all the galaxies)
# (1) Read in the master catalog (a SNANA HOSTLIB file) generated in Part I
# (2) For each simulated SN host galaxy, use the EAZY code to make a simulated host galaxy spectrum (from the best-fitting photoz template)
# (3) DOABLE, BUT NOT YET DONE: Store each simulated spectrum as an ascii .dat file with wavelength in nm and AB mag (suitable for input to the Subaru ETC).
# (4) STILL TBD :  Store the revised master catalog (now updated with SED .dat file names) as a modified SNANA HOSTLIB file (ascii text)
# * NOTE: At the moment, I'm working with a shortened library of galaxies, containing just 29 galaxies from the COSMOS field, selected because we think we have decent real-world DEIMOS spectra to compare to. (from the COSMOS DEIMOS paper by Gunther Hasinger, with data available here:
# https://irsa.ipac.caltech.edu/data/COSMOS/spectra/deimos/deimos.html)


import os
import numpy as np
from astropy.io import fits
#from glob import glob
from matplotlib import pyplot as plt

from astropy import table
from astropy.table import Table, Column
from astropy.io import ascii
#from astropy.coordinates import SkyCoord
#from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

import sncosmo
import snhostspec

#from scipy.interpolate import interp1d
#from scipy.integrate import trapz


# Adjust the `datadir` variable to reflect your path.
datadir = "/Users/rodney/Dropbox/SC-SN-DATA/cosmos_example_spectra/"

Data_overview_table_name = os.path.join(datadir, "gal_lib_short1.dat") #name of table which contains galaxy ID information and location for DEIMOS and vUDS spectra
hostlib_filename = os.path.join(datadir,'cosmos_example_hostlib.txt') #name of SNANA HOSTLIB file (includes observed and rest-frame-synthetic photometry)
eazy_templates_filename = os.path.join(datadir,"eazy_13_spectral_templates.dat")

vUDS_spec_location = os.path.join(datadir,"vUDS_spec/") 
DEIMOS_spec_location = os.path.join(datadir,'deimos_spec/') 
sim_spec_location = os.path.join(datadir,'sim_spec/') 

HST_table_name = os.path.join(datadir,'cosmos_match.dat') #name of table with HST ID information to locate and match spec files
HST_cosmos_folder_location = os.path.join(datadir,'COSMOS_3DHST_SPECTRA/') #location of all cosmos tile folders (i.e. directory which contains cosmos-02, cosmos-03, etc.)


medsmooth = lambda f,N : np.array(
    [np.median( f[max(0,i-N):min(len(f),max(0,i-N)+2*N)])
     for i in range(len(f))])

flcdm = FlatLambdaCDM(H0=73, Om0=0.27)


def load_eazypy_templates(eazytemplatefilename,
                          format='ascii.commented_header',
                          verbose=True,
                          **kwargs):
    """Read in the galaxy SED templates (basis functions for the
    eazypy SED fitting / simulation) and store as the 'eazytemplatedata'
    property.

    We read in an astropy Table object with N rows and M+1 columns, where
    N is the number of wavelength steps and M is the
    number of templates (we expect 13).
    The first column is the  wavelength array, common to all templates.

    We translate the Nx(M+1) Table data into a np structured array,
    then reshape as a (M+1)xN numpy ndarray, with the first row giving
    the wavelength array and each subsequent row giving a single
    template flux array.
    See the function simulate_eazy_sed_from_coeffs() to construct
    a simulated galaxy SED with a linear combination from this matrix.
    """
    eazytemplates = Table.read(eazytemplatefilename,
                               format=format, **kwargs)
    tempdata = eazytemplates.as_array()
    eazytemplatedata = tempdata.view(np.float64).reshape(
        tempdata.shape + (-1,)).T
    if verbose:
        print("Loaded Eazypy template SEDs from {0}".format(
            eazytemplatefilename))
    return eazytemplatedata


def scale_to_match_imag(wave, flam, imag, medsmooth_window=20):
    """KLUDGE!!  Using sncosmo to make this galaxy SED into a Source so 
    we can integrate into mags using the sncosmo bandmag, and rescale 
    to match a pre-defined mag
    
    wave: wavelength in angstroms
    flam: flambda in erg/s/cm2/A
    imag: sdss i band magnitude to scale to
    """    
    # check that we cover the i band
    if wave[0]>6600:
        wave = np.append([6580], wave)
        flam = np.append([1e-20], flam)
    if wave[-1]<8380:
        wave = np.append(wave, [8400])
        flam = np.append(flam, [1e-20])
    
    if medsmooth_window>1:
        # If a smoothing window size is given, use only the smoothed flux
        flam = medsmooth(flam, medsmooth_window)

    # Make a dummy sncosmo Source and scale it to the given sdss i band mag
    phase = np.array([-1, 0, 1, 2]) # need at least 4 phase positions for a source
    flux = np.array([flam, flam, flam, flam]) 
    galsource = sncosmo.TimeSeriesSource(phase, wave, flux)    
    galsource.set_peakmag(imag, 'sdssi', 'ab')

    fout = galsource.flux(0,wave)
    
    return(wave, fout)


def simulate_eazy_sed_from_coeffs(eazycoeffs, eazytemplatedata, z,
            returnfluxunit='flambda', returnwaveunit='A',
            limitwaverange=True, savetofile='', **outfile_kwargs):
    """
    Generate a simulated SED from a given set of input eazy-py coefficients
    and eazypy templates.

    NB: Requires the eazy-py package to apply the IGM absorption!
    (https://github.com/gbrammer/eazy-py)

    Optional Args:
    returnfluxunit: ['AB', 'flambda', 'fnu'] TODO: add Jy
        'AB'= return log(flux) as monochromatic AB magnitudes
        'AB25' = return AB mags, rescaled to a zeropoint of 25:  m=-2.5*log10(fnu)+25
        'flambda' = return flux density in erg/s/cm2/A
        'fnu' = return flux density in erg/s/cm2/Hz
    returnwaveunit: ['A' or 'nm'] limitwaverange: limit the output
    wavelengths to the range covered by PFS savetofile: filename for saving
    the output spectrum as a two-column ascii data file (suitable for use
    with the SubaruPFS ETC from C. Hirata.

    Returns
    -------
        wave : observed-frame wavelength, Angstroms or  nm
        flux : flux density of best-fit template, erg/s/cm2/A or AB mag
    """
    # the input data units are Angstroms for wavelength
    # and cgs for flux (flambda): erg s-1 cm-2 Ang-1
    wave_em = eazytemplatedata[0]  # rest-frame (emitted) wavelength
    wave_obs = wave_em * (1 + z)  # observer-frame wavelength
    obsfluxmatrix = eazytemplatedata[1:]
    flam = np.dot(eazycoeffs, obsfluxmatrix) # flux in erg/s/cm2/A
    
    if limitwaverange:
        # to simplify things, we only work with data over the Subaru PFS
        # + WFIRST prism wavelength range, from 200 to 2500 nm
        # (2000 to 25000 Angstroms)
        iuvoir = np.where((wave_obs>2000) & (wave_obs<25000))[0]
        wave_obs = wave_obs[iuvoir]
        wave_em = wave_em[iuvoir]
        flam = flam[iuvoir]
    
    # convert flux units to fnu using :  fnu=(lam^2/c)*flam  ;  c = 3.e18 A/s
    fnu = (wave_em * wave_em / 3.e18) * flam  # flux in erg/s/cm2/Hz
    
    # Confusing previous setup from GB, used to convert to AB mags w/ zpt=25
    #fnu_factor = 10 ** (-0.4 * (25 + 48.6))
    # flam_spec = 1. / (1 + z) ** 2
    # obsflux = sedsimflux * fnu_factor * flam_spec   
    
    try:
        import eazy.igm
        igmz = eazy.igm.Inoue14().full_IGM(z, wave_obs)
        fnu *= igmz
    except:
        pass
    
    if returnfluxunit=='AB':
        # convert from flux density fnu into monochromatic AB mag:
        returnflux = -2.5 * np.log10(fnu) - 48.6
    elif returnfluxunit=='AB25':
        # convert from flux density fnu into AB mags for zpt=25:
        returnflux = -2.5 * np.log10(fnu) + 25
    elif returnfluxunit=='fnu':
        returnflux = fnu
    elif returnfluxunit.startswith('flam'):
        returnflux = flam
    else:
        print("I don't recognize flux unit {}".format(returnfluxunit))
        return None,None
        
    if returnwaveunit=='nm':
        returnwave = wave_obs / 10.
    elif returnwaveunit.startswith('A'):
        returnwave = wave_obs
    else:
        print("I don't recognize wave unit {}".format(returnwaveunit))
        return None,None

    if savetofile:
        out_table = Table()
        outcol1 = Column(data=wave_obs, name='Angstroms')
        outcol2 = Column(data=flam, name='flambda')
        out_table.add_columns([outcol1, outcol2])
        out_table.write(savetofile, **outfile_kwargs)

    return returnwave, returnflux


def mAB_from_flambda(flambda, wave):
    """ Convert from flux density f_lambda in erg/s/cm2/A 
    into AB mag
    
    flambda: flux density f_lambda (erg/s/cm2/A)
    wave : wavelength in angstroms
    
    (see https://en.wikipedia.org/wiki/AB_magnitude)
    """
    return(-2.5 * np.log10(3.34e4 * wave * wave * (flambda / 3631)))


def getdeimosdat(galid, medsmoothpix=20, 
                 returnfluxunit='AB', returnwaveunit='A', 
                 extension='fits', imag=None):
    
    deimosfilename = os.path.join(
        DEIMOS_spec_location,
        "cosmos_example_spec1d_{0}.{1}".format(galid, extension))
   
    if extension=='fits':
        deimosdat = Table.read(deimosfilename, format='fits')
        f = np.array(deimosdat['FLUX'][0])
        w = np.array(deimosdat['LAMBDA'][0])
        ivalid = np.where((-1000<f) & (f<100))[0]
    elif extension=='txt':
        deimosdat = Table.read(deimosfilename, 
                               format='ascii.commented_header', 
                               header_start=-1, data_start=0)
        f = np.array(deimosdat['flux'])
        w = np.array(deimosdat['wavelength'])
        ivalid = np.where((-100<f) & (f<100))[0]
    else:
        raise RuntimeError("Extension {} for deimos file not known.".format(extension))

    f = f[ivalid] * 1e-17 # erg/s/cm2/A
    w = w[ivalid] 

    if imag is None:
        fsmooth = medsmooth(f, medsmoothpix)
        wsmooth = w
    else:
        wsmooth, fsmooth = scale_to_match_imag(
            w, f, imag, medsmooth_window=medsmoothpix)
        w, f = scale_to_match_imag(
            w, f, imag, medsmooth_window=1)

    if returnfluxunit=='AB':
        f = mAB_from_flambda(f, w)
        fsmooth = mAB_from_flambda(fsmooth, wsmooth)
    if returnwaveunit=='nm':
        w = w / 10

    inotnan = np.isfinite(f)
    f = f[inotnan]
    w = w[inotnan]

    inotnan = np.isfinite(fsmooth)
    fsmooth = fsmooth[inotnan]
    wsmooth = wsmooth[inotnan]

    return w, f, wsmooth, fsmooth


def getvudsdat(galid, medsmoothpix=20,
               returnfluxunit='AB', returnwaveunit='A', extension='fits',
               imag=None):
    """Read in the vUDS spectrum data for a single galaxy"""
    
    # Get the fits filename for the given GALID from the overview data file
    overview_table = ascii.read(Data_overview_table_name)
    ivuds = np.where(overview_table['GALID']==galid)[0][0]
    specfilename = overview_table["vUDS_spec_filename"][ivuds]
    if specfilename == str(0):
        print("No vUDS spectrum available for galid {:d}".format(galid))
        return(None)
    specfilename = os.path.join(vUDS_spec_location, specfilename)

    # extract flux data from the vUDS file data array
    hdulist = fits.open(specfilename)
    flux = hdulist[0].data

    # construct the vUDS wavelength array from header info
    lam_start = hdulist[0].header['CRVAL1']
    lam_delt = hdulist[0].header['CDELT1']
    lam_end = lam_start + lam_delt*(len(flux))
    wavelength = np.arange(lam_start, lam_end, lam_delt)

    # run a median filter to smooth the data
    fsmooth = medsmooth(flux, medsmoothpix)
    wsmooth = wavelength

    # Scale to match the given i-band mag
    if imag is not None:
        wsmooth, fsmooth = scale_to_match_imag(
            wsmooth, fsmooth, imag,
            medsmooth_window=0)
        wavelength, flux = scale_to_match_imag(
            wavelength, flux, imag, 
            medsmooth_window=0)
        
    # change units if needed
    if returnfluxunit=='AB':
        flux = mAB_from_flambda(flux, wavelength)
        fsmooth = mAB_from_flambda(fsmooth, wsmooth)
    if returnwaveunit=='nm':
        wavelength = wavelength / 10
        wsmooth = wsmooth/10

    # remove nans and infs
    inotnan = np.isfinite(flux)
    fnotnan = flux[inotnan]
    wnotnan = wavelength[inotnan]  
    
    # make smoothed flux array (if not done already by renormalization function)
    if imag is None:
        wsmooth = wnotnan
        fsmooth = medsmooth(fnotnan,medsmoothpix)
        
    return(wnotnan, fnotnan, wsmooth, fsmooth)


def gethstdat(galid, medsmoothpix=20, returnfluxunit='AB',
              returnwaveunit='A'):

    galid = int(galid)
    # The 'galid' being used here was constructed by adding an integer
    # factor of 10^5 to the galaxy ID from the 3DHST catalog paper
    # (Skelton et al. 201?).    The integer indicates which of the
    # CANDELS fields the galaxy was in.  For the COSMOS data (field #2)
    # this means that galid = 200000 + id

    # the cosmos_match table has a column telling us which
    # pointing each galaxy was in, and therefore which folder our #
    # target galaxy resides in.
    hstmatchtable = ascii.read(HST_table_name)
    if galid not in hstmatchtable['GALID']:
        print("no HST spec available for {:d}".format(galid))
        return None
    imatch = np.where(hstmatchtable['GALID']==galid)[0][0]
    hstid =  int(galid) - 200000

    #find the HST grism spectrum ascii data file
    folder = hstmatchtable['FOLDER'][imatch]
    filename = folder + '-G141_{:05d}'.format(hstid) +'.1D.ascii'
    fullpath = HST_cosmos_folder_location + folder + \
               '/1D/ASCII/' + filename
    if not os.path.isfile(fullpath):
        print("No HST spectrum for {:d}".format(galid))
        return None

    spectable = ascii.read(fullpath)
    cps = np.array(spectable['flux']) # measured flux (e- per sec)
    contam = np.array(spectable['contam']) # contaminating flux (e-/s)
    sens = np.array(spectable['sensitivity']) # sensitivity (e/s / 10-17 cgs)
    flux = ((cps-contam) / sens) * 1e-17 # galaxy flux (sans contam.)  in erg/s/cm2/A
    wave = np.array(spectable['wave']) # wavelength in Angstroms

    if returnfluxunit=='AB':
        ivalid = np.where((flux>0) & (sens>1.5))[0]
        flux = mAB_from_flambda(flux[ivalid], wave[ivalid])
        wave = wave[ivalid]
    if returnwaveunit=='nm':
        wave = wave / 10.

    fsmooth = medsmooth(flux, medsmoothpix)
    wsmooth = wave

    return wave, flux, wsmooth, fsmooth


# Make a composite catalog with all the info from the CANDELS hostlib 
def join_candels_to_deimos(deimoscatfile="DATA/cosmos_example_spectra/cosmos_example_spectra.txt",
                        candelscatfile=hostlib_filename):
    cat1 = ascii.read(deimoscatfile, format='fixed_width')
    cat2 = ascii.read(candelscatfile)
    catjoin = table.join(cat1, cat2, keys=['GALID'])
    return(catjoin)


# In[62]:


# read in the filter data, downloaded from http://svo2.cab.inta-csic.es/svo/theory//fps3/
# also available at: https:// wfirst.gsfc.nasa.gov/science/sdt_public/wps/references/ instrument/WFIRST-WFI-Transmission_160720.xlsm
J_hst = ascii.read("DATA/HST_WFC3_IR.F125W.dat")
H_hst = ascii.read("DATA/HST_WFC3_IR.F160W.dat")
J_wfirst = ascii.read("DATA/WFIRST_WFI.J129.dat")
H_wfirst = ascii.read("DATA/WFIRST_WFI.H158.dat")

#def getJHmags(wave, flambda):
#    """integrate over the J and H bandpasses and return an AB mag"""


# In[63]:


galid = 205925
deimoscat = join_candels_to_deimos()
ideimoscat = np.where(deimoscat['GALID']==galid)[0][0]

#sim1 = snhostspec.SnanaSimData()
#sim1.load_hostlib_catalog(hostlib_filename)
#sim1.load_eazypy_templates(eazy_templates_filename)

#eazycoeffs = np.array([deimoscat[col][ideimoscat]
#                       for col in sim1.simdata.colnames
#                       if col.startswith('coeff_specbasis')])


def plot_spec_comparison(galid, showphot=True, showvuds=True, showdeimos=True,
                         showhst=True, showeazy=True,
                         medsmooth_deimos=20, medsmooth_vuds=20,
                         medsmooth_hst=20,
                         rescaledeimos=True, rescalevuds=False, ax=None):
    """Plot flux vs wavelength for the given galaxy ID, showing the observed 
    DEIMOS, vUDS, and 3DHST spectra and the Eazy-simulated spectrum. """
    if ax is None:
        fig = plt.figure(figsize=[12,4])
        ax = fig.add_subplot(1,1,1)

    # row index for this galaxy in the simdata table
    # igal = np.where(sim1.simdata['GALID'] == galid)[0][0]

    # read in the eazy spectral templates data 
    # NOTE: could do this without loading the whole hostlib as a SnanaSimData object, would just need to grab
    # the code from snhostspec 
    #sim1 = snhostspec.SnanaSimData()
    #sim1.load_hostlib_catalog("DATA/cosmos_example_hostlib.txt")
    #sim1.
    eazytemplatedata = load_eazypy_templates(eazy_templates_filename)

    # ---------------------------------
    # Simulated and Observed photometry :
    # --------------------------------

    # read in the mag data for the DEIMOS spectra
    mastercat = join_candels_to_deimos()
    #mastercat = ascii.read("DATA/cosmos_example_spectra/cosmos_example_spectra.txt", format='fixed_width')
    ithisgal_mastercat = np.where(mastercat['GALID']==galid)[0][0]
    imag = mastercat['imag'][ithisgal_mastercat]
    kmag = mastercat['kmag'][ithisgal_mastercat]

    sdssu = mastercat['sdssu_fit'][ithisgal_mastercat]
    sdssg = mastercat['sdssg_fit'][ithisgal_mastercat]
    sdssr = mastercat['sdssr_fit'][ithisgal_mastercat]
    sdssi = mastercat['sdssi_fit'][ithisgal_mastercat]
    sdssz = mastercat['sdssz_fit'][ithisgal_mastercat]

    twomassj = mastercat['2massj_fit'][ithisgal_mastercat]
    twomassh = mastercat['2massh_fit'][ithisgal_mastercat]

    z = mastercat['ZTRUE'][ithisgal_mastercat]
    Jmag = mastercat['hst_f125w_obs'][ithisgal_mastercat]
    Hmag = mastercat['hst_f160w_obs'][ithisgal_mastercat]

    # TODO : make a separate function for plotting the photometry points
    maglist=[sdssu, sdssg, sdssr, sdssi, sdssz, imag, Jmag, Hmag]
    magmin = np.min(maglist)
    magmax = np.max(maglist)
    if showphot:
        # plot the SDSS, 2mass and HST mags
        ax.errorbar(7500, imag, xerr=500, marker='s', color='k', ms=8,
                    zorder=1000, label='_nolabel')
        ax.errorbar(12500, Jmag, xerr=12500 - 11000, marker='s', color='k',
                    ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(16000, Hmag, xerr=16000 - 13900, marker='s', color='k',
                    ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(21590, kmag, xerr=2620, marker='s', color='k', ms=8,
                    zorder=1000, label='observed i, HST-J, HST-H, Ks')

        ax.errorbar(3543 * (1 + z), sdssu, xerr=500 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(4770 * (1 + z), sdssg, xerr=500 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(6231 * (1 + z), sdssr, xerr=500 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(7625 * (1 + z), sdssi, xerr=600 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(9134 * (1 + z), sdssz, xerr=800 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='rest-frame SDSS/2MASS')
        ax.errorbar(12350 * (1 + z), twomassj, xerr=1620 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='_nolabel')
        ax.errorbar(16620 * (1 + z), twomassh, xerr=2510 * (1 + z), marker='d',
                    color='m', ms=8, zorder=1000, label='_nolabel')

    # ------------------------
    # DEIMOS Observed Spectrum :
    # ------------------------
    # read in the actual observed DEIMOS spectrum
    deimoslabel = 'Observed (DEIMOS)'
    if rescaledeimos:
        # rescale to match the observed SDSS i band mag
        wdeimos, mdeimos, wdeimos_smooth, mdeimos_smooth = getdeimosdat(
            galid, medsmooth_deimos, returnwaveunit='A', returnfluxunit='AB',
            extension='fits', imag=imag)
        deimoslabel += ' rescaled to match observed i mag'
    else:
        # Use the observed DEIMOS spectrum without rescaling
        wdeimos, mdeimos, wdeimos_smooth, mdeimos_smooth = getdeimosdat(
            galid, medsmooth_deimos, returnwaveunit='A', returnfluxunit='AB',
            extension='fits', imag=None)
    if showdeimos:
        # limit to wave<9000 Angstroms (DEIMOS response is unreliable redward)
        ilt9 = np.where(wdeimos_smooth<9000)[0]
        ax.plot(wdeimos_smooth[ilt9], mdeimos_smooth[ilt9],
                label=deimoslabel, color='g', zorder=20)

        
    # ------------------------
    # vUDS Observed Spectrum :
    # ------------------------
    # read in the actual observed vUDS spectrum
    vudslabel = 'Observed (vUDS)'
    if rescalevuds:
        imag_vuds = imag
        vudslabel += ' rescaled to match observed i mag'
    else:
        imag_vuds = None
    udsdat = getvudsdat(
            galid, medsmooth_vuds, returnwaveunit='A', returnfluxunit='AB',
            imag=imag_vuds)        
    if showvuds and udsdat is not None:
        wvuds, mvuds, wvuds_smooth, mvuds_smooth = udsdat
        ax.plot(wvuds_smooth, mvuds_smooth,
                label=vudslabel, color='b', zorder=30)

    # ------------------------
    # HST Observed Spectrum :
    # ------------------------
    hstdat = gethstdat(
        galid, medsmooth_hst,returnfluxunit='AB', returnwaveunit='A')
    if showhst and hstdat is not None:
        whst, mhst, whst_smooth, mhst_smooth = hstdat
        #ax.plot(whst, mhst, marker=' ',
        #        label='_nolabel', color='0.8', ls='-', zorder=30)
        ax.plot(whst_smooth, mhst_smooth,
                label='HST WFC3', color='r', ls='-', zorder=31)


    # plot the EAZY simulated spectrum
    eazycoeffs = np.array([mastercat[col][ithisgal_mastercat]
                           for col in mastercat.colnames
                           if col.startswith('coeff_specbasis')])
    outfilename = "DATA/cosmos_example_spectra/cosmos_example_host_simspec_" +\
                  "{:6d}.fits".format(galid)
    wobs, mobs = simulate_eazy_sed_from_coeffs(
        eazycoeffs, eazytemplatedata, z,
        returnwaveunit='A', returnfluxunit='AB25',
        savetofile=outfilename, overwrite=True)
    if showeazy:
        ax.plot(wobs, mobs, label='EAZY SED fit', color='0.5', zorder=10)
    
    ax.set_xlim(3000,19000)
    #ax.set_ylim(-0.25*1e-16,0.3*1e-16)
    #ax.set_ylim(27, 20)
    ax.text(0.95,0.95, galid, ha='right', va='top', transform=ax.transAxes)
    ax.text(0.95,0.88, "z={0}".format(z), ha='right', va='top', transform=ax.transAxes)

    ax = plt.gca()
    ax.set_xlim(3000, 19000)
    ax.set_ylim(magmin-2,magmax+1)

    ax.legend(loc='upper left')
    ax.invert_yaxis()
    ax.grid()
    ax.set_xlabel('Observed Wavelength (Angstroms)')
    ax.set_ylabel("AB mag")
    plt.tight_layout()
    #plt.savefig("cosmos_example_spec_eazysims.pdf")

    return(mdeimos_smooth)


def plot_all29():
    fig = plt.figure(figsize=[12,7])

    Ngal = len(deimoscat['GALID'])
    i = 0
    for galid in deimoscat['GALID']:
        i+=1
        plt.clf()
        fds = plot_spec_comparison(
            galid, showphot=True, showvuds=True, showdeimos=True,
            showhst=True, showeazy=True,
            rescaledeimos=True, rescalevuds=False,
            medsmooth_deimos=40, medsmooth_vuds=20, medsmooth_hst=5)
        plt.tight_layout()
        plt.savefig("DATA/cosmos_example_spectra/cosmos_example_specphot_comparison_{}.png".format(galid))



#for galid in deimoscat['GALID']:
#while True:
    #userin = input("Enter galid, or q to quit.")
    #if userin=='q':
    #    break
    #galid = int(userin)
if False:
    galid = 204489 #223627 #205346 #202800
    fds = plot_spec_comparison(
        galid, showphot=True, showvuds=True, showdeimos=True,
        showhst=True, showeazy=True,
        rescaledeimos=True, rescalevuds=False,
        medsmooth_deimos=40, medsmooth_vuds=20, medsmooth_hst=5)
    plt.show()
if True:
    plot_all29()



