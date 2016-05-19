import numpy as np
from matplotlib import pyplot as pl
import os
import sys
from astropy.io import ascii
from astropy.table import Table
import sncosmo
from scipy import integrate as scint, optimize as scopt
import exceptions

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__ :
    __THISFILE__ = __file__
__THISPATH__ = os.path.abspath(os.path.dirname(__THISFILE__))
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))

__LOWZLIST__ = ['sn1998bu_CfA', 'sn1999cl_CfA', 'sn1999cl_LCO', 'sn1999cl_LOSS',
                'sn1999cp_LCO', 'sn1999cp_LOSS', 'sn1999ek_LCO', 'sn1999ee_LCO',
                'sn1999gp_CfA', 'sn1999gp_LCO', 'sn1999gp_LOSS', 'sn2000E_ESO',
                'sn2000bh_LCO', 'sn2000ca_LCO', 'sn2000ce_LCO', 'sn2001ba_LCO',
                'sn2001bt_LCO', 'sn2001cn_LCO', 'sn2001cz_LCO', 'sn2001el_CTIO',
                'sn2002dj_ESO', 'sn2002dj_LOSS', 'sn2002fk_LOSS', 'sn2003cg_LOSS',
                'sn2003du_LOSS', 'sn2003hv_LOSS', 'sn2004S_CTIO', 'sn2003du_ESO',
                'sn2004S_LOSS', 'sn2004ef_CSP', 'sn2004ef_LOSS', 'sn2004eo_CSP',
                'sn2004eo_LOSS', 'sn2004ey_CSP', 'sn2004ey_LOSS', 'sn2004gs_CSP',
                'sn2004gs_LOSS', 'sn2005A_CSP', 'sn2005M_LOSS', 'sn2005M_CSP',
                'sn2005am_CSP', 'sn2005am_LOSS', 'sn2005bo_CSP', 'sn2005bo_LOSS',
                'sn2005cf_CfA', 'sn2005cf_LOSS', 'sn2005el_CSP', 'sn2005el_CfA',
                'sn2005el_LOSS', 'sn2005eq_CSP', 'sn2005eq_LOSS', 'sn2005eq_CfA',
                'sn2005eu_CfA', 'sn2005eu_LOSS', 'sn2005hc_CSP', 'sn2005hj_CSP',
                'sn2005iq_CSP', 'sn2005kc_CSP', 'sn2005ki_CSP', 'sn2005ls_CfA',
                'sn2005na_CSP', 'sn2005na_LOSS', 'sn2006D_CSP', 'sn2006D_CfA',
                'sn2006D_LOSS', 'sn2006X_CSP', 'sn2006X_CfA', 'sn2006X_LOSS',
                'sn2006ac_CfA', 'sn2006ax_CSP', 'sn2006bh_CSP', 'sn2006ax_CfA',
                'sn2006cp_LOSS', 'sn2006dd_CSP', 'sn2006ej_CSP', 'sn2006cp_CfA',
                'sn2006ej_LOSS', 'sn2006et_CSP', 'sn2006ev_CSP', 'sn2006gj_CSP',
                'sn2006gr_LOSS', 'sn2006hb_CSP', 'sn2006hb_LOSS','sn2006gr_CfA',
                'sn2006hx_CSP', 'sn2006is_CSP', 'sn2006kf_CSP', 'sn2006le_CfA',
                'sn2006le_LOSS', 'sn2006lf_CfA', 'sn2006ob_CSP','sn2006lf_LOSS',
                'sn2006os_CSP', 'sn2007A_CSP', 'sn2007S_CSP', 'sn2007S_CfA',
                'sn2007af_CSP', 'sn2007af_LOSS', 'sn2007as_CSP', 'sn2007bc_CSP',
                'sn2007bc_LOSS', 'sn2007bd_CSP', 'sn2007bm_CSP', 'sn2007ca_CSP',
                'sn2007ca_CfA', 'sn2007ca_LOSS', 'sn2007co_CfA', 'sn2007co_LOSS',
                'sn2007cq_LOSS', 'sn2007hx_CSP', 'sn2007jg_CSP', 'sn2007cq_CfA',
                'sn2007le_CSP', 'sn2007le_CfA', 'sn2007le_LOSS', 'sn2007nq_CSP',
                'sn2007on_CSP', 'sn2007qe_CfA', 'sn2007qe_LOSS', 'sn2007sr_CSP',
                'sn2007sr_LOSS', 'sn2008C_CSP', 'sn2008C_LOSS', 'sn2008R_CSP',
                'sn2008Z_CfA', 'sn2008Z_LOSS', 'sn2008bc_CSP', 'sn2008bq_CSP',
                'sn2008fp_CSP', 'sn2008gb_CfA', 'sn2008gp_CSP', 'sn2008gl_CfA',
                'sn2008hm_CfA', 'sn2008hv_CSP', 'sn2008hv_CfA', 'sn2008hs_CfA',
                'sn2008ia_CSP', 'sn2009D_CfA', 'sn2009al_CfA', 'sn2009ad_CfA',
                'sn2009bv_CfA', 'sn2009do_CfA', 'sn2009ds_CfA','sn2009an_CfA',
                'sn2009jr_CfA', 'sn2009kk_CfA', 'sn2009kq_CfA', 'sn2009fv_CfA',
                'sn2009le_CfA', 'sn2009lf_CfA', 'sn2009na_CfA', 'sn2010Y_CfA',
                'sn2010ag_CfA', 'sn2010dw_CfA', 'sn2010ju_CfA', 'sn2010ai_CfA',
                'sn2010kg_CfA', 'sn2011K_CfA', 'sn2011ae_CfA',  'sn2011B_CfA',
                'sn2011df_CfA', 'sn2011ao_CfA', 'sn2011by_CfA',
                'snf20080514-002_LOSS', 'snf20080522-000_CfA']


class LowzTemplate(object):
    def __init__(self, sedfile):
        self.sedfile = sedfile
        self.header = self.read_sed_template_header()
        self.phase, self.wave, self.flux = self.read_sed_template_data()

    def read_sed_template_header(self):
        """ read in metadata from the sed data file header
        :return:
        """
        fin = open(self.sedfile,'r')
        all_lines = fin.readlines()
        headerlines = []
        for hdrline in all_lines:
            hdrline = hdrline.strip()
            if len(hdrline)>0 and not hdrline.startswith('#'):
                break
            if '=' not in hdrline:
                continue
            hdrline = hdrline.strip('# ')
            key = hdrline.split('=')[0].strip()
            value = hdrline.split('=')[1].strip()
            if key not in ['name','survey']:
                value = float(value)
            self.__dict__[key] = value
            headerlines.append(hdrline)
        return headerlines

    def read_sed_template_data(self) :
        """ read in the phase, wavelength and flux grids from the given SN SED
        template data file (e.g. from a SALT2 template0.dat file)
        :rtype: np.ndarray, np.ndarray, np.ndarray
        :param sedfile:
        :return phase, wave, flux: phase is a 1D array with each entry giving the
           rest-frame phase relative to B band max. wave and flux are 2D arrays,
           each with a 1D array for every day in the phase array.
        """
        p,w,f = np.loadtxt( self.sedfile, unpack=True )
        phaselist = np.unique( p )
        phasearray = np.array(phaselist)
        wavearray = np.array([ w[ np.where( p == day ) ] for day in phaselist ])
        fluxarray = np.array([ f[ np.where( p == day ) ] for day in phaselist ])
        return phasearray, wavearray, fluxarray

    def plot_sed(self, phase=0, **kwargs):
        """ plot the SED template data from the given set of data arrays, only at
        the specified phase.
        :param phaselist: 1D array giving the list of phases (rel. to B band max)
        :param wavegrid: 2D array giving the wavelength values at each phase
        :param fluxgrid: 2D array giving the flux values at each phase
        :param phase: (int) the rest-frame phase (rel. to B band max) to be plotted
        :param kwargs: passed on to matplotlib.pyplot.plot()
        :return:
        """
        ithisphase = np.where(np.abs(self.phase-phase)<1)[0]
        wave = self.wave[ithisphase, :][0]
        flux = self.flux[ithisphase, :][0]
        pl.plot(wave, flux, **kwargs)
        return wave, flux



def load_sncosmo_models(modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                        salt2dir='salt2-4', salt2irdir='salt2ir'):
    """
    Load all the lowz template SEDs from sncosmo, along with the original
    salt2 model and our modified salt2 model that is extrapolated to the NIR.
    :param datadir:
    :return:
    """
    # read in all the low-z mangled SEDs
    modeldict = {}
    for name in __LOWZLIST__:
        sn = sncosmo.Model(name.lower())
        modeldict[name.lower()] = sn

    # read in the original and the revised salt2 models:
    salt2modeldir = os.path.join(modeldir, salt2dir)
    salt2irmodeldir = os.path.join(modeldir, salt2irdir)
    salt2 = sncosmo.models.SALT2Source(modeldir=salt2modeldir, name='salt2')
    salt2ir = sncosmo.models.SALT2Source(modeldir=salt2irmodeldir,
                                         name='salt2ir')
    modeldict['salt2'] = salt2
    modeldict['salt2ir'] = salt2ir

    return modeldict


def plot_template0_data(modeldict=None, phase=0, x1=0, c=0,
                        deltarange=[-0.2,0.2]):
    """
    Load all the lowz template SEDs from sncosmo

    Plot the template0 data at the given phase with


    :param datadir:
    :return:
    """
    if modeldict == None:
        modeldict = load_sncosmo_models()

    salt2mod = modeldict['salt2']
    salt2mod.set(x1=x1, c=c)

    salt2irmod = modeldict['salt2ir']
    salt2irmod.set(x1=x1, c=c)

    wave0 = np.arange(salt2mod.minwave(),salt2mod.maxwave(),10)
    wavelowz = np.arange(salt2mod.minwave(),18000,10)
    waveir = np.arange(2000, salt2irmod.maxwave())

    # plot all the low-z mangled SEDs at the given phase
    fluxlowzarray = []
    for name in __LOWZLIST__:
        lowzsn = modeldict[name.lower()]
        fluxlowz = lowzsn.flux(phase, wavelowz)
        iwavemax = np.where(wavelowz>=np.max(wave0))[0][0]
        # import pdb; pdb.set_trace()
        fluxlowz_norm = scint.trapz(fluxlowz[:iwavemax], wavelowz[:iwavemax])
        pl.plot(wavelowz, fluxlowz/fluxlowz_norm,
                'b-', alpha=0.1, lw=2 )
        # pl.plot(waveir, snflux/snflux.sum(), 'b-', alpha=0.3, lw=1 )
        fluxlowzarray.append(fluxlowz/fluxlowz_norm)

    # plot the normalized template0 data from the original SALT2 model
    # at the given phase:
    flux0 = salt2mod.flux(phase, wave0)
    flux0_norm = scint.trapz(flux0, wave0)
    pl.plot(wave0, flux0/flux0_norm, 'k-', lw=3,
            label='SALT2-4 template0' )

    # plot the median flux from all lowzIa templates
    fluxlowzarray = np.array(fluxlowzarray)
    pl.plot(wavelowz, np.median(fluxlowzarray, axis=0),
            'r--', alpha=1, lw=2,
            label='Median of low-z Sample' )

    # plot the normalized template0 flux from the modified SALT2-IR model
    # at the given phase:
    fluxir = salt2irmod.flux(phase, waveir)
    iwavemaxir = np.where(waveir>=np.max(wave0))[0][0]
    fluxir_norm = scint.trapz(fluxir[:iwavemaxir], waveir[:iwavemaxir])
    pl.plot(waveir, fluxir/fluxir_norm, 'g:', lw=3,
            label='SALT2-IR template0' )

    ax = pl.gca()
    ax.set_xlabel('wavelength (Angstroms)')
    ax.set_ylabel('flux (normalized)')
    # return temp0, modeldict
    pl.legend(loc='upper right')
    ax.set_xlim(3500,11000)
    ax.set_ylim(0,0.0005)


def fit_mlcs_to_salt2_parameter_conversion_functions(
        fitresfilename='lowz_salt.fitres', showfits=False, verbose=False):
    """ NOTE: this is a really crude kludge of a solution.

    Get the SALT2 x1,c and MLCS delta, Av values for all SNe for which we have
    both.  Fit a simple quadratic to each pair of corresponding parameters.
    :returns x1fitparam, cfitparam: the parameters of the quadratic functions
     that fit the x1 vs Delta and c vs Av functions.
    """
    # read in the low-z SN metadata from the file provided by Arturo
    metadata = load_metadata()
    # read in the low-z SN salt2 fit parameters from the file provided by Dan
    salt2fitfile = os.path.join(__THISDIR__, fitresfilename)
    salt2fitdata = ascii.read(salt2fitfile,
                              format='commented_header', data_start=0,
                              header_start=-1)
    x1list, x1errlist, clist, cerrlist = [],[],[], []
    deltalist, deltaerrlist, avlist, averrlist, namelist = [],[],[],[], []

    for snname in metadata['snname']:
        imeta = np.where(metadata['snname']==snname)[0]
        snname_stripped = snname.lstrip('sn')
        delta = float(metadata['Delta'][imeta])
        deltaerr = float(metadata['dDelta'][imeta])
        av = float(metadata['Av_mlcs'][imeta])
        averr = float(metadata['dAv'][imeta])
        if delta <= -0.5:
            continue
        if delta > 1:
            continue

        if av > 1.8:
            continue

        if snname_stripped not in salt2fitdata['CID']:
            if verbose:
                print "missing %s in salt2 fit data. Skipping" % snname_stripped
            continue

        isalt2 = np.where(salt2fitdata['CID']==snname_stripped)[0]
        x1 = np.median(salt2fitdata['x1'][isalt2])
        x1err = np.median(salt2fitdata['x1ERR'][isalt2])
        c = np.median(salt2fitdata['c'][isalt2])
        cerr = np.median(salt2fitdata['cERR'][isalt2])
        if x1<-3:
            continue

        x1list.append(x1)
        x1errlist.append(c)
        clist.append(c)
        cerrlist.append(cerr)
        deltalist.append(delta)
        deltaerrlist.append(deltaerr)
        avlist.append(av)
        averrlist.append(averr)
        namelist.append(snname)

    x1 = np.array(x1list)
    x1err = np.array(x1errlist)
    c = np.array(clist)
    cerr = np.array(cerrlist)
    delta = np.array(deltalist)
    deltaerr = np.array(deltaerrlist)
    av = np.array(avlist)
    averr = np.array(averrlist)

    # TODO : switch to using scipy.odr for fitting with errors in both
    #  dimensions.
    x1fit = scopt.curve_fit(quadratic, delta, x1,
                            p0=None, sigma=x1err,
                            absolute_sigma=True,
                            check_finite=True, )
    x1fitparam = x1fit[0]
    x1fitcov = np.sqrt(np.diag(x1fit[1]))

    cfit = scopt.curve_fit(quadratic, av, c,
                           p0=None, sigma=cerr,
                           absolute_sigma=True,
                           check_finite=True, )
    cfitparam = cfit[0]
    cfitcov = np.sqrt(np.diag(cfit[1]))

    if showfits:
        fig = pl.gcf()
        fig.clf()
        ax1 = fig.add_subplot(2,1,1)
        pl.errorbar(deltalist, x1list, x1errlist, deltaerrlist, marker='o',
                    color='k', ls=' ')
        ax = pl.gca()
        ax.set_xlabel('MLCS $\Delta$')
        ax.set_ylabel('SALT2 x$_1$')

        ax2 = fig.add_subplot(2,1,2)
        pl.errorbar(avlist, clist, cerrlist, averrlist, marker='d', color='g',
                    ls=' ')
        ax = pl.gca()
        ax.set_xlabel('MLCS $A_V$')
        ax.set_ylabel('SALT2 c')

        deltarange = np.arange(-0.4, 1.0, 0.01)
        ax1.plot( deltarange,
                  quadratic(deltarange, x1fitparam[0],
                            x1fitparam[1], x1fitparam[2]),
                  ls='-', color='r', marker=' ')


        avrange = np.arange(-0.1, 1.9, 0.01)
        ax2.plot( avrange,
                  quadratic(avrange, cfitparam[0],  cfitparam[1], cfitparam[2]),
                  ls='-', color='r', marker=' ')

        ax2.set_xlim(-0.1, 1.9)
        pl.draw()
    return x1fitparam, cfitparam


def quadratic(x, A, B, C):
    return A + B * x + C * x * x

def cubic(x, A, B, C, D):
    return A + B * x + C * x * x + D * x * x * x


def extend_template0_ir(modeldict = None,
                        modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                        salt2dir = 'salt2-4',
                        salt2irdir = 'salt2ir',
                        wavejoin = 8500, wavemax = 18000,
                        showplots=False):
    """ extend the salt2 Template_0 model component
    by adopting the IR tails from a collection of SN Ia template SEDs.
    Here we use the collection of CfA, CSP, and other low-z SNe provided by
    Arturo Avelino (2016, priv. comm.)
    The median of the sample is scaled and joined at the
    wavejoin wavelength, and extrapolated out to wavemax.
    """
    if modeldict == None:
        modeldict = load_models()
    salt2dir = os.path.join(modeldir, salt2dir)
    salt2irdir = os.path.join(modeldir, salt2irdir)

    temp0fileIN = os.path.join( salt2dir, 'salt2_template_0.dat' )
    temp0fileOUT = os.path.join( salt2irdir, 'salt2_template_0.dat' )
    templatein = LowzTemplate(temp0fileIN)

    # TODO : fix to use the lowztemplate class methods
    temp0phase, temp0wave, temp0flux = get_sed_template_data(temp0fileIN)


    wavestep = np.median(np.diff(temp0wave[0]))
    waveir = np.arange(wavejoin, wavemax+wavestep, wavestep)

    salt2mod = modeldict['salt2']
    # build up modified template0 data from day -20 to +50
    fscale = []
    outlines = []
    phaselist = np.unique(temp0phase)
    for iphase in np.arange(13,50,1): # len(phaselist)):
        # get the SALT2 template SED for this day
        phase0 = temp0phase[iphase]
        wave0 = temp0wave[iphase]
        flux0 = temp0flux[iphase]

        thisphase = phase0[0]

        ijoin = np.argmin(np.abs(wave0-wavejoin))
        fluxjoin = flux0[ijoin]
        print( 'splicing tail onto template for day : %i'%thisphase )

        # salt2_total_optical_flux = scint.trapz(flux0, wave0)

        # get the median of all the low-z mangled SEDs at this phase
        fluxlowzarray = []
        for name in __LOWZLIST__:
            lowzsn = modeldict[name.lower()]
            if lowzsn.mintime()>thisphase: continue
            if lowzsn.maxtime()<thisphase: continue

            fluxlowz = lowzsn.flux(thisphase, waveir)
            if np.sum(fluxlowz)==0: continue

            # determine the normalization factor that will normalize the flux
            # of this lowz mangled SED so that it integrates
            # to the same total flux as the salt2 template0 model, over
            # the wavelength span of the salt2 model
            #lowz_total_optical_flux = scint.trapz(
            #    lowzsn.flux(thisphase, wave0), wave0)

            normalization_factor = (fluxjoin / fluxlowz[0])
            #if thisphase == 0:
            #    import pdb; pdb.set_trace()
            fluxlowzarray.append(fluxlowz * normalization_factor)

        fluxlowzarray = np.array(fluxlowzarray)
        fluxlowzmedian = np.median(fluxlowzarray, axis=0)

        # extend the template0 SED into the IR for this phase
        # using the median of all the lowz templates
        ijoin0 = np.argmin(abs(wave0 - wavejoin))
        ijoinlowz = np.argmin(abs(waveir - wavejoin))

        imaxlowz = np.argmin(abs(waveir - wavemax))
        if flux0[ijoin0]>0:
            scalefactor = fluxlowzmedian[ijoinlowz]/flux0[ijoin0]
        else:
            scalefactor = 1

        phasenew = np.append(phase0[:ijoin0],
                             np.ones(len(waveir)) * thisphase)
        wavenew = np.append(wave0[:ijoin0], waveir.tolist())
        fluxnew = np.append(flux0[:ijoin0],
                            (scalefactor*fluxlowzmedian))
        #if thisphase == 0:
        #    import pdb; pdb.set_trace()

        # append to the list of output data lines
        for j in range( len(phasenew) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    phasenew[j], wavenew[j], fluxnew[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp0fileOUT, 'w' )
    fout.writelines( outlines )
    fout.close()

def plot_extended_template0():
    # TODO : update to use the lowztemplate class methods
    pl.clf()
    phasearray, wavearray, fluxarray = get_sed_template_data('salt2ir/salt2_template_0.dat')
    wave, flux = plot_sed_template_data(phasearray, wavearray, fluxarray, phase=0, color='r', ls='-')
    wave, flux = plot_sed_template_data(phasearray, wavearray, fluxarray, phase=-5, color='b', ls='-')
    wave, flux = plot_sed_template_data(phasearray, wavearray, fluxarray, phase=10, color='g', ls='-')
    wave, flux = plot_sed_template_data(phasearray, wavearray, fluxarray, phase=20, color='m', ls='-')
    wave, flux = plot_sed_template_data(phasearray, wavearray, fluxarray, phase=28, color='c', ls='-')




def ccm_unred(wave, flux, ebv, r_v=""):
    """ccm_unred(wave, flux, ebv, r_v="")
    Deredden a flux vector using the CCM 1989 (and O'Donnell 1994)
    parameterization. Returns an array of the unreddened flux

    INPUTS:
    wave - array of wavelengths (in Angstroms)
    dec - calibrated flux array, same number of elements as wave
    ebv - colour excess E(B-V) float. If a negative ebv is supplied
          fluxes will be reddened rather than dereddened

    OPTIONAL INPUT:
    r_v - float specifying the ratio of total selective
          extinction R(V) = A(V)/E(B-V). If not specified,
          then r_v = 3.1

    OUTPUTS:
    funred - unreddened calibrated flux array, same number of
             elements as wave

    NOTES:
    1. This function was converted from the IDL Astrolib procedure
       last updated in April 1998. All notes from that function
       (provided below) are relevant to this function

    2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
       ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.

    3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction
       can be represented with a CCM curve, if the proper value of
       R(V) is supplied.

    4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989, ApJ, 339,474)

    5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the
       original flux vector.

    6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1).    But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.

    7. For the optical/NIR transformation, the coefficients from
       O'Donnell (1994) are used

    >>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 )
    array([9.7976e+012, 1.12064e+07, 32287.1])
    """
    import numpy as np
    wave = np.array(wave, float)
    flux = np.array(flux, float)

    if wave.size != flux.size: raise TypeError, 'ERROR - wave and flux vectors must be the same size'

    if not bool(r_v): r_v = 3.1

    x = 10000.0/wave
    npts = wave.size
    a = np.zeros(npts, float)
    b = np.zeros(npts, float)

    ###############################
    #Infrared

    good = np.where( (x > 0.3) & (x < 1.1) )
    a[good] = 0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)

    ###############################
    # Optical & Near IR

    good = np.where( (x  >= 1.1) & (x < 3.3) )
    y = x[good] - 1.82

    c1 = np.array([ 1.0 , 0.104,   -0.609,    0.701,  1.137, \
                  -1.718,   -0.827,    1.647, -0.505 ])
    c2 = np.array([ 0.0,  1.952,    2.908,   -3.989, -7.985, \
                  11.102,    5.491,  -10.805,  3.347 ] )

    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    ###############################
    # Mid-UV

    good = np.where( (x >= 3.3) & (x < 8) )
    y = x[good]
    F_a = np.zeros(np.size(good),float)
    F_b = np.zeros(np.size(good),float)
    good1 = np.where( y > 5.9 )

    if np.size(good1) > 0:
        y1 = y[good1] - 5.9
        F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
        F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3

    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b

    ###############################
    # Far-UV

    good = np.where( (x >= 8) & (x <= 11) )
    y = x[good] - 8.0
    c1 = [ -1.073, -0.628,  0.137, -0.070 ]
    c2 = [ 13.670,  4.257, -0.420,  0.374 ]
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    # Applying Extinction Correction

    a_v = r_v * ebv
    a_lambda = a_v * (a + b/r_v)

    funred = flux * 10.0**(0.4*a_lambda)

    return funred

def load_metadata(metadatafilename='lowz_metadata.txt'):
    """read in the low-z SN metadata from the file provided by Arturo"""
    metadatafile = os.path.join(__THISDIR__, metadatafilename)
    metadata = ascii.read(metadatafile,
                          format='commented_header', data_start=0,
                          header_start=-1)
    return metadata


def deredden_template_sed(sedfile, sedfileout=None, snname=None,
                          metadatafilename='lowz_metadata.txt',
                          fitresfilename='lowz_salt2.fitres'):
    """
    For the given low-z SN, modify the SED template by correcting for
    the redshift and dust extinction (both in the SN host galaxy and in the
    Milky Way along the line of sight).   The result is an un-reddened
    SED template in the rest-frame.

    read in the metadata file
    for each low-z SN data file
    get the metadata
    correct for host galaxy E(B-V)   :  use the MLCS Av and Rv
    correct for redshift  :  use z_helio  (though zcmb is ~ equivalent)
    correct for MW A_V  : these are the E(B-V) and AV_gal columns

    :param sedfile: The SN SED file in the template library.
        If snname is None, then we assume that the root of the file name
        corresponds to the SN ID in the metadata file.
        e.g.  'sn2010ju_cfa'
    :param snname: the SN ID in the metadata file.
    :param sedfileout: filename in which to dump the output arrays
    :return phase, wave, flux: These arrays provide the data for the
        de-reddened template SED.  The array phase is a 1D array with each
        entry giving the rest-frame phase relative to B band max. wave and
        flux are 2D arrays, each with a 1D array for every day in <phase>.
    """
    if snname is None:
        snname = os.path.basename(sedfile).split('_')[0]

    if not os.path.isfile(sedfile):
        raise exceptions.RuntimeError("No such file %s"% sedfile)

    metadata = load_metadata(metadatafilename)

    if snname not in metadata['snname']:
        raise exceptions.RuntimeError(
            'No SN named %s in the metadata file' % snname)
    imeta = np.where(metadata['snname']==snname)[0]
    zhelio = metadata['z_helio'][imeta]
    zcmb = metadata['z_cmb'][imeta]
    zcmb = metadata['z_cmb'][imeta]
    EBVmw = metadata['E(B-V)'][imeta]
    Rvmw = 3.1
    Avhost = metadata['Av_mlcs'][imeta]
    Rvhost = metadata['Rv_mlcs'][imeta]
    Avhost = metadata['Av_mlcs'][imeta]
    EBVhost = Avhost / Rvhost
    Delta_mlcs = metadata['Delta'][imeta]

    #TODO: convert from Delta_mlcs to SALT2 x1

    # read in the low-z SN salt2 fit parameters from the file provided by Dan
    salt2fitfile = os.path.join(__THISDIR__, fitresfilename)
    salt2fitdata = ascii.read(salt2fitfile,
                              format='commented_header', data_start=0,
                              header_start=-1)
    snname_stripped = snname.lstrip('sn')
    if snname_stripped in salt2fitdata['CID']:
        isalt2 = np.where(salt2fitdata['CID']==snname_stripped)[0]
        x1 = np.median(salt2fitdata['x1'][isalt2])
        c = np.median(salt2fitdata['c'][isalt2])
        zHD = np.median(salt2fitdata['zHD'][isalt2])
        zsalt2 = np.median(salt2fitdata['z'][isalt2])
        if np.abs(zhelio - zcmb) > 0.01:
            print("Note significant difference in redshift for %s" % snname +
                  " \\n zhelio = %.5f    zsalt2= %.5f" % (zhelio, zsalt2))
    else:
        x1 = -9
        c = -9
        zHD = -9
        z = -9

    # read in the SED template file directly
    lowzsn = LowzTemplate(sedfile)
    snphase, snwave, snflux = lowzsn.phase, lowzsn.wave, lowzsn.flux

    # Define new arrays to hold the de-reddened template data
    snphaseout, snwaveout, snfluxout = [], [], []

    if sedfileout is not None:
        fout = open(sedfileout, 'w')
        # print a header that carries all the relevant metadata
        print >> fout, '# name = %s' % snname_stripped
        print >> fout, '# survey = %s' % snname.split('_')[-1]
        print >> fout, '# z = %.5f' % zhelio
        print >> fout, '# salt2x1 = %.5f' % x1
        print >> fout, '# salt2c = %.5f' % c
        print >> fout, '# mlcsDelta = %.5f' % Delta_mlcs
        print >> fout, '# E(B-V)mw = %.5f' % EBVmw
        print >> fout, '# RVmw = %.5f' % Rvmw
        print >> fout, '# AVhost = %.5f' % Avhost
        print >> fout, '# RVhost = %.5f' % Rvhost
        print >> fout, '# E(B-V)host = %.5f' % EBVhost
        print >> fout, '# phase                   wavelength               flux'

    for phase in sorted(snphase.tolist()) :
        iphase = np.where(snphase == phase)[0][0]
        # correct the host extinction, redshift and MW extinction
        phase = snphase[iphase]
        snflux1 = ccm_unred(snwave[iphase], snflux[iphase], EBVhost, Rvhost)
        snwave1 = snwave[iphase] / (1+zhelio)
        snflux2 = ccm_unred(snwave1, snflux1, EBVmw, Rvmw)
        snphaseout.append(phase)
        snwaveout.append(snwave1)
        snfluxout.append(snflux2)

        if sedfileout is not None:
            for w,f in zip(snwave1, snflux2):
                print >> fout, '%25.18e %25.18e %25.18e' % (phase, w, f)

    if sedfileout is not None:
        fout.close()

    snphaseout = np.array(snphaseout)
    snwaveout = np.array(snwaveout)
    snfluxout = np.array(snfluxout)

    return snphaseout, snwaveout, snfluxout


def plot_dereddened_template_comparison(sedfile0, sedfile, phase=0):
    lowzsn0 = LowzTemplate(sedfile0)
    lowzsn1 = LowzTemplate(sedfile)


    snphase0, snwave0, snflux0 = lowzsn0.phase, lowzsn0.wave, lowzsn0.flux
    snphase1, snwave1, snflux1 = lowzsn1.phase, lowzsn1.wave, lowzsn1.flux
    salt2mod = sncosmo.Model('salt2')

    iphase = np.argmin(np.abs(snphase0-phase))
    bestphase0 = snphase0[iphase]

    salt2wave = np.arange(salt2mod.minwave(),salt2mod.maxwave(),10)
    salt2flux = salt2mod.flux(bestphase0, salt2wave)
    salt2flux_norm = scint.trapz(salt2flux, salt2wave)
    pl.plot(salt2wave, salt2flux/salt2flux_norm, 'k--', lw=1,
            label='SALT2-4 template0' )
    minwave = np.min(salt2wave)

    for snphase, snwave, snflux, color, label in zip(
            [snphase0,snphase1],[snwave0,snwave1],[snflux0,snflux1],
            ['r','c'], ['original', 'dereddened']):
        iwavemax = np.where(snwave[iphase]>=np.max(salt2wave))[0][0]
        if minwave <= np.min(snwave):
            iwavemin = 0
        else:
            iwavemin = np.where(snwave[iphase] <= minwave)[0][-1]

        snflux_norm = scint.trapz(snflux[iphase][iwavemin:iwavemax],
                                  snwave[iphase][iwavemin:iwavemax])
        pl.plot(snwave[iphase], snflux[iphase]/snflux_norm,
                ls='-', color=color, alpha=1, lw=2, label=label )
    pl.legend(loc='upper right')
    ax = pl.gca()
    ax.set_xlabel('Wavelength ($\AA$)')
    ax.set_xlim(2000, 16000)
    snid = os.path.basename(sedfile0).split('_')[0].lstrip('sn')
    ax.text(8500, 0.00015, 'SN ' + snid, fontsize='large')
    pl.draw()

def deredden_all_templates(showfig=True, savefig=True, clobber=False):
    import glob
    sedfilelist = glob.glob('lowzIaObsFrame/sn*_*.dat')
    for sedfile0 in sedfilelist:
        sedfilename = os.path.basename(sedfile0)
        snname = sedfilename.split('_')[0]
        sedfile1 = os.path.join('lowzIa', sedfilename)
        if os.path.exists(sedfile1) and not clobber:
            print("%s exists. Not clobbering" % sedfile1)
            continue
        try:
            deredden_template_sed(sedfile0, sedfile1, snname=snname)
            print "Successfully Dereddened %s" % snname
        except:
            print "!!!  Failed to Deredden %s  !!!" % snname
            continue
        if showfig:
            pl.clf()
            plot_dereddened_template_comparison(sedfile0, sedfile1, phase=0)
        if savefig:
            figfilename = 'lowzIa/%s.png' % sedfilename.split('.')[0]
            pl.savefig(figfilename)

def load_all_templates():
    templatedict = {}
    for snname in __LOWZLIST__:
        sedfile = os.path.join('lowzIa', snname) + '.dat'
        templatedict[snname] = LowzTemplate(sedfile=sedfile)

    return templatedict

