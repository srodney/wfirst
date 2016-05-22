import numpy as np
from matplotlib import pyplot as pl
import os
import sys
from astropy.io import ascii
from astropy.table import Table
import sncosmo
from scipy import integrate as scint, interpolate as scinterp, optimize as scopt
import exceptions


_B_WAVELENGTH = 4302.57
_V_WAVELENGTH = 5428.55

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


class TimeSeriesGrid(object):
    def __init__(self, sedfile):
        self.sedfile = sedfile

        fin = open(self.sedfile,'r')
        all_lines = fin.readlines()
        self.headerlines = np.array([line for line in all_lines
                                     if len(line.strip()) > 0 and
                                     line.lstrip().startswith('#')])
        self.parse_lowz_template_header()
        self.phase, self.wave, self.value = self.read_timeseriesgrid_data()

    def parse_lowz_template_header(self):
        """ read in metadata from the sed data file header
        :return:
        """
        for hdrline in self.headerlines:
            hdrline = hdrline.strip()
            if '=' not in hdrline:
                continue
            hdrline = hdrline.strip('# ')
            key = hdrline.split('=')[0].strip()
            value = hdrline.split('=')[1].strip()
            if key not in ['name','survey']:
                value = float(value)
            self.__dict__[key] = value

    def read_timeseriesgrid_data(self) :
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

    def plot_at_phase(self, phase=0, **kwargs):
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
        flux = self.value[ithisphase, :][0]
        pl.plot(wave, flux, **kwargs)
        return wave, flux


    def extrapolate_flatline(self, outfilename,
                             maxwave=25000, showplots=False):
        """ use a super-crude flatline extension of the value array
        to extrapolate this model component to the IR at all phases """
        outlines = []
        for iphase in range(len(self.phase)):
            thisphase = self.phase[iphase]
            w = self.wave[iphase]
            v = self.value[iphase]
            wavestep = w[1] - w[0]

            # redward flatline extrapolation from last point
            wavenew = np.arange(w[0], maxwave, wavestep)
            valnew = np.append(v, np.zeros(len(wavenew)-len(w)) + v[-1])

            # append to the list of output data lines
            for j in range(len(wavenew)):
                outlines.append("%6.2f    %12i  %12.7e\n" % (
                    thisphase, wavenew[j], valnew[j]))

        # write it out to the new .dat file
        fout = open(outfilename, 'w')
        fout.writelines(outlines)
        fout.close()
        return

def load_sncosmo_models(modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                        salt2subdir='salt2-4', salt2irsubdir='salt2ir',
                        lowzsubdir='lowzIa'):
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
        lowzmodeldir = os.path.join(modeldir, lowzsubdir)
        lowzmodeldatfile = os.path.join(lowzmodeldir, '%s.dat' % name)

        # read the header info of the sed template file to
        # determine the x1,c, Delta, Av, z  and
        # store these as properties of the sncosmo Model object
        fin = open(lowzmodeldatfile,'r')
        all_lines = fin.readlines()
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
            sn.__dict__[key] = value

        modeldict[name.lower()] = sn

    # read in the original and the revised salt2 models:
    salt2modeldir = os.path.join(modeldir, salt2subdir)
    salt2irmodeldir = os.path.join(modeldir, salt2irsubdir)
    salt2 = sncosmo.models.SALT2Source(modeldir=salt2modeldir, name='salt2')
    salt2ir = sncosmo.models.SALT2Source(modeldir=salt2irmodeldir,
                                         name='salt2ir')
    modeldict['hsiao'] = sncosmo.Model('hsiao')
    modeldict['salt2'] = salt2
    modeldict['salt2ir'] = salt2ir

    return modeldict


def plot_template0_data(modeldict=None, phase=0, x1=0, c=0):
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


def get_mlcs_to_salt2_parameter_conversion_functions(
        fitresfilename='lowz_salt2.fitres', showfits=False, verbose=False):
    """ NOTE: this is a really crude kludge of a solution.

    Get the SALT2 x1,c and MLCS delta, Av values for all SNe for which we have
    both.  Fit a simple quadratic to each pair of corresponding parameters.
    :returns delta2x1, av2c: functions that convert from the MLCS parameter
        delta or Av to the SALT2 parameter x1 or c, respectively.
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

    cfit = scopt.curve_fit(linear, av, c,
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
                  linear(avrange, cfitparam[0],  cfitparam[1]),
                  ls='-', color='r', marker=' ')

        ax2.set_xlim(-0.1, 1.9)
        pl.draw()

    def mlcsdelta_to_salt2x1(delta):
        return quadratic(delta, x1fitparam[0], x1fitparam[1], x1fitparam[2])

    def mlcsav_to_salt2c(c):
        return linear(c, cfitparam[0], cfitparam[1])

    return mlcsdelta_to_salt2x1, mlcsav_to_salt2c


def linear(x, A, B):
    return A + B * x

def quadratic(x, A, B, C):
    return A + B * x + C * x * x

def cubic(x, A, B, C, D):
    return A + B * x + C * x * x + D * x * x * x


def ccm_unred(wave, flux, ebv, r_v=3.1):
    """ccm_unred(wave, flux, ebv, r_v="")
    Deredden a flux vector using the CCM 1989 (and O'Donnell 1994)
    parameterization. Returns an array of the unreddened flux
    """
    wave = np.array(wave, float)
    flux = np.array(flux, float)
    if wave.size != flux.size:
        raise TypeError, 'ERROR - wave and flux vectors must be the same size'

    a_lambda = ccm_extinction(wave, ebv, r_v)

    funred = flux * 10.0**(0.4*a_lambda)

    return funred


def ccm_extinction(wave, ebv, r_v=3.1):
    """
    The extinction (A_lambda) for given wavelength (or vector of wavelengths)
    from the CCM 1989 (and O'Donnell 1994) parameterization. Returns an
    array of extinction values for each wavelength in 'wave'

    INPUTS:
    wave - array of wavelengths (in Angstroms)
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
    scalar = not np.iterable(wave)
    if scalar:
        wave = np.array([wave], float)
    else:
        wave = np.array(wave, float)

    x = 10000.0/wave
    npts = wave.size
    a = np.zeros(npts, float)
    b = np.zeros(npts, float)

    #Infrared
    good = np.where( (x > 0.3) & (x < 1.1) )
    a[good] = 0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)

    # Optical & Near IR
    good = np.where( (x  >= 1.1) & (x < 3.3) )
    y = x[good] - 1.82

    c1 = np.array([ 1.0 , 0.104,   -0.609,    0.701,  1.137,
                  -1.718,   -0.827,    1.647, -0.505 ])
    c2 = np.array([ 0.0,  1.952,    2.908,   -3.989, -7.985,
                  11.102,    5.491,  -10.805,  3.347 ] )

    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

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

    # Far-UV
    good = np.where( (x >= 8) & (x <= 11) )
    y = x[good] - 8.0
    c1 = [ -1.073, -0.628,  0.137, -0.070 ]
    c2 = [ 13.670,  4.257, -0.420,  0.374 ]
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    # Defining the Extinction at each wavelength
    a_v = r_v * ebv
    a_lambda = a_v * (a + b/r_v)
    if scalar:
        a_lambda = a_lambda[0]
    return a_lambda



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
    EBVmw = metadata['E(B-V)'][imeta]
    Rvmw = 3.1
    Rvhost = metadata['Rv_mlcs'][imeta]
    Avhost = metadata['Av_mlcs'][imeta]
    Delta_mlcs = metadata['Delta'][imeta]
    if Delta_mlcs==-999:
        Delta_mlcs=0
    if Avhost==-999:
        Avhost= 0
    if Rvhost==-999:
        Rvhost= 3.1
    EBVhost = Avhost / Rvhost

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
        delta2x1, av2c = get_mlcs_to_salt2_parameter_conversion_functions(
            fitresfilename=fitresfilename)
        x1 = delta2x1(Delta_mlcs)
        c = av2c(Avhost)
        zHD = zcmb
        z = zhelio

    # read in the SED template file directly
    lowzsn = TimeSeriesGrid(sedfile)
    snphase, snwave, snflux = lowzsn.phase, lowzsn.wave, lowzsn.value

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
    lowzsn0 = TimeSeriesGrid(sedfile0)
    lowzsn1 = TimeSeriesGrid(sedfile)

    snphase0, snwave0, snflux0 = lowzsn0.phase, lowzsn0.wave, lowzsn0.value
    snphase1, snwave1, snflux1 = lowzsn1.phase, lowzsn1.wave, lowzsn1.value
    salt2mod = sncosmo.Model('salt2')

    iphase = np.argmin(np.abs(snphase0-phase))
    bestphase0 = snphase0[iphase]

    minwave = 3500
    salt2wave = np.arange(salt2mod.minwave(),salt2mod.maxwave(),10)
    if minwave <= np.min(salt2wave):
        iwavemin = 0
    else:
        iwavemin = np.where(salt2wave <= minwave)[0][-1]

    salt2flux = salt2mod.flux(bestphase0, salt2wave)
    salt2flux_norm = scint.trapz(salt2flux[iwavemin:], salt2wave[iwavemin:])

    pl.plot(salt2wave, salt2flux/salt2flux_norm, 'k--', lw=1,
            label='SALT2-4 template0' )

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
    ax.set_ylim(-0.00005, 0.00058)

    snid = os.path.basename(sedfile0).split('_')[0].lstrip('sn')
    ax.text(8500, 0.00015, 'SN ' + snid, fontsize='large')
    pl.draw()

def deredden_template_list(sedfilelist=None,
        showfig=True, savefig=True, clobber=False):
    import glob
    if sedfilelist is None:
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
        templatedict[snname] = TimeSeriesGrid(sedfile=sedfile)

    return templatedict



def extend_template0_ir(modeldict = None, x1min=-1, x1max=1,
                        modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                        salt2dir = 'salt2-4',
                        salt2irdir = 'salt2ir',
                        wavejoin = 8500, wavemax = 24990,
                        showplots=False):
    """ extend the salt2 Template_0 model component
    by adopting the IR tails from a collection of SN Ia template SEDs.
    Here we use the collection of CfA, CSP, and other low-z SNe provided by
    Arturo Avelino (2016, priv. comm.)
    The median of the sample is scaled and joined at the
    wavejoin wavelength, and extrapolated out to wavemax.
    """
    if modeldict == None:
        modeldict = load_sncosmo_models()
    salt2dir = os.path.join(modeldir, salt2dir)
    salt2irdir = os.path.join(modeldir, salt2irdir)

    temp0fileIN = os.path.join( salt2dir, 'salt2_template_0.dat' )
    temp0fileOUT = os.path.join( salt2irdir, 'salt2_template_0.dat' )
    temp0 = TimeSeriesGrid(temp0fileIN)
    wavestep = np.median(np.diff(temp0.wave[0]))
    waveir = np.arange(wavejoin, wavemax+wavestep, wavestep)

    outlines = []
    for iphase in range(len(temp0.phase)): # iphaselist:
        # get the SALT2 template0 SED for this day
        # phase0 = temp0.phase[iphase]
        wave0 = temp0.wave[iphase]
        flux0 = temp0.value[iphase]
        thisphase = temp0.phase[iphase]

        ijoin = np.argmin(np.abs(wave0-wavejoin))
        fluxjoin = flux0[ijoin]
        print( 'splicing tail onto template for day : %i'%thisphase )

        # get the median of all the low-z mangled SEDs that have data at
        # this phase and satisfy the x1 range requirements
        fluxlowzarray = []
        for name in __LOWZLIST__:
            lowzsn = modeldict[name.lower()]

            if lowzsn.mintime()>thisphase: continue
            if lowzsn.maxtime()<thisphase: continue
            if lowzsn.salt2x1<x1min: continue
            if lowzsn.salt2x1>x1max: continue

            fluxlowz = lowzsn.flux(thisphase, waveir)
            if np.sum(fluxlowz)==0: continue

            # normalize the flux of this lowz mangled SED so that it matches
            # the flux of the salt2 template0 model at the join wavelength
            normalization_factor = (fluxjoin / fluxlowz[0])
            fluxlowzarray.append(fluxlowz * normalization_factor)

        # extend the template0 SED into the IR for this phase
        # using the median of all the lowz templates (or the Hsiao model when
        # templates are not available
        if len(fluxlowzarray)<5:
            print("only %i templates for phase = %.1f. Using Hsiao model" % (
                len(fluxlowzarray), thisphase))
            fluxlowzmedian = modeldict['hsiao'].flux(thisphase, waveir)
        else:
            fluxlowzarray = np.array(fluxlowzarray)
            fluxlowzmedian = np.median(fluxlowzarray, axis=0)

        ijoin0 = np.argmin(abs(wave0 - wavejoin))
        ijoinlowz = np.argmin(abs(waveir - wavejoin))

        if flux0[ijoin0]>0:
            scalefactor = fluxlowzmedian[ijoinlowz]/flux0[ijoin0]
        else:
            scalefactor = 1
        wavenew = np.append(wave0[:ijoin0], waveir.tolist())
        fluxnew = np.append(flux0[:ijoin0], (scalefactor*fluxlowzmedian))

        # append to the list of output data lines
        for j in range( len(wavenew) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    thisphase, wavenew[j], fluxnew[j] ) )

    # write it out to the new template0 .dat file
    fout = open( temp0fileOUT, 'w' )
    fout.writelines( outlines )
    fout.close()

def plot_extended_template0():
    pl.clf()
    newtemp0 = TimeSeriesGrid('salt2ir/salt2_template_0.dat')
    for phase, color in zip([-15,-5,0,5,25],['m','b','g','r','k']):
        newtemp0.plot_at_phase(phase=phase, color=color, ls='-',
                               label='phase = %i'%int(phase))
    pl.legend(loc='upper right')




def extend_template1_ir(modeldict = None, cmin=-0.1, cmax=0.3,
                        x1min=-1, x1max=1,
                        modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                        salt2dir = 'salt2-4',
                        salt2irdir = 'salt2ir',
                        wavejoinstart = 8500, wavemax = 25000,
                        showphase0plots=False):
    """ extend the M1 component of the SALT2 model into the IR.  (this is the
    component that is multiplied by the SALT2 parameter x1, and therefore
    reflects changes in the shape of the light curve.

    :param modeldict:
    :param modeldir:
    :param salt2dir:
    :param salt2irdir:
    :param wavejoinstart:
    :param wavemax:
    :param showphase0plots:
    :return:
    """
    if modeldict == None:
        modeldict = load_sncosmo_models()
    salt2dir = os.path.join(modeldir, salt2dir)
    salt2irdir = os.path.join(modeldir, salt2irdir)

    # read in the extended template0 data (i.e. the M0 model
    # component that has already been extended to IR wavelengths
    temp0extfile = os.path.join( salt2irdir, 'salt2_template_0.dat' )
    temp0ext = TimeSeriesGrid(temp0extfile)

    temp1fileIN = os.path.join( salt2dir, 'salt2_template_1.dat' )
    temp1 = TimeSeriesGrid(temp1fileIN)

    temp1fileOUT = os.path.join( salt2irdir, 'salt2_template_1.dat' )
    outlines = []

    # define wavelength arrays. The same wavelength arrays are used
    # for every phase
    wavelist_old = temp1.wave[0]
    wavestep = wavelist_old[1] - wavelist_old[0]
    wavemin = wavelist_old[0]
    wavejoinend = wavelist_old[-1]
    wavelist_join = np.arange(wavejoinstart, wavejoinend+wavestep, wavestep)
    wavelist_opt = np.arange(4000, 8000, wavestep)
    wavelist_ir = np.arange(wavejoinstart, wavemax, wavestep)
    wavelist_all = np.arange(wavemin, wavemax, wavestep)

    # indices to pick out the join sections for the Old and new M1 models
    ijoinold = np.where((wavelist_old >= wavejoinstart) &
                        (wavelist_old <= wavejoinend))[0]
    ijoinnew = np.where((wavelist_ir >= wavejoinstart) &
                        (wavelist_ir <= wavejoinend))[0]

    assert (np.all(wavelist_all==temp0ext.wave[0]),
            "New M1 wavelength array must match extended M0"
            " wavelength array exactly.")

    assert (np.all(wavelist_all == np.append(wavelist_old, wavelist_ir)),
            "New M1 wavelength array must match the join of old (optical)"
            " and new (IR) wavelength arrays exactly.")

    # define weight functions to be applied when combining the new
    #  M1 curve with the old M1 curve.  It increases linearly from 0
    # at the start of the join window (8500 A) to 1 at the end of
    # the join window (9200 A), which is the last wavelength for the
    # old optical SALT2 model
    whtslope = 1.0 / (wavejoinend - wavejoinstart)
    whtintercept = -whtslope * wavejoinstart
    newM1weight = whtslope * wavelist_join + whtintercept
    oldM1weight = 1 - newM1weight

    for iphase1 in range(len(temp1.phase)): # iphaselist:
        thisphase = temp1.phase[iphase1]

        # get the SALT2 template0 and template1 data for this day
        # and define interpolating functions
        iphase0 = np.argmin(np.abs(temp0ext.phase-thisphase))
        assert iphase0 == iphase1

        M0func = scinterp.interp1d(
            temp0ext.wave[iphase0], temp0ext.value[iphase0],
            bounds_error=False, fill_value=0)
        M1func = scinterp.interp1d(
            temp1.wave[iphase1], temp1.value[iphase1],
            bounds_error=False, fill_value=0)

        print( 'solving for IR tail of template_1 for day : %i'%thisphase )

        M1extlist = []
        for name in __LOWZLIST__:
            lowzsn = modeldict[name.lower()]

            if lowzsn.mintime()>thisphase: continue
            if lowzsn.maxtime()<thisphase: continue
            if lowzsn.salt2c<cmin: continue
            if lowzsn.salt2c>cmax: continue
            if lowzsn.salt2x1<x1min: continue
            if lowzsn.salt2x1>x1max: continue

            fluxlowz_opt = lowzsn.flux(thisphase, wavelist_opt)
            if np.sum(fluxlowz_opt)==0: continue

            # Solve for x0 as a function of lambda (should be ~constant!)
            # over optical wavelengths. Then define a scalar x0 as the
            # median value over the  4000-8000 angstrom range
            x0 = np.median((fluxlowz_opt /
                            (M0func(wavelist_opt) +
                             lowzsn.salt2x1 * M1func(wavelist_opt))))

            # solve for the M1 array over the NIR wavelength range (8500+ A)
            fluxlowz_ir = lowzsn.flux(thisphase, wavelist_ir)
            M1ext = (1/lowzsn.salt2x1) * ((fluxlowz_ir/x0) - M0func(wavelist_ir))
            M1extlist.append(M1ext)

        if len(M1extlist)<3:
            print("only %i templates for phase = %.1f." % (
                len(M1extlist), thisphase))
            print("Using M1(lambda)=0 for all IR wavelengths")
            newM1median = np.zeros(len(wavelist_all))
        else:
            newM1median = np.median(M1extlist, axis=0)

        # apply the predefined weight functions to combine the new M1
        # curve with the old M1 curve in the "join window"
        # and then stitch together the old M1 curve up to the join
        # wavelength, the joining curve through the join window, and the
        # new M1 curve beyond that into the IR wavelengths.
        newM1joinvalues = (oldM1weight * temp1.value[iphase1][ijoinold] +
                           newM1weight * newM1median[ijoinnew])
        newM1values = np.append(
            np.append(temp1.value[iphase1][:ijoinold[0]], newM1joinvalues),
            newM1median[ijoinnew[-1]+1:])

        if showphase0plots and thisphase==0:
            fig1 = pl.figure(1)
            fig1.clf()
            ax1 = fig1.gca()
            for M1ext in M1extlist:
                ax1.plot(wavelist_ir, M1ext, lw=2, alpha=0.3, color='c')
            ax1.plot(wavelist_ir, newM1median, lw=2.5, color='0.5')
            ax1.plot(wavelist_all, newM1values, lw=1, color='k')
            ax1.plot(temp1.wave[0], M1func(temp1.wave[0]), color='r', lw=2.5)

        # append to the list of output data lines
        for j in range(len(wavelist_all)) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    thisphase, wavelist_all[j], newM1values[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp1fileOUT, 'w' )
    fout.writelines( outlines )
    fout.close()


def plot_extended_template1(phaselist=[-15,-5,0,5,25]):
    fig = pl.gcf()
    fig.clf()
    oldtemp1 = TimeSeriesGrid('salt2-4/salt2_template_1.dat')
    newtemp1 = TimeSeriesGrid('salt2ir/salt2_template_1.dat')
    iax = 0
    for phase in phaselist:
        iax+=1
        ax = fig.add_subplot(len(phaselist),1,iax)
        oldtemp1.plot_at_phase(phase=phase, color='0.5', ls='-', lw=2.5,
                               label='SALT2-4')
        newtemp1.plot_at_phase(phase=phase, color='m', ls='-', lw=1,
                               label='SALT2-IR')
        ax.text(0.5,0.9, 'phase = % i'%int(phase), ha='left', va='top',
                fontsize='large', transform=ax.transAxes)
        if iax==1:
            pl.legend(loc='upper right')
        if iax<len(phaselist):
            pl.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Wavelength ($\AA$)')
        if iax==int((len(phaselist)+1)/2.):
            ax.set_ylabel('SALT2 M1 model component value')
    fig.subplots_adjust(hspace=0, left=0.1, bottom=0.12, right=0.95)
    pl.draw()




def extendSALT2_flatline(
        modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
        salt2subdir='salt2-4', salt2irsubdir='salt2ir',
        showplots = False):
    """ extrapolate the *lc* and *spec* .dat files for SALT2
     using a flatline extension of the red tail to 2.5 microns
     """
    salt2modeldir = os.path.join(modeldir, salt2subdir)
    salt2irmodeldir = os.path.join(modeldir, salt2irsubdir)

    filelist = ['salt2_lc_dispersion_scaling.dat',
                'salt2_lc_relative_covariance_01.dat',
                'salt2_lc_relative_variance_0.dat',
                'salt2_lc_relative_variance_1.dat',
                'salt2_spec_covariance_01.dat',
                'salt2_spec_variance_0.dat',
                'salt2_spec_variance_1.dat']

    if showplots:
        fig = pl.gcf()
        fig.clf()
        iax=0

    for filename in filelist:
        infile = os.path.join(salt2modeldir, filename)
        outfile = os.path.join(salt2irmodeldir, filename)
        timeseries = TimeSeriesGrid(infile)
        if showplots:
            iax+=1
            ax = fig.add_subplot(7,1,iax)
            timeseries.plot_at_phase(0, color='k', lw=2.5, ls='-', marker=' ')
        timeseries.extrapolate_flatline(outfilename=outfile,
                                        maxwave=25000, showplots=False)
        if showplots:
            timeseriesNew = TimeSeriesGrid(outfile)
            timeseriesNew.plot_at_phase(0, color='r', lw=1,
                                        ls='--', marker=' ')
            ax.text(0.95, 0.05, filename, transform=ax.transAxes,
                    fontsize='large', ha='right', va='bottom')

    outinfo = os.path.join(salt2irmodeldir, 'SALT2.INFO')
    fout = open(outinfo, 'w')
    print >> fout, """
# open rest-lambda range WAAAY beyond nominal 2900-7000 A range.
RESTLAMBDA_RANGE:  2000. 25000.
COLORLAW_VERSION: 1
COLORCOR_PARAMS: 2800 7000 4 -0.537186 0.894515 -0.513865 0.0891927
COLOR_OFFSET:  0.0

MAG_OFFSET: 0.27
SEDFLUX_INTERP_OPT: 1  # 1=>linear,    2=>spline
ERRMAP_INTERP_OPT:  1  # 0=snake off;  1=>linear  2=>spline
ERRMAP_KCOR_OPT:    1  # 1/0 => on/off

MAGERR_FLOOR:   0.005            # don;t allow smaller error than this
MAGERR_LAMOBS:  0.1  2000  4000  # magerr minlam maxlam
MAGERR_LAMREST: 0.1   100   200  # magerr minlam maxlam
"""


def salt2_colorlaw(wave, params,
                   colorlaw_range=[2800,7000]):
    """Return the  extinction in magnitudes as a function of wavelength,
    for c=1. This is the version 1 extinction law used in SALT2 2.0
    (SALT2-2-0).

    Notes
    -----
    From SALT2 code comments:

    if(l_B<=l<=l_R):
        ext = exp(color * constant *
                  (alpha*l + params(0)*l^2 + params(1)*l^3 + ... ))
            = exp(color * constant * P(l))

        where alpha = 1 - params(0) - params(1) - ...

    if (l > l_R):
        ext = exp(color * constant * (P(l_R) + P'(l_R) * (l-l_R)))
    if (l < l_B):
        ext = exp(color * constant * (P(l_B) + P'(l_B) * (l-l_B)))
    """
    v_minus_b = _V_WAVELENGTH - _B_WAVELENGTH
    l = (wave - _B_WAVELENGTH) / v_minus_b
    l_lo = (colorlaw_range[0] - _B_WAVELENGTH) / v_minus_b
    l_hi = (colorlaw_range[1] - _B_WAVELENGTH) / v_minus_b

    alpha = 1. - sum(params)
    coeffs = [0., alpha]
    coeffs.extend(params)
    coeffs = np.array(coeffs)
    prime_coeffs = (np.arange(len(coeffs)) * coeffs)[1:]

    extinction = np.empty_like(wave)

    # Blue side
    idx_lo = l < l_lo
    p_lo = np.polyval(np.flipud(coeffs), l_lo)
    pprime_lo = np.polyval(np.flipud(prime_coeffs), l_lo)
    extinction[idx_lo] = p_lo + pprime_lo * (l[idx_lo] - l_lo)

    # Red side
    idx_hi = l > l_hi
    p_hi = np.polyval(np.flipud(coeffs), l_hi)
    pprime_hi = np.polyval(np.flipud(prime_coeffs), l_hi)
    extinction[idx_hi] = p_hi + pprime_hi * (l[idx_hi] - l_hi)

    # In between
    idx_between = np.invert(idx_lo | idx_hi)
    extinction[idx_between] = np.polyval(np.flipud(coeffs), l[idx_between])

    return -extinction


def fit_salt2colorlaw_to_ccm(c=0.1, Rv=3.1,
                             wmin=2000, wjoin=6750, wmax=25000,
                             salt2_colorlaw_range = [2800, 7000],
                             modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                             salt2subdir='salt2-4', salt2irsubdir='salt2ir',
                             uselowzfit=False, showfit=False):
    """ Find the SALT2 color law parameters that will cause the color law to
    approximately match the Cardelli+ 1989 extinction law (as extended
    by O'Donnell 1994) for IR wavelengths.
    :param c: SALT2 color parameter (~E(B-V))
    :return:
    """
    if uselowzfit:
        # Use the function that converts from MLCS A_V to SALT2 c
        # to find the value of E(B-V) for this value of c
        d2x1, av2c = get_mlcs_to_salt2_parameter_conversion_functions()
        avrange = np.arange(0,5,0.01)
        crange = av2c(avrange)
        ithisc = np.argmin(np.abs(crange-c))
        Av = avrange[ithisc]
        EBV = avrange/Rv
    else:
        # To first order (and for Rv=3.1) we can use: E(B-V) = c
        EBV = c

    # define a wavelength grid that extends from the red end of the SALT2
    # color law range out to
    # wave = np.append(np.arange(wmin, wjoin, 10.0), np.arange(wjoin,wmax,100.0))
    wave = np.arange(wmin, wmax, 10.0)
    alambda = lambda w: ccm_extinction(w, EBV, Rv)
    aB = ccm_extinction(_B_WAVELENGTH, EBV, Rv)

    def fitting_function(w, param0, param1, param2, param3):
        params = [param0, param1, param2, param3]
        return c * salt2_colorlaw(
            w, params, colorlaw_range=salt2_colorlaw_range)

    paramA = [-0.504294, 0.787691, -0.461715, 0.0815619]
    extinctioncurvetofit = np.where(wave < wjoin,
                                    c * salt2_colorlaw(wave, paramA),
                                    alambda(wave) - aB)
    fitres = scopt.curve_fit( fitting_function, wave,
                              extinctioncurvetofit, paramA)
    paramfit = fitres[0]

    if showfit:
        pl.clf()
        salt2colorlaw0 = salt2_colorlaw(wave, params=paramA[:4],
                                        colorlaw_range=[2800,7000])
        salt2colorlawfit = salt2_colorlaw(wave, params=paramfit[:4],
                                          colorlaw_range=salt2_colorlaw_range)

        pl.plot(wave, c * salt2colorlaw0, 'k-', label='SALT2-4 color law')
        pl.plot(wave, c * salt2colorlawfit, 'r--', label='SALT2ir color law')
        pl.plot(wave, alambda(wave) - aB, 'b-.', label='Dust with Rv=3.1')
        ax = pl.gca()
        ax.set_xlim(3000,20000)
        ax.set_ylim(-0.5,0.1)

        ax.text(0.95, 0.55,
                'SALT2-4 colorlaw range = [%i, %i]'%tuple([2800,7000]),
                ha='right', va='bottom', transform=ax.transAxes, color='k')
        ax.text(0.95, 0.45,
                ('SALT2-4 colorlaw parameters =\n'
                '[%.3f, %.3f, %.3f, %.3f]'%tuple(paramA)),
                ha='right', va='bottom', transform=ax.transAxes, color='k')

        ax.text(0.05, 0.15,
                '$\lambda_{\\rm join}$= %i'%wjoin,
                ha='left', va='bottom', transform=ax.transAxes, color='r')
        ax.text(0.05, 0.1,
                'SALT2ir colorlaw range = [%i, %i]'%tuple(salt2_colorlaw_range),
                ha='left', va='bottom', transform=ax.transAxes, color='r')
        ax.text(0.05, 0.05,
                ('SALT2ir colorlaw parameters ='
                '[%.3f, %.3f, %.3f, %.3f]'%tuple(paramfit)),
                ha='left', va='bottom', transform=ax.transAxes, color='r')
        ax.legend(loc='upper right')
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('A$_{\lambda}$ - A$_{B}$  or  c$\\times$ CL($\lambda$)')
        pl.draw()

    # write out the revised color law as a file
    outfile= os.path.join(modeldir,
                          salt2irsubdir + '/salt2_color_correction.dat')
    fout = open(outfile, 'w')
    print >> fout, '%i' % len(paramfit)
    for param in paramfit:
        print >> fout, '%.8f' % param
    print >> fout, 'Salt2ExtinctionLaw.version 1'
    print >> fout, 'Salt2ExtinctionLaw.min_lambda %i' % salt2_colorlaw_range[0]
    print >> fout, 'Salt2ExtinctionLaw.max_lambda %i' % salt2_colorlaw_range[1]
    fout.close()
    print "Updated SALT2 color law parameters written to %s" % outfile
    return


