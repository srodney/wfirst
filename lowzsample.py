"""
Read in data for the low-z SN sample.
Make some figures for testing the accuracy of the extrapolated SALT2ir model.
"""

import numpy as np
from matplotlib import pyplot as pl
import os
import sys
from astropy.io import ascii
from astropy.table import Table
import sncosmo
from scipy import integrate as scint, interpolate as scinterp, optimize as scopt
import exceptions
import json
import cPickle

_B_WAVELENGTH = 4302.57
_V_WAVELENGTH = 5428.55

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__ :
    __THISFILE__ = __file__
__THISPATH__ = os.path.abspath(os.path.dirname(__THISFILE__))
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

__LOWZNAMELIST__ = np.array([
    'sn1998bu', 'sn1999cl', 'sn1999cp', 'sn1999ee', 'sn1999ek',
    'sn1999gp', 'sn2000E', 'sn2000bh', 'sn2000ca', 'sn2000ce',
    'sn2001ba', 'sn2001bt', 'sn2001cn', 'sn2001cz', 'sn2001el',
    'sn2002dj', 'sn2002fk', 'sn2003cg', 'sn2003du', 'sn2003hv',
    'sn2004S', 'sn2004ef', 'sn2004eo', 'sn2004ey', 'sn2004gs',
    'sn2005A', 'sn2005M', 'sn2005am', 'sn2005bo', 'sn2005cf',
    'sn2005el', 'sn2005eq', 'sn2005eu', 'sn2005hc', 'sn2005hj',
    'sn2005iq', 'sn2005kc', 'sn2005ki', 'sn2005ls', 'sn2005na',
    'sn2006D', 'sn2006X', 'sn2006ac', 'sn2006ax', 'sn2006bh',
    'sn2006cp', 'sn2006dd', 'sn2006ej', 'sn2006et', 'sn2006ev',
    'sn2006gj', 'sn2006gr', 'sn2006hb', 'sn2006hx', 'sn2006is',
    'sn2006kf', 'sn2006le', 'sn2006lf', 'sn2006ob', 'sn2006os',
    'sn2007A', 'sn2007S', 'sn2007af', 'sn2007as', 'sn2007bc',
    'sn2007bd', 'sn2007bm', 'sn2007ca', 'sn2007co', 'sn2007cq',
    'sn2007hx', 'sn2007jg', 'sn2007le', 'sn2007nq', 'sn2007on',
    'sn2007qe', 'sn2007sr', 'sn2008C', 'sn2008R', 'sn2008Z', 'sn2008bc',
    'sn2008bq', 'sn2008fp', 'sn2008gb', 'sn2008gl', 'sn2008gp',
    'sn2008hm', 'sn2008hs', 'sn2008hv', 'sn2008ia', 'sn2009D',
    'sn2009ad', 'sn2009al', 'sn2009an', 'sn2009bv', 'sn2009do',
    'sn2009ds', 'sn2009fv', 'sn2009jr', 'sn2009kk', 'sn2009kq',
    'sn2009le', 'sn2009lf', 'sn2009na', 'sn2010Y', 'sn2010ag',
    'sn2010ai', 'sn2010dw', 'sn2010ju', 'sn2010kg', 'sn2011B',
    'sn2011K', 'sn2011ae', 'sn2011ao', 'sn2011by', 'sn2011df',
    'snf20080514-002', 'snf20080522-000'], dtype='|S15')



class LowzLightCurve(Table):
    def __init__(self, name,
                 datadir='/Users/rodney/Dropbox/WFIRST/SALT2IR/lightcurve_data',
                 metadatadir='/Users/rodney/Dropbox/WFIRST/SALT2IR/lowzIa',
                 magerrfloor=0.01):
        """
        :param magerrfloor: set a min magnitude error to be applied to all
        bands
        :return: a light curve object with a table of light curve data stored
        in the '.lcdata' property

        """
        self.name = name.lower()
        self.snid = name.lstrip('sn')
        self.propername = 'SN' + self.snid
        self.model_fixedx1c = None
        self.model_fitted = None
        self.fitbands = None
        self.magerrfloor = magerrfloor

        if datadir.endswith('.osc'):
            # Loading Open SN Catalog data in JSON formatas
            jsondatafilename = os.path.join(
                datadir, self.propername + '.json')
            if os.path.exists(jsondatafilename):
                self.jsondatfile = jsondatafilename
                self.rd_jsonfile()
            else:
                import pdb; pdb.set_trace()
        else:
            lcdatafilename = os.path.join(datadir, name + '_OIR.mag.dat')
            if os.path.exists(lcdatafilename):
                self.lcdatfile = lcdatafilename
                self.rd_datfile()
            else:
                import pdb; pdb.set_trace()

        if 'mjd' in self.__dict__:
            self.zpt = np.ones(len(self.magsys)) * 10
            self.flux = 10**(-0.4*(self.mag-self.zpt))
            self.fluxerr = 0.92103 * self.magerr * self.flux
            self.lcdata = Table({
                'mjd': self.mjd, 'mag': self.mag, 'magerr': self.magerr,
                'flux':self.flux, 'fluxerr':self.fluxerr,
                'band': self.band, 'magsys': self.magsys, 'zpt':self.zpt})

        self.metadatafilename = None
        for survey in ['CSP','CfA','ESO','LCO','LOSS','CTIO']:
            metadatafilename = os.path.join(metadatadir,
                                            name + '_%s.dat' % survey)
            if os.path.exists(metadatafilename):
                self.metadatafilename = metadatafilename
                self.get_metadata_from_file()
                break
        self.datatofit = self.lcdata


    def get_metadata_from_file(self):
        # TODO: read in the metadata file and get the salt2 fit parameters
        fin = open(self.metadatafilename,'r')
        datalines = fin.readlines()
        fin.close()
        for dline in datalines:
            if not dline.startswith('#'):
                break
            key = dline.lstrip('#').split()[0]
            val = dline.lstrip('#').split()[-1]
            if is_number(val):
                val = float(val)
            self.__dict__[key] = val

    def rd_datfile(self):
        """ read in the data from one of the low-z NIR sample data files
        provided by Arturo Avelino.
        """
        lcdata = Table.read(self.lcdatfile,
                            format='ascii.basic',
                            names=['band','mjd','mag','magerr','instr'])
        self.mjd = lcdata['mjd']
        self.mag = lcdata['mag']
        self.magerr = lcdata['magerr']
        self.instrument = lcdata['instr']
        self.band = lcdata['band']
        self.reformat_data()


    def rd_jsonfile(self):
        """ Read in the light curve data from a .json file provided by the
        open SN catalog.
        """
        fin = open(self.jsondatfile, 'r')
        jsondata = json.load(fin)
        fin.close()

        photdata = jsondata[self.propername]['photometry']
        imagdata = np.array([i for i in range(len(photdata))
                             if 'band' in photdata[i]
                             and 'time' in photdata[i]
                             and 'magnitude' in photdata[i]
                             and 'e_magnitude' in photdata[i]
                             and 'system' in photdata[i]
                             ])
        self.mjd = np.array([photdata[i]['time'] for i in imagdata],
                            dtype=float)
        self.mag = np.array([photdata[i]['magnitude'] for i in imagdata],
                            dtype=float)
        self.magerr = np.array([photdata[i]['e_magnitude'] for i in imagdata],
                               dtype=float)
        self.band = np.array([str(photdata[i]['band']) for i in imagdata],
                             dtype=str)
        self.instrument = np.array([photdata[i]['system'] for i in imagdata],
                                   dtype=basestring)
        self.reformat_data()

    def reformat_data(self, segregatecsp=False, mjdtrim=False):
        """ Reformat the data that has been read in from a data file.
        * Fix the passband names to conform to the sncosmo names
        * assign to a photometric system (Vega or AB) based on band name

        :param segregatecsp: for any band listed in the dat file as a CSP
                  band, assign it to the corresponding CSP filter from sncosmo
                  instead of using the generic Bessell band of the same name.
        """
        newbandnames, magsyslist = [], []
        for i in range(len(self.band)):
            bandi = self.band[i]
            if segregatecsp and bandi.endswith('CSP'):
                 bandname = 'csp' + bandi.split('_')[0].lower()
                 if bandi[0] in 'ugrizy':
                     magsys = 'AB'
                 elif bandi[0] in 'UBVRIYJHK':
                     magsys = 'Vega'
                 else:
                     import pdb; pdb.set_trace()
                 if bandname[-1] in ['y','j','h']:
                     if 'swo' in self.instrument[i].lower():
                         bandname += 's'
                     else:
                         bandname += 'd'
                 elif bandname[-1] == 'v':
                     bandname += '3009'
            # elif 'lco' in self.instrument[i].lower() and bandi[0] in 'YJHK':
            #     bandname = 'csp' + bandi[0].lower()
            #     if bandi[0] != 'K':
            #         bandname += 'd'
            #     magsys = 'AB'
            elif bandi in ['U','UX']:
                bandname = 'bessellux'
                magsys = 'Vega'
            elif bandi[0] in 'BVRI':
                bandname = 'bessell' + bandi[0].lower()
                magsys = 'Vega'
            elif bandi[0] in 'ugriz':
                bandname = 'sdss' + bandi[0]
                magsys = 'AB'
            elif bandi[0] in 'YJHK':
                bandname = 'csp' + bandi[0].lower()
                if bandi[0] != 'K':
                    bandname += 'd'
                magsys = 'Vega'
            else:
                print 'unrecognized band name %s' % bandi
                bandname = bandi
                import pdb; pdb.set_trace()
            newbandnames.append(bandname)
            magsyslist.append(magsys)
        self.band = np.array(newbandnames)
        self.magsys = np.array(magsyslist)

        if self.magerrfloor:
            ismallerr = np.where(self.magerr<self.magerrfloor)[0]
            igooderr = np.where(self.magerr>self.magerrfloor)[0]
            for ise in ismallerr:
                ithisband = np.where(self.band == self.band[ise])[0]
                igood = np.array([i for i in igooderr if i in ithisband])
                if len(igood)>0:
                    self.magerr[ise] = np.median(self.magerr[igood])
                else:
                    self.magerr[ise] = self.magerrfloor
        return


    def fitmodel(self, modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                 salt2subdir='salt2ir', useknownx1c=False):
        """ fit a new model to the data
        :param modeldir:
        :return:
        """
        salt2modeldir = os.path.join(modeldir, salt2subdir)
        salt2source = sncosmo.models.SALT2Source(
            modeldir=salt2modeldir, name=salt2subdir)
        salt2model = sncosmo.Model(source=salt2source)

        if useknownx1c:
            if 'salt2x1' not in self.__dict__:
                print("No SALT2 fitres data available for this SN."
                      " No predefined model")
                self.model_fitted = None
                salt2model.set(z=self.z, x1=0, c=0.1)
            else:
                # evaluate the SALT2 model for the given redshift, x1, c
                salt2model.set(z=self.z, t0=self.TBmax,
                                 x1=self.salt2x1, c=self.salt2c)
                salt2model.set_source_peakabsmag(-19.3, 'bessellb','AB')

                # fit the model without allowing x1 and c to vary
                res, model_fixedx1c = sncosmo.fit_lc(
                    self.datatofit, salt2model, ['t0', 'x0'], bounds={})
                self.model_fitted = model_fixedx1c
        else:
            # fit the model allowing all parameters to vary
            res, model_fitted = sncosmo.fit_lc(
                self.datatofit, salt2model, ['t0', 'x0', 'x1', 'c'],
                bounds={'x1':[-5,5], 'c':[-0.5,1.5]})
            self.model_fitted = model_fitted


    def trim_data_for_fitting(self, fitbands=None, mjdtrim=True):
        """
        :param fitbands: list of bands (or name of a set of bands) to be used
        in the fitting.
        :param mjdtrim: If True, trim the data to the rest-frame time
        range of [-10,+45]
        :return:
        """
        if fitbands is not None:
            print "SN %s : limiting fitting data to bands %s" % (
                self.name, ','.join(fitbands))
        if mjdtrim:
            print "SN %s : limiting fitting data to -10<t<+45" % self.name

        #TODO: clean up the MJD and band data trimming !
        if mjdtrim:

            # find the date of peak brightness by taking the median of the
            # peak mag epoch from all bands
            mjdpklist = []
            for band in np.unique(self.band):
                bandwave = sncosmo.get_bandpass(band).wave_eff
                if bandwave > 7000: continue
                if bandwave < 3000: continue
                ithisband = np.where((self.band==band) &
                                     (self.magerr>0) &
                                     (self.mag/self.magerr > 5))[0]
                ipeak = np.argmin(self.mag[ithisband])
                mjdpklist.append(self.mjd[ipeak])
            if len(mjdpklist):
                mjdpk = np.mean(mjdpklist)
            else:
                mjdpk = self.mjd[np.argmin(self.mag)]
            # limit the fit to epochs prior to +45 days
            trest = (self.mjd-mjdpk)/(1+self.z)
            iinside = np.where((trest>-10) & (trest<45))[0]
            ioutside = np.where((trest<-10) | (trest>45))[0]
            self.lcdata.remove_rows(ioutside)
            self.mjd = self.mjd[iinside]
            self.band = self.band[iinside]
            self.mag = self.mag[iinside]
            self.magerr = self.magerr[iinside]
            self.flux = self.flux[iinside]
            self.fluxerr = self.fluxerr[iinside]
            self.magsys = self.magsys[iinside]
            self.zpt = self.zpt[iinside]

        # else:
        #     iinside = np.arange(len(self.mjd))


        # and limit the fit to specific bands if requested
        if fitbands is None:
            self.fitbands = np.unique(self.band)
            ibandok = np.arange(len(self.band))
        else:
            self.fitbands = fitbands
            if isinstance(fitbands, basestring):
                fitbands = [fitbands]
            ibandok = np.array([], dtype=int)
            for fitband in fitbands:
                ithisband = np.where(self.band == fitband)[0]
                ibandok = np.append(ibandok, ithisband)

        # itofit = np.array([i for i in iinside if i in ibandok])
        self.datatofit = self.lcdata[ibandok]


    def plot_lightcurve(self, fixedx1c=False, **kwargs):
        """ Make a figure showing the observed light curve of the given low-z
        Type Ia SN, compared to the light curve from the SALT2ir model for the
        (predefined) x1, c values appropriate to that SN.
        """
        if self.model_fitted is None:
            self.fitmodel()
        if fixedx1c:
            model = self.model_fixedx1c
        else:
            model = self.model_fitted
        sncosmo.plot_lc(self.lcdata, model=model, **kwargs)


def mk_lightcurve_fig(lowzsn, fitbandset='all', usesalt2ir=True, clobber=False,
                      savefig=False, **plotlckwargs):
    """ Make a figure showing the observed light curve of the given low-z
    Type Ia SN, compared to the light curve from the SALT2ir model for the
    (best-fit) x1, c values appropriate to that SN.
    """

    if isinstance(lowzsn, basestring):
        if lowzsn not in __LOWZNAMELIST__:
            print("I don't know about %s" % lowzsn)
        # read in the observed light curve data
        lowzlc = LowzLightCurve(lowzsn)
    elif isinstance(lowzsn, Table):
        lowzlc = lowzsn
    else:
        raise exceptions.RuntimeError(
            'Give me a string or a LowzLightCurve object')

    bandlist = np.unique(lowzlc.band)
    if lowzlc.model_fitted is None or clobber:
        if isinstance(fitbandset, list):
            bandlisttofit = fitbandset
        elif fitbandset == 'sdss':
            bandlisttofit = ['sdssu', 'sdssg', 'sdssr', 'sdssi']
        elif fitbandset == 'bessell':
            bandlisttofit = ['bessellb', 'bessellv', 'bessellr', 'besselli']
        elif fitbandset == 'csp':
            bandlisttofit = ['cspu','cspb','cspv3009','cspr','cspi',
                             'cspyd','cspjd','csphd','cspk']
        elif fitbandset == 'ir':
            bandlisttofit = ['cspyd', 'cspjd', 'csphd', 'cspk']
        elif fitbandset in ['opt', 'optical']:
            bandlisttofit = ['cspb', 'cspv3009', 'cspr', 'cspi',
                             'bessellb', 'bessellv', 'besselr', 'besselli',
                             'sdssu', 'sdssg', 'sdssr', 'sdssi']
        elif fitbandset.lower() in ['oir','optir','bvrjhk']:
            bandlisttofit = ['bessellb', 'bessellv', 'bessellr',
                             'sdssg','sdssr',
                             'cspjd', 'csphd', 'cspk']
        elif fitbandset== 'all':
            bandlisttofit = bandlist
        else:
            raise exceptions.RuntimeError(
                "fitbands must be from ['sdss','csp','ir','opt','all']")

        fitbandlist = [band for band in bandlisttofit
                       if band in bandlist]
        dofit=True
    else:
        fitbandlist = lowzlc.fitbands
        dofit=False

    bandliststr = ''
    iwaveorder = np.argsort([sncosmo.get_bandpass(bandname).wave_eff
                             for bandname in fitbandlist])
    for bandname in np.array(fitbandlist)[iwaveorder]:
        if bandname.startswith('bessell'):
            bandliststr += bandname.upper().rstrip('X')[7:]
        elif bandname.startswith('csp'):
            bandliststr += bandname.upper().rstrip('DS')[3:]
        elif bandname.startswith('sdss'):
            bandliststr += bandname[4:]
        else:
            bandliststr += bandname.upper()
    if savefig:
        fname = '/Users/rodney/Desktop/sn%s_%s_salt2irfit.png' % (
            lowzlc.name, bandliststr)
    else:
        fname = None

    if fname is not None and os.path.isfile(fname) and not clobber:
        print("%s exists. Not clobbering." % fname)
    if dofit:
        # read in the metadata, including salt2 fit parameters
        lowzlc.trim_data_for_fitting(mjdtrim=True, fitbands=fitbandlist)
        if usesalt2ir:
            salt2subdir='salt2ir'
        else:
            salt2subdir='salt2-4'
        lowzlc.fitmodel(salt2subdir=salt2subdir)

    plotlcdefaults = dict(
        figtext='SN %s\n%s fit to:\n%s'%(
            lowzlc.name, salt2subdir, bandliststr),
        bands=bandlist, cmap_lims=(2500,20000), zp=25.0, zpsys='ab',
        pulls=True, errors=None, ncol=3, figtextsize=1.0,
        model_label=salt2subdir, show_model_params=True,
        tighten_ylim=True, color=None, fname=fname)
    plotlcdefaults.update(**plotlckwargs)

    lowzlc.plot_lightcurve(fixedx1c=False, **plotlcdefaults)

    fig = pl.gcf()
    for ax in fig.axes:
        ax.set_xlim(-12, 47)
    return


def mk_all_lightcurve_fit_figs(
        snlist=None, usesalt2ir=True, fitbands='optir',
        outfilename='/Users/rodney/Desktop/lowzIa_salt2ir_fits.pdf'):
    """ Make a single output pdf file showing summary light curve fits
    for all the SNe in the low-z sample, using the salt2ir model.
    :param outfilename:
    :return:
    """
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(outfilename)
    if snlist is None:
        print "No list of fitted SNe provided, so fitting on the fly"
        snlist = __LOWZNAMELIST__

    pl.ioff()
    for sn in snlist:
        mk_lightcurve_fig(sn, savefig=True, clobber=False,
                          usesalt2ir=usesalt2ir, fitbandset=fitbands,
                          fname=pp, format='pdf')
    pp.close()
    pl.ion()

