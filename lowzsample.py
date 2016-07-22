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

_B_WAVELENGTH = 4302.57
_V_WAVELENGTH = 5428.55

__THISFILE__ = sys.argv[0]
if 'ipython' in __THISFILE__ :
    __THISFILE__ = __file__
__THISPATH__ = os.path.abspath(os.path.dirname(__THISFILE__))
__THISDIR__ = os.path.abspath(os.path.dirname(__THISFILE__))


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
                 metadatadir='/Users/rodney/Dropbox/WFIRST/SALT2IR/lowzIa'):
        """
        :return: a light curve object with a table of light curve data stored
        in the '.lcdata' property
        """
        self.name = name.lower()
        self.snid = name.lstrip('sn')
        self.propername = 'SN' + self.snid
        self.model_fixedx1c = None
        self.model_fitted = None

        if datadir.endswith('.osc'):
            # Loading Open SN Catalog data in JSON format
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
        for survey in ['CSP','CfA','ESO','LCO','LOSS',]:
            metadatafilename = os.path.join(metadatadir,
                                            name + '_%s.dat' % survey)
            if os.path.exists(metadatafilename):
                self.metadatafilename = metadatafilename
                self.get_metadata_from_file()
                break

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

    def reformat_data(self, segregatecsp=False):
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
        return


    def fitmodel(self, modeldir='/Users/rodney/Dropbox/WFIRST/SALT2IR',
                 salt2subdir='salt2ir', fitbands=None):
        """ fit a new model to the data
        :param modeldir:
        :return:
        """
        salt2irmodeldir = os.path.join(modeldir, salt2subdir)
        salt2irsource = sncosmo.models.SALT2Source(
            modeldir=salt2irmodeldir, name=salt2subdir)
        salt2irmodel = sncosmo.Model(source=salt2irsource)

        # limit the fit to specific bands if requested
        if fitbands is None:
            datatofit = self.lcdata
        else:
            if isinstance(fitbands, basestring):
                fitbands = [fitbands]
            itofit = np.array([], dtype=int)
            for fitband in fitbands:
                ithisband = np.where(self.band == fitband)[0]
                itofit = np.append(itofit, ithisband)
            datatofit = self.lcdata[itofit]

        if 'salt2x1' not in self.__dict__:
            print("No SALT2 fitres data available for this SN."
                  " No predefined model")
            self.model_fixedx1c = None
            salt2irmodel.set(z=self.z, x1=0, c=0.1)
        else:
            # evaluate the SALT2 model for the given redshift, x1, c
            salt2irmodel.set(z=self.z, t0=self.TBmax,
                             x1=self.salt2x1, c=self.salt2c)
            salt2irmodel.set_source_peakabsmag(-19.3, 'bessellb','AB')

            # fit the model without allowing x1 and c to vary
            res, model_fixedx1c = sncosmo.fit_lc(
                datatofit, salt2irmodel, ['t0', 'x0'], bounds={})
            self.model_fixedx1c = model_fixedx1c

        # fit the model allowing all parameters to vary
        res, model_fitted = sncosmo.fit_lc(
            datatofit, salt2irmodel, ['t0', 'x0', 'x1', 'c'],
            bounds={'x1':[-3,3], 'c':[-0.5,0.5]})
        self.model_fitted = model_fitted


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


def mk_lightcurve_fig(lowzname, fitbands='all'):
    """ Make a figure showing the observed light curve of the given low-z
    Type Ia SN, compared to the light curve from the SALT2ir model for the
    (best-fit) x1, c values appropriate to that SN.
    """

    if lowzname not in __LOWZNAMELIST__:
        print("I don't know about %s" % lowzname)

    # read in the observed light curve data
    lowzlc = LowzLightCurve(lowzname)

    bandlist = np.unique(lowzlc.band)
    if isinstance(fitbands, list):
        bandlisttofit = fitbands
        fitbands = ','.join(fitbands)
    elif fitbands == 'sdss':
        bandlisttofit = ['sdssu', 'sdssg', 'sdssr', 'sdssi']
    elif fitbands == 'bessell':
        bandlisttofit = ['bessellb', 'bessellv', 'bessellr', 'besselli']
    elif fitbands == 'csp':
        bandlisttofit = ['cspu','cspb','cspv3009','cspr','cspi',
                         'cspyd','cspjd','csphd','cspk']
    elif fitbands == 'ir':
        bandlisttofit = ['cspyd', 'cspjd', 'csphd', 'cspk']
    elif fitbands.startswith('opt'):
        bandlisttofit = ['cspb', 'cspv3009', 'cspr', 'cspi',
                         'bessellb', 'bessellv', 'besselr', 'besselli',
                         'sdssu', 'sdssg', 'sdssr', 'sdssi']
    elif fitbands=='all':
        bandlisttofit = bandlist
    else:
        raise exceptions.RuntimeError(
            "fitbands must be from ['sdss','csp','ir','opt','all']")

    fitbandlist = [band for band in bandlisttofit
                   if band in bandlist]
    print "Fitting to bands %s" % ','.join(fitbandlist)

    # read in the metadata, including salt2 fit parameters
    lowzlc.fitmodel(fitbands=fitbandlist)
    lowzlc.plot_lightcurve(
        fixedx1c=False, bands=bandlist,
        figtext='%s\nfit to %s bands'%(lowzname, fitbands))
    pl.savefig('/Users/rodney/Desktop/%s_%s_salt2irfit.png' %
               (lowzname, fitbands))



