# /usr/bin/env python
# 2017.03.10  S. Rodney
# Reading in host galaxy data from WFIRST simulations
# and computing an estimated exposure time for Subaru+PFS
# to get a redshift for that host.

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
import os
import exceptions
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

class WfirstSimData(Table):

    def __init__(self, fitsfilename, *args, **kwargs):
        super(Table, self).__init__(*args, **kwargs)

        hdulist = fits.open(fitsfilename)
        bindata = hdulist[1].data
        self.zsim = bindata['SIM_REDSHIFT_HOST']
        if 'HOSTGAL_MAG_H' in [col.name for col in bindata.columns]:
            self.mag = bindata['HOSTGAL_MAG_H']
        else:
            self.mag = bindata['HOSTGAL_MAG_J']

        self.matchdata = None
        self.imatch = None
        self.mag3D = None
        self.id3D = None
        self.z3D = None
        self.eazydata = {}


    def load_matchdata(self, matchcatfilename=None):
        """Load a 3DHST catalog to identify galaxies that match the
        properties of the SN host galaxies.
        """
        if matchcatfilename is None:
            matchcatfilename = ('3DHST/3dhst_master.phot.v4.1/'
                                '3dhst_master.phot.v4.1.cat.FITS')

        if self.matchdata is None:
            # load the 3dHST catalog
            self.matchdata = fits.getdata(matchcatfilename)

        f160 = self.matchdata['f_F160W']
        zspec = self.matchdata['z_spec']
        zphot = self.matchdata['z_peak']
        zbest = np.where(zspec>0, zspec, zphot)
        usephot = self.matchdata['use_phot']
        ivalid = np.where(((f160>0) & (zbest>0)) & (usephot==1) )[0]
        isort = np.argsort(zbest[ivalid])

        self.z3D = zbest[ivalid][isort]
        idgal = self.matchdata['id'][ivalid][isort].astype(int)
        field = self.matchdata['field'][ivalid][isort]
        self.id3D = np.array(['{}.{:04d}'.format(field[i], idgal[i])
                              for i in range(len(field))])
        self.mag3D = (-2.5 * np.log10(f160[ivalid]) + 25)[isort]
        return


    def get_matchlists(self, dz=0.05, dmag=0.2):
        """For each simulated host galaxy, get a list of 3DHST galaxy IDs for
        all galaxies in the 3DHST catalog that have a redshift within dz of
        the simulated z, and an H band mag within dmag of the simulated H band
        mag."""
        if self.matchdata is None:
            self.load_matchdata()
        # TODO: find the nearest 10 or 100 galaxies, instead of all within
        # a specified dz and dmag range.
        allmatch_indexlist = np.array(
            [np.where((self.z3D + dz > self.zsim[i]) &
                      (self.z3D - dz < self.zsim[i]) &
                      (self.mag3D + dmag > self.mag[i]) &
                      (self.mag3D - dz < self.mag[i]))[0]
             for i in range(len(self.zsim))])
        self.nmatch = np.array([len(allmatch_indexlist[i])
                                for i in range(len(self.zsim))])

        self.allmatchids = np.array([self.id3D[allmatch_indexlist[i]]
                                     for i in range(len(self.zsim))])

        self.matchid = np.array([np.random.choice(self.allmatchids[i])
                                 for i in range(len(self.zsim))])


    def load_sed_data(self):
        """ load all the EAZY simulated SED data at once"""
        for field in ['aegis', 'cosmos', 'goodsn', 'goodss', 'uds']:
            fitsfilename = glob(
                '3DHST/{0}_3dhst.*.eazypy.data.fits'.format(field))[0]
            self.eazydata[field] = EazyData(fitsfilename=fitsfilename)


    def simulate_seds(self):
        """Use Gabe Brammer's EAZY code to simulate the host gal SED """

        for isim in range(len(self.matchid)):
            fieldidx = self.matchid[isim]
            fieldstr, idxstr = fieldidx.split('.')
            field = fieldstr.lower().replace('-', '')
            thiseazydat = self.eazydata[field]
            simsedfilename = ('3DHST/sedsim.output/' +
                              'wfirst_simsed.{:s}.dat'.format(fieldidx))
            simulate_eazy_sed( fieldidx=fieldidx, eazydata=thiseazydat,
                               savetofile=simsedfilename)


class SubaruSpecSim(Table):
    """ a class for handling the output of the C.Hirata SubaruPFS ETC code
    """
    def __init__(self, etcoutfilename, *args, **kwargs):
        super(Table, self).__init__(*args, **kwargs)

        etcoutdata = ascii.read(etcoutfilename, format='basic',
                                names=['arm', 'pix', 'wave','snpix',
                                       'signal_exp', 'var0', 'var',
                                       'mAB', 'flux_conversion',
                                       'samplingfactor', 'skybg'])

        self.wave = etcoutdata['wave']
        self.signaltonoise = etcoutdata['snpix']


    def check_line_detection(self, snthresh=5):
        """Test whether an emission line is detected """


class EazyData(object):
    """ EAZY data from gabe brammer """
    # TODO : this should probably just inherit from a fits BinTableHDU class
    def __init__(self, fitsfilename):
        hdulist = fits.open(fitsfilename)
        self.namelist = [hdu.name for hdu in hdulist]
        for name in self.namelist:
            self.__dict__[name] = hdulist[name].data
        hdulist.close()

class EazySpecSim(Table):
    """ a class for handling the output of the G.Brammer spec simulator code
    """
    def __init__(self, specsimfile):
        specsimdata = ascii.read(specsimfile, format='basic',
                                names=['wave', 'flux'])
        self.wave = specsimdata['wave']
        self.flux = specsimdata['flux']
        self.waveunit = 'nm'
        self.fluxunit = 'magAB'

    def plot(self, *args, **kwargs):
        plt.plot(self.wave, self.flux, *args, **kwargs)
        ax = plt.gca()
        ax.set_xlabel('observed wavelength (nm)')
        ax.set_ylabel('mag (AB)')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()


def simulate_eazy_sed(fieldidx='GOODS-S.21740', eazydata=None,
                      returnfluxunit='AB', returnwaveunit='nm',
                      limitwaverange=True, savetofile=''):
    """
    Pull best-fit SED from eazy-py output files.

    NB: Requires the eazy-py package to apply the IGM absorption!
    (https://github.com/gbrammer/eazy-py)

    Optional Args:
    returnfluxunit: ['AB', 'flambda']
      (TODO: add Jy)
    returnwaveunit: ['A' or 'nm']
    limitwaverange: limit the output wavelengths to the range covered by PFS
    savetofile: filename for saving the output spectrum as a two-column ascii
        data file (suitable for use with the SubaruPFS ETC from C. Hirata.

    Returns
    -------
        templz   : observed-frame wavelength, Angstroms or  nm
        tempflux : flux density of best-fit template, erg/s/cm2/A or AB mag
    """
    fieldstr, idxstr = fieldidx.split('.')
    field = fieldstr.lower().replace('-','')
    idx = int(idxstr)

    if eazydata is None:
        fitsfilename = glob(
            '3DHST/{0}_3dhst.*.eazypy.data.fits'.format(field))[0]
        eazydata = EazyData(fitsfilename)

    match = eazydata.ID == idx
    if match.sum() == 0:
        print('ID {0} not found.'.format(idx))
        return None, None

    ix = np.arange(len(match))[match][0]
    z = eazydata.ZBEST[ix]

    # the input data units are Angstroms for wavelength
    # and cgs for flux: erg/cm2/s/Ang
    templz = eazydata.TEMPL * (1 + z)
    templf = np.dot(eazydata.COEFFS[ix, :], eazydata.TEMPF)
    fnu_factor = 10 ** (-0.4 * (25 + 48.6))
    flam_spec = 1. / (1 + z) ** 2
    tempflux = templf * fnu_factor * flam_spec

    try:
        import eazy.igm
        igmz = eazy.igm.Inoue14().full_IGM(z, templz)
        tempflux *= igmz
    except:
        pass

    if limitwaverange:
        # to simplify things, we only write out the data over the Subaru PFS
        # wavelength range, from 300 to 1300 nm (3000 to 13000 Angstroms)
        ipfs = np.where((templz>3000) & (templz<13000))[0]
        templz = templz[ipfs]
        tempflux = tempflux[ipfs]

    if returnfluxunit=='AB':
        # convert from flux density f_lambda into AB mag:
        mAB_from_flambda = lambda f_lambda, wave: -2.5 * np.log10(
            3.34e4 * wave * wave * f_lambda / 3631)
        tempflux = mAB_from_flambda(tempflux, templz)
    if returnwaveunit=='nm':
        templz = templz / 10.

    if savetofile:
        fout = open(savetofile, 'w')
        for i in range(len(templz)):
            fout.write('{wave:.3e} {flux:.3e}\n'.format(
                wave=templz[i], flux=tempflux[i]))
        fout.close()
    else:
        return templz, tempflux





