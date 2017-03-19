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

class WfirstHostCatalog(Table):

    def __init__(self, *args, **kwargs):
        super(Table, self).__init__(*args, **kwargs)

    def mkcatfile(self, snanasimdir='SNANA.SIM.OUTPUT'):
        """Make a master catalog file to hold SN host galaxy data:
         # redshift hostmag matchid sedsimfile snrsimfile snrmed
         """
        simlist = []
        simfilelist_med = glob(
            os.path.join(snanasimdir, '*Z08*HEAD.FITS'))
        simfilelist_deep = glob('SNANA.SIM.OUTPUT/*Z17*HEAD.FITS')
        hostz_med, hostmag_med = np.array([]), np.array([])
        for simfile in simfilelist_med:
            sim = WfirstSimData(simfile)
            sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
            sim.get_matchlists()
            hostz_med = np.append(hostz_med, sim.zsim)
            hostmag_med = np.append(hostmag_med, sim.mag)
            simlist.append(sim)

        hostz_deep, hostmag_deep = np.array([]), np.array([])
        for simfile in simfilelist_deep:
            sim = WfirstSimData(simfile)
            sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
            sim.get_matchlists()
            hostz_deep = np.append(hostz_deep, sim.zsim)
            hostmag_deep = np.append(hostmag_deep, sim.mag)
            simlist.append(sim)


        return(simlist)



class WfirstSimData(object):

    def __init__(self, infilename=None, verbose=1, *args, **kwargs):
        # TODO: make this inherit Table properties, so we can treat it as a table directly
        # super(Table, self).__init__(*args, **kwargs)
        self.verbose = verbose
        self.matchdata = Table()
        self.simdata = Table()
        self.eazydata = {}
        if infilename:
            self.read(infilename, *args, **kwargs)
        elif self.verbose:
            print("Initiliazed an empty WfirstSimData object")
        return

    def read(self, infilename, format='snana',
             **kwargs):
            """read in a catalog of SN host galaxy data.
            Initialize a new catalog from a SNANA data.fits file
            (using format='snana') or
            load a modified catalog using the astropy table reading functions.
            Additional keywords are passed to the astropy.io.ascii.read
            function.
            """
            if format.lower() == 'snana':
                hdulist = fits.open(infilename)
                bindata = hdulist[1].data
                zsim = bindata['SIM_REDSHIFT_HOST']
                if 'HOSTGAL_MAG_H' in [col.name for col in bindata.columns]:
                    magsim = bindata['HOSTGAL_MAG_H']
                else:
                    magsim = bindata['HOSTGAL_MAG_J']
                self.simdata.add_column(
                    Table.Column(data=magsim, name='magsim'))
                self.simdata.add_column(Table.Column(data=zsim, name='zsim'))
            else:
                # read a processed catalog from some other data format.
                self.simdata.read(infilename, format=format, **kwargs)


    def write(self, outfilename, format='ascii.commented_header', **kwargs):
        """write out the catalog of SN host galaxy data
        Columns in the catalog will vary, depending on what other host gal
        simulation data have been collected and added to the table.
        Additional keywords are passed to the astropy.io.ascii.write
        function.
        """
        self.simdata.write(outfilename, format=format, **kwargs)


    def load_matchdata(self, matchcatfilename=None):
        """Load a 3DHST catalog to identify galaxies that match the
        properties of the SN host galaxies.
        """
        if len(self.matchdata) > 0:
            print("SNANA sim outputs already matched to 3DHST." +
                  "No changes done.")
            return

        if matchcatfilename is None:
            matchcatfilename = ('3DHST/3dhst_master.phot.v4.1/'
                                '3dhst_master.phot.v4.1.cat.FITS')

        # load the 3dHST catalog
        matchdata = fits.getdata(matchcatfilename)

        f160 = matchdata['f_F160W']
        zspec = matchdata['z_spec']
        zphot = matchdata['z_peak']
        zbest = np.where(zspec>0, zspec, zphot)
        usephot = matchdata['use_phot']
        ivalid = np.where(((f160>0) & (zbest>0)) & (usephot==1) )[0]
        isort = np.argsort(zbest[ivalid])

        z3d = zbest[ivalid][isort]
        idgal = matchdata['id'][ivalid][isort].astype(int)
        field = matchdata['field'][ivalid][isort]
        mag3d = (-2.5 * np.log10(f160[ivalid]) + 25)[isort]
        id3d = np.array(['{}.{:04d}'.format(field[i], idgal[i])
                         for i in range(len(field))])

        self.matchdata.add_column(Table.Column(data=z3d, name='z3D'))
        self.matchdata.add_column(Table.Column(data=mag3d, name='mag3D'))
        self.matchdata.add_column(Table.Column(data=id3d, name='id3D'))
        return


    def get_matchlists(self, dz=0.05, dmag=0.2):
        """For each simulated host galaxy, get a list of 3DHST galaxy IDs for
        all galaxies in the 3DHST catalog that have a redshift within dz of
        the simulated z, and an H band mag within dmag of the simulated H band
        mag."""
        if self.matchdata is None:
            self.load_matchdata()
        zsim = self.simdata['zsim']
        magsim= self.simdata['magsim']
        zmatch = self.matchdata['z3D']
        magmatch = self.matchdata['mag3D']
        idmatch = self.matchdata['id3D']

        # TODO: find the nearest 10 or 100 galaxies, instead of all within
        # a specified dz and dmag range.
        allmatch_indexlist = np.array(
            [np.where((zmatch + dz > zsim[i]) &
                      (zmatch - dz < zsim[i]) &
                      (magmatch + dmag > magsim[i]) &
                      (magmatch - dz < magsim[i]))[0]
             for i in range(len(zsim))])
        nmatch = np.array([len(allmatch_indexlist[i])
                                for i in range(len(zsim))])

        allmatchids = np.array([idmatch[allmatch_indexlist[i]]
                                for i in range(len(zsim))])

        matchid = np.array([np.random.choice(allmatchids[i])
                                 for i in range(len(zsim))])

        # TODO: don't use add_column... we should update columns if they already exist.
        self.simdata.add_column(Table.Column(data=matchid, name='matchid'))
        self.simdata.add_column(Table.Column(data=nmatch, name='nmatch'))


    def load_sed_data(self):
        """ load all the EAZY simulated SED data at once"""
        for field in ['aegis', 'cosmos', 'goodsn', 'goodss', 'uds']:
            fitsfilename = glob(
                '3DHST/{0}_3dhst.*.eazypy.data.fits'.format(field))[0]
            self.eazydata[field] = EazyData(fitsfilename=fitsfilename)


    def simulate_seds(self):
        """Use Gabe Brammer's EAZY code to simulate the host gal SED """

        sedsimfilelist = []
        for isim in range(len(self.simdata['matchid'])):
            fieldidx = self.simdata['matchid'][isim]
            fieldstr, idxstr = fieldidx.split('.')
            field = fieldstr.lower().replace('-', '')
            thiseazydat = self.eazydata[field]
            sedsimfilename = ('3DHST/sedsim.output/' +
                              'wfirst_simsed.{:s}.dat'.format(fieldidx))
            simulate_eazy_sed( fieldidx=fieldidx, eazydata=thiseazydat,
                               savetofile=sedsimfilename)
            sedsimfilelist.append(sedsimfilename)
        self.simdata.add_column(
            Table.Column(data=sedsimfilelist, name='sedsimfile'))



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





