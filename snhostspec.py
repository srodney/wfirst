# /usr/bin/env python
# 2017.03.10  S. Rodney
# Reading in host galaxy data from WFIRST simulations
# and computing an estimated exposure time for Subaru+PFS
# to get a redshift for that host.

from astropy.io import fits
from astropy.table import Table
from astropy import table
from astropy.io import ascii
import os
import subprocess
import exceptions
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from StringIO import StringIO
import time



class SnanaSimData(object):
    # TODO : needs some checks to make sure that we don't rerun unneccessary
    # host galaxy SED simulations or S/N calculations.

    def __init__(self, infilename=None, verbose=1, *args, **kwargs):
        self.verbose = verbose
        self.matchdata = Table()
        self.simdata = Table()
        self.simfilelist = []
        self.eazydata = {}
        if infilename:
            self.add_snana_simdata(infilename, *args, **kwargs)
        elif self.verbose:
            print("Initiliazed an empty WfirstSimData object")
        return

    def add_snana_simdata(self, infilename):
        """read in a catalog of SN host galaxy data. Initialize a new
        catalog from a SNANA head.fits file
        """
        simdata = Table()
        hdulist = fits.open(infilename)
        bindata = hdulist[1].data
        zsim = bindata['SIM_REDSHIFT_HOST']
        if 'HOSTGAL_MAG_H' in [col.name for col in bindata.columns]:
            magsim = bindata['HOSTGAL_MAG_H']
        else:
            magsim = bindata['HOSTGAL_MAG_J']
        simdata.add_column(Table.Column(data=magsim, name='magsim'))
        simdata.add_column(Table.Column(data=zsim, name='zsim'))
        self.simdata = table.vstack([self.simdata, simdata])
        self.simfilelist.append(infilename)


    def load_simdata_catalog(self, infilename, **kwargs):
        """Load an ascii commented_header catalog using the astropy table
        reading functions. Additional keywords are passed to the
        astropy.io.ascii.read function. """
        simdata = ascii.read(infilename, format='commented_header', **kwargs)
        if len(self.simdata):
            self.simdata = table.vstack([self.simdata, simdata])
        else:
            self.simdata = simdata
        self.simfilelist.append(infilename)


    def add_all_snana_simdata(self, snanasimdir='SNANA.SIM.OUTPUT'):
        """Load all the snana simulation data.
        """
        simfilelist = glob(os.path.join(snanasimdir, '*HEAD.FITS'))
        # for simfile in simfilelist:
        for simfile in simfilelist:
            if self.verbose:
                print("Adding SNANA sim data from {:s}".format(simfile))
            self.add_snana_simdata(simfile)
        return


    def write_catalog(self, outfilename, format='ascii.commented_header',
                      **kwargs):
        """Write out the master catalog of SN host galaxy data
        Columns in the catalog will vary, depending on what other host gal
        simulation data have been collected and added to the table.
        Additional keywords are passed to the astropy.io.ascii.write
        function.
        """
        self.simdata.write(outfilename, format=format, **kwargs)
        if self.verbose:
            print('Wrote sim data catalog to {:s}'.format(outfilename))


    def load_matchdata(self, matchcatfilename=None):
        """Load a 3DHST catalog to identify galaxies that match the
        properties of the SN host galaxies.
        """
        if len(self.matchdata) > 0:
            print("SNANA sim outputs already matched to 3DHST." +
                  "No changes done.")
            return
        if matchcatfilename is None:
            matchcatfilename = '3DHST/3dhst_master.phot.v4.1.cat.FITS'

        if self.verbose:
            print("Loading observed galaxy data from the 3DHST catalogs")
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


    def pick_random_matches(self, dz=0.05, dmag=0.2):
        """For each simulated SN host gal, find all observed galaxies (from
        the 3DHST catalogs) that have similar redshift and magnitude---i.e.,
        a redshift within dz of the simulated z, and an H band mag within
        dmag of the simulated H band mag.

        Pick one at random, and adopt it as the template for our simulated SN
        host gal (to be used for simulating the host gal spectrum).
        """
        if self.matchdata is None:
            self.load_matchdata()
        zsim = self.simdata['zsim']
        magsim= self.simdata['magsim']
        z3d = self.matchdata['z3D']
        mag3d = self.matchdata['mag3D']
        id3d = self.matchdata['id3D']

        nsim = len(zsim)
        if self.verbose:
            print("Finding observed galaxies that ~match simulated SN host" +
                  "\ngalaxy properties (redshift and magnitude)...")

        # TODO: find the nearest 10 or 100 galaxies, instead of all within
        # a specified dz and dmag range.

        nmatch, magmatch, zmatch, idmatch = [], [], [], []
        for i in range(nsim):
            isimilar = np.where((z3d + dz > zsim[i]) &
                                (z3d - dz < zsim[i]) &
                                (mag3d + dmag > magsim[i]) &
                                (mag3d - dz < magsim[i]))[0]
            nmatch.append(len(isimilar))
            irandmatch = np.random.choice(isimilar)
            magmatch.append(mag3d[irandmatch])
            zmatch.append(z3d[irandmatch])
            idmatch.append(id3d[irandmatch])

        # record the 3DHST data for each galaxy we have randomly picked:
        #   z, mag, id (field name + 3DHST catalog index)
        # TODO: don't use add_column... we should update columns if they
        # already exist.
        self.simdata.add_column(
            Table.Column(data=np.array(idmatch), name='idmatch'))
        self.simdata.add_column(
            Table.Column(data=np.array(nmatch), name='nmatch'))
        self.simdata.add_column(
            Table.Column(data=np.array(magmatch), name='magmatch'))
        self.simdata.add_column(
            Table.Column(data=np.array(zmatch), name='zmatch'))


    def load_sed_data(self):
        """ load all the EAZY simulated SED data at once"""
        if self.verbose:
            print("Loading data for best-fit SEDs from 3DHST fits "
                  "to observed photometry for all galaxies in all "
                  "five CANDELS fields...")

        for field in ['aegis', 'cosmos', 'goodsn', 'goodss', 'uds']:
            fitsfilename = glob(
                '3DHST/{0}_3dhst.*.eazypy.data.fits'.format(field))[0]
            self.eazydata[field] = EazyData(fitsfilename=fitsfilename)


    def simulate_host_spectra(self, indexlist=None,
                              outdir='3DHST/sedsim.output',
                              clobber=False):
        """Use Gabe Brammer's EAZY code to simulate the host gal spectrum
        for every host galaxy in the sample.
        """
        if indexlist is None:
            indexlist = range(len(self.simdata['idmatch']))

        if self.verbose:
            print("Using Gabe Brammer's EAZY code to generate "
                  "the best-fit SEDs of the observed galaxies that "
                  "we have matched up to the SNANA simulation hostgal data.")
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        sedoutfilelist = []
        for isim in indexlist:
            fieldidx = self.simdata['idmatch'][isim]
            fieldstr, idxstr = fieldidx.split('.')
            field = fieldstr.lower().replace('-', '')
            thiseazydat = self.eazydata[field]
            sedoutfilename = os.path.join(
                outdir, 'wfirst_simsed.{:s}.dat'.format(fieldidx))
            sedoutfilelist.append(sedoutfilename)
            if clobber or not os.path.isfile(sedoutfilename):
                if self.verbose>1:
                    print("Generating {:s}".format(sedoutfilename))
                simulate_eazy_sed(fieldidx=fieldidx, eazydata=thiseazydat,
                                  savetofile=sedoutfilename)
            else:
                if self.verbose>1:
                    print("{:s} exists. Not clobbering.".format(
                        sedoutfilename))
        assert(len(self.simdata['zsim']) == len(sedoutfilelist))
        self.simdata.add_column(
            Table.Column(data=sedoutfilelist, name='sedoutfile'))


    def get_host_percentile_indices(self, zlist=[0.8, 1.2, 1.5, 2.0],
                                    percentilelist=[50, 80, 95]):
        """For each redshift in zlist, identify all simulated host galaxies
        within dz of that redshift.   Sort them by "observed" host magnitude
        (the mag of the observed 3DHST galaxy that has been matched to each
        simulated host gal).   Identify which simulated host is closest to
        each percentile point in percentilelist.  Returns a list of indices
        for those selected galaxies.
        """
        index_array = []
        for z in zlist:
            dz = np.abs(self.simdata['zmatch'] - z)
            iznearest = np.argsort(dz)[:100]
            magnearest = self.simdata['magmatch'][iznearest]
            mag_percentiles = np.percentile(magnearest, percentilelist)
            index_array.append(
                [iznearest[np.abs(magnearest - mag_percentiles[i]).argmin()]
                for i in range(len(percentilelist))])
        return(np.ravel(index_array))


    def simulate_subaru_snr_curves(self, indexlist=[],
                                   exposuretimelist=[1, 5, 10, 40],
                                   clobber=False):
        """ Run the subaru PSF ETC to get a S/N vs wavelength curve.

        indexlist : select a subset of the master catalog to simulate.
           defaul = simulate S/N for all.

        exposuretimelist : exposure times in hours to use for the
          Subaru PFS ETC.
        """
        if not os.path.isdir("etcout"):
            os.mkdir("etcout")
        if not len(indexlist):
            indexlist = np.arange(len(self.simdata['zsim']))
        for idx in indexlist:
            for et in exposuretimelist:
                defaultsfile = os.path.join(
                    '/Users/rodney/src/wfirst',
                    'wfirst_subarupfsetc.{:d}hr.defaults'.format(et))
                sedoutfile = self.simdata['sedoutfile'][idx]
                snroutfile = "etcout/subaruPFS_SNR_{:s}_z{:.2f}_m{:.2f}_{:d}hrs.dat".format(
                    self.simdata['idmatch'][idx], self.simdata['zmatch'][idx],
                    self.simdata['magmatch'][idx], et)
                if os.path.isfile(snroutfile) and not clobber:
                    if self.verbose:
                        print("{:s} exists. Not clobbering.".format(
                            snroutfile))
                else:
                    if self.verbose:
                        print(
                        "Running the PFS ETC for "
                        "{:s} at z {:.2f} with mag {:.2f}"
                        "for {:d} hrs, sedfile {:s}.\n output: {:s}".format(
                            self.simdata['idmatch'][idx],
                            self.simdata['zmatch'][idx],
                            self.simdata['magmatch'][idx],
                            et, self.simdata['sedoutfile'][idx], snroutfile))

                    start = time.time()
                    etcerr = subprocess.call(["python",
                                              "/Users/rodney/src/subarupfsETC/run_etc.py",
                                              "@{:s}".format(defaultsfile),
                                              "--MAG_FILE={:s}".format(sedoutfile),
                                              "--OUTFILE_SNC={:s}".format(snroutfile)
                                              ])
                    end = time.time()
                    print("Finished in {:.1f} seconds".format(end-start))





class SubaruObsSim(object):
    """ a class for handling the output of the C.Hirata SubaruPFS ETC code
    """
    # TODO: record metadata in the .dat file header and read it in here
    # TODO:  better yet--- use a fits bintable instead of an ascii file.
    def __init__(self, etcoutfilename, verbose=1):
        # super(Table, self).__init__(*args, **kwargs)
        # KLUDGE!  parsing filename to get redshift, host mag and exptime
        filenameparselist = os.path.basename(etcoutfilename).split('_')
        self.matchid = filenameparselist[2]
        self.z = float(filenameparselist[3][1:])
        self.mag = float(filenameparselist[4][1:])
        self.exptime_hours = float(filenameparselist[5].strip('hrs.dat'))
        self.exptime_seconds = self.exptime_hours * 3600
        etcoutdata = ascii.read(etcoutfilename, format='basic',
                                names=['arm', 'pix', 'wave','snpix',
                                       'signal_exp', 'var0', 'var',
                                       'mAB', 'flux_conversion',
                                       'samplingfactor', 'skybg'])

        self.wave_obs = etcoutdata['wave']
        self.wave_rest = self.wave_obs / (1 + self.z)
        self.signaltonoise = etcoutdata['snpix']
        self.specsim = None
        self.verbose = verbose
        self.redshift_detected = -1
        self.bestsnr = 0
        self.bestsnr_waverest = 0
        self.bestbinsize = 0
        self.redshift_detection_string = ""

    def load_specdata(self, specdatadir='3DHST/sedsim.output'):
        """ read in the simulated spectrum data, generated with EAZY """
        specsimfile = os.path.join(specdatadir,
                                   'wfirst_simsed.{:s}.dat'.format(
                                       self.matchid))
        self.specsim = EazySpecSim(specsimfile)

    def plot(self, frame='rest', showspec=False, **kwargs):
        """ frame = 'rest' : show restframe wavelengths
        """
        ax = plt.gca()
        if frame=='rest':
            wsubaru = self.wave_rest
            xlabel = 'rest-frame wavelength (nm)'
        else:
            wsubaru = self.wave_obs
            xlabel = 'obs-frame wavelength (nm)'
        ax.plot(wsubaru, self.signaltonoise, color='k', **kwargs)
        xlab = ax.set_xlabel(xlabel)
        ylab = ax.set_ylabel('S/N per pix with Subaru PFS')

        if not showspec:
            return
        if self.specsim is None:
            self.load_specdata()
        ax2 = ax.twinx()
        if frame=='rest':
            weazy = self.specsim.wave / (1 + self.z)
        else:
            weazy = self.specsim.wave
        ax2.plot(weazy, self.specsim.flux, color='r', **kwargs)
        ax2.invert_yaxis()
        ax2.set_ylabel('AB mag', color='r', rotation=-90)

        if self.redshift_detected >= 0:
            ax.text(0.05, 0.95, self.redshift_detection_string,
                    ha='left', va='top', transform=ax.transAxes)

    def check_redshift(self, snrthresh=4, showplot=False, **kwargs):
        """Test whether a redshift can be determined from the spectrum"""
        self.redshift_detected = 0
        self.bestsnr = 0
        self.bestbinsize=1
        for binsize in [2,3,4,5,6,7,8,9,10]:
            ibinlist = np.arange(0, len(self.signaltonoise), binsize)
            snrbinned = np.array(
                [self.signaltonoise[ibin:ibin + binsize].sum()/np.sqrt(binsize)
                 for ibin in ibinlist[:-1]])
            snrbinmax = np.max(snrbinned)
            waverestbinned = self.wave_rest[ibinlist[:-1] + binsize / 2]
            snrbinmaxwaverest = waverestbinned[np.argmax(snrbinned)]
            if snrbinmax>=snrthresh:
                self.redshift_detected = 1
            if snrbinmax > self.bestsnr:
                self.bestsnr = snrbinmax
                self.bestsnr_waverest = snrbinmaxwaverest
                self.bestbinsize = binsize

        if self.redshift_detected == 0:
            self.redshift_detection_string = "No "
        elif self.redshift_detected == 1:
            self.redshift_detection_string = ""
        self.redshift_detection_string += (
            "Redshift detected. Max S/N={:.1f} at rest wave={:d} nm".format(
                self.bestsnr, int(self.bestsnr_waverest)))

        if self.verbose:
            print(self.redshift_detection_string)

        if showplot:
            binpix = self.bestbinsize
            ibinlist = np.arange(0, len(self.signaltonoise), binpix)
            snrbinned = np.array(
                [self.signaltonoise[ibin:ibin + binpix].sum() / np.sqrt(binpix)
                 for ibin in ibinlist[:-1]])
            waverestbinned = self.wave_rest[ibinlist[:-1] + binpix / 2]
            ax = plt.gca()
            ax.plot(waverestbinned, snrbinned, color='b', **kwargs)


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





