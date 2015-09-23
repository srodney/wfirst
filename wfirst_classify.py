#! /usr/bin/env python
import sncosmo
from sncosmo import classify
from astropy.table import Table, Column
import numpy as np
import os
import time
from glob import glob
from random import choice
import pyfits
from matplotlib import pyplot as pl
from astropy.io import ascii
import argparse

_snanamodeldata = ascii.read( """
# snanamodelnumber  snanasubclass  snananame  sncosmoname sncosmosubclass  sncosmosedfile
101         Ic   CSP-2004fe      snana-2004fe     Ic      CSP-2004fe.SED
102         Ic   CSP-2004gq   	 snana-2004gq     Ic      CSP-2004gq.SED
103         Ib   CSP-2004gv   	 snana-2004gv     Ib      CSP-2004gv.SED
104         Ib   CSP-2006ep   	 snana-2006ep     Ib      CSP-2006ep.SED
105         Ib   CSP-2007Y    	 snana-2007Y      Ib      CSP-2007Y.SED
201         IIP  SDSS-000018  	 snana-2004hx     IIP     SDSS-000018.SED
202         Ib   SDSS-000020  	 snana-2004ib     Ib      SDSS-000020.SED
203         Ib   SDSS-002744  	 snana-2005hm     Ib      SDSS-002744.SED
204         IIP  SDSS-003818  	 snana-2005gi     IIP     SDSS-003818.SED
205         Ic   SDSS-004012  	 snana-sdss004012 Ic      SDSS-004012.SED
206         IIN  SDSS-012842  	 snana-2006ez     IIn     SDSS-012842.SED
207         Ic   SDSS-013195  	 snana-2006fo     Ic      SDSS-013195.SED
208         IIP  SDSS-013376  	 snana-2006gq     IIP     SDSS-013376.SED
209         IIN  SDSS-013449  	 snana-2006ix     IIn     SDSS-013449.SED
210         IIP  SDSS-014450  	 snana-2006kn     IIP     SDSS-014450.SED
211         Ic   SDSS-014475  	 snana-sdss014475 Ic      SDSS-014475.SED
212         Ib   SDSS-014492  	 snana-2006jo     Ib      SDSS-014492.SED
213         IIP  SDSS-014599  	 snana-2006jl     IIP     SDSS-014599.SED
214         IIP  SDSS-015031  	 snana-2006iw     IIP     SDSS-015031.SED
215         IIP  SDSS-015320  	 snana-2006kv     IIP     SDSS-015320.SED
216         IIP  SDSS-015339  	 snana-2006ns     IIP     SDSS-015339.SED
217         Ic   SDSS-015475  	 snana-2006lc     Ic      SDSS-015475.SED
218         Ic   SDSS-017548  	 snana-2007ms     II      SDSS-017548.SED
219         IIP  SDSS-017564  	 snana-2007iz     IIP     SDSS-017564.SED
220         IIP  SDSS-017862  	 snana-2007nr     IIP     SDSS-017862.SED
221         IIP  SDSS-018109  	 snana-2007kw     IIP     SDSS-018109.SED
222         IIP  SDSS-018297  	 snana-2007ky     IIP     SDSS-018297.SED
223         IIP  SDSS-018408  	 snana-2007lj     IIP     SDSS-018408.SED
224         IIP  SDSS-018441  	 snana-2007lb     IIP     SDSS-018441.SED
225         IIP  SDSS-018457  	 snana-2007ll     IIP     SDSS-018457.SED
226         IIP  SDSS-018590  	 snana-2007nw     IIP     SDSS-018590.SED
227         IIP  SDSS-018596  	 snana-2007ld     IIP     SDSS-018596.SED
228         IIP  SDSS-018700  	 snana-2007md     IIP     SDSS-018700.SED
229         IIP  SDSS-018713  	 snana-2007lz     IIP     SDSS-018713.SED
230         IIP  SDSS-018734  	 snana-2007lx     IIP     SDSS-018734.SED
231         IIP  SDSS-018793  	 snana-2007og     IIP     SDSS-018793.SED
232         IIP  SDSS-018834  	 snana-2007ny     IIP     SDSS-018834.SED
233         IIP  SDSS-018892  	 snana-2007nv     IIP     SDSS-018892.SED
234         Ib   SDSS-019323  	 snana-2007nc     Ib      SDSS-019323.SED
235         IIP  SDSS-020038  	 snana-2007pg     IIP     SDSS-020038.SED
021         Ibc  SNLS-04D1la  	 snana-04d1la     Ic      SNLS-04D1la.SED
022         Ic   SNLS-04D4jv  	 snana-04d4jv     Ic      SNLS-04D4jv.SED
002         IIL  Nugent+ScolnicIIL nugent-sn2l	  IIL     sn2l_flux.v1.2.dat
""", format='commented_header', header_start=-1, data_start=0)

def get_snana_modelname(sn):
    if sn.meta['SIM_NON1a']==0:
        return sn.meta['SIM_MODEL_NAME'].strip()
    snanamodelnumber = sn.meta['SIM_NON1a']
    if snanamodelnumber not in _snanamodeldata['snanamodelnumber']:
        return '%03i'%int(snanamodelnumber)
    imod = np.where(_snanamodeldata['snanamodelnumber']==snanamodelnumber)[0]
    return _snanamodeldata['sncosmoname'][imod].item()

def get_sncosmo_excludelist(sn):
    if sn.meta['SIM_NON1a']==0:
        return []
    snanamodelnumber = sn.meta['SIM_NON1a']
    if snanamodelnumber not in _snanamodeldata['snanamodelnumber']:
        return []
    imod = np.where(_snanamodeldata['snanamodelnumber']==snanamodelnumber)[0]
    sncosmo_model_name = _snanamodeldata['sncosmoname'][imod].item()
    excludelist = [sncosmo_model_name]

    # also include Sako 2011 PSNID versions of 8 models:
    sncosmo_model_root = sncosmo_model_name.split('-')[-1]
    if sncosmo_model_root in ['2004hx', '2005lc', '2005hl', '2005hm',
                            '2005gi', '2006fo', '2006jo', '2006jl']:
        excludelist.append('s11-%s' % sncosmo_model_root)

    return excludelist

def standardize_sn_data(sn, headfile=None):
    """ prep a freshly-loaded SNANA-sim data table so that it
    can be handled by sncosmo.classify
    """

    if 'ZEROPT' in sn.colnames and 'ZPT' not in sn.colnames:
        sn['ZEROPT'].name = 'ZPT'
        sn['ZEROPT_ERR'].name = 'ZPTERR'

    if 'FLUXCAL' in sn.colnames and 'FLUX' not in sn.colnames:
        fluxdata = sn['FLUXCAL'] * 10 ** (0.4 * (sn['ZPT'] - 27.5))
        fluxerrdata = sn['FLUXCALERR'] * 10 ** (0.4 * (sn['ZPT'] - 27.5))
        fluxcolumn = Column(data=fluxdata, name='FLUX')
        sn.add_column(fluxcolumn)
        fluxerrcolumn = Column(data=fluxerrdata, name='FLUXERR')
        sn.add_column(fluxerrcolumn)

    if 'FLT' in sn.colnames and 'FILTER' not in sn.colnames:
        filterdata = np.where(sn['FLT']=='Y','Y106',
                              np.where(sn['FLT']=='J','J129','H158'))
        filtercolumn = Column(data=filterdata, name='FILTER')
        sn.add_column(filtercolumn)
        sn.remove_column('FLT')

    if 'MAGSYS' not in sn.colnames:
        magsysdata = np.ones(len(sn), dtype='S2')
        magsysdata.fill('AB')
        magsyscol = Column(data=magsysdata, name='MAGSYS')
        sn.add_column(magsyscol)

    snstd = sncosmo.fitting.standardize_data(sn)
    if headfile:
        sn.meta['HEADFILE'] = headfile
        sn.meta['PHOTFILE'] = headfile.replace('HEAD', 'PHOT')

    return Table(snstd, meta=sn.meta, copy=False)

def strip_post_detection_epochs(sn, ndetections, epochspan=2):
    """ strip the SN down to the first ndetections epochs where both
    J and H have >4sigma detections.
    :param sn:
    :return:
    """
    signaltonoise = sn['flux']/sn['fluxerr']

    # find all observations where S/N > 4 in the Y, J or H bands
    bandlist = np.unique(sn['band'])
    if 'Y106' in bandlist and 'H158' not in bandlist:
        band1 = 'Y106'
        band2 = 'J129'
    else:
        band1 = 'J129'
        band2 = 'H158'

    idet1_all = np.where((signaltonoise>4) & (sn['band']==band1))[0]
    idet2_all = np.where((signaltonoise>4) & (sn['band']==band2))[0]

    # find the epoch in which we get to the Nth detection
    ndetepoch=0
    ilastdetection = 0
    for idet1 in idet1_all:
        idet2_all_index = np.where(
            np.abs(sn['time'][idet2_all] - sn['time'][idet1]) < epochspan)[0]
        if len(idet2_all_index):
            ilastdetection = max([ilastdetection, idet1,
                                  idet2_all[idet2_all_index[0]]])
            ndetepoch += 1
            if ndetepoch == ndetections:
                break
    sntrimmed = sn[:ilastdetection+1]
    return Table(sntrimmed, meta=sn.meta, copy=False)

def get_test_sn(headfile='random', snid='random', verbose=True):
    if headfile=='random':
        headfilelist = glob('data/*HEAD.FITS')
        headfile = choice(headfilelist)

    if snid=='random':
        # Get metadata for all the SNe
        head_data = pyfits.getdata(headfile, 1, view=np.ndarray)

        # Strip trailing whitespace characters from SNID.
        if 'SNID' in head_data.dtype.names:
            snidlist = np.char.strip(head_data['SNID'])
            snid = choice(snidlist)

    sndataset = sncosmo.read_snana_fits(
        headfile, headfile.replace('HEAD','PHOT'),
        snids=[str(snid)])
    sn = sndataset[0]
    if verbose:
        print "SNID=%s from %s" % (snid, os.path.basename(headfile))
        excludelist = get_sncosmo_excludelist(sn)
        if excludelist :
            modelname = excludelist[0]
        elif sn.meta['SIM_NON1a'] :
            modelname = 'NON1a.%03i' % sn.meta['SIM_MODEL_INDEX']
        else :
            modelname = sn.meta['SIM_MODEL_NAME']
        print "Type=%s  model=%s" % (
            sn.meta['SIM_TYPE_NAME'].strip(), modelname)
    return sn

def test_classify(templateset='PSNID', verbose=3):
    """  run a test classification
    :return:
    """
    os.chdir(os.path.expanduser("~/sandbox/wfirst"))
    sn = get_test_sn(verbose=verbose)
    sn = standardize_sn_data(sn)
    sn = strip_post_detection_epochs(sn, 4)
    sncosmo.plot_lc(sn)
    pl.draw()

    z = sn.meta['SIM_REDSHIFT_CMB']
    tpk = sn.meta['PEAKMJD']
    excludetemplates = get_sncosmo_excludelist(sn)
    start = time.time()
    snclassdict = sncosmo.classify.classify(
        sn, zhost=z, zhosterr=0.0001, zminmax=[z-0.01,z+0.01],
        t0_range=[tpk-10, tpk+10], templateset=templateset,
        excludetemplates=excludetemplates,
        nobj=20, maxiter=1000, nsteps_pdf=51, verbose=verbose)
    end = time.time()
    print "P(Ia)=%.2f ; Type = %s ; best model = %s ; %.3f seconds" % (
        snclassdict['pIa'], sn.meta['SIM_TYPE_NAME'].strip(),
        snclassdict['bestmodel'], (end-start))

    return sn, snclassdict

def get_sndataset(headfile):
    photfile = headfile.replace('HEAD','PHOT')
    sndataset = sncosmo.read_snana_fits(headfile, photfile)
    for i in xrange(len(sndataset)):
        sndataset[i] = standardize_sn_data(
            sndataset[i], headfile=os.path.basename(headfile))
    return sndataset

def wfirst_classification_sequence(
        sn, outfile, epochspan=2, ndetepochs=['all',7,4,1],
        templateset='PSNID', nobj=20, maxiter=1000,
        detection_threshold = 4, verbose=3, clobber=False):
    """  for the given SN data table, run the classifier using only the number
     of detection epochs specified
    :param sn: sncosmo SN data table, assumed to be already standardized
    :return:
    """
    if os.path.isfile(outfile):
        outdat = ascii.read(outfile, format='commented_header',
                            header_start=-1, data_start=0)
        donesnidlist = outdat['snid']
        donendetlist = outdat['ndet']
    else :
        donesnidlist = np.array([])
        donendetlist = np.array([])

    # sort the observation dates into epochs and count the total
    # number of detection epochs
    signaltonoise = sn['flux']/sn['fluxerr']
    epochnumbers = np.zeros(len(sn))
    thisepoch = 0
    ndetepochtot = 0
    for i in xrange(len(sn)):
        if epochnumbers[i]:
            continue
        thisepoch += 1
        ithisepoch = np.where(np.abs(sn['time'] - sn['time'][i]) < epochspan)[0]
        epochnumbers[ithisepoch] = thisepoch
        if np.all(signaltonoise[ithisepoch]>detection_threshold):
            ndetepochtot += 1

    if 'all' in ndetepochs:
        ndetepochs.remove('all')
        if len(sn) not in ndetepochs:
            ndetepochs.append(ndetepochtot)
    ndetepochs.sort(reverse=True)

    fileroot = os.path.splitext(sn.meta['HEADFILE'])[0]
    z = sn.meta['SIM_REDSHIFT_CMB']
    tpk = sn.meta['PEAKMJD']
    excludetemplates = get_sncosmo_excludelist(sn)
    modelname = get_snana_modelname(sn)

    if not os.path.isfile(outfile):
        fout = open(outfile,'w')
        hdrstr = ("# file snid sim_type sim_model redshift nepochs ndet "
                  "pIa pIbc pII bestmodel nobj maxiter time")
        print >> fout, hdrstr
        fout.close()

    # strip out epochs and classify... repeat
    timelist = [time.time()]
    for ndet in ndetepochs:
        alreadydone = len(np.where(
            (donesnidlist==int(sn.meta['SNID'])) & (ndet==donendetlist))[0])
        if alreadydone :
            print('%s already classified with n=%i epochs.  Skipping' %
                  (sn.meta['HEADFILE'], ndet))
            continue

        if ndet < ndetepochtot:
            snstripped = strip_post_detection_epochs(
                sn, ndet, epochspan=epochspan)
        else :
            snstripped = sn
        t0min = min(sn['time'].max(), tpk - 10 * (1+z))
        t0max = tpk + 10 * (1+z)
        try:
            snclassdict = sncosmo.classify.classify(
                snstripped, zhost=z, zhosterr=0.0001, zminmax=[z-0.01,z+0.01],
                t0_range=[t0min,t0max], templateset=templateset,
                excludetemplates=excludetemplates,
                nobj=nobj, maxiter=maxiter, nsteps_pdf=0, verbose=verbose)
        except:
            print("WARNING : %s SNID=%s failed to classify" % (
                fileroot, sn.meta['SNID']))
            snclassdict = {'pIa': -1, 'pIbc': -1, 'pII': -1,
                           'bestmodel': 'failed'}
        timelist.append(time.time())

        classtime = timelist[-1] - timelist[-2]

        outstr = ("%s " % fileroot +
                  "%s " % str(sn.meta['SNID']) +
                  "%5s " % sn.meta['SIM_TYPE_NAME'].strip() +
                  "%5s " % modelname +
                  "%8.5f " % sn.meta['SIM_REDSHIFT_CMB'] +
                  "%2i " % len(np.unique(epochnumbers)) +
                  "%2i " % ndet +
                  "%6.2f " % snclassdict['pIa'] +
                  "%6.2f " % snclassdict['pIbc'] +
                  "%6.2f " % snclassdict['pII'] +
                  "%12s " % snclassdict['bestmodel'] +
                  "%i " % nobj +
                  "%i " % maxiter +
                  "%8.1f" % classtime)
        fout = open(outfile, 'a')
        print >> fout, outstr
        fout.close()

    return

def do_classify_fulldatfile(headfile, outfile='',
                            ndetepochs=['all',7,4,1],
                            nobj=50, maxiter=2000):
    """ Run classifications on all SN objects in the given .dat file.
    :param datfilename: name of the HEAD.FITS file with the SN data set
    to classify.
    :return:
    """
    if not outfile:
        outfileroot = os.path.splitext(os.path.basename(headfile))[0]
        outfile = outfileroot + '_CLASS.dat'
    sndataset = get_sndataset(headfile)
    isn = 0

    for sn in sndataset :
        isn += 1
        print("Classifying %s %s :  %i of %i" % (
            sn.meta['HEADFILE'], sn.meta['SNID'], isn, len(sndataset)))
        wfirst_classification_sequence(
            sn, outfile, epochspan=2, ndetepochs=ndetepochs,
            templateset='snana', nobj=nobj, maxiter=maxiter,
            detection_threshold=4, verbose=1)
    return

def do_classify_fulldatfile_list(headfilelist, outfile='',
                                 ndetepochs=['all',7,4,1],
                                 nobj=50, maxiter=2000):
    """
    :param headfilelist: name of a file containing a list of HEAD.FITS files,
        or an explicit list of headfiles
    :return:
    """
    headfiles = np.loadtxt(headfilelist, dtype=str, unpack=True)
    if not np.iterable(headfiles):
        headfiles = [headfiles.item()]
    for headfile in headfiles :
        do_classify_fulldatfile(headfile, outfile=outfile,
                                ndetepochs=ndetepochs,
                                nobj=nobj, maxiter=maxiter)
    return


def main():
    parser = argparse.ArgumentParser(
        description='Run an sncosmo-based classifier on simulated WFIRST SNe '
                    ' using user-specified numbers of detection epochs.')

    # Required positional argument
    parser.add_argument('headfiles',
                        help='Comma-separated list of file names for '
                             'HEAD.FITS files generated in SNANA sims, '
                             'with the SNe to classify - OR - a single '
                             'filename for a text file with a list of '
                             'HEAD.FITS files, one per line.')
    parser.add_argument('--outfile', type=str, default='',
                        help="Output file for classification results. "
                        "(If unspecified, we use the headfile root with "
                        "_CLASS.dat replacing the .FITS suffix)")
    parser.add_argument('--ndetepochs', type=str, default='1,4,7,all',
                        help="Comma-separated list of detection epochs to "
                        "allow the classifier to use.")
    parser.add_argument('--nobj', type=int, default=50,
                        help="Number of objects for sncosmo nested sampler. "
                        "(bigger = slower and more precise)")
    parser.add_argument('--maxiter', type=int, default=2000,
                        help="Max number of iterations for sncosmo nested "
                             "sampler. (bigger = slower and more accurate)")
    argv = parser.parse_args()
    if ',' in argv.headfiles:
        headfilelist = argv.headfiles.split(',')
    else:
        headfilelist = argv.headfiles

    ndetepochs = argv.ndetepochs.split(',')
    for i in xrange(len(ndetepochs)):
        try:
            ndetepochs[i] = int(ndetepochs[i])
        except ValueError:
            pass

    do_classify_fulldatfile_list(headfilelist, outfile=argv.outfile,
                                 ndetepochs=ndetepochs,
                                 nobj=argv.nobj, maxiter=argv.maxiter)


if __name__ == "__main__":
    main()


