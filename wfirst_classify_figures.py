import os
import numpy as np
from matplotlib import pyplot as pl
from astropy.io import ascii
from pytools import plotsetup, colorpalette as cp


def mk_class_fig(datfile, zbins=np.arange(0.01,2.02,0.25), **kwargs):
    """ Plot the classification accuracy as a function of number of
    detection epochs used, in a given redshift range and with a
    given number of S/N>4 detections.
    :param datfile: filename of a .dat file produced by wfirst_classify
    :param zrange: redshift min and max
    :param ndetrange: number of detections min and max
    :return:
    """
    plotkwargs = dict(marker='o', mfc='0.5', mec='k', mew=0.8,
                      color='k', ms=8, lw=1.5, capsize=0)
    plotkwargs.update(**kwargs)
    fig = pl.gcf()

    classdata = ascii.read(datfile, format='commented_header',
                           header_start=-1, data_start=0)

    icol = 0
    ndetrangelist = [(1,2),(4,5),(7,8),(8,1000)]
    ncol = len(ndetrangelist)
    for ndetrange in ndetrangelist:
        icol += 1
        if icol == 1 :
            axtop0 = fig.add_subplot(2, ncol, 1)
            axtop = axtop0
            axbot0 = fig.add_subplot(2, ncol, ncol+icol, sharex=axtop0, sharey=axtop0)
            axbot = axbot0
        else :
            axtop = fig.add_subplot(2, ncol, icol, sharex=axtop0, sharey=axtop0)
            axbot = fig.add_subplot(2, ncol, ncol+icol, sharex=axtop0, sharey=axtop0)

        fcorrect = []
        fcorhigh = []
        fcorlow  = []
        purity = []
        purityhigh = []
        puritylow  = []

        fcor_zbinmid = []
        purity_zbinmid = []
        for iz in range(len(zbins)-1):
            zmin=zbins[iz]
            zmax=zbins[iz+1]
            ithisbin = np.where(
                (classdata['pIa']>-1) &
                (classdata['redshift']>=zmin) &
                (classdata['redshift']<zmax) &
                (classdata['ndet']>=ndetrange[0]) &
                (classdata['ndet']<ndetrange[1])
            )[0]

            nthisbin = float(len(ithisbin))
            if nthisbin < 1 :
                continue
            classdatabin = classdata[ithisbin]
            fcor_thisbin = []
            purity_thisbin = []
            for threshold in [0.50,0.80,0.95]:
                icorrect = np.where(
                    ((classdatabin['sim_type']=='Ia') &
                     (classdatabin['pIa']>threshold)) |
                     ((classdatabin['sim_type']!='Ia') &
                      (classdatabin['pIa']<threshold)
                      & (classdatabin['pIa']>-1) ))[0]
                fcor_thisbin.append(len(icorrect)/nthisbin)

                icalledIa = np.where(
                    classdatabin['pIa']>threshold)[0]
                itrueIa = np.where(
                    (classdatabin['sim_type']=='Ia') &
                    (classdatabin['pIa']>threshold))[0]
                if len(icalledIa)<1 :
                    continue
                purity_thisbin.append(
                    len(itrueIa)/float(len(icalledIa)))

            if len(fcor_thisbin):
                fcorrect.append(np.median(fcor_thisbin))
                fcorhigh.append(np.max(fcor_thisbin))
                fcorlow.append(np.min(fcor_thisbin))
                fcor_zbinmid.append((zmin+zmax)/2.)
            if len(purity_thisbin):
                purity.append(np.median(purity_thisbin))
                purityhigh.append(np.max(purity_thisbin))
                puritylow.append(np.min(purity_thisbin))
                purity_zbinmid.append((zmin+zmax)/2.)
            #else :
            #    import pdb; pdb.set_trace()
            #    print('missing bin')

        fcor_errhigh = np.array(fcorhigh) - np.array(fcorrect)
        fcor_errlow = np.array(fcorrect) - np.array(fcorlow)
        purity_errhigh = np.array(purityhigh) - np.array(purity)
        purity_errlow = np.array(purity) - np.array(puritylow)
        axtop.errorbar( fcor_zbinmid, fcorrect,
                     np.array([fcor_errlow, fcor_errhigh]),
                     **plotkwargs)
        axbot.errorbar( purity_zbinmid, purity,
                     np.array([purity_errlow, purity_errhigh]),
                     **plotkwargs)

        axtop.set_title('%i Detections' % ndetrange[0])
        if icol>1:
            pl.setp(axtop.get_yticklabels(), visible=False)
            pl.setp(axbot.get_yticklabels(), visible=False)

        pl.setp(axtop.get_xticklabels(), visible=False)
        axbot.set_xlabel('Redshift')
    axtop.set_title('Full Light Curve ($>$7 det)')
    fig.subplots_adjust(left=0.07, bottom=0.12, right=0.97, top=0.92,
                        hspace=0, wspace=0)
    axtop0.set_ylabel("Fraction Correctly Classified")
    axbot0.set_ylabel("Type Ia Sample Purity")
    axtop0.set_ylim(0.01,1.1)
    axtop0.set_xlim(0, 1.99)

    return





def mk_multisurvey_fig():
    fig1 = plotsetup.fullpaperfig(figsize=[12,6])
    fig1.clf()
    for survey, zbins, mfc, mec in zip(
            ['Deep','Med','Wide'],
            [np.arange(0.8,2.51,0.25),np.arange(0.3,1.25,0.2),
             np.arange(0.01,0.62,0.2)],
            [cp.lightred, cp.teal, cp.lightblue],
            [cp.darkred, cp.darkgreen, cp.darkblue]):
        mk_class_fig('%s_class.dat' % survey.lower(), zbins=zbins,
                         mfc=mfc, mec=mec, color=mec, marker='D')

    ax = fig1.add_subplot(2,4,1)
    txt = ax.text(0.4, 0.7, 'Wide', ha='right', va='top',
                  fontsize=15, color=cp.darkblue)
    txt = ax.text(0.6, 0.65, 'Med', ha='left', va='top',
                  fontsize=15, color=cp.teal)
    txt = ax.text(1.5, 0.55, 'Deep', ha='center', va='top',
                  fontsize=15, color=cp.darkred)
    pl.savefig(os.path.expanduser("~/Desktop/wfirst_classification_test.pdf"))

