def extend_template0_ir( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR',
                      tailsedfile = 'snsed/Hsiao07.extrap.dat',
                      wjoinred = 8500 ,
                      wmax = 20000, tmin=-20, tmax=100,
                      showplots=False ):
    """ extend the salt2 Template_0 model component
    by adopting the IR tails from a collection of SN Ia template SEDs.
    Here we use the collection of CfA, CSP, and other low-z SNe provided by
    Arturo Avelino (2016, priv. comm.)
    The median of the sample is scaled and joined at the
    wjoin wavelength, and extrapolated out to wmax.
    """
    import shutil
    sndataroot = os.environ['SNDATA_ROOT']

    salt2dir = os.path.join( sndataroot, salt2dir )

    temp0fileIN = os.path.join( salt2dir, '../SALT2.Guy10_LAMOPEN/salt2_template_0.dat' )
    temp0fileOUT = os.path.join( salt2dir, 'salt2_template_0.dat' )
    temp0dat = getsed( sedfile=temp0fileIN )

    tailsedfile = os.path.join( sndataroot, tailsedfile )

    taildat = getsed( sedfile=tailsedfile )

    dt,wt,ft = loadtxt( tailsedfile, unpack=True )
    taildays = unique( dt )

    fscale = []
    # build up modified template from day -20 to +100
    outlines = []
    daylist = range( tmin, tmax+1 )
    for i in range( len(daylist) ) :
        thisday = daylist[i]

        if thisday < 50 :
            # get the tail SED for this day from the Hsiao template
            it = where( taildays == thisday )[0]
            dt = taildat[0][it]
            wt = taildat[1][it]
            ft = taildat[2][it]

            # get the SALT2 template SED for this day
            d0 = temp0dat[0][i]
            w0 = temp0dat[1][i]
            f0 = temp0dat[2][i]
            print( 'splicing tail onto template for day : %i'%thisday )

            i0blue = argmin(  abs(w0-wjoinblue) )
            itblue = argmin( abs( wt-wjoinblue))

            i0red = argmin(  abs(w0-wjoinred) )
            itred = argmin( abs( wt-wjoinred))

            itmin = argmin( abs( wt-wmin))
            itmax = argmin( abs( wt-wmax))

            bluescale = f0[i0blue]/ft[itblue]
            redscale = f0[i0red]/ft[itred]

            d0new = dt.tolist()[itmin:itblue] + d0.tolist()[i0blue:i0red] + dt.tolist()[itred:itmax+1]
            w0new = wt.tolist()[itmin:itblue] + w0.tolist()[i0blue:i0red] + wt.tolist()[itred:itmax+1]
            f0newStage = (bluescale*ft).tolist()[itmin:itblue] + f0.tolist()[i0blue:i0red] + (redscale*ft).tolist()[itred:itmax+1]

            # compute the flux scaling decrement from the last epoch (for extrapolation)
            if i>1: fscale.append( np.where( np.array(f0newStage)<=0, 0, ( np.array(f0newStage) / np.array(f0new) ) ) )
            f0new = f0newStage

        # elif thisday < 85 :
        #     # get the full SED for this day from the Hsiao template
        #     it = where( taildays == thisday )[0]
        #     dt = taildat[0][it]
        #     wt = taildat[1][it]
        #     ft = taildat[2][it]
        #     d0new = dt
        #     w0new = wt
        #     f0new = ft * (bluescale+redscale)/2.  * (fscaleperday**(thisday-50))
        else  :
            print( 'scaling down last template to extrapolate to day : %i'%thisday )
            # linearly scale down the last Hsiao template
            d0new = zeros( len(dt) ) + thisday
            w0new = wt
            #f0new = f0new * (bluescale+redscale)/2. * (fscaleperday**(thisday-50))

            f0new = np.array(f0new) * np.median( np.array(fscale[-20:]), axis=0 )
            #f0new = np.array(f0new) * ( np.median(fscale[-20:])**(thisday-50))

        if showplots:
            # plot it
            print( 'plotting modified template for day : %i'%thisday )
            clf()
            plot( w0, f0, ls='-',color='b', lw=1)
            plot( wt, (bluescale+redscale)/2. * ft, ls=':',color='r', lw=1)
            plot( w0new, f0new, ls='--',color='k', lw=2)
            ax = gca()
            ax.grid()
            ax.set_xlim( 500, 13000 )
            ax.set_ylim( -0.001, 0.02 )
            draw()
            raw_input('return to continue')

        # append to the list of output data lines
        for j in range( len( d0new ) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    d0new[j], w0new[j], f0new[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp0fileOUT, 'w' )
    fout.writelines( outlines )
    fout.close()


