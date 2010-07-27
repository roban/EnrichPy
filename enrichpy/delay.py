"""Class Delay represents the delay distribution for metals emerging from stars.

"""
import math
import sys
import cPickle
import os

import scipy
import scipy.integrate
import scipy.optimize
import numpy

import cosmolopy.utils as utils
import yields
import initialmassfunction
import lifetime

class Delay:
    """Represents the delay distribution for metals emerging from stars.

    Takes the IMF, stellar metal yield function, and stellar lifetime
    function and calculates various the distribution of ejected metal
    mass with respect to stellar mass and stellar lifetime. 
    """

    def __init__(self, metallicity,
                 imfFunc = initialmassfunction.imf_number_Chabrier,
                 yieldClass = yields.Data_GBM,
                 ejections=False,
                 lifetimeFunc = lifetime.main_sequence_life_K97,
                 lowMass = None,
                 highMass = None,
                 minTime = None,
                 maxTime = None,
                 ):
        """Initialize the distribution.
    
        Parameters
        ----------
        
        metallicity: scalar
            metallicity value for the yield calculation.

        imfFunc: function
            should return IMF by number given stellar mass in Msun.

        yieldClass: 
            class encapsulating the yield data (see yields.py)

        lifetimeFunc: function
            should return lifetime of a star given mass in Msun.

        lowMass, highMass: float
            define the interval over which to normalize the distributions.
            Default to the range of masses in yieldClass().mVals

        """
        
        self.metallicity = metallicity
        self.yieldClass = yieldClass
        self.lifetimeFunc = lifetimeFunc
        self.ejections = ejections
        
        ## Stellar mass as a function of lifetime. ##
        self._m_star_func = \
                         lifetime.mass_from_main_sequence_life_function(self.lifetimeFunc)

        ### Initialize some functions that require interpolation grids. ###

        ## Yield function.
        self._yields = yieldClass()
        self._yieldFunc_2d = \
                         self._yields.interpolate_function(ejections=ejections)

        ## Normalization constant (determined later) to turn yieldFunc
        ## * imf into the fraction of carbon produced by stars between
        ## M and M+dm.
        self._fCnorm = 1.0

        ## Convert minTime and maxTime to lowMass and highMass. ##
        if minTime is not None:
            highMass = self.m_star(minTime)
        if maxTime is not None:
            lowMass = self.m_star(maxTime)
        
        ## Choose low and high masses using the range of yield data.##
        self.lowMass = lowMass
        if lowMass is None:
            self.lowMass = numpy.min(self._yieldFunc_2d.mVals)
        self.highMass = highMass
        if highMass is None:
            self.highMass = numpy.max(self._yieldFunc_2d.mVals)

        ## Calculate the equivalent range of stellar lifetimes. ##
        self.maxTime = self.t_life(self.lowMass)
        self.minTime = self.t_life(self.highMass)

        print "Normalizing delay function from %.3g -- %.3g Msun or " %\
        (self.lowMass, self.highMass)
        print "                           from %.3g -- %.3g years." %\
        (self.maxTime, self.minTime)

        ## Normalize the IMF. ##
        self.imfFunc = utils.Normalize(self.lowMass, self.highMass)(imfFunc)
        
        ## Normalize the CCDF to unity at lowMass. ##
        # Use an array of masses because the integration works better
        # when split into pieces.
        marray = []
        if (lowMass is None) and (highMass is None):
            # Use masses at which yield is defined (rather than
            # interpolated).
            marray = self._yieldFunc_2d.mVals
        else:
            # Or just use an array of values covering the given mass range.
            marray = numpy.linspace(self.lowMass, self.highMass, 1000.)
        ccdf = self.metal_mass_CCDF(marray)
        self._fCnorm = ccdf[0]
        print " delay curve normalization = %.3g" % self._fCnorm
        
        mass_limit_error = self.mass_limit_error()
        print "Estimated error from exclusion of m_star > %.3g Msun = %.3g" % \
        (self.highMass, mass_limit_error)
        self.printstats()
        sys.stdout.flush()

#     def dumb_pickle_filter(self, prefix='temp_pic_test'):
#         """Filter out attributes that can't be pickled.
#         """
#         picname = prefix + '.pkl'
#         picfile = open(picname, 'w')
#         for k, v in self.__dict__.items():
#             # Avoid self references (and thereby infinite recurion).
#             if v  is self:
#                 delattr(self, k)
#                 continue
#             # Remove any attributes that can't be pickled.
#             try:
#                 cPickle.dump(v, picfile)
#             except TypeError:
#                 print "Can't pickle: ",
#                 print k, ": ",
#                 print type(v), " "
#                 delattr(self, k)
#         picfile.close()
#         os.remove(picname)

#     def __getstate__(self):
#         """Prepare a state of pickling."""
#         self.dumb_pickle_filter()
#         return self.__dict__        

    def mass_limit_error(self, m_max=None, imf_index=-2.3):
        """Estimate the error incurred by excluding stars above m_max.

        m_max defaults to self.highMass.

        imf_index is the powerlaw index of the imf at high masses.
        """
        if m_max is None:
            m_max = self.highMass

        (exfuncHigh, exfuncLow,
         highSlope, highIntercept, lowSlope, lowIntercept) = \
         yields.extrapolate_function(self._yieldFunc_2d,
                                     self.metallicity,
                                     return_coeffs=True)
        
        phi0 = self.imf(m_max) * (m_max**2.3)
        phi0 *= self._fCnorm
        plustwo = imf_index + 2.
        plusone = imf_index + 1.

        ## The error is simply the integral of the linear yield times
        ## the powelaw IMF from m_max to infinity.
        error = ((highSlope * phi0 * m_max**(plustwo) / (-1.* plustwo)) +
                 (highIntercept * phi0 * m_max**(plusone) / (-1.* plusone)))
        return error

    def metal_mass_dist_postinterp(self, m):
        """Like metal_mass_dist, but with interpolation after IMF
        multiplication.
        """
        if not hasattr(self, '_fCfunc_2d'):
            self._fCfunc_2d = \
                self.yieldClass(interp_args={'weight_function':self.imf},
                                ).interpolate_function(ejections=self.ejections)
        return (self._fCfunc_2d(m, self.metallicity).flatten() / self._fCnorm)

    def metal_mass_dist(self, m):
        """Fraction of total carbon yield from stars of mass m,
        dfC/dm, i.e. the distribution of carbon production as a
        function of stellar mass.
        
        This is weighted by the stellar initial mass function (IMF),
        so that metal_mass_dist(m) * dm is the fraction of carbon
        emitted by stars with masses between m and m+dm.

        Note that this is normalized over the interval from
        self.lowMass to self.highMass.
        """
        dist = (self.imf(m) *
                self._yieldFunc_2d(m, self.metallicity).flatten() /
                self._fCnorm)
        #dist[m<self.lowMass] = None
        #dist[m>self.highMass] = None
        return dist

    def yield_mass(self, m):
        """Carbon yield from a star of mass m.

        This is the total mass of carbon emitted by a single star of
        mass m over its lifetime.
        
        """
        return self._yieldFunc_2d(m, self.metallicity).flatten()

    def metal_mass_CCDF(self, m, highMass=None, method='romberg', **kwargs):
        """Complementary cumulative distribution of metal yield by mass,
        i.e. the fraction of the total metal output produced by stars
        above mass m.

        This is the integral of metal_mass_dist from m to highMass (or
        self.highMass).

        Note that this is normalized over the interval from
        self.lowMass to self.highMass, so values may be negative above
        highMass and greater than one below lowMass.

        kwargs get passed to utils.ccumulate, which may passes them to
        utils.integrate_piecewise, which may pass them to the
        integration routine.

        See also: metal_mass_dist.
        
        """
        if highMass is None:
            highMass = self.highMass
        ccdf = utils.ccumulate(self.metal_mass_dist, m, max=highMass,
                               method=method,
                               **kwargs)
        ccdf[m<self.lowMass] = numpy.nanmax(ccdf)
        ccdf[m>self.highMass] = 0.0
        return ccdf


    def metal_mass_dist_binned(self, m, ccdf=None, **kwargs):
        """Calculate the values of a binned enrichment function.
        
        Each returned value bdf[i] is the integral of the mass dist
        function from m[i-1] to m[i]. bdf[0] is the integral from
        self.lowMass to m[0].
        
        Also returned are delta_m values (the width of each bin in
        units of mass), so the average derivative can be calculated,
        if desired.

        Note that this is normalized over the interval from
        self.lowMass to self.highMass.

        See also: metal_mass_CCDF
        """

        if ccdf is None:
            # Cumulative delay function.
            cdf = 1. - numpy.nan_to_num(self.metal_mass_CCDF(m, **kwargs))
        else:
            cdf = 1. - numpy.nan_to_num(ccdf)
            
        # Break up into bins.
        bdf = numpy.empty(cdf.shape)
        bdf[0] = cdf[0]
        bdf[1:] = cdf[1:] - cdf[:-1]

        # Width of the bins in time.
        dm = numpy.empty(cdf.shape)
        dm[0] = m[0] - self.lowMass
        dm[1:] = m[1:] - m[:-1]
        return bdf, dm

    def metal_time_CDF(self, t, ccdf=None, return_mass=False, **kwargs):
        """The delay CDF of metal yield as a function of stellar age.

        I.e. the fraction of the total metal yield produced by stars
        with a lifetime of t or less.

        Note that this is normalized over the interval from
        self.lowMass to self.highMass, so values may be negative above
        the lifetime corresponding to highMass and greater than one
        below the lowMass lifetime.

        Calculated using metal_mass_CCDF and m_star.

        kwargs get passed to metal_mass_CCDF.
        """
        m = self.m_star(t)
        if ccdf is None:
            cdf = self.metal_mass_CCDF(m, **kwargs)
        else:
            cdf = ccdf

        if return_mass:
            return cdf, m
        return cdf

    def metal_time_dist_binned(self, t, ccdf=None, **kwargs):
        """Calculate the values of a binned delay function.
        
        Each returned value bdf[i] is the integral of the delay
        function from t[i-1] to t[i]. bdf[0] is the integral from
        self.minTime to t[0].
        
        Also returned are delta_t values (the width of each bin in
        units of time), so the average derivative can be calculated,
        if desired.

        Note that this is normalized over the interval from
        self.lowMass to self.highMass.

        See also: metal_time_CDF
        """

        if ccdf is None:
            # Cumulative delay function.
            cdf = numpy.nan_to_num(self.metal_time_CDF(t, **kwargs))
        else:
            cdf = ccdf
        # Break up into bins.
        bdf = numpy.empty(cdf.shape)
        bdf[0] = cdf[0]
        bdf[1:] = cdf[1:] - cdf[:-1]

        # Width of the bins in time.
        dt = numpy.empty(cdf.shape)
        dt[0] = t[0] - self.minTime
        dt[1:] = t[1:] - t[:-1]
        return bdf, dt

    def metal_time_dist_unbinned(self, t, **kwargs):
        """Delay function inferred from metal_time_dist_binned.

        Simply returns the bin values divided by the bin size. Note
        that the first bin is from self.minTime to t[0].

        Notes
        -----
        
        This is the most accurate way to get the distribution of metal
        emission delay times. A more analytical treatment would be to
        convert directly from the continuous metal-mass distribution,
        but this involves the derivative of the stellar lifetime as a
        funtion of mass, which can be numerically problematic.

        Note that this is normalized over the interval from
        self.lowMass to self.highMass.
        """

        bdf, dt = self.metal_time_dist_binned(t, **kwargs)
        df = bdf/dt
        return df, dt

    def imf(self, mass):
        """The initial mass function.

        Note that the normalization may include stars that do not
        contribute to the yield of metals.
        """
        return self.imfFunc(mass)

    def m_star(self, t_life):
        """Stellar mass as a function of lifetime. """
        return self._m_star_func(t_life)

    def t_life(self, m_star):
        """Stellar lifetime as a function of mass. """
        return self.lifetimeFunc(m_star)

    def stats(self, meantol=1e-3):
        """Calculate median and mean delay."""

        npieces = 1./meantol
        tarray = numpy.logspace(numpy.log10(self.minTime),
                                numpy.log10(self.maxTime),
                                npieces)
        marray = self.m_star(tarray)

        ccdf = self.metal_mass_CCDF(marray)
        cdf = 1. - ccdf

        # Find median:
        medindex = numpy.max(numpy.where(ccdf<0.5))
        # Ambiguity of the exact index:
        mediana = tarray[medindex] 
        medianb = tarray[medindex+1]
        medianMassa = marray[medindex]
        medianMassb = marray[medindex+1]

        ## Calculate mean time. ###
        dist, dt = self.metal_time_dist_binned(tarray, ccdf=ccdf)
        tarrayb = numpy.empty(tarray.shape)
        tarrayb[0] = 0
        tarrayb[1:] = tarray[:-1]
        meana = numpy.sum(dist * tarray) / numpy.sum(dist)
        meanb = numpy.sum(dist * tarrayb) / numpy.sum(dist)

        ## Calculate mean mass. ###
        marray = marray[::-1]
        ccdf = ccdf[::-1]
        dist, dm = self.metal_mass_dist_binned(marray, ccdf=ccdf)
        marrayb = numpy.empty(marray.shape)
        marrayb[0] = 0
        marrayb[1:] = marray[:-1]
        meanMassa = numpy.sum(dist * marray) / numpy.sum(dist)
        meanMassb = numpy.sum(dist * marrayb) / numpy.sum(dist)
        return (mediana, (medianb-mediana)/min(mediana,medianb),
                medianMassb, (medianMassa-medianMassb)/min(medianMassb,mediana),
                meana, 
                (meanb-meana)/min(meana,meanb),
                meanMassa,
                (meanMassb-meanMassa)/min(meanMassa,meanMassb))

    def printstats(self):
        median, medianerr, medianMass, medianMasserr, mean, meanerr, meanMass, meanMasserr = self.stats()
        print "Median and mean delays are %.4g and %.4g Gyr" % (median/1e9,
                                                                mean/1e9)
        print "                  (frac. med err  = %.3g)" % medianerr
        print "                  (frac. mean err = %.3g)" % meanerr

        print "   Mass from mean delay is %.3g Msun" % self.m_star(mean)
        print "Median and mean masses are %.4g and %.4g Msun" % (medianMass,
                                                                 meanMass)
        print "                  (frac. med err  = %.3g)" % medianMasserr
        print "                  (frac. mean err = %.3g)" % meanMasserr
        print "   Age from mean mass is %.3g Gyr" % (self.t_life(meanMass)/1e9)

        

class MockDelay(Delay):
    _fCnorm = numpy.nan
    def __init__(self, binmasses, fractions,
                 lifetimeFunc = lifetime.main_sequence_life_K97,
                 maxTime=None,
                 **args):
        """ Mimics the Delay class.
        """

        self.lifetimeFunc = lifetimeFunc
        ## Stellar mass as a function of lifetime. ##
        self._m_star_func = \
             lifetime.mass_from_main_sequence_life_function(self.lifetimeFunc)


        if maxTime is not None:
            lowMass = self.m_star(maxTime)
            print "Resetting low mass limit to %.3g (from %.3g)." % \
                  (lowMass, binmasses[0])
            binmasses[0] = lowMass

        if not sum(fractions) == 1:
            norm = 1./sum(fractions)
            print "Normalizing fractions with a factor of %.3g" % norm
            fractions = fractions * norm

        assert len(fractions) == len(binmasses) - 1,\
                  "len(fractions) must be one less than len(splitmasses)."
        assert numpy.all(numpy.sort(binmasses) == binmasses),\
                  "binmasses must be ordered from low to high."

        self.binmasses = binmasses
        self.fractions = fractions

        ## Choose low and high masses using the range of yield data.##
        self.lowMass = numpy.min(self.binmasses)
        self.highMass = numpy.max(self.binmasses)

        self.minTime = self.t_life(self.highMass)
        self.maxTime = self.t_life(self.lowMass)
        self.printstats()
        sys.stdout.flush()

    def metal_mass_dist(self, m):
        # Cumulative delay function.
        ccdf = numpy.nan_to_num(self.metal_mass_CCDF(m))
        cdf = numpy.nanmax(ccdf) - ccdf

        # Break up into bins.
        bdf = numpy.empty(cdf.shape)
        bdf[0] = cdf[0]
        bdf[1:] = cdf[1:] - cdf[:-1]

        # Width of the bins in time.
        dm = numpy.empty(cdf.shape)
        dm[0] = m[0] - self.lowMass
        dm[1:] = m[1:] - m[:-1]

        df = bdf/dm
        return df

    def metal_mass_CCDF(self, mass):
        ccdf = numpy.zeros(mass.shape)

        # Step through the fraction indices in reverse
        for ibin in range(len(self.fractions)-1, -1, -1):
            # Add the current fraction to all lower bins.
            ccdf[mass < self.binmasses[ibin]] += self.fractions[ibin]

            # And add the correct portion of the current fraction to
            # points in the current bin.
            binmask = numpy.logical_and(mass >= self.binmasses[ibin],
                                        mass < self.binmasses[ibin+1])
            binwidth = self.binmasses[ibin+1] - self.binmasses[ibin]
            ccdf[binmask] += (self.fractions[ibin] *
                              (self.binmasses[ibin+1] - mass[binmask]) /
                              binwidth)
        return ccdf
    def metal_mass_dist_postinterp(self, m):
        return self.metal_mass_dist(m)

    def imf(self, mass):
        return numpy.nan * mass

    def yield_mass(self, m):
        return numpy.nan * m

def test_plot_binned_delay_function():
    ddist = Delay(0.02)
    t = numpy.arange(0., 5.e10, 1.33e6)
    bdf, dt = ddist.metal_time_dist_binned(t)
    cbdf = numpy.cumsum(bdf)
    print "Sum of binned function = ", numpy.sum(bdf)
    pylab.subplot(221)
    pylab.plot(t, t * bdf/dt,'.', alpha=0.2)
    pylab.subplot(222)
    pylab.plot(t, cbdf,'.', alpha=0.2)

def test_plot_delay_function(dlogt=1e-3):
    Z = 0.02
    ftfunc = delay_function(0.02)
    t = 10**numpy.arange(6.0,10.2,dlogt)
    dt = t[1:] - t[:-1]
    ft = ftfunc(t[:-1], dt)

    ft2 = ft
    dlnt = numpy.log(t[1:]) - numpy.log(t[:-1])
    ft2[numpy.logical_not(numpy.isfinite(ft))] = 0
    ftcum = scipy.integrate.cumtrapz(ft2 * dt)
    
    lifetimeFunc = lifetime.main_sequence_life_K97
    m_star_func = \
        lifetime.mass_from_main_sequence_life_function(lifetimeFunc,
                                                       logmmin=10.)
    m = m_star_func(t[:-1])
    dmdt = (m - m_star_func(t[:-1] + dt))/dt

#    print 'ft=',ft
    Z = 0.02

    import pylab
    #pylab.figure()
    pylab.subplot(221)
    pylab.plot(t[:-1], t[:-1]*ft, alpha=0.7)
    ax = pylab.gca()
    ax.set_xscale('log')
    pylab.xlim(t.min(), t.max(), alpha=0.7)

    #pylab.twinx()
    pylab.subplot(222)
    pylab.plot(t[1:-1], ftcum, alpha=0.7)
    if dlogt > 1e-2:
        pylab.plot(t[1:-1], ftcum, '.', alpha=0.7)
    pylab.plot(t[1:-1], 1.0 - ftcum, alpha=0.7)
    pylab.axhline(y=0.5)
    ax = pylab.gca()
    ax.set_xscale('log')
    pylab.xlim(t.min(), t.max(), alpha=0.7)
    #pylab.gca().set_yscale('log')

    pylab.subplot(223)
    pylab.plot(t[:-1], m, alpha=0.7)
    pylab.gca().set_xscale('log')
    pylab.xlim(t.min(), t.max(), alpha=0.7)
    pylab.gca().set_yscale('log')
    pylab.axhline(y=100.0)
    #pylab.twinx()

    pylab.subplot(224)
    pylab.plot(t[:-1], dmdt, alpha=0.7)
    pylab.gca().set_xscale('log')
    pylab.gca().set_yscale('log')
    pylab.xlim(t.min(), t.max(), alpha=0.7)

def test_plot_converge_delay_function():
    test_plot_delay_function(1.e-4)
    test_plot_delay_function(1.e-3)
    test_plot_delay_function(1.e-2)
    #test_plot_binned_delay_function()

def plot_delay_function(metallicity,
                        imfFunc = initialmassfunction.imf_number_Chabrier,
                        yieldClass = yields.Data_GBM,
                        lifetimeFunc = lifetime.main_sequence_life_K97):
                        #lifetimeFunc = lifetime.main_sequence_life_MM89):
    """ Plot delay function, yield, IMF, and lifetime.
    
    """

    dm = 0.01
    mass = numpy.arange(0, 110., dm)

    time = 10**numpy.arange(6., 10.5, 0.01)
    dt = time[1:] - time[:-1]
    time = time[1:]

    m_star_func = \
        lifetime.mass_from_main_sequence_life_function(lifetimeFunc)
    mass_from_time = m_star_func(time)

    imf = imfFunc(mass)
    
    mCfunc_weighted = yieldClass(weight_function=imfFunc)
    mCfunc_noweight = yieldClass()

    mC_weighted = mCfunc_weighted(mass, metallicity)
    mC_noweight = mCfunc_noweight(mass, metallicity)

    tLife = lifetimeFunc(mass)
    dmdt = (mass[1:] - mass[:-1])/(tLife[1:] - tLife[:-1])

    ddist = Delay(0.02,
                  imfFunc = imfFunc,
                  yieldClass = yieldClass,
                  lifetimeFunc = lifetimeFunc)
    fdelay = ddist.metal_time_dist_unbinned(time)
    fdelay[numpy.isnan(fdelay)] = 0.0
    cumfdelay = numpy.cumsum(fdelay * dt)

    print mass_from_time
    print "Min time = %.3g" % min(time[numpy.isfinite(cumfdelay)])
    print "Max time = %.3g " % max(time[numpy.isfinite(cumfdelay)])
    print "Max mass = %.5g" % numpy.nanmax(mass_from_time[numpy.isfinite(cumfdelay)])
    print "Min mass = %.5g" % numpy.nanmin(mass_from_time[numpy.isfinite(cumfdelay)])
    print "Min cdf = %.5g " % min(cumfdelay[numpy.isfinite(cumfdelay)])
    print "Max cdf = %.5g" % max(cumfdelay[numpy.isfinite(cumfdelay)])

    delay_fig = pylab.figure(figsize=(11,6))
    delay_fig.set_label('vsMass')

    pylab.subplot(221)
    pylab.plot(mass, imf)
    pylab.xlabel('mass (Msun)')
    pylab.ylabel('IMF')
    pylab.gca().set_yscale('log')

    pylab.subplot(222)
    pylab.plot(mass, mC_noweight, label='unweighted', ls='--')
    pylab.xlabel('mass (Msun)')
    pylab.ylabel(r'$m_C$ unweighted (--)')
    pylab.gca().set_yscale('log')
    #pylab.legend(loc='best')
    
    pylab.twinx()
    pylab.plot(mass, mC_weighted, label='IMF weighted', ls=':')
    pylab.ylabel(r'$m_C$ weighted (:)')
    pylab.gca().set_yscale('log')
    #pylab.legend(loc='best')

    pylab.subplot(223)
    pylab.plot(mass, tLife)
    pylab.xlabel('mass (Msun)')
    pylab.ylabel('tLife (years)')
    pylab.gca().set_yscale('log')

    pylab.twinx()
    pylab.plot(mass[1:], -1. * dmdt, label='-dmdt', ls=':')
    pylab.gca().set_yscale('log')
    pylab.ylabel(r'dmdt(:)')

    pylab.subplot(224)
    pylab.plot(mass_from_time, fdelay)
    pylab.gca().set_yscale('log')
    pylab.xlabel('mass (Msun)')
    pylab.ylabel('fdelay')

    pylab.twinx()
    pylab.plot(mass_from_time, cumfdelay)
    pylab.ylabel('cumfdelay')
    pylab.gca().set_yscale('log')
    pylab.xlim(numpy.min(mass), numpy.max(mass))

    pylab.subplots_adjust(wspace=0.5, hspace=0.29)

    delay_fig_time = pylab.figure(figsize=(11,6))
    delay_fig_time.set_label('vsTime')

    pylab.subplot(221)
    pylab.plot(tLife, imf)
    pylab.xlabel('tLife (yr)')
    pylab.ylabel('IMF')
    pylab.gca().set_yscale('log')
    pylab.gca().set_xscale('log')

    pylab.twinx()
    pylab.plot(tLife[1:], -1. * tLife[1:] * imf[1:] * dmdt, ls=':')
    pylab.ylabel('IMF * t * -dm/dt')
    pylab.gca().set_yscale('log')

    pylab.subplot(222)
    pylab.plot(tLife, mC_noweight, label='unweighted', ls='--')
    pylab.xlabel('tLife (yr)')
    pylab.ylabel(r'$m_C$ unweighted (--)')
    pylab.gca().set_yscale('log')
    pylab.gca().set_xscale('log')
    #pylab.legend(loc='best')
    
    pylab.twinx()
    pylab.plot(tLife, mC_weighted, label='IMF weighted', ls=':')
    pylab.ylabel(r'$m_C$ weighted (:)')
    pylab.gca().set_yscale('log')
    #pylab.legend(loc='best')

    pylab.subplot(223)
    pylab.plot(tLife, mass)
    pylab.xlabel('tLife (yr)')
    pylab.ylabel('mass (Msun)')
    pylab.gca().set_yscale('log')
    pylab.gca().set_xscale('log')

    pylab.twinx()
    pylab.plot(tLife[1:], -1. * dmdt, label='-dmdt', ls=':')
    pylab.gca().set_yscale('log')
    pylab.ylabel(r'-dmdt(:)')

    pylab.subplot(224)
    pylab.plot(time, time * fdelay)
    #pylab.gca().set_yscale('log')
    pylab.gca().set_xscale('log')
    pylab.xlabel('tLife (years)')
    pylab.ylabel('t * fdelay')
    pylab.subplots_adjust(wspace=0.5, hspace=0.29)

    pylab.twinx()
    pylab.plot(time, cumfdelay)
    pylab.ylabel('cumfdelay')
    #pylab.gca().set_yscale('log')

def test_Delay(plots=True, skip_asserts=False, delayfunc=None):
    import numpy.testing.utils as ntest
    
    maxTime = delayfunc.maxTime
    minTime = delayfunc.minTime

    # Set up mass and time coordinates.
    npoints = 1000
    #dmass = 0.01
    #mass = numpy.arange(delayfunc.lowMass, delayfunc.highMass + dmass, dmass)
    mass, dmass = numpy.linspace(delayfunc.lowMass, delayfunc.highMass, npoints,
                          retstep=True)
    
    #dlogt = 0.01
    #time = 10**numpy.arange(numpy.log10(minTime), numpy.log10(maxTime) + dlogt,
    #                        dlogt)
    #time = numpy.logspace(numpy.log10(minTime), numpy.log10(maxTime), npoints)
    time = numpy.linspace(0, maxTime, npoints)

    mass_from_time = delayfunc.m_star(time)
    time_from_mass = delayfunc.t_life(mass)
    
    # Calculate delay-function-related quantities.
    metal_mass_dist = delayfunc.metal_mass_dist(mass)
    metal_mass_dist_postinterp = delayfunc.metal_mass_dist_postinterp(mass)
    metal_mass_CCDF = delayfunc.metal_mass_CCDF(mass)
    metal_time_CDF = delayfunc.metal_time_CDF(time)
    metal_time_CDF_time_from_mass = delayfunc.metal_time_CDF(time_from_mass)
    metal_time_dist_binned, dtime_bin = delayfunc.metal_time_dist_binned(time)
    metal_time_dist_unbinned, dtime_ubin = \
                              delayfunc.metal_time_dist_unbinned(time)

    imf = delayfunc.imf(mass)
    yield_mass = delayfunc.yield_mass(mass)

    ### Test some relationships between the quantities. ###
    import scipy.integrate

    # Do a simple integral over the PDF:
    test_mass_CCDF = scipy.integrate.cumtrapz((dmass *
                                               metal_mass_dist)[::-1])[::-1]
    diff_mass_CCDF = test_mass_CCDF - metal_mass_CCDF[:-1]

    sum_metal_time_dist = \
                        numpy.cumsum(numpy.nan_to_num(metal_time_dist_binned))

    test_time_CDF = numpy.cumsum(metal_time_dist_unbinned * dtime_ubin)

    test_metal_mass_dist = imf * yield_mass / delayfunc._fCnorm
    
    if not skip_asserts:

        ### The CDF should be the integral of the metal dist. ###

        # The CDF should be the integral of the metal dist:
        ntest.assert_almost_equal(metal_mass_CCDF[:-1], test_mass_CCDF, 2)

        # The metal time CDF and metal mass CCDF should be
        # equal. (This could be 10 digits rather than four, except for
        # a small hicup in the time from mass conversion.)#
        ntest.assert_almost_equal(metal_time_CDF_time_from_mass,
                                  metal_mass_CCDF, 4)

        ### The CCDF and CDF should be close one at lowMass. ###
        marray = numpy.linspace(delayfunc.lowMass, delayfunc.highMass, 100)
        ccdf = delayfunc.metal_mass_CCDF(marray)
        cdf = delayfunc.metal_time_CDF(delayfunc.t_life(marray))
        ntest.assert_approx_equal(ccdf[0], 1., 4)
        ntest.assert_approx_equal(cdf[0], 1., 4)

        # The sum of the binned dist should be one over the norm
        # range.
        ntest.assert_approx_equal(sum_metal_time_dist[-1], 1., 4)

        ### The sum of the binned dist should be equal to the CDF ###
        ntest.assert_almost_equal(\
            numpy.cumsum(numpy.nan_to_num(metal_time_dist_binned)),
                         numpy.nan_to_num(metal_time_CDF))

        ### The integral of the time dist should be equal to the CDF ###
        ntest.assert_almost_equal(numpy.nan_to_num(metal_time_CDF),
                                  test_time_CDF, 10)

        ### The product of IMF and yield should be proportional to the metal
        ### mass distribution. ###
        ntest.assert_almost_equal(metal_mass_dist, test_metal_mass_dist, 10)

        ### The metal mass dist should be similar if we interpolate before
        ### or after the IMF multiplication. ###
        ntest.assert_almost_equal(metal_mass_dist,
                                  metal_mass_dist_postinterp, 2)

    ### Make some plots ###

    if plots:
        pylab.figure().set_label('metal_mass_CCDF')

        pylab.subplot(221)
        pylab.plot(mass[:-1], test_mass_CCDF, label="test")
        pylab.plot(mass, metal_mass_CCDF, label="CCDF")
        pylab.legend(loc='best')
        pylab.axvline(x=delayfunc.lowMass)
        pylab.axvline(x=delayfunc.highMass)
        pylab.twinx()
        pylab.plot(mass, metal_mass_dist, ':', label="PDF")

        pylab.subplot(222)
        pylab.plot(mass[:-1], diff_mass_CCDF, label="test diff")
        pylab.legend(loc='best')
        pylab.twinx()
        pylab.plot(mass, metal_mass_dist, ":", label="PDF")

        pylab.subplot(223)
        pylab.plot(time_from_mass, metal_mass_CCDF, label="CCDF")
        pylab.xscale('log')
        pylab.legend(loc='best')

        pylab.subplot(224)
        pylab.plot(mass, metal_mass_dist, ':', label="PDF")
        pylab.axvline(x=delayfunc.lowMass)
        pylab.axvline(x=delayfunc.highMass)
        pylab.legend(loc='best')

        pylab.figure().set_label('metal_mass_PDF')
        pylab.subplot(211)
        pylab.plot(mass, metal_mass_dist, ':', label="PDF")
        pylab.plot(mass, metal_mass_dist_postinterp, '-.', label="postinterp")
        pylab.plot(mass, test_metal_mass_dist, '--', label="~ IMF * yield")
        pylab.axvline(x=delayfunc.lowMass)
        pylab.axvline(x=delayfunc.highMass)
        pylab.legend(loc='best')
        pylab.subplot(212)
        pylab.plot(mass, metal_mass_dist -
                   test_metal_mass_dist, ':', label="PDF - c*(IMF * yield)")
        pylab.plot(mass, metal_mass_dist -
                   metal_mass_dist_postinterp, '-', label="PDF - postinterp")
        pylab.axvline(x=delayfunc.lowMass)
        pylab.axvline(x=delayfunc.highMass)
        pylab.legend(loc='best')

        pylab.figure().set_label('mass_time')
        pylab.subplot(211)
        pylab.plot(time, mass_from_time, label="mass from time")
        pylab.plot(time_from_mass, mass, label="time from mass")
        pylab.xscale('log')
        pylab.subplot(212)
        pylab.plot(mass_from_time, time, label="mass from time")
        pylab.plot(mass, time_from_mass, label="time from mass")
        pylab.yscale('log')
        pylab.legend(loc='best')
                
        pylab.figure().set_label('metal_time_CDF')

        pylab.subplot(221)
        pylab.plot(time, metal_time_CDF, label="CDF")
        pylab.plot(time, sum_metal_time_dist, label="sum of binned")
        pylab.plot(time, test_time_CDF, label="int of unbinned")
        pylab.legend(loc='best')
        pylab.xscale('log')
        pylab.axvline(x=minTime)
        pylab.axvline(x=maxTime)

        pylab.subplot(222)
        pylab.plot(mass_from_time, metal_time_CDF, label="CDF")
        pylab.plot(mass_from_time, sum_metal_time_dist, label="sum of binned")
        pylab.plot(mass_from_time, test_time_CDF, label="int of unbinned")
        pylab.axvline(x=delayfunc.lowMass)
        pylab.axvline(x=delayfunc.highMass)
        pylab.legend(loc='best')

        pylab.subplot(223)
        pylab.plot(time, sum_metal_time_dist - metal_time_CDF, label="sum-CDF")
        pylab.plot(time, test_time_CDF - metal_time_CDF, label="int-CDF")
        pylab.legend(loc='best')
        pylab.xscale('log')
        pylab.axvline(x=minTime)
        pylab.axvline(x=maxTime)

        pylab.subplot(224)
        pylab.plot(mass_from_time, sum_metal_time_dist - metal_time_CDF,
                   label="sum-CDF")
        pylab.plot(mass_from_time,
                   test_time_CDF - metal_time_CDF, label="int-CDF")
        pylab.legend(loc='best')
        pylab.axvline(x=delayfunc.lowMass)
        pylab.axvline(x=delayfunc.highMass)


        pylab.figure().set_label('metal_time_PDF_bin')
        pylab.plot(time, metal_time_dist_binned, label="binned")
        pylab.xscale('log')
        pylab.axvline(x=minTime)
        pylab.axvline(x=maxTime)
        pylab.legend(loc='best')

        pylab.figure().set_label('metal_time_PDF_unbin')
        pylab.plot(time, metal_time_dist_unbinned, label="unbinned")
        pylab.xscale('log')
        pylab.legend(loc='best')

        pylab.figure().set_label('CDF_CCDF')
        pylab.subplot(211)
        pylab.plot(mass, metal_mass_CCDF, label='mass CCDF')
        pylab.plot(mass, metal_time_CDF_time_from_mass, label='time CDF')
        pylab.legend(loc='best')
        pylab.subplot(212)
        pylab.plot(mass, metal_time_CDF_time_from_mass - metal_mass_CCDF, ':',
                   label='CDF - CCDF')
        pylab.legend(loc='best')


def compare_dist_times():
    import time
    timelist = []
    names = []

    names.append('init')
    timelist.append(time.time())
    delayfunc = Delay(0.02)

    names.append('linspace')
    timelist.append(time.time())
    m = numpy.linspace(delayfunc.lowMass, delayfunc.highMass, 1e3)

    names.append('postinterp')
    timelist.append(time.time())
    delayfunc.metal_mass_dist_postinterp(m)

    names.append('dist')
    timelist.append(time.time())
    delayfunc.metal_mass_dist(m)

    names.append('ccumulate dist')
    timelist.append(time.time())
    utils.ccumulate(delayfunc.metal_mass_dist, m, max=delayfunc.highMass)

    names.append('ccumulate dist romberg')
    timelist.append(time.time())
    utils.ccumulate(delayfunc.metal_mass_dist, m, max=delayfunc.highMass,
                    method='romberg')

    names.append('ccumulate postintepr')
    timelist.append(time.time())
    utils.ccumulate(delayfunc.metal_mass_dist_postinterp, m,
                    max=delayfunc.highMass)

    names.append('ccumulate postintepr romberg')
    timelist.append(time.time())
    utils.ccumulate(delayfunc.metal_mass_dist_postinterp, m,
                    max=delayfunc.highMass,
                    method='romberg')

    timelist.append(time.time())
    timearray = numpy.asarray(timelist)

    time_diffs = 1e3 * (timearray[1:] - timearray[:-1])

    for diff, name in zip(time_diffs, names):
        print "%s: %.4g ms" % (name, diff)

def run_profs():
    import cProfile
    cProfile.run('delayfunc = Delay(0.02)', 'Delay.prof')
    cProfile.run('m = numpy.linspace(delayfunc.lowMass, delayfunc.highMass, 1e3)')

    cProfile.run("""utils.ccumulate(delayfunc.metal_mass_dist, m, max=delayfunc.highMass)""", 'cumulate_dist.prof')

    cProfile.run("""utils.ccumulate(delayfunc.metal_mass_dist_postinterp, m, max=delayfunc.highMass)""", 'cumulate_dist_postinterp.prof')


if __name__ == '__main__':
    import matplotlib.pyplot as pylab
    import os
    import sys
    
    ### Argument parsing. ###
    if len(sys.argv)==1:
        print "Run with a filename argument to produce image files, e.g.:"
        print " python delay.py delay.png"
        print " python delay.py delay.eps"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        prefix, extension = os.path.splitext(filename)
    else:
        filename = None

    ### Main code area. ###

    #run_profs()
    #compare_dist_times()

    test_Delay(skip_asserts=False,
               delayfunc = Delay(0.02, maxTime=2e9, ejections=False))
    test_Delay(skip_asserts=False,
               delayfunc = Delay(0.02, maxTime=2e9, ejections=True))
         
    ##test_Delay(skip_asserts=False, delayfunc = Delay(0.02))

    #test_Delay(skip_asserts=True,
    #           delayfunc = Delay(0.0001, maxTime=2e9,
    #                             yieldClass = yields.Data_H))

    #bins = numpy.array([1.4, 5, 100.])
    #fracs = numpy.array([0.5, 0.5])
    #test_Delay(skip_asserts=True, delayfunc = MockDelay(bins, fracs))


    #test_plot_converge_delay_function()
    #plot_delay_function(0.02)
    #test_plot_binned_delay_function()

    ### Plot output code. ###
    if filename is None:
        pylab.show()
    else:
        from matplotlib import _pylab_helpers
        for manager in _pylab_helpers.Gcf.get_all_fig_managers():
            fig = manager.canvas.figure
            if len(fig.get_label()) > 0:
                label = fig.get_label()
            else:
                label = '_Fig' + str(manager.num)
            newfilename = prefix + '_' + label + extension
            fig.savefig(newfilename, bbox_inches="tight")


