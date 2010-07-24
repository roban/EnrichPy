import warnings
import math
import pickle

import numpy
import scipy
import scipy.integrate
import scipy.interpolate

import cosmolopy.distance as cd
import cosmolopy.constants as cc

class NullWriter:
    """Dummy file-like object that does nothing.

    From
    http://stackoverflow.com/questions/1809958/hide-stderr-output-in-unit-tests
    """
    def write(self, s):
        pass

def loadSaveable(filename):
    """Return an instance of an object unpickled from a file.
    """
    picfile = open(filename)
    loaded = pickle.load(picfile)
    picfile.close()
    return loaded

class Saveable(object):
    """An object with methods to save and restore.

    Unpickleable attributes will simply be deleted from the object.
    """

    def __init__(self, filename=None):
        if filename is not None:
            self.load(filename)
                  
    def save(self, filename):
        """Save object to a file."""
        picfile = open(filename, 'w')
        pickle.dump(self, picfile)
        picfile.close()

    def load(self, filename):
        """Return an instance of an object unpickled from a file.
        """
        picfile = open(filename)
        loaded = pickle.load(picfile)
        picfile.close()
        self.__dict__.update(loaded.__dict__)
        return self

    def dumb_pickle_filter(self):
        """Filter out attributes that can't be pickled.

        Returns a copy of dict with 
        """
        picfile = NullWriter()
        sdict = self.__dict__.copy()
        for k, v in sdict.items():
            # Avoid self references (and thereby infinite recurion).
            if v is self:
                del sdict[k]
                continue
            # Remove any attributes that can't be pickled.
            try:
                pickle.dump(v, picfile)
            except (TypeError, pickle.PicklingError) as err:
                if hasattr(self, 'verbose') and self.verbose:
                    print "Won't pickle", k, type(v), ": "
                    print "'", err, "'"
                del sdict[k]
        return sdict
    def __getstate__(self):
        """Prepare a state of pickling."""
        return self.dumb_pickle_filter()

    def __setstate__(self, dict):
        """Unpickle."""
        self.__dict__.update(dict)

class AgeSpacedRedshift(Saveable):
    """Set up uniform time array and corresponding redshift array.
    """
    def __init__(self, z_min, z_max, dt_yr=2e6, **cosmo):
        self.z_min = z_min
        self.z_max = z_max
        self.dt_yr = dt_yr
        self.cosmo = cosmo
        
        self.agefunc, self.redshiftfunc, e_f, e_t  = \
                      cd.quick_age_function(zmax=1.1 * z_max, 
                                            zmin=z_min-0.05, 
                                            zstep=0.01,
                                            logspacing=True,
                                            return_inverse=True,
                                            **cosmo)
        self.tmax = self.agefunc(z_min)
        self.tmin = self.agefunc(z_max)
        self.dt = self.dt_yr * cc.yr_s
        self.t = numpy.arange(self.tmin, self.tmax + 1.01 * self.dt, self.dt)
        self.t_yr = self.t / cc.yr_s
        self.z = self.redshiftfunc(self.t)
        print " Using %i points in t, dt = %.3g yr." % (len(self.t_yr),
                                                        self.dt_yr)

    def age_Gyr(self, z):
        return self.agefunc(z)/cc.yr_s/1e9
    def __setstate__(self, dict):
        """Unpickle."""
        self.__dict__.update(dict)
        self.agefunc, self.redshiftfunc, e_f, e_t  = \
                      cd.quick_age_function(zmax=1.1 * self.z_max, 
                                            zmin=self.z_min-0.05, 
                                            zstep=0.01,
                                            logspacing=True,
                                            return_inverse=True,
                                            **self.cosmo)
        
        
class Extrapolate1d:
    """Interpolate/Extrapolate 1d data.
    """

    def linear_coefficients(self, x=None, y=None, slope=None,
                            match_index=0):
        if x is None:
            x = self.x
        if y is None:
            y=self.y
        if slope is None:
            slope = (y[0]-y[-1]) / (x[0] - x[-1])
        intercept = y[match_index] - slope * x[match_index]
        return slope, intercept

    def __init__(self, x, y,
                 bounds_behavior=['extrapolate', 'extrapolate'],
                 slopes=[None, None],
                 npoints = [2, 2],
                 **interpargs):
        """

        Parameters
        ----------

        x, y:

          sequences of data. Will be sorted by x value before use.

        bound_behavior:

          length-2 sequence specifying behavior below the lower and
          above the upper boungs of the data, respectively. Each
          element can be 'extrapolate', 'constant', or a numerical
          value.

        npoints:

          Linear extrapolation uses the slope between x[0] and
          x[npoints-1] or x[-npoints] and x[-1]. Note: this is not a
          linear fit over that range. It Ignores points within the
          interval

        interpargs:

          Extra keywords passed to scipy.interpolate.interp1d.

        """
        order = numpy.argsort(numpy.nan_to_num(x))
        self.x = x[order]
        self.y = y[order]

        self.bounds_behavior = bounds_behavior
        
        self._interpfunc = scipy.interpolate.interp1d(self.x, self.y,
                                                      **interpargs)

        if self.bounds_behavior[1] == 'constant':
            self._exfuncHigh = lambda x1: self.y[-1]
        elif self.bounds_behavior[1] == 'extrapolate':
            n1 = npoints[1]
            highSlope, highIntercept = self.linear_coefficients(self.x[-n1:],
                                                                self.y[-n1:],
                                                                slope=slopes[1],
                                                                match_index=-1)
            self._exfuncHigh = lambda x1: x1 * highSlope + highIntercept
            self.highSlope = highSlope
            self.highIntercept = highIntercept
        else:
            self._exfuncHigh = lambda x1: self.bounds_behavior[1]

        if self.bounds_behavior[0] == 'constant':
            self._exfuncLow = lambda x1: self.y[0]
        elif self.bounds_behavior[0] == 'extrapolate':
            n0 = npoints[0]
            lowSlope, lowIntercept = self.linear_coefficients(self.x[:n0],
                                                              self.y[:n0],
                                                              slope=slopes[0],
                                                              match_index=0)
            self._exfuncLow = lambda x1: x1 * lowSlope + lowIntercept
            self.lowSlope = lowSlope
            self.lowIntercept = lowIntercept
        else:
            self._exfuncLow = lambda x1: self.bounds_behavior[0]

    def extrap_string(self):
        extstr = ""
        if hasattr(self, 'lowSlope'):
            extstr += "y = %g x + %g for x <= %g" % (self.lowSlope,
                                                     self.lowIntercept,
                                                     self.x[0])
        if hasattr(self, 'highSlope'):
            if hasattr(self, 'lowSlope'):
                extstr += "\n"
            extstr += "y = %g x + %g for x >= %g" % (self.highSlope,
                                                     self.highIntercept,
                                                     self.x[-1])
        return extstr

    def __call__(self, x1):
        if numpy.isscalar(x1) or x1.shape==():
            if x1 <= self.x[0]:
                return self._exfuncLow(x1)
            elif x1 >= self.x[-1]:
                return self._exfuncHigh(x1)
            else:
                return self._interpfunc(x1)
        lowmask = x1 <= self.x[0]
        highmask = x1 >= self.x[-1]
        inmask = numpy.logical_not(numpy.logical_or(lowmask,highmask))
        if numpy.all(inmask):
            return self._interpfunc(x1)
        
        y1 = numpy.empty(x1.shape)
        y1[inmask] = self._interpfunc(x1[inmask])
        y1[lowmask] = self._exfuncLow(x1[lowmask])
        y1[highmask] = self._exfuncHigh(x1[highmask])

        return y1

class PiecewisePowerlaw(object):
    """A piecewise powerlaw function.

    You can specify the intervals and power indices, and this class
    will figure out the coefficients needed to make the function
    continuous and normalized to unit integral.

    Notes
    -----

    Intervals are defined by an array l

    Powerlaw indicies by and array p

    a_n are the coefficients.
    
    f(x) = a_n x^{p_n} for l_{n-1} <= x < l_n

    Recursion relation for continuity:

    a_n = a_{n-1} l_n^{p_{n-1} - p_n}

    Integral of a piece:

    I_n = a_n p_n (l_{n+1}^{p_n - 1} - l_n^{p_n - 1})

    Total integral:

    I_tot = Sum_0^N I_n

    """

    def __init__(self, limits, powers,
                 coefficients=None,
                 externalval=0.0,
                 norm=True):
        """Defined a piecewise powerlaw.

        If coefficients is None then the coefficients are determined
        by requiring the function to be continuous and normalized to
        an integral of one.

        The function is composed of N powerlaws, where N = len(powers).

        len(limits) must be one greated than len(powers)

        Parameters
        ----------

        limits: array (length n+1)
            boundaries of the specified powerlaws. Must be one greater in
            length than coefficents and powers. Specify -numpy.infty for
            the first limit or numpy.infty for the last limit for
            unbounded powerlaws.

        coefficients: optional array (length n)
            values of the coefficient a_i

        powers: array (length n)
            values of the powerlaw indices p_i

        externalval: scalar
            Value to return outside the defined domain. None
            correspons to 'NaN'.

        norm: boolean
            Whether to normalize the integral of the function over the
            defined domain to unity.

        The resulting function takes a single, one-dimensional array of
        values on which to operate.

        """

        limits = numpy.atleast_1d(limits)
        powers = numpy.atleast_1d(powers)

        if not len(limits) == len(powers)+1:
            raise ValueError("limits must be one longer than powers.")

        if coefficients is None:
            coefficients = numpy.ones(len(powers))

            # Leaving a_0 = 1, apply the recurence relation.
            for n in range(1,len(powers)):
                coefficients[n] = (coefficients[n-1] *
                                   limits[n]**(powers[n-1] - powers[n]))
        else:
            coefficients = numpy.atleast_1d(coefficients)
            if not len(coefficients) == len(powers):
                raise ValueError("coefficients and powers must be"+
                                 " the same length.")

        # Find the integral of each piece.
        integrals = ((coefficients / (powers + 1.)) *
                     (limits[1:]**(powers + 1.) -
                      limits[:-1]**(powers + 1.)))
        if norm:
            # The total integral over the function.
            integralTot = numpy.sum(integrals)
            
            coefficients = coefficients / integralTot
            integrals = integrals /  integralTot

        for array in [limits, coefficients, powers]:
            if array.ndim > 1:
                raise ValueError("arguments must be a 1D arrays or scalars.")
        self._integrals = integrals
        self._limits = limits.reshape((-1,1))
        self._coefficients = coefficients.reshape((-1,1))
        self._powers = powers.reshape((-1,1))
        self._externalval = externalval
    
    def __call__(self, x):
        """Evaluate the powerlaw at values x.
        """
        x = numpy.atleast_1d(x)
        if x.ndim > 1:
            raise ValueError("argument must be a 1D array or scalar.")
        y = numpy.sum((self._coefficients * x**self._powers) *
                      (x >= self._limits[0:-1]) * (x < self._limits[1:]),
                      axis=0)
        y[x < self._limits[0]] = self._externalval
        y[x >= self._limits[-1]] = self._externalval
        return y

    def integrate(self, low, high, weight_power=None):
        """Integrate the function from low to high.

        Optionally weight the integral by x^weight_power.

        """
        limits = self._limits.flatten()
        coefficients = self._coefficients.flatten()
        powers = self._powers.flatten()
        if weight_power is not None:
            powers += weight_power
            # Integral of each piece over its domain.
            integrals = ((coefficients / (powers + 1.)) *
                         (limits[1:]**(powers + 1.) -
                          limits[:-1]**(powers + 1.)))
        else:
            integrals = self._integrals
        
        pairs = numpy.broadcast(low, high)
        integral = numpy.empty(pairs.shape)
        for (i, (x0, x1)) in enumerate(pairs):
            # Sort the integral limits.
            x0, x1 = list(numpy.sort([x0,x1]))

            # Select the pieces contained entirely in the interval.
            mask = numpy.logical_and(x0 < limits[:-1],
                                     x1 >= limits[1:]).flatten()
            indices = numpy.where(mask)
            if not numpy.any(mask):
                integral.flat[i] = 0

                # If the interval is outside the domain.
                if x0 > limits[-1] or x1 < limits[0]:
                    integral.flat[i] = 0
                    continue

                # Find out if any piece contains the entire interval:
                containedmask = numpy.logical_and(x0 >= limits[:-1],
                                                  x1 < limits[1:])
                # Three possibilites:
                if numpy.any(containedmask):
                    # The interval is contained in a single segment.
                    index = numpy.where(containedmask)[0][0]
                    integral.flat[i] = ((coefficients[index] /
                                         (powers[index] + 1.)) *
                                        (x1**(powers[index] + 1.) -
                                         x0**(powers[index] + 1.)))
                    continue
                elif x1 >= limits[0] and x1 < limits[1]:
                    # x1 is in the first segment.
                    highi = 0
                    lowi = -1
                elif x0 < limits[-1] and x0 >= limits[-2]:
                    # x0 is in the last segment:
                    lowi = len(limits) - 2
                    highi = len(limits)
                else:
                    # We must be spanning the division between a pair of pieces.
                    lowi = numpy.max(numpy.where(x0 >= limits[:-1]))
                    highi = numpy.min(numpy.where(x1 < limits[1:]))
                insideintegral = 0
            else:
                # Add up the integrals of the pieces totally inside the interval.
                insideintegral = numpy.sum(integrals[indices])

                lowi = numpy.min(indices) - 1
                highi = numpy.max(indices) + 1

            # Check that the integral limits are inside our domain.
            if x0 < limits[0] or lowi < 0:
                lowintegral = 0.
            else:
                lowintegral = ((coefficients[lowi] / (powers[lowi] + 1.)) *
                               (limits[lowi + 1]**(powers[lowi] + 1.) -
                                x0**(powers[lowi] + 1.)))
            if x1 > limits[-1] or highi > len(coefficients) - 1:
                highintegral = 0.
            else:
                highintegral = ((coefficients[highi] / (powers[highi] + 1.)) *
                                (x1**(powers[highi] + 1.) -
                                 limits[highi]**(powers[highi] + 1.)))
            integral.flat[i] = highintegral + insideintegral + lowintegral
        return integral

def ccumulate(function, x, max=None, **kwargs):
    """Integrate a function from x to max, where x can be an array.

    Parameters
    ----------

    function: callable

    x: array-like

    max: float
        defaults to max(x)

    Notes
    -----

    This can be used to find the complementary cumulative distribution
    function (CCDF) given the probability distribution function (PDF).

    Unlike integrate_piecewise, the x values don't have to be in
    order, though a warning will be issued if any are greater than
    max, if max is specified.
    """

    x = numpy.atleast_1d(x)
    # Sort the x values and get a list of non-NaN values in order.
    order = numpy.argsort(numpy.nan_to_num(x))
    bad_mask = numpy.isnan(x[order])
    bad_order = order[bad_mask]
    good_order = order[numpy.logical_not(bad_mask)]
    x0 = x[good_order]

    if len(x0) == 1:
        integral = [0]
    else:
        integral = integrate_piecewise(function, x0, **kwargs)

    # If we have a max value, we need to add the integral from max(x0) to max.
    postintegral = 0.
    if max is not None:
        sign = 0.
        if max >= x0[-1]:
            sign = 1.
            x1 = [x0[-1], max]
        elif(max < x0[-1]):
            warnings.warn("max %s is less than maximum x %s" % (max, x0[-1]))
            sign = -1.
            x1 = [max, x0[-1]]
        postintegral += sign * integrate_piecewise(function, x1, **kwargs)[-1]

    # Reverse the direction of the integration.
    cintegral = postintegral + integral[-1] - integral

    # Put the integral values back in the right order.
    ordintegral = numpy.empty(len(x))
    ordintegral[good_order] = cintegral
    ordintegral[bad_order] = numpy.nan
    return ordintegral

def integrate_piecewise(function, x, method='romberg', return_pieces=False,
                        **kwargs):
    """Integrate function and return the integral at a sequence of points.

    Integration methods available are: quad, romberg.

    Parameters
    ----------
    function : callable
        User defined function. Should take a single vector argument
        and return q vector of the same shape.

    x : array_like
        Array of points at which to evaluate the integral. 

    method : str, optional
        Name of the method to use to integrate each segment. 'quad' or
        'romberg'.

    return_pieces : bool, optional
        Return the individual segments rather than the sum of all
        preceding segments.
   
    Returns
    -------
    integral : ndarray
        The value of the integral at each x. The first value is always zero.
    """
    
    x = numpy.asarray(x)
    if numpy.any(x[1:] - x[:-1] < 0):
        raise ValueError, "Array x must increase monotonically."
    if numpy.any(numpy.isnan(x)):
        raise ValueError, "Array x must not include NaN values." 
    integral_list = [0.0]
    if method is None:
        method = 'quad'
    if method=='quad':
        args = {'limit':200}
        args.update(kwargs)
        for i in xrange(1, len(x)):
            a, b = x[i-1], x[i]
            integral, error = scipy.integrate.quad(function, a, b,
                                                   **args)
            integral_list.append(integral)
    elif method=='romberg':
        args = {'divmax':100, 'vec_func':True}
        args.update(kwargs)
        for i in xrange(1, len(x)):
            a, b = x[i-1], x[i]
            integral = scipy.integrate.romberg(function, a, b,
                                               **args)
            integral_list.append(integral)
    else:
        raise ValueError, "Method '%s' unknown." % method

    integrals = numpy.asarray(integral_list)
    if return_pieces:
        return integrals

    integral = numpy.cumsum(numpy.nan_to_num(integrals))
    return integral

@numpy.vectorize
def vecquad(function, low, high, kwargs={}):
    """Integrate a function from low to high.

    Vectorized convenience function.
    """
    integral, error = scipy.integrate.quad(function, 
                                           low,
                                           high,
                                           **kwargs)
    return integral, error

@numpy.vectorize
def logquad(function, low, high, **kwargs):
    """Integrate a function from low to high using a log transform.

    The log transform is applied to the variable over which the
    integration is being performed.
    """
    
    # Transform the function to log space.
    def func_dlnx (lnx):
        x = numpy.exp(lnx)
        return function(x) * x
    integral, error = scipy.integrate.quad(func_dlnx, 
                                           math.log(low),
                                           math.log(high),
                                           **kwargs)
    return integral, error

class Normalize:
    """A decorator that normalizes a function.

    Only works for functions of a single variable.

    The new function is normalized over the interval from min to max,
    i.e. the integral of the new function from low to high is one.
    """

    def __init__(self, min, max, quiet=False, **kwargs):
            self.min = min
            self.max = max
            self.quiet = quiet
            self.kwargs = kwargs

    def __call__(self, function):
        integral = logquad(function, self.min, self.max, **self.kwargs)[0]
        newfunction = lambda x: function(x)/integral

        if not self.quiet:
            print "Normalization factor for %s is %.3g" % (function.__name__,
                                                           1./integral)
        # inspired by
        # http://wiki.python.org/moin/PythonDecoratorLibrary#DifferentDecoratorForms

        newfunction.__name__ = function.__name__
        newfunction.__dict__.update(function.__dict__)
        newfunction.__doc__ = function.__doc__
        newfunction.min = self.min
        newfunction.max = self.max
        return newfunction

######### Testing ############

def test_integrate_piecewise(pieces=2, method='quad'):
    # Test modified from scipy test_quadrature.py.
    n = 2
    z = 1.8
    def func(x):       # Bessel function integrand
        return numpy.cos(n*x-z*numpy.sin(x))/numpy.pi
    x = numpy.linspace(0, numpy.pi, pieces)
    val = integrate_piecewise(func,x, method=method)
    table_val = 0.30614353532540296487
    diff = val[-1] - table_val
    print "Error with %i %s pieces = %.3g" % (pieces, method, diff)
    numpy.testing.assert_almost_equal(val[-1], table_val, decimal=7)

def test_PiecewisePowerlaw(n=4, plot=False):

    powers = 1.5 * (0.5 - numpy.random.rand(n))
    limits = 10. * numpy.cumsum(numpy.random.rand(n+1))

    pfunc = PiecewisePowerlaw(limits, powers, externalval=0)
    x = numpy.linspace(limits[0] - 0.1, limits[-1] + 0.1, 20.)
    #x = numpy.linspace(limits[0], limits[-1], 20.)
    y = pfunc(x)

    integral = pfunc.integrate(0, x)
    numintegral = vecquad(pfunc, 0, x)[0]

    integral2 = pfunc.integrate(x, x[-1])
    numintegral2 = vecquad(pfunc, x, x[-1])[0]

    # Weighted integral
    integral3 = pfunc.integrate(0, x, weight_power=1.5)
    weightedfunc = lambda x: (x**1.5) * pfunc(x)
    numintegral3 = vecquad(weightedfunc, 0, x)[0]

    if plot:
        import pylab

        pylab.subplot(221)
        pylab.plot(x,y)
        pylab.xlim(min(x), max(x))
        for xlim in pfunc._limits.flat:
            pylab.axvline(x=xlim)
        
        pylab.subplot(223)
        pylab.plot(x, integral)
        pylab.plot(x, numintegral)
        pylab.plot(x, integral2)
        pylab.plot(x, numintegral2)
        pylab.plot(pfunc._limits.flat[1:], numpy.cumsum(pfunc._integrals), '.')
        pylab.xlim(min(x), max(x))

        pylab.subplot(222)
        pylab.plot(x, (integral - numintegral)/integral)
        pylab.plot(x, (integral2 - numintegral2)/integral2)
        pylab.plot(x, (integral3 - numintegral3)/integral3)

        pylab.subplot(224)
        pylab.plot(x, integral3)
        pylab.plot(x, numintegral3)
        
        pylab.show()
    assert numpy.all(numpy.abs(integral - numintegral) < 1e-4)
    assert numpy.all(numpy.abs(integral2 - numintegral2) < 1e-4)
    assert numpy.all(numpy.abs(integral3[integral3>0] -
                               numintegral3[integral3>0]) /
                     integral3[integral3>0] < 1e-3)
    assert numpy.abs(integral[-1] - 1.) < 1e-4
    assert numpy.abs(integral2[0] - 1.) < 1e-4


def test_Extrapolate1d():

    slope = 2. * (0.5 - numpy.random.rand(1))
    intercept = 20. * (0.5 - numpy.random.rand(1))
    x = numpy.array([3., 4.,
                     5., 6., 7.,
                     8., 9.])
    y = slope * x + intercept

    # Random deviations in the midde.
    y[2:5] = y[2:5] + (5. * (0.5 - numpy.random.rand(3)))
    x1 = numpy.linspace(0., 15., 100.)

    extrap = Extrapolate1d(x,y)
    print extrap.extrap_string()
    y1 = extrap(x1)
    ytrue = slope * x1 + intercept

    # Test extrapolation with a fixed slope.
    newslopes = [3.0, 2.0]
    extrap2 = Extrapolate1d(x, y, slopes=newslopes)
    print extrap2.extrap_string()
    y2 = extrap2(x1)

    mask = numpy.logical_or(x1 >= x[5],
                            x1 <= x[1])

    assert numpy.all(numpy.abs((y1 - ytrue)[mask]) < 1e-10)

    pylab.plot(x, y, 'o')
    pylab.plot(x1, y1, '-')
    pylab.plot(x1, ytrue, ':')
    pylab.plot(x1, y2, '--')

if __name__ == '__main__':

    import pylab
    pylab.figure()
    for i in range(4):
        test_Extrapolate1d()
    pylab.show()


    test_PiecewisePowerlaw(plot=True)
    
    import numpy.testing
    for method in ['quad', 'romberg']:
        for pieces in xrange(2,1000,300):
            test_integrate_piecewise(pieces, method)
