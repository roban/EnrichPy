"""Implements stellar initial mass functions from Romano et al. 2005.

Full Citation: Romano D., Chiappini C., Matteucci F., Tosi M., 2005,
A\&A, 430, 491 (2005A&A...430..491R)

http://adsabs.harvard.edu/abs/2005A&A...430..491R
"""

import math

import numpy
import scipy
import scipy.integrate
import utils
from utils import Normalize

##################### Utilities #####################################

def by_number(imf_mass_function):
    """Convert an IMF by mass to an IMF by number.
    """
    def imf_number_function(mass):
        return imf_mass_function(mass)/mass
    imf_number_function.__name__ = \
                        imf_mass_function.__name__.replace('imf_mass_',
                                                           'imf_number_')
    imf_number_function.__dict__.update(imf_mass_function.__dict__)
    imf_number_function.__doc__ = imf_mass_function.__doc__
    return imf_number_function

def broken_powerlaw_function(limits, coefficients, powers):
    """Return a function consisting of seperate powerlaws.


    F(x) = a_i x^{p_i} for l_i < x < l_{i+1}

    Parameters
    ----------

    limits: array (length n+1)
        boundaries of the specified powerlaws. Must be one greater in
        length than coefficents and powers. Specify -numpy.infty for
        the first limit or numpy.infty for the last limit for
        unbounded powerlaws.

    coefficients: array (length n)
        values of the coefficient a_i

    powers: array (length n)
        values of the powerlaw indices p_i

    The returned function takes a single, one-dimensional array of
    values on which to operate.

    """
    
    limits = numpy.atleast_1d(limits)
    coefficients = numpy.atleast_1d(coefficients)
    powers = numpy.atleast_1d(powers)
    for array in [limits, coefficients, powers]:
        if array.ndim > 1:
            raise ValueError("arguments must be a 1D arrays or scalars.")

    if not len(coefficients) == len(powers):
        raise ValueError("coefficients and powers must be the same length.")

    if not len(limits) == len(powers)+1:
        raise ValueError("limits must be one longer than powers.")

    limits = limits.reshape((-1,1))
    coefficients = coefficients.reshape((-1,1))
    powers = powers.reshape((-1,1))
    
    def pf_func(x):
        x = numpy.atleast_1d(x)
        if x.ndim > 1:
            raise ValueError("argument must be a 1D array or scalar.")
        y = numpy.sum((coefficients * x**powers) *
                      (x > limits[0:-1]) * (x < limits[1:]),
                      axis=0)
        y[x < limits[0]] = None
        y[x > limits[-1]] = None
        return y
    return pf_func

###################### IMFs ##########################################

### Scalo86 ###
coeff_Scalo86 = numpy.array([0.19, 0.24])
pow_Scalo86 = numpy.array([-1.35, -1.70])
lims_Scalo86 = numpy.array([0, 2., numpy.infty])
_imf_mass_Scalo86 = broken_powerlaw_function(lims_Scalo86, 
                                             coeff_Scalo86, 
                                             pow_Scalo86)
#@Normalize(0.1,100)
def imf_mass_Scalo86(mass):
    """Scalo 1986 stellar Initial Mass function by mass via Romano et al. (2005).
    
    Parameters
    ----------
    
    mass: 1d array
        mass in Solar masses.

    """
    return _imf_mass_Scalo86(mass)

### Scalo98 ###
coeff_Scalo98 = numpy.array([0.39, 0.39, 0.16])
pow_Scalo98 = numpy.array([-0.2, -1.7, -1.3])
lims_Scalo98 = numpy.array([0, 1., 10., numpy.infty])
_imf_mass_Scalo98 = broken_powerlaw_function(lims_Scalo98, 
                                             coeff_Scalo98, 
                                             pow_Scalo98)
#@Normalize(0.1,100)
def imf_mass_Scalo98(mass):
    """Scalo 1998 stellar Initial Mass function by mass via Romano et al. (2005).
    
    Parameters
    ----------
    
    mass: 1d array
        mass in Solar masses.

    """
    return _imf_mass_Scalo98(mass)


### Tinsley ###
coeff_Tinsley = numpy.array([0.21, 0.26, 2.6])
pow_Tinsley = numpy.array([-1., -1.3, -2.3])
lims_Tinsley = numpy.array([0, 2., 10., numpy.infty])
_imf_mass_Tinsley = broken_powerlaw_function(lims_Tinsley, 
                                             coeff_Tinsley, 
                                             pow_Tinsley)
#@Normalize(0.1,100)
def imf_mass_Tinsley(mass):
    """Tinsley stellar Initial Mass function by mass via Romano et al. (2005).
    
    Parameters
    ----------
    
    mass: 1d array
        mass in Solar masses.

    """
    return _imf_mass_Tinsley(mass)


### Kroupa ###
coeff_Kroupa = numpy.array([0.58, 0.31, 0.31])
pow_Kroupa = numpy.array([-0.3, -1.2, -1.7])
lims_Kroupa = numpy.array([0, 0.5, 1., numpy.infty])
_imf_mass_Kroupa = broken_powerlaw_function(lims_Kroupa, 
                                             coeff_Kroupa, 
                                             pow_Kroupa)
#@Normalize(0.1,100)
def imf_mass_Kroupa(mass):
    """Kroupa stellar Initial Mass function by mass via Romano et al. (2005).
    
    Parameters
    ----------
    
    mass: 1d array
        mass in Solar masses.

    """
    return _imf_mass_Kroupa(mass)

#@Normalize(0.1,100)    
def imf_mass_Salpeter(mass):
    """Salpeter stellar Initial Mass function by mass via Romano et al. (2005).

    """
    return 0.17 * mass**-1.35

mc = 0.079
logmc = math.log10(mc)
sigma = 0.69
twosigsq = 2. * sigma**2

fa = 0.85
fb = 0.24
#@Normalize(0.1,100)
def _imf_lowmass_Chabrier(mass):
    return fa * numpy.exp(-1 * ((numpy.log10(mass) -
                                 logmc)**2. / 
                                twosigsq))
def _imf_highmass_Chabrier(mass):
    return fb * mass**-1.3
def imf_mass_Chabrier(mass):
    """Chabrier stellar Initial Mass function by mass via Romano et al. (2005).

    With slope -1.3 above 1 MSun.

    """
    if numpy.isscalar(mass):
        if mass < 1.:
            return _imf_lowmass_Chabrier(mass)
        else:
            return _imf_highmass_Chabrier(mass)
    mass = numpy.asarray(mass)
    imf = numpy.empty(mass.shape)
    lowmask = mass < 1.
    imf[lowmask] = _imf_lowmass_Chabrier(mass[lowmask])
    imf[mass >= 1.] = _imf_highmass_Chabrier(mass[mass >= 1.])

    return imf

#imf_num_Chabrier = Normalize(0.1,100)(by_number(imf_mass_Chabrier))
imf_number_Chabrier = by_number(imf_mass_Chabrier)

######################## Tests ############################################

def test_plot_norm_imf(imf_function):
    """Plot and test the normalization of imf_function."""
    label = imf_function.__name__.replace('imf_', '')
    label = label.replace('_', ' ')
    print label
    dm = 0.005
    mmin = 0.1
    mmax = 100.
    mass = numpy.arange(mmin, mmax + 1.1 * dm, dm)
    imf = imf_function(mass)

    # Test that the normalization is correct.
    liimf = utils.logquad(imf_function, mmin, mmax)
    print liimf[0], ", ",
    print 1.-liimf[0]
    assert abs(liimf[0] - 1) < 1e-10

    # Test normalization using a different integration method.
    cimf = scipy.integrate.cumtrapz(imf * dm)
    print cimf[-1], ", ",
    print 1.-cimf[-1]
    assert abs(cimf[-1] - 1) < 1e-3
    
    pylab.subplot(121)
    l = pylab.plot(mass, imf, label=label)
    pylab.plot(mass[1:], cimf, l[0].get_c())
    pylab.gca().set_yscale('log')
    pylab.legend(loc='best')

    pylab.subplot(122)
    pylab.plot(mass, imf, l[0].get_c(), label=label)
    pylab.plot(mass[1:], cimf, l[0].get_c())
    pylab.gca().set_yscale('log')
    pylab.gca().set_xscale('log')

def test_plot_norm_all():
    # Get a list of all the imf_mass_* functions.
    import sys
    module = sys.modules[__name__]
    functionlist = [getattr(module, name) for name in dir(module) if name.startswith('imf_mass_')]

    import pylab
    import scipy
    import scipy.integrate
    pylab.figure(figsize=(11,6))

    # Plot the functions by mass.
    for function in functionlist:
        test_plot_norm_imf(Normalize(0.1,100.)(function))

    # Plot the functions by number.
    pylab.figure(figsize=(11,6))
    for function in functionlist:
        test_plot_norm_imf(Normalize(0.1,100.)(by_number(function)))    

if __name__ == '__main__':
    import matplotlib.pyplot as pylab
    import os
    import sys
    
    ### Argument parsing. ###
    if len(sys.argv)==1:
        print "Run with a filename argument to produce image files, e.g.:"
        print " python initialmassfunction.py imf.png"
        print " python initialmassfunction.py imf.eps"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        prefix, extension = os.path.splitext(filename)
    else:
        filename = None

    ### Main code area. ###

    test_plot_norm_all()

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

