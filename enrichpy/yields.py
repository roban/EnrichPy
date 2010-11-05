"""Encapsulates stellar chemical yield data, currently only carbon 12.

General scheme:

For each data source, a Yield_Data subclass handles reading in the
data and converting it into numpy array of yields, stellar masses, and
stellar metallicites.

The Interpolate_Yields class take a Yield_Data subclass instance and
handles the creation of an interpolation function that returns the
yield as a function of stellar mass and metallicity.

"""
import re
import optparse
import os
from pkg_resources import resource_stream

import scipy
import scipy.interpolate as si
import scipy.interpolate.fitpack2 as sif2
import numpy

import initialmassfunction

def ejection_from_yield(mYield, mIni, mRem, xIni):
    """Convert net yield of an element to total ejected mass.

    mYield, mIni, and mRem should be in the same units (which will also
    be the units of the returned ejected mass.

    xIni is the initial mass fraction of the element.

    See GBM equation 11.

    Gavilan M., Buell J.F., Molla M. Astron. Astrophys. 2005, 432, 861
    (2005A&A...432..861G)
    """
    return mYield + ((mIni - mRem) * xIni)
    
def yield_from_ejection(mEj, mIni, mRem, xIni):
    """Convert total ejected mass to net yield of an element.

    mEj, mIni, and mRem should be in the same units (which will also
    be the units of the returned yield.

    xIni is the initial mass fraction of the element.

    See GBM equation 11.

    Gavilan M., Buell J.F., Molla M. Astron. Astrophys. 2005, 432, 861
    (2005A&A...432..861G)
    """
    return mEj - ((mIni - mRem) * xIni)

class Yield_Data:
    """Base class for yield data."""

    ### Subclasses should define the following, where applicable: ###
    #shortname =
    #filename =
    #metal_col = # stellar inition metallicity
    #mass_col = # stellar initial mass
    #rem_col = # stellar remnant mass
    #yield_col = # net mass yield of element
    #ejection_col = # total mass of element ejected
    ###

    def __init__(self, interp_args={}):
        self.interp_args=interp_args
        self.loadtxt_args={}
        self.metal_mass_frac_C = 0.178
        self.load_data()

    def load_data(self):
        self.read_table()
        self.select_data()
        self.regrid_data()

    def read_table(self):
        """Read data in self.filename.

        Uses the pkg_resources.resource_stream function to access the
        data file.
        """
        stream = resource_stream(__name__, self.filename)
        self.data = numpy.loadtxt(stream, **self.loadtxt_args)
        stream.close()

    def select_data(self):
        if hasattr(self, 'mass_col'):
            self.mass_star = self.data[:,self.mass_col]
        if hasattr(self, 'rem_col'):
            self.mass_rem = self.data[:,self.rem_col]
        if hasattr(self, 'metal_col'):
            self.metal_star = self.data[:,self.metal_col]
        if hasattr(self, 'yield_col'):
            self.mass_C = self.data[:,self.yield_col]
        if hasattr(self, 'ejection_col'):
            self.mass_C_total = self.data[:,self.ejection_col]

        # If we have net yield and remnant mass, but no total ejection mass:
        if (hasattr(self, 'yield_col') and hasattr(self, 'rem_col') and
            not hasattr(self, 'ejection_col')):
            self.mass_C_total = ejection_from_yield(self.mass_C,
                                                    self.mass_star,
                                                    self.mass_rem,
                                                    self.metal_mass_frac_C *
                                                    self.metal_star)
        # If we total ejection and remnant mass, but no net yield mass:
        if (hasattr(self, 'ejection_col') and hasattr(self, 'rem_col') and
            not hasattr(self, 'yield_col')):
            self.mass_C = yield_from_ejection(self.mass_C_total,
                                              self.mass_star,
                                              self.mass_rem,
                                              self.metal_mass_frac_C *
                                              self.metal_star)


    def regrid_data(self):
        """Subclasses can define this if they want."""
        pass
        
    def __call__(self, ejections=False):
        if ejections:
            return self.mass_star, self.metal_star, self.mass_C_total
        else:
            return self.mass_star, self.metal_star, self.mass_C

    def interpolate_function(self, ejections=False):
        return Interpolate_Yields(*self(ejections=ejections),
                                  **self.interp_args)
        #return Interpolate_Yields(*self(ejections=ejections))

class Data_vdHG(Yield_Data):
    """van den Hoek and Groenewegen (1997): 0.8 -- 8 Msun; Z = 0.001 -- 0.04

    van den Hoek, L.B., \& Groenewegen, M.A.T.\ 1997, \aaps, 123, 305
    (1997A&AS..123..305V)

    [http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+AS/123/305]

    Notes
    -----

    metal_mass_frac_C is infered to be 0.224 from the table below:
    
    Table1: Initial element abundances adopted
    --------------------------------------------------------------
    Element.  Z=0.001   0.004     0.008     0.02      0.04
    --------------------------------------------------------------
    H       0.756     0.744     0.728     0.68      0.62
    4He      0.243     0.252     0.264     0.30      0.34
    12C      2.24E-4   9.73E-4   1.79E-3   4.47E-3   9.73E-3
    13C      0.04E-4   0.16E-4   0.29E-4   0.72E-4   1.56E-4
    14N      0.70E-4   2.47E-4   5.59E-4   1.40E-3   2.47E-3
    16O      5.31E-4   2.11E-3   4.24E-3   1.06E-2   2.11E-2

    """

    shortname = 'vdH&G1997'
    filename = 'yield_data/1997A&AS__123__305V_tab2-22_total.dat'
    metal_col=0
    mass_col=1
    yield_col=4
    mass_Ej_col = 10

    def __init__(self, interp_args={}):
        self.interp_args=interp_args
        self.loadtxt_args = {'usecols' : range(1,13)}
        self.metal_mass_frac_C = 0.224
        self.load_data()

    def select_data(self):
        self.mass_star = self.data[:,self.mass_col]
        self.mass_star_ejected = self.data[:,self.mass_Ej_col]
        self.mass_rem = self.mass_star - self.mass_star_ejected
        self.metal_star = self.data[:,self.metal_col]
        self.mass_C = self.data[:,self.yield_col]

        # If we have net yield and remnant mass, but no total ejection mass:
        self.mass_C_total = ejection_from_yield(self.mass_C,
                                                self.mass_star,
                                                self.mass_rem,
                                                self.metal_mass_frac_C *
                                                self.metal_star)


class Data_H(Yield_Data):
    """Herwig 2004: 2-6 Msun; Z=0.0001 

    Herwig, F.\ 2004, \apjs, 155, 651 (2004ApJS..155..651H)

    2-6 Msun; Z=0.0001 
    [http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/ApJS/155/651]

      Intermediate-mass stellar evolution tracks from the main
      sequence to the tip of the AGB for five initial masses (2-6
      Msolar) and metallicity Z=0.0001 have been computed. ...yields
      are presented.

    """

    shortname = 'H2004'
    filename = 'yield_data/2004ApJS__155__651H_table5.dat'
    def _H_iso_converter(*args):
        """Strip element names from isotope in ChLi table."""
        isotope = args[-1]
        alphanum = re.compile('(\D*)(\d*)') # non-digit followed by digit 
        element, number = alphanum.match(isotope).groups()
        if number == '':
            number=1
        return int(number)

    def __init__(self, interp_args={}):
        self.interp_args=interp_args
        self.loadtxt_args={'converters' : {1: self._H_iso_converter}}
        self.metal_mass_frac_C = 0.178
        self.load_data()

    def select_data(self):
        #self.mass_star = self.data[:,self.mass_col]
        #self.metal_star = self.data[:,self.metal_col]
        self.mass_C = self.data[2,2:]
        self.mass_star = numpy.array([2.,3.,4.,5.,6.])
        self.metal_star = numpy.ones(self.mass_star.shape) * 0.0001
    
class Data_GBM(Yield_Data):
    """Gavilan, Buell, & Molla (2005): 0.8 -- 100 Msun; log(Z/Zsun)=-0.2--+0.2

    Gavilan M., Buell J.F., Molla M. Astron. Astrophys. 2005, 432, 861
    (2005A&A...432..861G)

    log(Z/Zsun)=-0.2, -0.1, 0.0, +0.1 and +0.2

    [http://cdsarc.u-strasbg.fr/viz-bin/ftp-index?J/A%2bA/432/861]

    """

    shortname = 'GB&M2005'
    filename = 'yield_data/2005A&A__432__861G/table2.dat'
    metal_col=0
    mass_col=1
    rem_col=2
    ejection_col=5

    def regrid_data(self):
        """ Interpolate data onto a common mass grid.
        """
        mass_C = self.mass_C
        mass_C_total = self.mass_C_total
        mass_star = self.mass_star
        metal_star = self.metal_star

        ### Set up a mass scale with reasonable spacing. ###
            
        # Get all unique mass values in the data set.
        mass_scale = numpy.sort(numpy.unique(mass_star))

        # Now step through the values, discarding ones that are too close
        # together.

        last_mass = mass_scale[0]
        new_mass_scale = [last_mass]
        numavg = 0
        for m in list(mass_scale[1:]):
            if m - last_mass < (0.04 * m):
                # If the next value is too close, we average the positions.
                numavg = numavg + 1
                new_mass_scale[-1] = (numavg * new_mass_scale[-1] + m)/(numavg+1)
            else:
                numavg = 0
                new_mass_scale.append(m)
                last_mass = m
        mass_scale = numpy.array(new_mass_scale)

        ZVals = numpy.unique(metal_star)

        new_len = len(ZVals) * len(mass_scale)
        new_mass_C = numpy.zeros(new_len)
        new_mass_C_total = numpy.zeros(new_len)
        repeated_inds = range(len(mass_scale)) * len(ZVals)
        new_mass_star = mass_scale[repeated_inds]
        new_metal_star = numpy.zeros(new_len)

        doplot = False
        if doplot:
            print new_mass_star
            import pylab

        for Z, Zind in zip(list(ZVals), range(len(ZVals))):
            mask =  metal_star == Z 
            ifunc = si.interp1d(numpy.log10(mass_star[mask]),
                                mass_C[mask],
                                kind=3)
            ifunc_total = si.interp1d(numpy.log10(mass_star[mask]),
                                      mass_C_total[mask],
                                      kind=3)
            i0 = Zind * len(mass_scale) 
            i1 = (Zind+1) * len(mass_scale)

            new_mass_C[i0:i1] = ifunc(numpy.log10(mass_scale))
            new_mass_C_total[i0:i1] = ifunc_total(numpy.log10(mass_scale))
            new_metal_star[i0:i1] = Z

            if doplot:
                print mass_scale.shape
                print new_mass_C[i0:i1].shape
                pylab.figure()
                pylab.scatter(mass_star[mask], mass_C[mask], marker='x')
                pylab.plot(new_mass_star[i0:i1], new_mass_C[i0:i1], '-+')
                pylab.title(str(Z))

        self.orig_mass_star = self.mass_star
        self.orig_metal_star = self.metal_star
        self.orig_mass_C = self.mass_C
        self.orig_mass_C_total = self.mass_C_total

        self.mass_star = new_mass_star
        self.metal_star = new_metal_star
        self.mass_C = new_mass_C
        self.mass_C_total = new_mass_C_total

class Exponentiate:
    """Decorator to exponentiate some arguments before calling the function.
    """
    def __init__(self, xexp=True, yexp=True):
        self.xexp = xexp
        self.yexp = yexp

    def __call__(self, function):
        if (not self.xexp) and (not self.yexp):
            return function

        if self.xexp and (not self.yexp):
            newfunction = lambda x, y: function(10**x, y)
        elif (not self.xexp) and self.yexp:
            newfunction = lambda x, y: function(x, 10**y)
        elif self.xexp and self.yexp:
            newfunction = lambda x, y: function(10**x, 10**y)

        newfunction.__name__ = function.__name__
        newfunction.__dict__.update(function.__dict__)
        newfunction.__doc__ = function.__doc__
        return newfunction

class Interpolate_Yields:
    """The ejected carbon mass is interpolated on a logarithmic
    metalicity scale and mass scale.

    Essentially a convenience class to encapsulate my current choices
    about the interpolation.

    """

    def __init__(self, mass_star, metal_star, mass_C, weight_function=None,
                 mass_scale='linear', metal_scale='linear',
                 kx=1,
                 ky=1,
                 swapxy = False):
        self.swapxy = swapxy
        self.mass_scale = mass_scale
        self.metal_scale = metal_scale

        # Find unique m and Z values.
        mVals = numpy.unique(mass_star)
        mVals.sort()

        ZVals = numpy.unique(metal_star)
        ZVals.sort()

        self.x = mass_star
        if not mass_scale=='linear':
            self.x = numpy.log10(x)

        self.y = metal_star
        if not metal_scale=='linear':
            self.y = numpy.log10()
        
        self.xknots = numpy.unique(self.x)
        self.xknots.sort()
        self.yknots = numpy.unique(self.y)
        self.yknots.sort()

        self.mVals = mVals
        self.ZVals = ZVals

        self.mass_C = mass_C
        self.z = self.mass_C
        if weight_function is not None:
            self.z *= weight_function(self.x)

        ### bicubic spline interpolation ###

        if self.swapxy:
           ## Note that x and y are swapped due to an error I encountered
           ## trying to use kx > ky.
            self._interpfunc = sif2.LSQBivariateSpline(self.y, self.x,
                                                       self.z,
                                                       self.yknots,
                                                       self.xknots, 
                                                       kx=ky, ky=kx)
            self._swap_mCfunc = Exponentiate(not metal_scale=='linear',
                                             not mass_scale=='linear')\
                                             (self._interpfunc)
            self._mCfunc = lambda x, y: self._swap_mCfunc(y,x).transpose()
        else:
            self._interpfunc = sif2.LSQBivariateSpline(self.x, self.y,
                                                       self.mass_C,
                                                       self.xknots,
                                                       self.yknots, 
                                                       kx=kx, ky=ky)
            self._mCfunc = Exponentiate(not mass_scale=='linear',
                                        not metal_scale=='linear')\
                                        (self._interpfunc)

    def __call__(self, m, Z):
        ## Note that x and y are swapped due to an error I encountered
        ## trying to use kx > ky.
        return self._mCfunc(m, Z)

def linear_coefficients(x,y):
    slope = (y[1:]-y[:-1]) / (x[1:] - x[:-1])
    intercept = y[:-1] - slope * x[:-1]
    return slope, intercept

def print_linear_info(slope, intercept):
    print " mass_C = %.3g m_star + %3.g Msun" % (slope, intercept)
    print "                            mass_C = 0 when m = %.3g Msun" % \
          (-1. * intercept / slope)
    
def extrapolate_function(yield_function, metallicity, return_coeffs=False):
    """Linearly extrapolate yield data to high and low masses.

    Returns two functions: the high mass extrapolation and the low
    mass extrapolation.
    """

    # High mass extrapolation
    highM = numpy.sort(yield_function.mVals)[-2:]
    high_mass_C = yield_function(highM, metallicity).flatten()
    highSlope, highIntercept = linear_coefficients(highM, high_mass_C)
    print "Extrapolating from points at m_star = %.3g and %.3g Msun:" % \
          (highM[0], highM[1])
    print_linear_info(highSlope, highIntercept)
    exfuncHigh = lambda mass: mass * highSlope + highIntercept

    # Low mass extrapolation
    lowM = numpy.sort(yield_function.mVals)[:2]
    low_mass_C = yield_function(lowM, metallicity).flatten()
    lowSlope, lowIntercept = linear_coefficients(lowM, low_mass_C)
    print "Extrapolating from points at m_star = %.3g and %.3g Msun:" % \
          (lowM[0], lowM[1])
    print_linear_info(lowSlope, lowIntercept)
    exfuncLow = lambda mass: mass * lowSlope + lowIntercept

    if return_coeffs:
        return (exfuncHigh, exfuncLow,
                highSlope, highIntercept, lowSlope, lowIntercept)
    else:
        return exfuncHigh, exfuncLow

class Data_ChLi(Yield_Data):
    """Chieffi & Limongi (2004) yields: 13--35 Msun; Z = 0, 1e-6 -- 0.02
    
    Chieffi, A., \& Limongi, M.\ 2004, \apj, 608, 405
    (2004ApJ...608..405C)

    [http://vizier.cfa.harvard.edu/viz-bin/Cat?J/ApJ/608/405]

    """
    shortname = 'Ch&Li2004'
    filename = 'yield_data/2004ApJ__608__405C_yields.dat'
    def _ChLi_iso_converter(*args):
        """Strip element names from isotope in ChLi table."""
        isotope = args[-1]
        alphanum = re.compile('(\d*)(\D*)') # digit followed by non-digit
        number, element = alphanum.match(isotope).groups()
        return int(number)

    def __init__(self, interp_args={}):
        self.interp_args=interp_args
        self.loadtxt_args={'converters' : {2: self._ChLi_iso_converter}}
        self.metal_mass_frac_C = 0.178
        self.load_data()

    def select_data(self):
        """Return mass, metallicity, and C yield.
        """
        data = self.data
        isonumber = data[:,2]
        # Select 12C.
        cmask = isonumber == 12

        # Set of mass values for the columns 4-9.
        mass_star = numpy.array([13., 15., 20., 25., 30., 35.])

        # Metallicity values.
        metal_star = data[cmask][:,0:1]

        # Yield data
        mass_C = data[cmask][:,3:]

        # Form compatible grid of mass_star and metal_star values.
        ones = numpy.ones(mass_C.shape)
        mass_star = ones * mass_star
        metal_star = ones * metal_star
        self.mass_star = mass_star.flatten()
        self.metal_star = metal_star.flatten()
        self.mass_C = mass_C.flatten()
        return self.mass_star, self.metal_star, self.mass_C

def test_plot_data_interp(data, nm=1e3, nZ=1e3, **args):

    mCfunc = data.interpolate_function()

    mass_star, metal_star, mass_C = data()
    
    m = numpy.linspace(0.75 * min(mass_star), 1.05 * max(mass_star), nm)
    from collections import deque
    marks = deque(['.', 'o', 'v','+','x','*'])
    colors = deque(['r', 'y', 'g','c','b','k', 'm'])

    import matplotlib.pyplot as pylab
    pylab.figure()
    for Z in numpy.unique(metal_star):
        mark = marks[0]
        col = colors[0]
        marks.rotate()
        colors.rotate()
        datamask = metal_star==Z
        #print mass_star[datamask], mass_C[datamask]
        pylab.plot(mass_star[datamask], mass_C[datamask], 
                   marker=mark, label=str(Z), c=col, ls='')
        pylab.plot(m, mCfunc(m, Z), c=col, ls=':')

    pylab.legend(loc='best')

    Z = numpy.logspace(numpy.min(numpy.log10(metal_star[metal_star>0])) - 1,
                       numpy.max(numpy.log10(metal_star)) + 0.5, nZ)
    if numpy.any(metal_star <= 0):
        Z = numpy.hstack(([0], Z))

    logZ = numpy.log10(Z)
    pylab.figure()
    for M in numpy.unique(mass_star):
        mark = marks[0]
        col = colors[0]
        marks.rotate()
        colors.rotate()
        datamask = numpy.logical_and(mass_star==M, metal_star>0)
        pylab.plot(numpy.log10(metal_star[datamask]), mass_C[datamask], 
                   marker=mark, label=str(M), c=col, ls='')
        pylab.plot(logZ[Z>0], mCfunc(M, Z).flatten()[Z>0], c=col, ls=':')

    pylab.legend(loc='best')


    pylab.figure()
    for M in numpy.unique(mass_star):
        mark = marks[0]
        col = colors[0]
        marks.rotate()
        colors.rotate()
        datamask = mass_star==M
        pylab.plot(metal_star[datamask], mass_C[datamask], 
                   marker=mark, label=str(M), c=col, ls='')
        pylab.plot(Z, mCfunc(M, Z).flatten(), c=col, ls=':')
        #print mCfunc(M, 10**logZ)

    #pylab.xlim(Z[0], Z[-1])
    pylab.legend(loc='best')


def data_overlap(data_list):
    names = ['m*', 'Z*', 'mC']
    mins = [numpy.inf, numpy.inf, numpy.inf]
    maxa = [None, None, None]
    maxmin = [None, None, None]
    minmax = [None, None, None]
    for data in data_list:
        print data.shortname
        datarrays = data()
        for i, dat in enumerate(datarrays):
            imin = numpy.nanmin(dat)
            imax = numpy.nanmax(dat)
            print(("%5.3g <= " + names[i] + " <= %5.3g ") % (imin, imax)),
            mins[i] = min(imin, mins[i])
            maxa[i] = max(imax, maxa[i])
            maxmin[i] = max(imin, maxmin[i])
            minmax[i] = min(imax, maxmin[i]) 
        print
    print mins
    print maxa
    print maxmin
    print minmax
    return mins, maxa, maxmin, minmax
    
def test_plot_interp(data_list, Z_list, nm=1e3,
                     maxmass=None, ejections=False, title=None,
                     logx=True, logy=True):
    import matplotlib.pyplot as pylab
    fig = pylab.figure(figsize=(9,6))
    if title is not None:
        fig.set_label(title)
    ax = fig.add_subplot(111)

    from collections import deque
    styles = deque(['-', '--', ':', '-.'])

    mins, maxa, maxmin, minmax = data_overlap(data_list)

    for data in data_list:
        print data.shortname
        style = styles[0]
        styles.rotate(-1)

        if ejections:
            if not hasattr(data, 'mass_C_total'):
                print "No total ejected masses for " + data.shortname
                continue
            mCfunc_ej = data.interpolate_function(ejections=True)
            mass_star, metal_star, mass_C_ej = data(ejections=True)

        mCfunc = data.interpolate_function()
        mass_star, metal_star, mass_C = data()
        
        #m = numpy.linspace(0.75 * min(mass_star), 1.05 * max(mass_star), nm)
        m = numpy.linspace(min(mass_star), max(mass_star), nm)
        colors = deque(['k','b', 'c', 'g','r'])
        for Z in Z_list:
            color = colors[0]
            colors.rotate(-1)
            if (Z < min(metal_star)) or (Z > max(metal_star)):
                print "skipping ", 
                print Z,
                print data.shortname

                continue
            Zlab = "Z = %.3g" % (Z)
            pylab.plot(m, mCfunc(m, Z), ls=style, c=color,
                       label=data.shortname + " " + Zlab)
            if ejections:
                color = colors[0]
                colors.rotate(-1)
                pylab.plot(m, mCfunc_ej(m, Z), ls=style, c=color,
                           label=data.shortname + " ej " + Zlab)
    if maxmass is not None:
        pylab.xlim(xmax=maxmass)
    #pylab.legend(loc='best')
    labels=[]
    lines=[]
    for line in ax.lines:
        label = line.get_label()
        if label[0] == '_':
            continue
        labels.append(label)
        lines.append(line)
    if logx:
        pylab.xscale('log')
    if logy:
        pylab.yscale('log')
    fig.legend(lines, labels, 'lower right')
    fig.subplots_adjust(left=0.07, bottom=0.07, top=0.95, right=0.62)


class Data_CaLa(Yield_Data):
    """Campbell & Lattanzio 2008 0.85 -- 3 Msun; Z = 0, & [Fe/H] = -6.5 -- -3.0

    Campbell, S.~W., \& Lattanzio, J.~C.\ 2008, \aap, 490, 769
    (2008A&A...490..769C)
  
    [http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/490/769]

    """
    shortname = 'Ca&La2008'
    basefilename = 'yield_data/2008A&A__490__769C_table'

    def __init__(self, interp_args={}):
        self.interp_args={'swapxy':True}
        self.interp_args.update(interp_args)
        self.loadtxt_args={}
        self.metal_mass_frac_C = 0.178
        self.load_data()

#    def _CaLa_iso_converter(self, isotope):
#         return 0.0
#     def _CaLa_iso_converter(self, isotope):
#         """Strip element names from isotope in CaLa table."""
#         alphanum = re.compile('(\D*)(\d*)') # non-digit  followed by  digit
#         element, number = alphanum.match(isotope).groups()
#         print isotope
#         print element
#         print number
#         return int(number)

    def read_table(self):
        self.tables = []
        for i in range(1,6):
            cols = [1,2,3,4,5,6]
            if i==4:
                cols = [1,2,3,4,5]
            stream = resource_stream(__name__,
                                     self.basefilename + str(i) + '.dat')
            table = numpy.loadtxt(stream,
                                  #converters={0: self._CaLa_iso_converter},
                                  usecols=cols
                                  )
            stream.close()
            if i==4:
                table = numpy.hstack((table,
                                     numpy.nan * numpy.ones((len(table),1))))
            self.tables.append(table)
            #print "Table " + str(i)
            #print table.shape
        #import newio
        #table = newio.loadtxt(self.basefilename + '6.dat')
        stream = resource_stream(__name__, self.basefilename + '6.dat')
        table = numpy.loadtxt(stream)
        stream.close()
        self.tables.append(table)
        self.data = numpy.vstack(self.tables[0:5])
        self.remnant_data = self.tables[5]
        return self.tables
        
    def select_data(self):
        """Return mass, metallicity, and C yield.
        """
        data = self.data
        remnant_data = self.remnant_data
        
        # Set of mass values for the columns 2-5.
        mass_star = numpy.array([0.85, 1.0, 2.0, 3.0])

        # Metallicity values.
        Z_solar = 0.02

        metal_star = Z_solar * numpy.array([[0,
                                             10**-6.5,
                                             10**-5.45,
                                             10**-4.0,
                                             10**-3.0]]).transpose()
        
        # Select 12C.
        isonumber = data[:,0]
        cmask = isonumber == 12

        # Fractional mass values. Columns are:
        # A; Initial; 0.85 Msun; 1.0 Msun; 2.0 Msun; 3.0 Msun
        init_frac_mass_C = data[cmask][:,1:2]
        frac_mass_C = data[cmask][:,2:]

        # Table 6 contains Remnant Masses:
        # Met   0.85   1.0    2.0    3.0
        mass_Re = remnant_data[:, 1:]
        # Ejected mass = init mass - remnant mass
        mass_Ej = mass_star - mass_Re

        # initial mass of carbon
        init_mass_C = init_frac_mass_C * mass_star

        # total ejected mass of carbon
        mass_C_tot = frac_mass_C * mass_Ej

        # net mass of carbon produced
        mass_C_net = mass_C_tot - init_mass_C

        # Form compatible grid of mass_star and metal_star values.
        ones = numpy.ones(mass_C_tot.shape)
        mass_star = ones * mass_star
        metal_star = ones * metal_star

        # Convert to 1D arrays and mask out missing data:
        data_mask = numpy.isfinite(mass_Ej.flatten())
        self.mass_star = mass_star.flatten()[data_mask]
        self.metal_star = metal_star.flatten()[data_mask]
        self.mass_Ej = mass_Ej.flatten()[data_mask]
        self.mass_C_total = mass_C_tot.flatten()[data_mask]
        self.mass_C = mass_C_net.flatten()[data_mask]
        
        return self.mass_star, self.metal_star, self.mass_C


def test_all():
    imf = initialmassfunction.imf_number_Chabrier
    args = {'weight_function' : imf}
    weighted_list = [Data_GBM(args),
                     Data_ChLi(args),
                     Data_CaLa(args),
                     Data_vdHG(args),
                     Data_H(args)]
    unweighted_list = [Data_GBM(),
                       Data_ChLi(),
                       Data_CaLa(),
                       Data_vdHG(),
                       Data_H()]
    metal_list = [0.02, 1e-3, 1e-4, 1e-5, 0.0]
    for ej in [True, False]:
        if ej:
            pt = 'Ej'
        else:
            pt = 'Yi'
        test_plot_interp(unweighted_list, metal_list,
                         ejections=ej, title='unwFull'+pt)
#        test_plot_interp(unweighted_list, metal_list,
#                         maxmass=8.0,
#                         ejections=ej, title='unwZoom'+pt)

        test_plot_interp(weighted_list, metal_list,
                         ejections=ej, title='wFull'+pt)
#        test_plot_interp(weighted_list, metal_list,
#                         maxmass=8.0,
#                         ejections=ej, title='wZoom'+pt)
    
    
#    test_plot_data_interp(Data_GBM())
#    test_plot_data_interp(Data_ChLi())
#    test_plot_data_interp(Data_CaLa())

if __name__ == '__main__':

    usage = """ """
    
    ### Argument parsing. ###
    parser = optparse.OptionParser(usage)
    parser.add_option("-f", "--file", action='store_true', dest="filename",
                      default=None)
    (options, args) = parser.parse_args()
    if (len(args) > 0) and (options.filename is None):
        options.filename = args[0]
    if options.filename is None:
        print "No filename given."
        print usage
    else:
        prefix, extension = os.path.splitext(options.filename)

    ### Main code ###

    import pylab
    test_all()

    ### Plot output code. ###
    if options.filename is None:
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
            fig.savefig(newfilename, dpi=75)#, bbox_inches="tight")
    
