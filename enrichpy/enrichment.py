import copy

import numpy
import scipy
import scipy.constants.codata as scc

import cosmolopy.reionization as cr
import cosmolopy.utils as utils
from cosmolopy.utils import Saveable

from cosmolopy import luminosityfunction
import padconvolve

def tophat_delay_kernel(t, tmin, tmax):
    """Make a normalized tophat function (nonzero from tmin to tmax).

    t: array-like

    The result is normalized so that the sum of the bins is equal to
    norm (not the integral), i.e. there's no multiplication by the
    width of the bins in the normalization.

    Returns the kernel and the cdf.
    """

    kernel = numpy.ones(t.shape)
    kernel[t>tmax] = 0.
    kernel[t<tmin] = 0.
    ksum = numpy.sum(kernel)
    kernel = kernel/ksum
    cdf = numpy.cumsum(kernel)
    return kernel, cdf

def get_omega_CIV_data(filename):
    data = numpy.loadtxt(filename, skiprows=1, delimiter=',', 
                         usecols=range(5))
    return data

class LFIonHistory(Saveable):
    """Calculate an ionization history based on luminosity function evolution.
    
    LF parameters enter here, e.g. choice of LF, f_esc_gamma, M*-z slope, alpha

    Calculate ionizing emissivity history (from LF, for example) Ndot(z).
    Integrate ionizing emissivity to find x_tot(z). 
    Integrate recombinations to find x(z).
    """
    
    def __init__(self, z, lf_params, cosmo,
                 lf_args={}, lf_maglim=None, MStar_z_slope=None, alpha=None,
                 f_esc_gamma=None, clump_fact=None, clump_fact_func=None):
        """
        z - array of redshift values
        lf_params - dictionary with keys 'z', 'MStar', 'phiStar', 'alpha'
        f_esc_gamma - ionizing photon escape fraction
        cosmo - cosmological parameter dict (see cosmolopy.parameters)
        lf_args - dict of keyword args for luminosityfunction.LFHistory
        lf_maglim - magnitude limit for interating LF
        MStar_z_slope - slope of high-z M* extrapolation
        alpha - faint-end slope of the LF
        """

        self.z = z
        self.lf_params = lf_params
        self.cosmo = cosmo
        self.lf_args = lf_args
        self.lf_maglim = lf_maglim
        self.MStar_z_slope = MStar_z_slope
        self.alpha = alpha
        self.f_esc_gamma = f_esc_gamma
        self.clump_fact = clump_fact
        self.clump_fact_func = clump_fact_func

        ### Set up galaxy luminosity function history. ###
        if alpha is not None:
            lf_params = copy.deepcopy(lf_params)
            lf_params['alpha'] = numpy.array(lf_params['alpha'])
            lf_params['alpha'][:] = alpha
            self.lf_params = lf_params

        if MStar_z_slope is not None:
            lf_args = copy.deepcopy(lf_args)
            if not 'extrap_args' in lf_args:
                lf_args['extrap_args'] = {}
            lf_args['extrap_args']['slopes'] = [0, MStar_z_slope]
            lf_args['extrap_var'] = 'z'
            self.lf_args = lf_args

        args = copy.deepcopy(self.cosmo)
        args.update(lf_args)
        self.lfh = luminosityfunction.LFHistory(params=lf_params,**args)

        # Ionizing photon emissivity. 
        self.NdotTot = self.lfh.iPhotonRateDensity_z(z, lf_maglim)

        # Integrate ionization.
        self.xTot = self.lfh.ionization(z, lf_maglim)

        if f_esc_gamma is not None:
            self.xEsc = self.xTot * f_esc_gamma
            self.integrate_ion_recomb()
            self.optical_depth()
            self.optical_depth(use_recomb=False)

    def make_ion_func(self):
        self.xEsc = self.xTot * self.f_esc_gamma
        ion_func = utils.Extrapolate1d(self.z, self.xEsc,
                                       bounds_behavior=[0.0,
                                                        'extrapolate']
                                       )
        return ion_func

    def make_clump_fact_func(self):
        if self.clump_fact is not None:
            self.clump_fact_func = lambda z1: self.clump_fact
        if self.clump_fact_func is None:
            self.clump_fact_func = cr.clumping_factor_HB
        return self.clump_fact_func
    
    def integrate_ion_recomb(self):
        """Integrate the recombination rate.

        Returns and stores results.
        """
        self.make_clump_fact_func()
        ion_func = self.make_ion_func()
        
        x, w, t = cr.integrate_ion_recomb(self.z,
                                          ion_func=ion_func,
                                          clump_fact_func =
                                          self.clump_fact_func,
                                          **self.cosmo)
        self.xr = x
        self.w = w
        self.t = t
        return x, w, t

    def optical_depth(self, use_recomb=True, store=True):
        """Calculate Thompson electron scattering optical depth."""

        # First calculate the optical depth from the fully-ionized
        # epoch up to our minimum redshift.
        tau_z0 = cr.optical_depth_instant(self.z[-1], **self.cosmo)

        # Then the contribution from the partially-ionized epoch.
        if use_recomb:
            xPhys = self.xr.copy()
        else:
            xPhys = self.xEsc.copy()
        xPhys[xPhys > 1.0] = 1.0
        tau_later = (tau_z0 +
                     cr.integrate_optical_depth(xPhys[...,::-1], 
                                                xPhys[...,::-1], 
                                                self.z[::-1],
                                                **self.cosmo))
        tau = tau_later[...,::-1]
        if store:
            if use_recomb:
                self.tau = tau
            else:
                self.tauEsc = tau
        return tau

    def reionization_redshift(self):
        """Find the redshift at with xr reaches unity."""
        z_ion_func = utils.Extrapolate1d(self.xr, self.z)
        self.z_reion = z_ion_func(1.0)
        return self.z_reion
                        
class EnrichmentHistory(Saveable):
    """Calculate CIV histories based on ionization.
    """

    def __init__(self, ionHist, f_x_Z, z_change=None, f_change=None,
                 delay_kernel=None,
                 ):
        """Calculate instantaneos histories based on ionization.

        Enrichment parameters enter here, e.g. f_x_Z, z_change, f_change.
        
        Omega_Z is fraction of critical density contributed by metals
        = nz * mz / rho_crit.

        Z can just as well represent a specific metal element or
        ionization species given the appropriatly decreased f_x_Z.
        
        """
        self.ionHist = ionHist
        self.f_x_Z = f_x_Z
        self.z_change = z_change
        self.f_change=f_change
        self.z = ionHist.z
        self.calculate_Omega_Z()
        self.delay_kernel = delay_kernel
        if delay_kernel is not None:
            self.convolve()

    def calculate_Omega_Z(self):
        if self.z_change is not None:
            print "f_x_Z changes from %.3g to %.3g (f = %.3g) at z<%.2f" % \
                  (self.f_x_Z,
                   self.f_x_Z * self.f_change,
                   self.f_change, self.z_change)
            xfunc = scipy.interpolate.interp1d(self.ionHist.z[::-1],
                                               self.ionHist.xTot[::-1])
            x_change = xfunc(self.z_change)
            xTot = self.ionHist.xTot
            
            f_early = self.f_x_Z
            f_late = self.f_x_Z * self.f_change
            self.Omega_Z = ((self.z > self.z_change) * f_early * xTot +
                            (self.z <= self.z_change) *
                            (f_late * xTot + (f_early - f_late) * x_change))
        else:
            print "f_x_Z = %.3g" % (self.f_x_Z)
            self.Omega_Z = self.f_x_Z * self.ionHist.xTot
        return self.Omega_Z
    
    def convolve(self):
        """Convolve delay curve with metal histories.
        Convolve Omega_Z with delay curve to produce Omega_Z_convolved. 
        """
        self.Omega_Z_convolved = padconvolve.padded_convolve(self.Omega_Z,
                                                             self.delay_kernel, 
                                                             origin=0,
                                                             value=0.0)
        return self.Omega_Z_convolved

    def binned_Omega_Z(self, zmin, zmax, useConvolved=True):
        if useConvolved:
            o = self.Omega_Z_convolved
        else:
            if hasattr(self, 'Omega_Z_unconvolved'):
                o = self.Omega_Z_unconvolved
            else:
                o = self.Omega_Z
        mask = numpy.logical_and(self.z >= zmin,
                                 self.z < zmax)
        return numpy.mean(o[mask])

    def fit_f_change(self, z_bin, Omega_Z_obs,
                     min_f_change=0.0,
                     max_f_change=None,
                     useConvolved=True):
        """Fit f_change to an observed value of Omega_Z_obs in redshift bin
        z_bin[0]--z_bin[1].
        """
        if not hasattr(self, 'original_f_change'):
            self.original_f_change = self.f_change
        def fitfunc(fc):
            self.f_change = fc
            self.calculate_Omega_Z()
            if useConvolved:
                self.convolve()
            return (self.binned_Omega_Z(z_bin[0], z_bin[1], useConvolved) -
                    Omega_Z_obs)

        if fitfunc(min_f_change) > Omega_Z_obs:
            print "Warning: can't match Omega_Z with f_change > %.3g" \
                  % (min_f_change)
            self.f_change = min_f_change
            return min_f_change
        
        if max_f_change is None:
            max_f_change = 3. * self.original_f_change
            ff = fitfunc(max_f_change)
            while ff < 0:
                print " max_f_change = %.3g too small (diff = %.3g)..." \
                      % (max_f_change, ff)
                max_f_change = 3. * max_f_change
                ff = fitfunc(max_f_change)
        print " considering f_change between %.3g and %.3g" % (min_f_change,
                                                               max_f_change)
        f_change_fit = scipy.optimize.minpack.bisection(fitfunc,
                                                       min_f_change,
                                                       max_f_change,
                                                       xtol=self.original_f_change*1e-3)
        if useConvolved:
            self.f_change = f_change_fit
        else:
            self.f_change_unconvolved = f_change_fit
            self.Omega_Z_unconvolved = self.Omega_Z
        return f_change_fit

    def fit_f_x_Z(self, z_bin, Omega_Z_obs,
                    min_f_x_Z=0.0,
                    max_f_x_Z=None,
                    useConvolved=True):
        """Fit f_x_Z to an observed value of Omega_Z_obs in redshift bin
        z_bin[0]--z_bin[1].
        """
        if not hasattr(self, 'original_f_x_Z'):
            self.original_f_x_Z = self.f_x_Z
        def fitfunc(fxc):
            self.f_x_Z = fxc
            self.calculate_Omega_Z()
            if useConvolved:
                self.convolve()
            return (self.binned_Omega_Z(z_bin[0], z_bin[1], useConvolved) -
                    Omega_Z_obs)

        if fitfunc(min_f_x_Z) > Omega_Z_obs:
            print "Warning: can't match Omega_Z with f_x_Z > %.3g" \
                  % (min_f_x_Z)
            self.f_x_Z = min_f_x_Z
            return min_f_x_Z
        
        if max_f_x_Z is None:
            max_f_x_Z = 3. * self.original_f_x_Z
            ff = fitfunc(max_f_x_Z)
            while ff < 0:
                print " max_f_x_Z = %.3g too small (diff = %.3g)..." \
                      % (max_f_x_Z, ff)
                max_f_x_Z = 3. * max_f_x_Z
                ff = fitfunc(max_f_x_Z)
        print " considering f_x_Z between %.3g and %.3g" % (min_f_x_Z,
                                                               max_f_x_Z)
        f_x_Z_fit = scipy.optimize.minpack.bisection(fitfunc,
                                                       min_f_x_Z,
                                                       max_f_x_Z,
                                                       xtol=self.original_f_x_Z*1e-3)
        self.f_x_Z = f_x_Z_fit
        return self.f_x_Z

class EnrichmentHistoryCollection(Saveable):
    def __init__(self, ionHists, delay_kernels, f_x_Zs,
                 z_changes, f_changes,
                 lf_names=None, dl_names=None, en_names=None):
        self.ionHists = ionHists
        self.delay_kernels = delay_kernels
        self.f_x_Zs = f_x_Zs
        self.z_changes = z_changes
        self.f_changes = f_changes
        self.lf_names=lf_names
        self.dl_names=dl_names
        self.en_names=en_names
        
        self.nLF = len(ionHists)
        self.nDl = len(delay_kernels)
        self.nEn = len(f_x_Zs)

        EnHists = []
        for iLF in range(self.nLF):
            if lf_names is not None:
                print "Using %s." % lf_names[iLF]
            square = []
            for iDl in range(self.nDl):
                if lf_names is not None:
                    print "Using %s." % dl_names[iDl]
                line = []
                for iEn in range(self.nEn):
                    if en_names is not None:
                        print "Using %s." % en_names[iEn]
                    line.append(EnrichmentHistory(ionHists[iLF],
                                                  f_x_Zs[iEn],
                                                  z_changes[iEn],
                                                  f_changes[iEn],
                                                  delay_kernels[iDl]))
                square.append(line)
            EnHists.append(square)
        self.EnHists = EnHists

    def fit_f_change(self, z_bin, Omega_Z_obs,
                     min_f_change=0.0, max_f_change=None):
        f_changes = []
        f_changes_unconvolved = []
        for iLF in range(self.nLF):
            square = []
            line_unc = []
            for iDl in range(self.nDl):
                line = []
                for iEn in range(self.nEn):
                    if iDl == 0:
                        line_unc.append(\
                            self.EnHists[iLF][iDl][iEn].fit_f_change(z_bin,
                                                                   Omega_Z_obs,
                                                                   min_f_change,
                                                                   max_f_change,
                                                            useConvolved=False))
                    line.append(\
                        self.EnHists[iLF][iDl][iEn].fit_f_change(z_bin,
                                                                 Omega_Z_obs,
                                                                 min_f_change,
                                                                 max_f_change,
                                                                 useConvolved=True))

                square.append(line)
            f_changes.append(square)
            f_changes_unconvolved.append(line_unc)
        self.f_changes = numpy.array(f_changes)
        self.f_changes_unconvolved = numpy.array(f_changes_unconvolved)
        return f_changes, f_changes_unconvolved

    def fit_f_x_Z(self, z_bin, Omega_Z_obs,
                    min_f_x_Z=0.0, max_f_x_Z=None,
                    useConvolved=True):
        f_x_Zs = []
        for iLF in range(self.nLF):
            square = []
            for iDl in range(self.nDl):
                line = []
                for iEn in range(self.nEn):
                    line.append(\
                        self.EnHists[iLF][iDl][iEn].fit_f_x_Z(z_bin,
                                                                Omega_Z_obs,
                                                                min_f_x_Z,
                                                                max_f_x_Z,
                                                                useConvolved))
                square.append(line)
            f_x_Zs.append(square)
        self.f_x_Zs = f_x_Zs
        return f_x_Zs
    
    def convolve(self):
        Omega_Zs_convolved = []
        for iLF in range(self.nLF):
            square = []
            for iDl in range(self.nDl):
                line = []
                for iEn in range(self.nEn):
                    line.append(\
                        self.EnHists[iLF][iDl][iEn].convolve())
                square.append(line)
            Omega_Zs_convolved.append(square)
        self.Omega_Zs_convolved = Omega_Zs_convolved
        return self.Omega_Zs_convolved
