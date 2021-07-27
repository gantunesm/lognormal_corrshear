from scipy.special import erf
from scipy.optimize import brentq
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import constants as const
from astropy import units as u


#Dict with some default precision params
limits = {
    "dNdz_zmin"     : 0.0,
    "dNdz_zmax"     : 2.5,
    "lens_zmax"   : 5.,
    "magbias_npts"  : 300,
    "lens_npts"     : 300,
    "kernel_npts"   : 30,
    "dNdz_precision": 1.48e-8,
    "lens_precision": 1.48e-6,
    "global_precision": 1.48e-32, 
    "divmax":10 }
    
class Properties(object):
    """
    Base class for redshift distributions.

    Attributes
    ----------
    z_min : float 
        Minimum redshift  

    z_max : float
        Maximum redshift 

    Derived quantities:
    --------------------
    norm : float
        Normalization factor of the redshift distribution

    z_med : float
        Median redshift  

    z_mean : float
        Mean redshift  

    """
    def __init__(self, z_min= limits['dNdz_zmin'], z_max=limits['dNdz_zmax']):

        self.z_min   = z_min
        self.z_max   = z_max
        self.norm    = 1.0
        self._z_med  = None
        self._z_mean = None

        self.normalize()

    def normalize(self):
        """
        Compute the normalization factor for the redshift distribution in the range [z_min, z_max]
        """

        norm = integrate.quad( self.raw_dndz, self.z_min, self.z_max, epsabs=0., epsrel=1e-5, points=[0.2,0.4,0.6,0.8,1.])[0]  
        self.norm = 1.0/norm

    def raw_dndz(self, z):
        """
        Raw definition of the redshift distribution (overwritten in the derived classes)
        """
        return 1.0

    def dndz(self, z):
        """
        Normalized dn/dz PDF
        """
        return np.where(np.logical_and(z <= self.z_max, z >= self.z_min), self.norm*self.raw_dndz(z), 0.0)

    @property
    def z_med(self):
        """ 
        Median of the redshift distribution 
        """
        if self._z_med is None:
            f = lambda x: integrate.romberg(self.dndz, self.z_min, x) - 0.5
            self._z_med = brentq(f, self.z_min, self.z_max)

        return self._z_med

    @property
    def z_mean(self):
        """ 
        Mean of the redshift distribution
        """
        if self._z_mean is None:
            self._z_mean = integrate.romberg(lambda z: z * self.dndz(z), self.z_min, self.z_max)
        return self._z_mean
        


class Tomography(Properties):
    """
    Class for a tomographic redshift distribution derived from the Properties class.

    Attributes
    ----------
    z_min : float 
        Minimum redshift considered in the survey

    z_max : float
        Maximum redshift considered in the survey    

    b_zph : float
        Photo-z error bias

    sigma_zph : float
        Photo-z error scatter (=> sigma_zph * (1+z))

    nbins : float
        Number of equally spaced redshift bins to consider

    bins : array-like
        List of redshift bins edges (=> [z_min, z_1, z_2, ..., z_n, ..., z_max])
    * Input can be either nbins or bins

    """
    def __init__(self, z_min=limits['dNdz_zmin'], 
                       z_max=limits['dNdz_zmax'],
                       b_zph=0., 
                       sigma_zph=0.,
                       nbins=1,
                       bins=None):
            
        super(Tomography, self).__init__(z_min, z_max)

        self.b_zph     = b_zph
        self.sigma_zph = sigma_zph

        if bins is None:
            dz   = (z_max - z_min) / nbins
            bins =  np.asarray([z_min + dz*i for i in range(nbins+1)]) 
        else:
            nbins = len(bins) - 1

        self.bounds = [(bins[i],bins[i+1]) for i in range(nbins)]
        self.bins   = bins
        self.nbins  = nbins

        self._z_med_bin  = None
        self._z_mean_bin = None

        self.normalize_bins()

    def raw_dndz_bin(self, z, i):
        """
        Un-normalized redshift distribution within the photo-z bin i, see Eq.(6) and (7) from astro-ph/0506614
        """
        x_min = (self.bounds[i][0] - z + self.b_zph) / (np.sqrt(2.) * self.sigma_zph*(1+z))
        x_max = (self.bounds[i][1] - z + self.b_zph) / (np.sqrt(2.) * self.sigma_zph*(1+z))
        n_z   = self.raw_dndz(z)

        return np.nan_to_num( 0.5 * n_z * ( erf(x_max) - erf(x_min) ))

    def dndz_bin(self, z, i):
        """
        Normalized PDF for the photo-z bin i
        """
        return np.where(np.logical_and(z <= self.z_max, z >= self.z_min), self.norm_bin[i]*self.raw_dndz_bin(z,i), 0.0)

    def normalize_bins(self):
        """
        Compute the normalization factors for the photo-z bins the range [z_min, z_max]
        """
        self.norm_bin = np.ones(self.nbins)
        for i in range(self.nbins):
            f = lambda z: self.raw_dndz_bin(z, i)

            norm = integrate.simps(f(np.linspace(self.z_min,self.z_max,1000)), x=np.linspace(self.z_min,self.z_max,1000))

            
            self.norm_bin[i] = 1.0/norm
            print(self.norm_bin[i])

    
    def z_med_bin(self, i):
        """ 
        Median of the redshift distribution for bin i
        """
        if self._z_med_bin is None:
            self._z_med_bin = np.zeros(self.nbins)
            for i in range(self.nbins):
                u = lambda y: self.dndz_bin(y, i)
                f = lambda x: integrate.romberg(u, self.bounds[i][0], x) - 0.5
                self._z_med_bin[i] = brentq(f, self.bounds[i][0], self.bounds[i][1])

        return self._z_med_bin[i]

    
    def z_mean_bin(self, i):
        """ 
        Mean of the redshift distribution for bin i
        """
        if self._z_mean_bin is None:
            self._z_mean_bin = np.zeros(self.nbins)
            for i in range(self.nbins):
                f = lambda z: z * self.dndz_bin(z,i)
                self._z_mean_bin[i] = integrate.quad(f, self.z_min, self.z_max)[0]

        return self._z_mean_bin[i]        
        
        
        
class dNdz(Tomography):
    """
    Class for a p(z) derived from real data assuming:
    - array of redshifts 
    - dN/dz (or probabilities) for each redshift

    Attributes
    ----------
    z_array   : Array of redshifts
    
    dndz_array: Array of weights

    z_min : Minimum redshift 

    z_max : Maximum redshift 

    b_zph : Photo-z error bias

    sigma_zph : Photo-z error scatter (=> sigma_zph * (1+z))

    nbins : Number of equally spaced redshift bins to consider

    bins : List of redshift bins edges (=> [z_min, z_1, z_2, ..., z_n, ..., z_max])
 
    """

    def __init__(self, z_array, dndz_array,
                       z_min= limits['dNdz_zmin'], 
                       z_max= limits['dNdz_zmax'],
                       b_zph=0., 
                       sigma_zph=0.,
                       nbins=1,
                       bins=None):

        self.z_array    = z_array
        self.dndz_array = dndz_array

        self._p_of_z = interpolate.interp1d(z_array, dndz_array, bounds_error=False, fill_value=0.)

        super(dNdz, self).__init__(z_min, z_max, b_zph, sigma_zph, nbins, bins)

    def raw_dndz(self, z):
        return self._p_of_z(z)        
        
 
 
 

class Kernels(object):
    """
    Class for observables kernels.
    
    A kernel is defined over the redshift range [z_min, z_max] and can be composed by more than one bin.
     
    Attributes
    ----------
    z_min : float
        Minimum redshift defined by the kernel

    z_max : float
        Maximum redshift defined by the kernel

    nbins : float
        Number of z-bins
    """
    def __init__(self, z_min, z_max, nbins=1):

        self.initialized_spline = False

        self.z_min = z_min
        self.z_max = z_max
        self.nbins = nbins

        self._z_array  = np.linspace(self.z_min, self.z_max, limits['kernel_npts'])
        self._wz_array = np.zeros((len(self._z_array),self.nbins), dtype='float128')

    def _initialize_spline(self):
        self._wz_spline = []

        for idb in range(self.nbins):
            for idz in range(self._z_array.size):
                self._wz_array[idz,idb] = self.raw_W_z(self._z_array[idz], i=idb)
            self._wz_spline.append(InterpolatedUnivariateSpline(self._z_array, self._wz_array[:,idb], k=1))

        self.initialized_spline = True

    def raw_W_z(self, z, i=0):
        """
        Raw window function.
        """
        return 1.0

    def W_z(self, z, i=0):
        """
        Wrapper for splined window function.
        """
        if not self.initialized_spline:
            self._initialize_spline()
        return np.where(np.logical_and(z >= self.z_min, z <= self.z_max), self._wz_spline[i](z), 0.0)
 


class LensGal(Kernels):
    """ 
    Redshift kernel for a galaxy lensing survey.
    """
    def __init__(self, cosmo, tomo, z_max_lens= limits['lens_zmax'], npts=limits['lens_npts']):
        """
        Attributes
        ----------
        cosmo :  Cosmology object (from Cosmology class)

        tomo : Properties object

        b :  Linear bias parameter 

        alpha : Magnification bias parameter ( slope of the integrated number counts as function
            of the flux N(>S) \propto S^{-\alpha}) 

        z_max_lens : Upper limit of magnification bias integral (performed with Simpson integration)	

        npts : int
            Number of points to sample the magnification bias integral (performed with Simpson integration)	
        """
        self.cosmo = cosmo
        self.tomo = tomo
        self.z_max_lens = z_max_lens
        self.npts = npts
        self.tomo.normalize()

        self.fac = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.H0**2.) / (const.c.to('km/s').value)  

        super(LensGal, self).__init__(1e-5, tomo.z_max)

    def raw_W_z(self, z, i=0):
        return self.fac / self.cosmo.H_z(z) * self.cosmo.f_K(z) * (1.+z) * self.W_lens(z, zmax=self.z_max_lens, npts=self.npts)

    def W_lens(self, z, zmax= limits['lens_zmax'], npts=limits['lens_npts']):
        """
        Computes the galaxy lensing kernel
        """
        if np.isscalar(z) or (np.size(z) == 1):
            lens_integrand = lambda zprime:  (1.-self.cosmo.f_K(z)/self.cosmo.f_K(zprime)) * self.tomo.dndz(zprime)
            return integrate.quad(lens_integrand, z, zmax, epsabs=limits["global_precision"], epsrel=limits["lens_precision"])[0]
 
        else:
            return np.asarray([ self.W_lens(tz) for tz in z ])

  
