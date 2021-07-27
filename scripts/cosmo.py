import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import optimize
from astropy import constants as const
from astropy import units as u
import copy
from scipy.interpolate import RectBivariateSpline
import camb
from camb import model, initialpower


#Some default precision params

limits = {
    "pk_kmin"     : 1e-4, # 1/Mpc
    "pk_kmax"     : 40.,  # 1/Mpc
}


planck2018Params = {
    "ombh2"                 :0.0223, #  baryon physical density at z=0
    "omch2"                 : 0.1200,
    "omk"                   : 0.,     # Omega_K curvature paramter
    "mnu"                   : 0.06,   # sum of neutrino masses [eV]
    "nnu"                   : 3.046,  # N_eff, # of effective relativistic dof
    "TCMB"                  : 2.725,  # temperature of the CMB in K at z=0
    "H0"                    : 67.3,   # Hubble's constant at z=0 [km/s/Mpc]
    "w"                     : -1.0,   # dark energy equation of state (fixed throughout cosmic history)
    "wa"                    : 0.,     # dark energy equation of state (fixed throughout cosmic history)
    "cs2"                   : 1.0,    # dark energy equation of state (fixed throughout cosmic history)
   "tau"                   : 0.054,   #  optical depth
    "deltazrei"             : None,   # z-width of reionization
    "YHe"                   : None,   # Helium mass fraction
    "As"                    : 2.1e-9, # comoving curvature power at k=piveo_scalar
    "ns"                    : 0.964,  # scalar spectral index
    "nrun"                  : 0.,     # running of scalar spectral index
   "nrunrun"               : 0.,     # running of scalar spectral index
    "r"                     : 0.,     # tensor to scalar ratio at pivot scale
    "nt"                    : None,   # tensor spectral index
    "ntrun"                 : 0.,     # running of tensor spectral index
   "pivot_scalar"          : 0.05,   # pivot scale for scalar spectrum 
    "pivot_tensor"          : 0.05,   # pivot scale for tensor spectrum 
   "meffsterile"           : 0.,     # effective mass of sterile neutrinos
    "num_massive_neutrinos" : 1,      # number of massive neutrinos (ignored unless hierarchy == 'degenerate')
    "standard_neutrino_neff": 3.046,
    "gamma0"                : 0.55,   # growth rate index
   "gammaa"                : 0.,     # growth rate index (series expansion term)
    "neutrino_hierarchy": 'degenerate', # degenerate', 'normal', or 'inverted' (1 or 2 eigenstate approximation)
    }
 


class Cosmology(object):
	""" 
	Class to compute cosmological quantities according to the cosmological parameters. 
	Most of the functions are wrapper around CAMB outputs
 	Attributes
	----------
	params : dictionary
		Cosmological parameters to initialize the class

 	"""
	def __init__(self, params=None, lmax=5000, cmbonly=False, **kwargs): 

		self.lmax = lmax

		# Setting cosmo params
		if params is None:
			params = planck2018Params.copy()
		else:
			for key, val in planck2018Params.items():#.iteritems():
				params.setdefault(key,val)

		self.params_dict = params.copy()

        #To avoid future issues with camb- it may not interpret this params as the inputs:
		for par in list(params): 
			if par == 'gamma0':
				params.pop(par)
			if par == 'gammaa':
				params.pop(par)
			if par == 'w':
				params.pop(par)
			if par == 'wa':
				params.pop(par)
			if par == 'cs2':
				params.pop(par)
			if par == '100theta':
				params.pop(par)


		# Initialize CAMB
		pars = camb.CAMBparams()
		pars.set_cosmology(H0=params['H0'],
						   ombh2=params['ombh2'],
						   omch2=params['omch2'],
						   omk=params['omk'],
						   mnu=params['mnu'],
						   tau=params['tau'],
						   nnu=params['nnu'],
						   TCMB=params['TCMB'],
						   YHe=params['YHe'],
						   meffsterile=params['meffsterile'],
						   standard_neutrino_neff=params['standard_neutrino_neff'],
						   num_massive_neutrinos=params['num_massive_neutrinos'],
						   neutrino_hierarchy=params['neutrino_hierarchy'],
						   deltazrei=params['deltazrei'])

		pars.InitPower.set_params(As=params['As'], 
								  ns=params['ns'],
								  nrun=params['nrun'], 
								  nrunrun=params['nrunrun'], 
								  r=params['r'], 
								  nt=params['nt'], 
								  ntrun=params['ntrun'],
								  pivot_scalar=params['pivot_scalar'], 
								  pivot_tensor=params['pivot_tensor'],)
 
		pars.set_for_lmax(lmax=self.lmax, lens_potential_accuracy=1.0)

		if params['r'] != 0:
			pars.WantTensors = True

		self.pars = pars.copy()

		# Background quantities such as distances etc: 
		self.bkd = camb.get_background(pars)

		# Derived params
		for parname, parval in self.bkd.get_derived_params().items():
			setattr(self, parname, parval) 

		# Initializing P(z,k) Matter Power spectrum spline up to LSS
		self.kmin = limits['pk_kmin']
		self.kmax = limits['pk_kmax']
		nonlinear = False # Linear matter power spectrum as a defaults 
 
		self.pars.NonLinear_lens = 0

		if not cmbonly:
			for kw in kwargs:
				if kw == 'pk_kmin':
					self.kmin = kwargs[kw]
				if kw == 'pk_kmax':
					self.kmax = kwargs[kw]
				if kw == 'NonLinear' or kw == 'nonlinear':
					nonlinear = True
					self.pars.NonLinearModel.set_params(halofit_version='takahashi')  
				if kw == 'NonLinearLens' or kw == 'nonlinearlens':
					self.pars.NonLinear_lens = 3           
					self.pars.NonLinear = model.NonLinear_both

			self.pkz = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zstar)

			# Shortcuts for some params 
			self.omegab = self.pars.omegab
			self.omegac = self.pars.omegac
			self.omegam = self.omegab + self.omegac
			self.omnuh2 = self.pars.omnuh2
			self.omk = self.pars.omk
			self.H0     = self.pars.H0
			self.h      = self.H0/100.
			self.ns     = self.pars.InitPower.ns
			self.As     = self.pars.InitPower.As
			self.r      = self.pars.InitPower.r
			self.w      = self.params_dict['w']
			self.wa     = self.params_dict['wa']
			self.gamma0 = self.params_dict['gamma0']
			self.gammaa = self.params_dict['gammaa']

			# Get background quantities splines
			chis = np.linspace(0., self.bkd.comoving_radial_distance(2000), 300)
			zs   = self.bkd.redshift_at_comoving_radial_distance(chis)
			zs[0] = 0.
			res  = self.bkd.get_background_redshift_evolution(zs) 
			self.spline_f_K = interpolate.UnivariateSpline(zs, [self.bkd.comoving_radial_distance(z) for z in zs], k=2) #transverse comoving radial distance
			self.spline_d_L = interpolate.UnivariateSpline(zs, [self.bkd.luminosity_distance(z) for z in zs], k=2) #luminosity distance out to redshift z
			self.spline_d_A = interpolate.UnivariateSpline(zs, [self.bkd.angular_diameter_distance(z) for z in zs], k=2) #angular diameter distance
			self.spline_t_z = interpolate.UnivariateSpline(zs, [self.bkd.physical_time(z) for z in zs], k=2)
			self.spline_H_z = interpolate.UnivariateSpline(zs, [self.bkd.hubble_parameter(z) for z in zs], k=2)
			self.spline_x_e = interpolate.interp1d(zs, res['x_e'], 'cubic') #ionization fraction
			self.spline_z_chi = interpolate.UnivariateSpline(chis, zs, k=2)

			del pars, 
		else:
			pass

 

 	#Get some useful quantities:
	def d_A(self, z):  
		""" 
		Returns the angular diameter distance out to redshift z [Mpc].
		"""
		return  self.spline_d_A(z)

	def d_A12(self, z1, z2):  
		""" 
		Returns the angular diameter distance between redshift z1 and z2 [Mpc].
		"""
		return  self.bkd.angular_diameter_distance2(z1,z2)

	def f_K(self, z): # [Mpc]
		""" 
		Returns the transverse comoving radial distance out to redshift z [Mpc].
		"""
		return  self.spline_f_K(z)

	def H_a(self, a): # [km/s/Mpc]
		""" 
		Returns the hubble factor at scale factor a=1/(1+z) [km/s/Mpc]. 
		"""
		return self.spline_H_z(np.nan_to_num(1./a - 1.))

	def H_z(self, z): # [km/s/Mpc]
		""" 
		Returns the hubble factor at redshift z [km/s/Mpc]. 
		"""
		return self.spline_H_z(z)

	def H_x(self, x): # [km/s/Mpc] 
		""" 
		Returns the hubble factor at conformal distance x (in Mpc) [km/s/Mpc]. 
		"""
		return self.spline_H_z(self.spline_z_chi)

	def E_z(self, z): # [unitless]
		""" 
		Returns the unitless Hubble expansion rate at redshift z 
		"""
		return  self.H_z(z) / self.H0


	def D_z_norm(self, z, gamma0=0.55, gammaa=0.):
		""" 
		Returns the normalized growth factor at redshift z
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			def func(x, gamma0, gammaa): 
				return self.f_z(x, gamma0=gamma0,gammaa=gammaa)/(1+x)
			return np.exp( -integrate.quad( func, 0, z, args=(gamma0,gammaa,))[0])
		else:
			return np.asarray([ self.D_z_norm(tz, gamma0=gamma0, gammaa=gammaa) for tz in z ])

	def f_z(self, z, gamma0=None, gammaa=None): 
		"""
		Returns the growth rate (eq. Linder) [unitless]

		f(z) = d\ln{D}/d\ln{a} = \Omega_m(z)^\gamma(z),
		
		where the growth index (gamma) can be expanded as

		\gamma(z) = \gamma_0 - z/(1+z)\gamma_a
		"""
		if gamma0 is None: 
			gamma0 = self.gamma0

		if gammaa is None: 
			gammaa = self.gammaa

		gamma = gamma0 + z/(1+z)*gammaa

		return (self.omegam*(1+z)**3/self.E_z(z)**2)**gamma

 

 
