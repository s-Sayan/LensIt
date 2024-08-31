import numpy as np
from lensit.clusterlens import constants as const
from scipy.special import sici
from lensit.misc.misc_utils import Freq
from camb import CAMBdata
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d


class profile(object):
    def __init__(self, cosmo:CAMBdata, pname='nfw'):
        r"""Lensing convergence field from a dark matter halo.

            The profile follows the definitions fo Geach and Peacock 2017

            Args:
                cosmo: an instantiated camb.CAMBdata object
                pname: String, could use different halo profile (for now only NFW)
        """
        
        self.pname = pname
        self.cosmo = cosmo
        self.h = self.cosmo.Params.H0 / 100  
        chi_cmb = cosmo.conformal_time(0)- cosmo.tau_maxvis
        self.zcmb = cosmo.redshift_at_comoving_radial_distance(chi_cmb)
        self.sigmat = 6.6524587158e-29  # Thomson cross section in m^2
        self.me = 9.10938356e-31  # electron mass in kg
        self.c = 299792458  # speed of light in m/s
        self.tcmb = 2.7255  # CMB temperature in K
        self.h = 6.62607015e-34  # Planck constant in J·s
        self.k_B = 1.380649e-23  # Boltzmann constant in J/K
        self.h70 = self.cosmo.Params.H0 / 100

        if pname == 'nfw':
            self.kappa = self.kappa_nfw

    def kappa_nfw(self, M200, z, R, xmax=None):
        """Convergence profile of a cluster for a NFW density profile
            Args:
                M200: mass (in Msol) of the cluster in a sphere of density 
                    200 times the critical density of the universe
                z: redshift of the cluster
                R: array, distance from the center of the cluster in Mpc 
        """
        return self.sigma_nfw(M200, z, R, xmax=xmax) / self.sigma_crit(z)
        #return self.sigma_int(M200, z, R) / self.sigma_crit(z)

    def get_rho_s(self, M200, z, const_c=None):
        c = self.get_concentration(M200, z, const_c)
        return 200* self.rho_crit(z)/3 * c**3 / (np.log(1+c) - c/(1+c))

    def get_rs(self, M200, z, const_c=None):
        return self.get_r200(M200, z) / self.get_concentration(M200, z, const_c)
    
    def get_thetas_amin(self, M200,z, const_c=None):
        return self.r_to_theta(z, self.get_rs(M200, z, const_c))

    def get_r200(self, M200, z):
        """Get r200 in Mpc"""
        mthreshold = 200
        return (M200 * 3 / (4 * np.pi * mthreshold * self.rho_crit(z)))**(1/3)        

    def rho_crit(self, z):
        """The critical density of the universe at redshift z
        in units of :math:`M_{\odot} / {\\rm Mpc}^3
        """
        return const.rhocrit * (self.cosmo.hubble_parameter(z))**2

    def get_concentration(self, M200, z, const_c=None):
        """Get the concentration parameter as in Geach and Peacock eq. 5.
        M200: In units of solar mass."""
        if const_c is None:
            const_c = 5.71 * (1 + z)**(-0.47) * (M200 / (2E12 / self.h))**(-0.084)

        return const_c

    def sigma_crit(self, z):
        """Critical surface mass density, in Msun/Mpc^2"""
        Dang_OS = self.cosmo.angular_diameter_distance(self.zcmb)
        Dang_OL = self.cosmo.angular_diameter_distance(z)
        Dang_LS = self.cosmo.angular_diameter_distance2(z, self.zcmb)
        return const.c_Mpcs**2 /(4*np.pi*const.G_Mpc3_pMsol_ps2) *  Dang_OS/Dang_OL/Dang_LS
    
    def get_kappa0(self, M200, z, xmax=None):
        """Return the value of the lensing convergence profile for x = 1, i.e. R = rs"""
        thsc = self.get_thetas_amin(M200,z)
        return self.kappa_theta(M200, z, thsc, xmax=xmax)[0]

    def sigma_nfw(self, M200, z, R, xmax=None):
        """Analytic expression for the surface mass desinty of a NFW profile 
        From Equation 7 of Bartelmann 1996
        Args:
            M200: mass (in Msol) of the cluster in a shpere of density 200 times the critical density
            z: redshift of the cluster
            R: distance from the center of the cluster in Mpc
        Returns:
            sigma: surface mass density  in Msun / Mpc^2
        """
        rs = self.get_rs(M200, z)
        rhos = self.get_rho_s(M200, z)
        R = np.atleast_1d(R)
        x = R / rs 
        if xmax is None:
            f = self.fx(x)
        else:
            f = self.gx(x, xmax)
        c = self.get_concentration(M200, z)
        sigma = 2*rhos*rs*f
        return sigma

    def fx(self, x, tol=6):
        """This integral of the NFW profile along the line of sight is integrated up to infinity
        See Bartelmann 1996 or Wright et al 1999"""
        ## Need to fix for M200, z, now it is for M200, z = 2*1e14, 0.7
        f = np.zeros_like(x)
        xp = np.where(x>1)
        xo = np.where(np.abs(x-1.) < 10**(-tol)) 
        xm = np.logical_and(x<1, x>=self.theta_amin_to_x(2*1e14, 0.7, 0.12))
        #xm = np.where(x<1)
        xd = np.where(x<self.theta_amin_to_x(2*1e14, 0.7, 0.12))
        idx = self.theta_amin_to_x(2*1e14, 0.7, 0.12)
        f[xp] = (1 - 2/np.sqrt(x[xp]**2 - 1) * np.arctan(np.sqrt((x[xp] - 1)/(x[xp] + 1))))/(x[xp]**2-1)
        f[xm] = (1 - 2/np.sqrt(1 - x[xm]**2) * np.arctanh(np.sqrt((1 - x[xm])/(1 + x[xm]))))/(x[xm]**2-1)
        f[xo] = 1/3
        f[xd] = (1 - 2/np.sqrt(1 - idx**2) * np.arctanh(np.sqrt((1 - idx)/(1 + idx))))/(idx**2-1)
        return f


    def gx(self, x, xmax, tol=5):
        """We apply a cutoff in the halo profile for x>xmax, i.e. R>rs*xmax
        See equation 27 of Takada and Jain 2003"""
        ## Need to fix for M200, z, now it is for M200, z = 2*1e14, 0.7
        g = np.zeros_like(x)
        xp = np.logical_and(x>1, x<=xmax)
        xo = np.where(np.abs(x-1.) < 10**(-tol))       
        xm = np.logical_and(x<1, x>=self.theta_amin_to_x(2*1e14, 0.7, 0.12))
        #xm = np.where(x<1)
        xd = np.where(x<self.theta_amin_to_x(2*1e14, 0.7, 0.12))
        idx = self.theta_amin_to_x(2*1e14, 0.7, 0.12)
        g[xp] = (np.sqrt(xmax**2 - x[xp]**2)/(1+xmax) - 1/np.sqrt(x[xp]**2 - 1) * np.arccos((x[xp]**2 + xmax) / (x[xp] * (1+xmax))) )/(x[xp]**2-1)
        g[xm] = (np.sqrt(xmax**2 - x[xm]**2)/(1+xmax) - 1/np.sqrt(1-x[xm]**2) * np.arccosh((x[xm]**2 + xmax) / (x[xm] * (1+xmax))) )/(x[xm]**2-1)
        g[xo] = np.sqrt(xmax**2 - 1)/(3*(1+xmax)) * (1+ 1/ (1+xmax))
        g[xd] = (np.sqrt(xmax**2 - idx**2)/(1+xmax) - 1/np.sqrt(1-idx**2) * np.arccosh((idx**2 + xmax) / (idx * (1+xmax))) )/(idx**2-1)
        return g



    def rho_nfw(self, M200, z, r):
        r"""Navarro Frenk and White density profile 
            Args:
                M200: mass (in Msol) of the cluster in a sphere of density 
                    200 times the critical density of the universe
                z: redshift of the cluster
                r: array, distance from the center of the cluster in Mpc 
        """
        rho_s = self.get_rho_s(M200, z)
        rs = self.get_rs(M200, z)
        rho = rho_s / ((r/rs)*(1+r/rs)**2)
        return rho

    def sigma_int(self, M200, z, R, xmax = 100, npoints=1000):
        """Integrate density over the line of sight 
        Args:
            M200: M200c mass of the cluster in Msol
            z: redshift of the cluster
            R: distance from the center of the cluster in Mpc
            xmax: maximum radius of the integration in factors of rs
            npoints: number of points in the integration
        Returns:
            sigma: surface mass density  in Msun / Mpc^2
        """
        rs = self.get_rs(M200, z)
        # assert rs*xmax >= np.max(R), "Need to increase integration limit"
        assert R[0] > 0, "Can't integrate on center of NFW profile, please give R>0"
        R = np.atleast_1d(R)
        sigma = np.zeros_like(R)
        for i, iR in enumerate(R):
            if iR>rs*xmax:
                sigma[i] = 0
            else:
                r_arr = np.geomspace(iR, xmax*rs, num = npoints)
                dr_arr = np.diff(r_arr, axis=0)
                # I skip the point r= R because the integrand is not defined, should check if corect?
                r_arr = r_arr[1:]
                rho_arr = self.rho_nfw(M200, z, r_arr)
                sigma[i] = np.trapz( 2 * r_arr * rho_arr / np.sqrt(r_arr**2 - iR**2), r_arr)
        return sigma

    def analitic_kappa_ft(self, M200, z, xmax, ell, const_c=None):
        """Analytic Fourier transform of the convergence fiels for a NFW profile
            from Oguri&Takada 2010, Eq.28"""
        c1 = self.get_concentration(M200, z, const_c)
        mu_nfw = np.log(1. + c1) - c1 / (1. + c1)
        rs = self.get_rs(M200, z)
        chi = self.cosmo.comoving_radial_distance(z)
        k = ell / chi
        x = ((1. + z) * k * rs)
        Six, Cix = sici(x)
        Sixpc, Cixpc = sici(x * (1. + xmax))
        Sidiff = Sixpc - Six
        Cidiff = Cixpc - Cix
        u0 = np.sin(x) * Sidiff + np.cos(x) * Cidiff - np.sin(x * xmax) / (x * (1. + xmax))
        ufft = 1. / mu_nfw * u0

        kappaft = M200 * ufft * (1+z)**2 /chi**2 / self.sigma_crit(z)
        return kappaft

    def kappa_theta(self, M200, z, theta, xmax=None):
        """Get the convergence profile
            Args:
                M200: Cluster mass defined as in a sphere 200 times the critical density 
                z: redshift of teh cluster
                theta: amgle in arcminute
                
            Returns:
                kappa: convergence profile
        """
        R = self.theta_amin_to_r(z, theta)
        return self.kappa(M200, z, R, xmax=xmax)

    def r_to_theta(self, z, R):
        """Convert a transverse distance into an angle
            Args:
                R: transverse distance in Mpc
                z: redshift of the cluster
            
            Returns:
                theta: angle on the sky in arcmin, with R = Dang(z) * theta"""

        Dang = self.cosmo.angular_diameter_distance(z)
        theta_rad = R / Dang
        return theta_rad *180*60/np.pi

    def theta_amin_to_r(self, z, theta_amin):
        """Convert a transverse distance into an angle
        Args:
            theta: angle on the sky in arcmin
            z: redshift of the cluster
        
        Returns:
            R: transverse distance in Mpc, with R = Dang(z) * theta"""

        Dang = self.cosmo.angular_diameter_distance(z)
        R = Dang * theta_amin * np.pi/180/60
        return R

    def theta_amin_to_x(self, M200, z, th_amin, const_c=None):
        """Angle substended at chararcteric scale x = r/rs

            Args:
                M200: cluster M200 mass in solar masses (?)
                z: cluster redshift
                x:  dimensionless R / Rs

        """
        return th_amin/self.get_thetas_amin(M200, z, const_c=const_c)



    def x_to_theta_amin(self,M200, z, x, const_c=None):
        """Angle substended at chararcteric scale x = r/rs

            Args:
                M200: cluster M200 mass in solar masses (?)
                z: cluster redshift
                x:  dimensionless R / Rs

        """
        return self.r_to_theta(z, x * self.get_rs(M200, z, const_c=const_c))

    def pix_to_theta(self, x, y, dtheta, c0):
        """Return the angle between the center of the map and the pixel (x,y)
        Args:
            x, y: pixel coordinates of the map
            dtheta: size 2 array, physical size of the pixels (dx, dy) 
            c0: size 2 array, central pixel of the map
        Returns:
            angle between the pixel and the center 
            """
        return np.sqrt((x-c0[0])**2 * dtheta[0]**2 + (y-c0[1])**2 * dtheta[1]**2)


    def kappa_map(self, M200, z, shape, lsides, xmax=None):
        """Get the convergence map of the cluster
            Args:
                M200: Cluster mass defined as in a sphere 200 times the critical density 
                z: redshift of the cluster
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
                xmax: cutoff scale in factors of rs
            Returns:
                kappa_map: numpy array defining the convergence field
        """
        dtheta_x = lsides[0]/shape[0] * 180/np.pi*60
        dtheta_y = lsides[1]/shape[1] * 180/np.pi*60
        # Center of the cluster
        #x0 = shape[0]/2
        #y0 = shape[1]/2       
        x0 = 0.
        y0 = 0.

        #X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        if shape[0] % 2 == 0 and shape[1] % 2 == 0:
            X, Y =  np.meshgrid(np.concatenate((np.arange(0,shape[0]//2), np.arange(-shape[0]//2,0))), np.concatenate((np.arange(0,shape[1]//2), np.arange(-shape[1]//2,0))))
        else:
            assert 0, "pixel size is not right"
        
        #X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        theta_amin = self.pix_to_theta(X, Y, (dtheta_x,dtheta_y),  (x0, y0))

        return self.kappa_theta(M200, z, theta_amin, xmax=xmax)


    def kmap2deflmap(self, kappamap, shape, lsides):
        """Transforms a kappa map into a deflection map 
            Args: 
                kappamap: numpy array defining the convergence field
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
            Returns:
                (dx, dy): 2 numpy arrays defining the deflection field
        """

        rfft_kappa = np.fft.rfft2(kappamap)
        rs = rfft_kappa.shape

        ky = (2. * np.pi) / lsides[0] * Freq(np.arange(shape[0]), shape[0])
        ky[int(shape[0] / 2):] *= -1.
        kx = (2. * np.pi) / lsides[1] * Freq(np.arange(rs[1]), shape[1])
        KX, KY = np.meshgrid(kx, ky)
        dx_lm = 2 * rfft_kappa * 1.j * KX / (KX**2+KY**2)
        dy_lm = 2 * rfft_kappa * 1.j * KY / (KX**2+KY**2)
        dx_lm[0, 0] = 0  ## 0 or rfft_kappa[0, 0]
        dy_lm[0, 0] = 0

        dx = np.fft.irfft2(dx_lm, shape)
        dy = np.fft.irfft2(dy_lm, shape)
        return (dx, dy)



    def phimap2kappamap(self, phimap, shape, lsides):
        """Transforms a phi map into a kappa map 
            Args: 
                phimap: numpy array defining the convergence field
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
            Returns:
                kappa_map: numpy array of the convergence field
        """

        rfft_phi = np.fft.rfft2(phimap)
        rs = rfft_phi.shape

        ky = (2. * np.pi) / lsides[0] * Freq(np.arange(shape[0]), shape[0])
        ky[int(shape[0] / 2):] *= -1.
        kx = (2. * np.pi) / lsides[1] * Freq(np.arange(rs[1]), shape[1])
        KX, KY = np.meshgrid(kx, ky)

        rfft_kappa = 1/2 * (KX**2 + KY**2) * rfft_phi

        return np.fft.irfft2(rfft_kappa, shape)

    def P500(self, M500, z):
        # Compute the scaling factor
        scaling_factor = 1.65e-3
        h_z = self.cosmo.hubble_parameter(z) / self.cosmo.Params.H0
    
        # Compute the M500 term
        M500_term = (M500 / (3e14 * self.h70**-1))**(2/3)
    
        # Compute P500
        P500_value = scaling_factor * h_z**(8/3) * M500_term * self.h70**2
    
        return P500_value

    
    def P_gnfw(self, M200, z, r):
        # Use the provided values
        h_z = self.cosmo.hubble_parameter(z) / self.cosmo.Params.H0
        R200 = self.get_r200(M200, z)
        P0 = 8.403 * np.power(self.h70, -3/2)
        rs = self.get_rs(M200, z)
        gamma = 0.3081
        alpha = 1.0510
        beta = 5.4905
        
        R500, M500, _ = self.solve_R500(R200, M200, rs)
    
        # Calculate P500
        P500_value = self.P500(M500, h_z)
    
        # Calculate the numerator
        numerator = P500_value * P0
    
        # Calculate the denominator
        term1 = (r / rs)**gamma
        term2 = (1 + (r / rs)**alpha)**((beta - gamma) / alpha)
    
        denominator = term1 * term2
    
        # Calculate P(r)
        P_r_value = numerator / denominator
    
        return P_r_value
    
    # Define the integrand for line-of-sight integration
    def integrand(self, zx, R, M200, z):
        #xmax = self.
        R_trunc = 3*self.get_r200(M200, z)
        r = np.sqrt(R**2 + zx**2)
        if r <= R_trunc:
            return self.P_gnfw(M200, z, r)
        else:
            return 0  # Return 0 outside the truncation radius
        
        #return self.P_gnfw(M200, z, r) * zx / np.sqrt(r**2 - R**2)
    
    # Define the integral for line-of-sight integration
    def compute_y_theta(self, M200, z, R_values, x_trunc = 3):
        R200 = self.get_r200(M200, z)
        R_trunc = x_trunc * R200  # Calculate R_trunc from x_trunc and R200
        y_theta_values = []
        for R in R_values:
            upper_limit = np.sqrt(R_trunc**2 - R**2) if R < R_trunc else 0
            integral, error = quad(self.integrand, 0, upper_limit, args=(R, M200, z))
            y_theta_values.append(2 * integral)  # Factor of 2 accounts for integration from -∞ to +∞
        return np.array(y_theta_values)


    def tsz_cl(self, M200, z, nu, shape, lsides, xmax=None):
        """Get the tsz map of the cluster
            Args:
                M200: Cluster mass defined as in a sphere 200 times the critical density 
                z: redshift of the cluster
                shape(2-tuple): pair of int defining the number of pixels on each side of the box
                lsides(2-tuple): physical size (in radians) of the box sides
                xmax: cutoff scale in factors of rs
                nu: frequency in GHz
            Returns:
                kappa_map: numpy array defining the convergence field
        """
        R_values = np.linspace(0, 5, 500)  
        y_theta_values = self.compute_y_theta(M200, z, R_values)
        y_theta_interp = interp1d(R_values, y_theta_values, kind='cubic', fill_value='extrapolate')
        
        x = (self.h * nu * 1e9) / (self.k_B * self.tcmb)  # converting frequency from GHz to Hz by multiplying by 1e9
        freq_res = x * ((np.exp(x/2) + np.exp(-x/2)) / (np.exp(x/2) - np.exp(-x/2)))

        dtheta_x = lsides[0]/shape[0] * 180/np.pi*60
        dtheta_y = lsides[1]/shape[1] * 180/np.pi*60
        # Center of the cluster
        #x0 = shape[0]/2
        #y0 = shape[1]/2       
        x0 = 0.0
        y0 = 0.0

        #X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        if shape[0] % 2 == 0 and shape[1] % 2 == 0:
            X, Y =  np.meshgrid(np.concatenate((np.arange(0,shape[0]//2), np.arange(-shape[0]//2,0))), np.concatenate((np.arange(0,shape[1]//2), np.arange(-shape[1]//2,0))))
        else:
            assert 0, "pixel size is not right"
        
        #X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        theta_amin = self.pix_to_theta(X, Y, (dtheta_x,dtheta_y),  (x0, y0))
        
        R = self.theta_amin_to_r(z, theta_amin)
        y = self.sigmat * y_theta_interp(R) / self.me**2 / self.c**2
        #kappa = self.sigma_nfw(M200, z, R, xmax=xmax)
        #kappa_theta = kappa(M200, z, R, xmax=xmax)

        return y * self.tcmb * freq_res
    
    def solve_R500(self, R200, M200, rs):
        r"""function to solve for R500, M500 given R200, M200, and rs
            Args:
                R200: radius (in Mpc) of the cluster in a sphere of density 
                    200 times the critical density of the universe
                M200: mass (in Msol) of the cluster in a sphere of density 
                    200 times the critical density of the universe
                rs: scale radius of the cluster in Mpc
            Returns: 
                R500: radius (in Mpc) of the cluster in a sphere of density 
                    500 times the critical density of the universe
                M500: mass (in Msol) of the cluster in a sphere of density 
                    500 times the critical density of the universe
                x_solution: R500/R200
        """
        def equation(x):
            f_x = np.log((x * R200 + rs) / rs) - (x * R200) / (rs + x * R200)
            f_R200 = np.log((R200 + rs) / rs) - R200 / (rs + R200)
            return f_x / x**3 - (5 / 2) * f_R200

        # Initial guess for x (R500/R200)
        x_initial_guess = 0.5
    
        # Solve for x using fsolve
        x_solution = fsolve(equation, x_initial_guess)[0]
    
        # Calculate R500
        R500 = x_solution * R200
        M500 = (5/2) * M200 * x_solution**3
    
        return R500, M500, x_solution
    
    def convert_M200_to_M500(self, M200, z):
        # Constants for the conversion based on an NFW profile
        delta_200 = 200
        delta_500 = 500
        c200 = self.get_concentration(M200, z)
    
        # Calculate concentration parameters
        # For simplicity, assume c500 = 0.7 * c200 (This is a common assumption in cosmological simulations)
        c500 = 0.7 * c200
        
    
        # Convert M200 to M500
        M500 = M200 * (delta_500 / delta_200) * ((1 + c200) / (1 + c500))**2
    
        return M500    
    

    def sigma_int_v2(self, M200, z, R, xmax=100, npoints=1000):
        """
        Integrate density over the line of sight.

        Args:
            M200: M200c mass of the cluster in Msol
            z: redshift of the cluster
            R: distance from the center of the cluster in Mpc
            xmax: maximum radius of the integration in factors of rs
            npoints: number of points in the integration

        Returns:
            sigma: surface mass density in Msun / Mpc^2
        """
        rs = self.get_rs(M200, z)
        R = np.atleast_1d(R)  # Ensure R is treated as an array

        # Ensure that we're not integrating at the center (R = 0)
        if np.any(R <= 0):
            raise ValueError("Can't integrate at the center of NFW profile, please provide R > 0")

        sigma = np.zeros_like(R)
    
        for i, iR in enumerate(R):
            if np.any(iR > rs * xmax):
                sigma[i] = 0
            else:
                r_arr = np.geomspace(iR, xmax * rs, num=npoints)
                # Exclude the first point to avoid the singularity at r = R
                r_arr = r_arr[1:]
                rho_arr = self.rho_nfw(M200, z, r_arr)
                integrand = 2 * r_arr * rho_arr / np.sqrt(r_arr**2 - iR**2)
                sigma[i] = np.trapz(integrand, r_arr)
    
        return sigma
    
    