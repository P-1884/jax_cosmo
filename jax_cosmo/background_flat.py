# This module implements various functions for the background COSMOLOGY
import jax.numpy as np
from jax import lax

import jax_cosmo.constants as const
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.scipy.ode import odeint
from jax_cosmo.background import w,f_de,Esqr,H,Omega_m_a,Omega_de_a,radial_comoving_distance,dchioverda,transverse_comoving_distance,angular_diameter_distance,growth_factor,growth_rate,_growth_factor_ODE,_growth_rate_ODE,_growth_factor_gamma,_growth_rate_gamma

def transverse_comoving_distance(cosmo, a):
    r"""Transverse comoving distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    f_k : ndarray, or float if input scalar
        Transverse comoving distance corresponding to the specified
        scale factor.

    Notes
    -----
    The transverse comoving distance depends on the curvature of the
    universe and is related to the radial comoving distance through:

    .. math::

        f_k(a) = \left\lbrace
        \begin{matrix}
        R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for }\Omega_k > 0 \\
        \chi(a)&
            \mbox{for } \Omega_k = 0 \\
        R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for } \Omega_k < 0
        \end{matrix}
        \right.
    """
    '''
    PH:
    Important Notes on Omega_k:
    Omega_k is not an integer, but 'k' is. Omega_k and 'k' have opposite signs.
    Open: k=-1,Omega_k>0
    Closed: k=1,Omega_k<0.
    See this webpage: https://ned.ipac.caltech.edu/level5/Carroll/Carroll1.html
    and less usefully, here: https://en.wikipedia.org/wiki/Friedmann_equations
    '''
    index = cosmo.k + 1 

    def open_universe(chi): #k=-1, O_k>0
        return const.rh / cosmo.sqrtk * np.sinh(cosmo.sqrtk * chi / const.rh)

    def flat_universe(chi):
        return chi

    def close_universe(chi): #k=+1, O_k<0
        return const.rh / cosmo.sqrtk * np.sin(cosmo.sqrtk * chi / const.rh)

    branches = (open_universe, flat_universe, close_universe)

    chi = radial_comoving_distance(cosmo, a)
    #TEMPORARILY putting this in, to see if can fix Omega_k issue
    return flat_universe(chi)
