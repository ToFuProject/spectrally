# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:06:21 2024

@author: dvezinet
"""


import numpy as np


#############################################
#############################################
#       MODEL FUNC ORDER
#############################################


_LMODEL_ORDER = [
    # background
    'poly', 'exp_lamb', # 'exp_E',
    # spectral lines
    'gauss', 'lorentz', 'pvoigt', 'voigt',
    # pulse shape
    'pulse_exp', 'pulse_gauss', 'lognorm',
]


#############################################
#############################################
#       MODEL DICT
#############################################


_DMODEL = {

    # -------------------
    # background
    # -------------------

    # -----------
    # poly

    'poly': {
        'var': ['a0', 'a1', 'a2'],
        'description': 'polynomial (up to deg = 2)',
        'expressions': {
            'main': r"$\left\{ \begin{array}{ll} a_0 + a_1\frac{\lambda - <\lambda>}{\Delta\lambda} + a_2\left(\frac{\lambda - <\lambda>}{\Delta\lambda}\right)^2\\ <\lambda> = \frac{\lambda[0] + \lambda[-1]}{2} \\ \Delta\lambda = \lambda[-1] - \lambda[0]\end{array} \right.$",
            'argmax': r"$\lambda_{max} = <\lambda> - \frac{a_1}{2a_2}\Delta\lambda$",
            'max': r"$f(\lambda_{max}) = a_0 - \frac{a_1^2}{4a_2}$",
        },
    },

    # ------------
    # exponential

    'exp_lamb': {
        'var': ['amp', 'rate'],
        'description': 'Bremsstrahlung-like exponential',
        'expressions': {
            'main': r"$\frac{A}{\lambda} e^{-\frac{rate}{\lambda}}$",
            'Te': r"$k_BT_e [eV] = \frac{hc}{rate} \times \frac{1}{e}$",
        },
    },

    # -------------------
    # spectral lines
    # -------------------

    # -----------
    # gaussian

    'gauss': {
        'var': ['amp', 'vccos', 'sigma'],
        'param': [('lamb0', float), ('mz', float, np.nan)],
        'description': 'gaussian',
        'expressions': {
            'main': r"$A e^{-\frac{\left(\lambda - \lambda_0(1 + \frac{v}{c}\cos)\right)^2}{2\sigma^2}}$",
            'argmax': r"$\lambda_{max} = \lambda_0(1 + \frac{v}{c}\cos)$",
            'max': r"$f(\lambda_{max}) = A$",
            'Ti': r"$k_BT_i [eV] = m_z c^2 \left(\frac{\sigma}{\lambda_0}\right)^2 \times \frac{1}{e}$",
            "FWHM": "",
        },
    },

    # -----------
    # lorentzian

    'lorentz': {
        'var': ['amp', 'vccos', 'gam'],
        'param': [('lamb0', float), ('mz', float, np.nan)],
        'description': 'lorentzian',
        'expressions': {
            'main': r"$\frac{A}{1 + \left(\frac{\lambda - \lambda_0(1 + \frac{v}{c}\cos)}{\gamma}\right)^2}$",
            'argmax': r"$\lambda_{max} = \lambda_0(1 + \frac{v}{c}\cos)$",
            'max': r"$f(\lambda_{max}) = A$",
        },
    },

    # ------------
    # pseudo-voigt

    'pvoigt': {
        'var': ['amp', 'vccos', 'sigma', 'gam'],
        'param': [('lamb0', float), ('mz', float, np.nan)],
        'description': 'pseudo-voigt',
        'expressions': {
            'main': (
                r"$\left\{ \begin{array}{ll} A \left[\frac{\eta}{1 + \left(\frac{\lambda - \lambda_0(1 + \frac{v}{c}\cos)}{\gamma}\right)^2} + (1-\eta)e^{-\frac{\left(\lambda - \lambda_0(1 + \frac{v}{c}\cos)\right)^2}{2\sigma^2}}\right] \\ \eta = g(\sigma, \gamma) \end{array} \right.$"
            ),
            'ref': "https://en.wikipedia.org/wiki/Voigt_profile",
            'argmax': r"$\lambda_{max} = \lambda_0(1 + \frac{v}{c}\cos)$",
            'max': r"$f(\lambda_{max}) = A$",
            'Ti': r"$k_BT_i [eV] = m_z c^2 \left(\frac{\sigma}{\lambda_0}\right)^2 \times \frac{1}{e}$",
        },
    },

    # -----------
    # voigt

    'voigt': {
        'var': ['amp', 'vccos', 'sigma', 'gam'],
        'param': [('lamb0', float), ('mz', float, np.nan)],
        'description': 'voigt',
        'expressions': {
            'main': r"$$",
            'argmax': "",
            'max': "",
            'Ti': r"$k_BT_i [eV] = m_z c^2 \left(\frac{\sigma}{\lambda_0}\right)^2 \times \frac{1}{e}$",
        },
    },

    # ------------------
    # pulse shape
    # ------------------

    # -----------
    # pulse with exponentials

    'pulse_exp': {
        'var': ['amp', 'tau', 't_up', 't_down'],
        'description': 'asymmetric pulse, 2 exponentials',
        'expressions': {
            'main': (
                r"$\left\{ \begin{array}{ll} f(t>= t_0) = A\left( e^{-\frac{t-t_0}{t_{down}}} - e^{-\frac{t-t_0}{t_{up}}} \right) \\ t_0 = t[0] + \tau\Delta t  \\ \Delta t = t[-1] - t[0] \end{array} \right.$"
            ),
            'argmax': r"$t_{max} = t_0 + \frac{t_{down}t_{up}}{t_{down} - t_{up}}\ln\left({\frac{t_{down}}{t_{up}}}\right)$",
            'max': r"$f(t_{max}) = A\left( e^{-\frac{t_{up}}{t_{down} - t_{up}}\ln\left({\frac{t_{down}}{t_{up}}}\right)} - e^{-\frac{t_{down}}{t_{down} - t_{up}}\ln\left({\frac{t_{down}}{t_{up}}}\right)}\right)$",
        },
    },

    # -----------
    # pulse with gaussians

    'pulse_gauss': {
        'var': ['amp', 'tau', 't_up', 't_down'],
        'description': 'asymmetric pulse, 2 gaussians',
        'expressions': {
            'main': (
                r"$\left\{ \begin{array}{ll} f(t < t_0) = A e^{-\left(\frac{t-t_0}{t_{up}}\right)^2} \\ f(t>=t_0) = A e^{-\left(\frac{t-t_0}{t_{down}}\right)^2} \\ t_0 = t[0] + \tau\Delta t  \\ \Delta t = t[-1] - t[0] \end{array} \right.$"
            ),
            'argmax': r"$t_{max} = t_0$",
            'max': r"$f(t_{max}) = A$",
        },
    },

    # -----------
    # lognorm

    'lognorm': {
        'var': ['amp', 'tau', 'mu', 'sigma'],
        'description': 'asymmetric pulse, lognorm',
        'expressions': {
            'main': r"$\left\{ \begin{array}{ll} f(t > t_0) = \frac{A}{t-t_0} e^{-\frac{\left(\ln(t-t_0) - \mu\right)^2}{2\sigma^2}} \\ t_0 = t[0] + \tau\Delta t  \\ \Delta t = t[-1] - t[0] \end{array} \right.$",
            'argmax': r"$t_{max} = t_0 + e^{\mu - \sigma^2}$",
            'max': r"$f(t_{max}) = A e^{\frac{\sigma^2}{2} - \mu}$",
            'skewness': r"$skew = \left(e^{\sigma^2} + 2\right) \sqrt{exp(\sigma^2) - 1}$",
            'variance': r"$var = \left(e^{\sigma^2} - 1\right) e^{2\mu + \sigma^2}$"
        },
    },
}