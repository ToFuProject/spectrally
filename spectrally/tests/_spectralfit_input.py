# -*- coding: utf-8 -*-


import os
import itertools as itt


import numpy as np
import scipy.constants as scpct
import matplotlib.pyplot as plt


# ###################################################
# ###################################################
#               DEFAULTS
# ###################################################


# PATH
_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUT = os.path.join(_PATH_HERE, 'input')


# PFE
_MASK_1D = os.path.join(_PATH_INPUT, 'mask1d.npy')


# LAMB
_NLANB = 300
_LAMB = lamb = np.linspace(3.9, 4, _NLANB)*1e-10
_LAMB0 = _LAMB[0] + (_LAMB[-1] - _LAMB[0]) * np.r_[0.25, 0.55, 0.75]


# ###################################################
# ###################################################
#               data
# ###################################################


def add_data(coll=None, lamb=_LAMB, lamb0=_LAMB0):

    # ------------------
    # reference vectors
    # ------------------

    # lamb
    coll.add_ref('nlamb', size=lamb.size)
    coll.add_data(
        'lamb',
        data=lamb,
        ref='nlamb',
        units='m',
        dim='dist',
        quant='wavelength',
    )

    # derive useful quantities
    Dlamb = lamb[-1] - lamb[0]
    lambm = 0.5*(lamb[-1] + lamb[0])

    # phi
    nphi = 100
    coll.add_ref('nphi', size=nphi)
    phi = np.linspace(-0.1, 0.1, nphi)
    coll.add_data(
        'phi',
        data=phi,
        ref='nphi',
        units='rad',
        dim='angle',
        quant='phi',
    )

    # time
    nt = 50
    coll.add_ref('nt', size=nt)
    t = np.linspace(0, 10, nt)
    coll.add_data(
        't',
        data=t,
        ref='nt',
        units='s',
        dim='time',
        quant='t',
    )

    # ------------------
    # model-specific data
    # ------------------

    # --------
    # poly

    lamb_rel = (lamb - lambm) / Dlamb

    a0 = 5000
    a1 = 1500
    a2 = -3000

    coll.add_data(
        key='data_poly',
        data=np.random.poisson(a0 + a1*lamb_rel + a2*lamb_rel**2),
        ref='nlamb',
        units='counts',
    )

    # --------
    # exp_lamb

    Te = 1e3
    rate = scpct.h * scpct.c / (Te * scpct.e)
    amp = 500 * lambm * np.exp(rate/lambm)

    coll.add_data(
        key='data_exp',
        data=np.random.poisson(amp * np.exp(-rate/lamb) / lamb),
        ref='nlamb',
        units='counts',
    )

    # --------
    # gauss

    sigma = Dlamb/20.
    vccos = Dlamb/10. / lamb0[0]
    amp = 100 * np.exp((lambm - lamb0[0]*(1+vccos))**2/(2*sigma**2))

    coll.add_data(
        key='data_gauss',
        data=np.random.poisson(
            amp * np.exp(-(lamb - lamb0[0]*(1+vccos))**2/(2*sigma**2))
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # double gauss

    coll.add_data(
        key='data_gauss2',
        data=np.random.poisson(
            amp * np.exp(-(lamb - lamb0[0]*(1+vccos))**2/(2*sigma**2))
            + 0.5*amp * np.exp(-(lamb - lamb0[1]*(1+vccos))**2/(2*sigma**2))
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # lorentz

    gam = Dlamb/20.
    vccos = Dlamb/10. / lamb0[0]
    amp = 100 * (1 + ((lambm - lamb0[0]*(1 + vccos)) / gam)**2)

    coll.add_data(
        key='data_lorentz',
        data=np.random.poisson(
            amp / (1 + ((lamb - lamb0[0]*(1 + vccos)) / gam)**2)
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # double lorentz

    coll.add_data(
        key='data_lorentz2',
        data=np.random.poisson(
            amp / (1 + ((lamb - lamb0[0]*(1 + vccos)) / gam)**2)
            + 0.5* amp / (1 + ((lamb - lamb0[1]*(1 + vccos)) / gam)**2)
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # pvoigt

    sigma = Dlamb/20.
    gam = Dlamb/20.
    vccos = Dlamb/10. / lamb0[1]
    eta = 0.2
    amp = 100 / (
        eta / (1 + ((lambm - lamb0[0]*(1 + vccos)) / gam)**2)
        + (1-eta) * np.exp(-(lambm - lamb0[0]*(1 + vccos))**2/(2*sigma**2))
    )

    pvoigt = amp * (
        eta / (1 + ((lamb - lamb0[0]*(1 + vccos)) / gam)**2)
        + (1-eta) * np.exp(
            -(lamb - lamb0[0]*(1 + vccos))**2
            / (2*sigma**2)
        )
    )

    coll.add_data(
        key='data_pvoigt',
        data=np.random.poisson(pvoigt),
        ref='nlamb',
        units='counts',
    )

    # --------
    # double pvoigt

    pvoigt2 = amp * (
        eta / (1 + ((lamb - lamb0[1]*(1 + vccos)) / gam)**2)
        + (1-eta) * np.exp(
            -(lamb - lamb0[1]*(1 + vccos))**2
            / (2*sigma**2)
        )
    )

    coll.add_data(
        key='data_pvoigt2',
        data=np.random.poisson(pvoigt + 0.5 * pvoigt2),
        ref='nlamb',
        units='counts',
    )

    # --------
    # pulse_exp

    t0 = lamb0[0]
    tup = Dlamb / 30
    tdown = Dlamb / 5
    amp = 100 / (np.exp(-(lambm-t0)/tdown) - np.exp(-(lambm-t0)/tup))

    coll.add_data(
        key='data_pulse_exp',
        data=np.random.poisson(
            amp * (lamb >= t0) * (
                np.exp(-(lamb-t0)/tdown) - np.exp(-(lamb-t0)/tup)
            )
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # double pulse_exp

    deltat = 0.04e-10

    coll.add_data(
        key='data_pulse_exp2',
        data=np.random.poisson(
            amp * (lamb >= t0) * (
                np.exp(-(lamb-t0)/tdown) - np.exp(-(lamb-t0)/tup)
            )
            + 0.5*amp * (lamb >= (t0 + deltat)) * (
                np.exp(-(lamb-(t0 + deltat))/tdown) - np.exp(-(lamb-(t0 + deltat))/tup)
            )
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # pulse_gauss

    t0 = lamb0[0]
    tup = Dlamb / 30
    tdown = Dlamb / 5
    indup = (lamb < t0)
    inddown = (lamb >= t0)
    amp = 100

    coll.add_data(
        key='data_pulse_gauss',
        data=np.random.poisson(
            amp * (
                indup * np.exp(-(lamb - t0)**2/tup**2)
                + inddown * np.exp(-(lamb - t0)**2/tdown**2)
            )
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # double pulse_gauss

    coll.add_data(
        key='data_pulse_gauss2',
        data=np.random.poisson(
            amp * (
                indup * np.exp(-(lamb - t0)**2/tup**2)
                + inddown * np.exp(-(lamb - t0)**2/tdown**2)
            )
            + 0.5 * amp * (
                (lamb < (t0+deltat)) * np.exp(-(lamb - (t0+deltat))**2/tup**2)
                + (lamb >= (t0+deltat)) * np.exp(-(lamb - (t0+deltat))**2/tdown**2)
            )
        ),
        ref='nlamb',
        units='counts',
    )

    # --------
    # lognorm

    t0 = lamb0[0]
    sigma = 0.5
    mu = 0.5 * (np.log((Dlamb / 30)**2/(np.exp(sigma**2) - 1)) - sigma**2)

    amp = (
        100 * (lambm - t0)
        * np.exp((np.log(lambm - t0) - mu)**2 / (2.*sigma**2))
    )
    data = np.zeros((lamb.size,))
    iok = (lamb >= t0)
    data[iok] = (amp / (lamb[iok] - t0)) * np.exp(
        -(np.log(lamb[iok] - t0) - mu)**2 / (2.*sigma**2)
    )

    coll.add_data(
        key='data_lognorm',
        data=np.random.poisson(data),
        ref='nlamb',
        units='counts',
    )

    # --------
    # double lognorm

    iok = (lamb >= (t0 + deltat))
    data2 = np.zeros((lamb.size,))
    data2[iok] = (amp / (lamb[iok] - (t0 + deltat))) * np.exp(
        -(np.log(lamb[iok] - (t0 + deltat)) - mu)**2 / (2.*sigma**2)
    )

    coll.add_data(
        key='data_lognorm2',
        data=np.random.poisson(data + 0.5 * data2),
        ref='nlamb',
        units='counts',
    )

    # ------------------
    # data 1d
    # ------------------

    # lamb0
    amp0 = 700
    width = Dlamb * np.r_[0.015, 0.035, 0.025]
    amp = amp0 * np.r_[1, 0.5, 2]

    amp1 = np.max(amp) * 0.10
    amp2 = np.max(amp) * 0.02
    dlamb = -(lamb[0] - lamb[-1]) / np.log(amp1/amp2)
    A = amp1 / np.exp(-lamb[0]/dlamb)

    # data 1d
    data = np.random.poisson(
        A * np.exp(-lamb / dlamb)
        + np.sum(
            [
                amp[ii] * np.exp(-(lamb-lamb0[ii])**2 / (2*width[ii]**2))
                for ii in range(len(lamb0))
            ],
            axis=0,
        ),
        size=lamb.size,
    ).astype(float)     # + amp0 * 0.10 * np.random.random((nlamb,))

    # store
    coll.add_data(
        'data1d',
        data=data,
        ref='nlamb',
        units='ph',
    )

    # ------------------
    # data 2d (1d vs t)
    # ------------------

    ampt = np.exp(-(t-np.mean(t))**2 / (0.3*(t[-1] - t[0]))**2)
    At = A * ampt[:, None]
    lambt = lamb[None, :]
    dv = np.r_[0.1, -0.05, 0.08]

    # data 1d
    data = np.random.poisson(
        At * np.exp(-lambt / dlamb)
        + ampt[:, None] * np.sum(
            [
                amp[ii]
                * np.exp(
                    - (lambt-lamb0[ii] - dv[ii] * Dlamb * ampt[:, None])**2
                    / (2*ampt[:, None] * width[ii]**2)
                )
                for ii in range(len(lamb0))
            ],
            axis=0,
        ),
        size=(nt, lamb.size),
    ).astype(float)     # + amp0 * 0.10 * np.random.random((nlamb,))

    # store
    coll.add_data(
        'data2d',
        data=data,
        ref=('nt', 'nlamb'),
        units='ph',
    )

    # ------------------
    # data 2d (1d + bs)
    # ------------------

    # ------------------
    # data 3d (1d + bs + t)
    # ------------------

    return


# ####################################################
# ####################################################
#            Add model-specific data (xfree)
# ####################################################


def add_xfree(coll=None):

    # --------------
    # prepare xfree

    t = coll.ddata['t']['data']
    dxfree = _get_dxfree(t, lamb)

    # ---------------
    # add model data

    for ii, kmodel in enumerate(dxfree.keys()):

        # ref_nx
        ref_nx = f'nx_{kmodel}'

        # xfree
        xfree = dxfree[kmodel]
        if xfree.ndim == 2:
            ref = (coll.ddata['t']['ref'][0], ref_nx)
        else:
            ref = (ref_nx,)

        # add_data
        kdata = f"xfree_{kmodel}"
        coll.add_data(
            key=kdata,
            data=xfree,
            ref=ref,
        )

    return


def _get_dxfree(t=None, lamb=None):

    lambD = lamb[-1] - lamb[0]
    tm = np.mean(t)

    rate = np.log(2*lamb[0]/lamb[-1]) / (1/lamb[-1] - 1/lamb[0])

    tmax = 3.960e-10
    fmax = 1
    delta = lambD / 30

    sigma = 0.5
    mu = 0.5 * (np.log(delta**2/(np.exp(sigma**2) - 1)) - sigma**2)
    # mu = -28

    t0 = (tmax - np.exp(mu - sigma**2))
    tau = (t0 - lamb[0]) / lambD
    amp = fmax / np.exp(0.5*sigma**2 - mu)

    # sm00
    # 'bck0_a0', 'bck0_a1', 'bck_a2',
    # 'l00_amp', 'l00_vccos', 'l00_sigma',
    # 'l08_amp', 'l08_vccos', 'l08_gam',
    # 'l12_amp', 'l12_vccos', 'l12_sigma', 'l12_gam',

    # sm01
    # 'bck0_amp', 'bck0_rate',
    # 'l00_amp', 'l00_vccos', 'l00_sigma',
    # 'l01_vccos', 'l01_gam',
    # 'l02_amp', 'l02_vccos',

    # sm02
    # 'bck0_amp', 'bck0_rate',
    # 'l00_amp', 'l00_vccos', 'l00_sigma',
    # 'l01_vccos',
    # 'l02_amp', 'l02_vccos', 'l02_sigma', 'l02_gam'

    # sm03
    # 'bck0_a0', 'bck0_a1', 'bck_a2',
    # 'l00_amp', 'l00_tau', 'l00_t_up', 'l00_t_down',
    # 'l01_amp', 'l01_tau', 'l01_t_up', 'l01_t_down',
    # 'l02_amp', 'l02_tau', 'l02_mu', 'l02_sigma'

    dxfree = {
        # testing elementary models
        # 'smpoly': np.r_[0.1 - lamb[0] * 0.1/lambD, 0.1 / lambD],
        'smpoly': np.r_[0.2, 0.01, 0.02],
        'smexp': np.r_[0.2*lamb[0]*np.exp(rate/lamb[0]), rate],
        'smgauss': np.r_[1, 0.004, 0.003e-10],
        'smlorentz': np.r_[0.9, -0.006, 0.003e-10],
        'smpvoigt': np.r_[1.1, -0.001, 0.003e-10, 0.003e-10],
        'smvoigt': np.r_[1.1, -0.001, 0.003e-10, 0.003e-10],
        'smpulse_exp': np.r_[2, 3.905e-10, 0.001e-10, 0.004e-10],
        'smpulse_gauss': np.r_[1, 3.935e-10, 0.001e-10, 0.007e-10],
        'smlognorm': np.r_[amp, tau, mu, sigma],

        # double simple
        'smgauss2': np.r_[1, 0.004, 0.003e-10],
        'smlorentz2': np.r_[0.9, -0.006, 0.003e-10],
        'smpvoigt2': np.r_[1.1, -0.001, 0.003e-10, 0.003e-10],
        'smvoigt2': np.r_[1.1, -0.001, 0.003e-10, 0.003e-10],
        'smpulse_exp2': np.r_[2, 3.905e-10, 0.001e-10, 0.004e-10],
        'smpulse_gauss2': np.r_[1, 3.935e-10, 0.001e-10, 0.007e-10],
        'smlognorm2': np.r_[amp, tau, mu, sigma],

        # testing complex models
        'sm00': np.r_[
           0.2, 0.01, 0.02,
            1, 0.004, 0.003e-10,
            0.9, -0.006, 0.003e-10,
            1.1, -0.001, 0.003e-10, 0.003e-10,
        ],
        'sm01': np.r_[
            0.2*lamb[0]*np.exp(rate/lamb[0]), rate,
            1, 0.001, 0.003e-10,
            -0.001, 0.001e-10,
            1.e-12, 0.001,
        ][None, :] * np.exp(-(t[:, None] - tm)**2 / 2**2),
        'sm02': np.r_[
            0.2*lamb[0]*np.exp(rate/lamb[0]), rate,
            1, 0.001, 0.003e-10,
            -0.001,
            1e-12, 0.001, 0.003e-10, 0.001e-10,
        ],
        'sm03': np.r_[
            0.2, 0.01, 0.02,
            2, 3.905e-10, 0.001e-10, 0.004e-10,
            1, 3.935e-10, 0.001e-10, 0.007e-10,
            amp, tau, mu, sigma,
            amp, tau + 0.015e-10, mu, sigma,
        ],
    }
    return dxfree




# ###################################################
# ###################################################
#               dmodels
# ###################################################


def add_models(coll=None, models=None, lamb=_LAMB, lamb0=_LAMB0):

    # ---------------
    # check

    if coll.dobj.get('spect_model') is not None:
        return

    # --------------------------------------------------------------
    # add spectral lines just for testing automated loading of lamb0

    coll.add_spectral_line(
        key='sl00',
        ion='Ar16+',
        lamb0=3.96e-10,
    )

    lambD = lamb[-1] - lamb[0]

    # ---------------
    # dmodels

    dmodel = {
        # Testing single fit only
        'smpoly': {
            'bck0': 'poly',
        },
        'smexp': {
            'bck0': 'exp_lamb',
        },
        'smgauss': {
            'l00': 'gauss',
        },
        'smlorentz': {
            'l00': 'lorentz',
        },
        'smpvoigt': {
            'l00': 'pvoigt',
        },
        'smvoigt': {
            'l00': 'voigt',
        },
        'smpulse_exp': {
            'l00': 'pulse_exp',
        },
        'smpulse_gauss': {
            'l00': 'pulse_gauss',
        },
        'smlognorm': {
            'l00': 'lognorm',
        },

        # Testing double fits with contraints
        'smgauss2': {
            'l00': {'type': 'gauss', 'lamb0': lamb0[0]},
            'l01': {'type': 'gauss', 'lamb0': lamb0[0]},
        },
        'smlorentz2': {
            'l00': {'type': 'lorentz', 'lamb0': lamb0[0]},
            'l01': {'type': 'lorentz', 'lamb0': lamb0[0]},
        },
        'smpvoigt2': {
            'l00': {'type': 'pvoigt', 'lamb0': lamb0[0]},
            'l01': {'type': 'pvoigt', 'lamb0': lamb0[0]},
        },
        'smvoigt2': {
            'l00': {'type': 'voigt', 'lamb0': lamb0[0]},
            'l01': {'type': 'voigt', 'lamb0': lamb0[0]},
        },
        'smpulse_exp2': {
            'l00': {'type': 'pulse_exp', 'lamb0': lamb0[0]},
            'l01': {'type': 'pulse_exp', 'lamb0': lamb0[0]},
        },
        'smpulse_gauss2': {
            'l00': {'type': 'pulse_gauss', 'lamb0': lamb0[0]},
            'l01': {'type': 'pulse_gauss', 'lamb0': lamb0[0]},
        },
        'smlognorm2': {
            'l00': {'type': 'lognorm', 'lamb0': lamb0[0]},
            'l01': {'type': 'lognorm', 'lamb0': lamb0[0]},
        },

        # testing model population
        'sm-1': {
            'bck0': 'poly',
            'll0': 'gauss',
            'll1': 'gauss',
            'll2': 'lorentz',
        },
        'sm00': {
            'bck0': 'poly',
            'l00': {'type': 'gauss'},     # from spectral line
            'l08': {'type': 'lorentz'},   # from spectral line
            'l12': {'type': 'pvoigt'},    # from spectral line
        },
        'sm01': {
            'bck0': 'exp_lamb',
            'l00': {
                'type': 'gauss',
                'lamb0': 3.92e-10,
                'mz': 39.948*scpct.m_u,
            },
            'l01': {
                'type': 'lorentz',
                'lamb0': 3.95e-10,
                'mz': 39.948*scpct.m_u,
            },
            'l02': {
                'type': 'gauss',
                'lamb0': 3.97e-10,
                'mz': 39.948*scpct.m_u,
            },
        },
        'sm02': {
            'bck0': 'exp_lamb',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'l01': {'type': 'lorentz', 'lamb0': 3.95e-10},
            'l02': {'type': 'voigt', 'lamb0': 3.97e-10},
        },
        'sm03': {
            'bck0': 'poly',
            'l00': {'type': 'pulse_exp'},
            'l01': {'type': 'pulse_gauss'},
            'l02': {'type': 'lognorm'},
            'l03': {'type': 'lognorm'},
        },
    }

    dconstraints = {
        # double
        'smgauss2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_sigma', 'l01_sigma': [0, 1, 0]},
            'g02': {'ref': 'l00_vccos', 'l01_vccos': [(lamb0[1] - lamb0[0])/lamb0[0], 1, 0]},
        },
        'smlorentz2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_gam', 'l01_gam': [0, 1, 0]},
            'g02': {'ref': 'l00_vccos', 'l01_vccos': [(lamb0[1] - lamb0[0])/lamb0[0], 1, 0]},
        },
        'smpvoigt2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_sigma', 'l01_sigma': [0, 1, 0]},
            'g02': {'ref': 'l00_gam', 'l01_gam': [0, 1, 0]},
            'g03': {'ref': 'l00_vccos', 'l01_vccos': [(lamb0[1] - lamb0[0])/lamb0[0], 1, 0]},
        },
        'smvoigt2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_sigma', 'l01_sigma': [0, 1, 0]},
            'g02': {'ref': 'l00_gam', 'l01_gam': [0, 1, 0]},
            'g03': {'ref': 'l00_vccos', 'l01_vccos': [(lamb0[1] - lamb0[0])/lamb0[0], 1, 0]},
        },
        'smpulse_exp2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_t_up', 'l01_t_up': [0, 1, 0]},
            'g02': {'ref': 'l00_t_down', 'l01_t_down': [0, 1, 0]},
            'g03': {'ref': 'l00_tau', 'l01_tau': [0.04e-10 / lambD, 1, 0]},
        },
        'smpulse_gauss2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_t_up', 'l01_t_up': [0, 1, 0]},
            'g02': {'ref': 'l00_t_down', 'l01_t_down': [0, 1, 0]},
            'g03': {'ref': 'l00_tau', 'l01_tau': [0.04e-10 / lambD, 1, 0]},
        },
        'smlognorm2': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 0.5, 0]},
            'g01': {'ref': 'l00_mu', 'l01_mu': [0, 1, 0]},
            'g02': {'ref': 'l00_sigma', 'l01_sigma': [0, 1, 0]},
            'g03': {'ref': 'l00_tau', 'l01_tau': [0.04e-10 / lambD, 1, 0]},
        },

        # multi
        'sm01': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_sigma', 'l02_sigma': [0, 1, 0]},
        },
        'sm02': {
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_sigma', 'l01_gam': [0, 1, 0]},
        },
    }

    if models is None:
        models = sorted(dmodel.keys())

    lout = [k0 for k0 in models if k0 not in dmodel.keys()]
    if len(lout):
        lstr = [f"\t- {k0}" for k0 in lout]
        msg = "Requested models not available for tests:\n" + "\n".join(lstr)
        raise Exception(msg)

    # ---------------
    # add models
    # ---------------

    for k0 in dmodel.keys():

        if k0 not in models:
            continue

        # check err
        if k0 == 'sm-1':
            try:
                coll.add_spectral_model(
                    key='sm-1',
                    dmodel=dmodel['sm-1'],
                    dconstraints=None,
                )
                raise Exception('sucess')
            except Exception as err:
                if "For model" not in str(err):
                    raise Exception("Wrong error raised!") from err

        else:
            # no constraints
            coll.add_spectral_model(
                key=k0,
                dmodel=dmodel[k0],
                dconstraints=dconstraints.get(k0),
            )

    return


# ###################################################
# ###################################################
#               spectral model - func
# ###################################################


def get_spectral_model_func(coll=None):

    # ---------------
    # check

    if coll.dobj.get('spect_model') is None:
        add_models(coll)

    # ---------------
    # get func models

    wsm = coll._which_model
    for kmodel in coll.dobj[wsm].keys():

        for ff in ['sum', 'cost']:  # , 'details', 'jac']:

            try:
                _ = coll.get_spectral_fit_func(
                    key=kmodel,
                    func=ff,
                )

            except Exception as err:
                msg = (
                    "Could not get func for:\n"
                    f"\t- spectral model: {kmodel}\n"
                    f"\t- func: {ff}\n"
                )
                raise Exception(msg) from err

    return


# ###################################################
# ###################################################
#               spectral model - interpolate
# ###################################################


def interpolate_spectral_model(coll=None):

    # ---------------
    # check

    if coll.dobj.get('spect_model') is None:
        add_models(coll)

    # -------------
    # lamb

    lamb = np.linspace(3.9, 4, 100)*1e-10

    # ---------------
    # add model data

    wsm = coll._which_model
    lkstore = []
    for ii, kmodel in enumerate(coll.dobj[wsm].keys()):

        # interpolate
        for jj, details in enumerate([False, True]):
            for kk, store in enumerate([False, True]):

                store_key = f'interp_{kmodel}_{jj}_{kk}'
                lambi = ('lamb' if store else lamb)

                _ = coll.interpolate_spectral_model(
                    key_model=kmodel,
                    key_data=f'xfree_{kmodel}',
                    lamb=lambi,
                    # details
                    details=details,
                    # store
                    returnas=None,
                    store=store,
                    store_key=store_key,
                )

                if store:
                    lkstore.append(store_key)

    # remove stored output
    coll.remove_data(lkstore)

    return



# ###################################################
# ###################################################
#               spectral model - moments
# ###################################################


def get_spectral_model_moments(coll=None):

    # ------------
    # check

    if coll.dobj.get('spect_model') is None:
        add_models(coll)

    # ------------
    # get moments

    wsm = coll._which_model
    for kmodel in coll.dobj[wsm].keys():
        _ = coll.get_spectral_model_moments(
            kmodel,
            key_data=f"xfree_{kmodel}",
            lamb='lamb',
        )

    return


# ###################################################
# ###################################################
#               spectral model - plot
# ###################################################


def plot_spectral_model(coll=None):

    # ---------------
    # check

    if coll.dobj.get('spect_model') is None:
        add_models(coll)

    # ---------------
    # get func models

    wsm = coll._which_model
    for kmodel in coll.dobj[wsm].keys():
        pass

    return


# ###################################################
# ###################################################
#               spectral fit - add
# ###################################################


def add_fit(coll=None, key_model=None, key_data=None):

    # ---------------
    # check
    # ---------------

    add_models(coll)

    if coll.dobj.get('spect_fit') is not None:
        lk = [
            k0 for k0, v0 in coll.dobj['spect_fit'].items()
            if v0['key_data'] == key_data
            and v0['key_model'] == key_model
        ]
        if len(lk) > 0:
            return

    # ---------------
    # add 1d
    # ---------------

    mask = [None, _MASK_1D]
    domain = [
        None,
        {'lamb': [(3.96e-10, 3.97e-10)]},
        {'lamb': [
            [3.91e-10, 3.94e-10],
            (3.96e-10, 3.97e-10),
            [3.965e-10, 3.995e-10]
        ]},
    ]
    focus = [
        (None, None),
        ([[3.925e-10, 3.94e-10], [3.97e-10, 3.99e-10]], 'min'),
        ([[3.925e-10, 3.94e-10], [3.97e-10, 3.99e-10]], 'max'),
        ([[3.925e-10, 3.94e-10], [3.97e-10, 3.99e-10]], 'sum'),
    ]

    for ii, ind in enumerate(itt.product(mask, domain, focus)):

        try:
            coll.add_spectral_fit(
                key=None,
                key_model=key_model,
                key_data=key_data,
                key_sigma=None,
                key_lamb='lamb',
                # params
                dparams=None,
                dvalid={
                    'mask': ind[0],
                    'domain': ind[1],
                    'focus': ind[2][0],
                    'focus_logic': ind[2][1],
                },
            )

        except Exception as err:
            msg = (
                "Failed add_spectral_fit for 'data1d':\n"
                f"\t- ii = {ii}\n"
                f"\t- mask = {ind[0]}\n"
                f"\t- domain = {ind[1]}\n"
                + "-"*20 + "\n"
            )
            print(msg)
            raise err

    return


def add_fit_single(coll=None):

    lk = [
        ('smpoly', 'data_poly'),
        ('smexp', 'data_exp'),
        ('smgauss', 'data_gauss'),
        ('smlorentz', 'data_lorentz'),
        ('smpvoigt', 'data_pvoigt'),
        ('smpulse_exp', 'data_pulse_exp'),
        ('smpulse_gauss', 'data_pulse_gauss'),
        ('smlognorm', 'data_lognorm'),
    ]

    for (key_model, key_data) in lk:
        add_fit(coll=coll, key_model=key_model, key_data=key_data)

    return


def add_fit_double(coll=None):

    lk = [
        ('smgauss2', 'data_gauss2'),
        ('smlorentz2', 'data_lorentz2'),
        ('smpvoigt2', 'data_pvoigt2'),
        ('smpulse_exp2', 'data_pulse_exp2'),
        ('smpulse_gauss2', 'data_pulse_gauss2'),
        ('smlognorm2', 'data_lognorm2'),
    ]

    for (key_model, key_data) in lk:
        add_fit(coll=coll, key_model=key_model, key_data=key_data)

    return


def add_fit_multi(coll=None):

    lk = [
        # ('sm00', 'data1d'),
        ('sm01', 'data1d'),
    ]

    for (key_model, key_data) in lk:
        add_fit(coll=coll, key_model=key_model, key_data=key_data)

    return


# ###################################################
# ###################################################
#           plot spectral fit input validity
# ###################################################


def plot_input_validity(coll=None, key_data=None):

    # ---------------
    # check

    add_models(coll)

    wsm = coll._which_model
    for k0 in coll.dobj[wsm].keys():
        add_fit(coll, key_model=k0, key_data=key_data)

    # ---------------
    # select data

    lk = [
        k0 for k0, v0 in coll.dobj['spect_fit'].items()
        if v0['key_data'] == key_data
    ]

    # ---------------
    # plot

    for k0 in lk:
        _ = coll.plot_spectral_fit_input_validity(k0)
        plt.close('all')

    return


# ###################################################
# ###################################################
#               spectral fit - compute
# ###################################################


def compute_fit(coll=None, key_data=None, binning=None):

    # ---------------
    # select data
    # ---------------

    wsf = coll._which_fit
    lk = [
        k0 for k0, v0 in coll.dobj[wsf].items()
        if v0['key_data'] == key_data
    ]

    # ---------------
    # compute fit
    # ---------------

    for ii, k0 in enumerate(lk):

        if ii % 2 == 0:
            solver = 'scipy.least_squares'
            ftol = 1e-6
        else:
            solver = 'scipy.curve_fit'
            ftol = 1e-3

        coll.compute_spectral_fit(
            key=k0,
            strict=True,
            binning=binning,
            solver=solver,
            dsolver_options={'ftol': ftol},
            verb=1,
            timing=None,
        )

    return


def compute_fit_single(coll=None, binning=None):

    # --------------------
    # add models if needed
    # --------------------

    add_models(coll)

    # ---------------
    # add fits if needed
    # ---------------

    add_fit_single(coll)

    # --------------------
    # compute
    # --------------------

    lk = [
        'data_poly', 'data_exp',
        'data_gauss', 'data_lorentz', 'data_pvoigt',
        'data_pulse_exp', 'data_pulse_gauss', 'data_lognorm',
    ]

    for key_data in lk:
        compute_fit(coll, key_data=key_data, binning=binning)

    return


def compute_fit_double(coll=None, binning=None):

    # --------------------
    # add models if needed
    # --------------------

    add_models(coll)

    # ---------------
    # add fits if needed
    # ---------------

    add_fit_double(coll)

    # --------------------
    # compute
    # --------------------

    lk = [
        'data_gauss2', 'data_lorentz2', 'data_pvoigt2',
        'data_pulse_exp2', 'data_pulse_gauss2', 'data_lognorm2',
    ]

    for key_data in lk:
        compute_fit(coll, key_data=key_data, binning=binning)

    return


def compute_fit_multi(coll=None, key_data=None, binning=None):

    # --------------------
    # add models if needed
    # --------------------

    add_models(coll)

    # ---------------
    # add fits if needed
    # ---------------

    add_fit_multi(coll)

    # --------------------
    # compute
    # --------------------

    compute_fit(coll, key_data=key_data, binning=binning)

    return