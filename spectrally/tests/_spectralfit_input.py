# -*- coding: utf-8 -*-


import numpy as np


# ###################################################
# ###################################################
#               data
# ###################################################


def add_data(coll=None):

    # ------------------
    # reference vectors
    # ------------------

    # lamb
    nlamb = 150
    coll.add_ref('nlamb', size=nlamb)
    lamb = np.linspace(3.9, 4, nlamb)*1e-10
    coll.add_data(
        'lamb',
        data=lamb,
        ref='nlamb',
        units='m',
        dim='dist',
        quant='wavelength',
    )

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
    nt = 20
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
    # data 1d
    # ------------------

    # lamb0
    amp0 = 1000
    Dlamb = lamb[-1] - lamb[0]
    lamb0 = lamb[0] + Dlamb * np.r_[0.25, 0.55, 0.7]
    width = Dlamb * np.r_[0.1, 0.05, 0.02]
    amp = amp0 * np.r_[1, 0.5, 2]

    # data 1d
    data = (
        amp0 * 0.1
        + np.sum(
            [
                amp[ii] * np.exp(-(lamb-lamb0[ii])**2 / (2*width[ii]**2))
                for ii in range(len(lamb0))
            ],
        axis=0,
        )
    ) + amp0 * 0.05 * np.random.random((nlamb,))

    coll.add_data(
        'data1d',
        data=data,
        ref='nlamb',
        units='ph/s',
    )

    # ------------------
    # data 1d vs t
    # ------------------

    # ------------------
    # data 2d
    # ------------------

    # ------------------
    # data 2d vs t
    # ------------------

    return


# ###################################################
# ###################################################
#               dmodels
# ###################################################


def add_models(coll=None):

    # ---------------
    # dmodels

    dmodel = {
        'model00': {
            'bck0': 'linear',
            'l00': 'gauss',
            'l01': 'gauss',
            'l02': 'lorentz',
        },
        'model01': {
            'bck0': 'exp',
            'l00': 'gauss',
            'l01': 'lorentz',
            'l02': 'pvoigt',
        },
        'model02': {
            'bck0': 'exp',
            'l00': 'gauss',
            'l01': 'lorentz',
            'l02': 'voigt',
        },
    }

    # ---------------
    # add models

    # no constraints
    coll.add_spectral_model(
        key='model00',
        dmodel=dmodel['model00'],
        dconstraints=None,
    )

    # with constraints
    coll.add_spectral_model(
        key='model01',
        dmodel=dmodel['model01'],
        dconstraints={
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_width', 'l01_gamma': [0, 1, 0]},
        },
    )

    # with voigt
    coll.add_spectral_model(
        key='model02',
        dmodel=dmodel['model02'],
        dconstraints={
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_width', 'l01_gamma': [0, 1, 0]},
        },
    )

    return


# ###################################################
# ###################################################
#               spectral fit
# ###################################################


def add_fit1d(coll=None):

    coll.add_spectral_fit(
        key_model='model00',
        key_data='data1d',
        key_sigma=None,
        key_lamb='lamb',
        # params
        dparams=None,
        dvalid=None,
    )

    return
