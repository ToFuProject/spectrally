# -*- coding: utf-8 -*-


import os
import traceback
import itertools as itt


import numpy as np
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


# ###################################################
# ###################################################
#               data
# ###################################################


def add_data(coll=None):

    # ------------------
    # reference vectors
    # ------------------

    # lamb
    nlamb = 300
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
    # data 1d
    # ------------------

    # lamb0
    amp0 = 700
    Dlamb = lamb[-1] - lamb[0]
    lamb0 = lamb[0] + Dlamb * np.r_[0.25, 0.55, 0.75]
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
        size=nlamb,
    ).astype(float) # + amp0 * 0.10 * np.random.random((nlamb,))

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
        size=(nt, nlamb),
    ).astype(float) # + amp0 * 0.10 * np.random.random((nlamb,))

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


# ###################################################
# ###################################################
#               dmodels
# ###################################################


def add_models(coll=None):

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

    # ---------------
    # dmodels

    dmodel = {
        'model-1': {
            'bck0': 'linear',
            'l00': 'gauss',
            'l01': 'gauss',
            'l02': 'lorentz',
        },
        'model00': {
            'bck0': 'linear',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'l01': {'type': 'gauss', 'lamb0': 3.95e-10},
            'l02': {'type': 'lorentz', 'lamb0': 3.97e-10},
        },
        'model01': {
            'bck0': 'exp',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'sl00': 'lorentz',
            'l02': {'type': 'pvoigt', 'lamb0': 3.97e-10},
        },
        'model02': {
            'bck0': 'exp',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'l01': {'type': 'lorentz', 'lamb0': 3.95e-10},
            'l02': {'type': 'voigt', 'lamb0': 3.97e-10},
        },
    }

    # ---------------
    # add models

    # check err
    try:
        coll.add_spectral_model(
            key='model-1',
            dmodel=dmodel['model-1'],
            dconstraints=None,
        )
        raise Exception('sucess')
    except Exception as err:
        if "For model" not in str(err):
            raise Exception("Wrong error raised!")

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
            'g00': {'ref': 'l00_amp', 'sl00_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_width', 'sl00_gamma': [0, 1, 0]},
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
#               spectral fit - add
# ###################################################


def add_fit(coll=None, key_data=None):

    # ---------------
    # check

    add_models(coll)

    if coll.dobj.get('spect_fit') is not None:
        lk = [
            k0 for k0, v0 in coll.dobj['spect_fit'].items()
            if v0['key_data'] == key_data
        ]
        if len(lk) > 0:
            return

    # -------------------
    # add 1d

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
                key_model='model00',
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


# ###################################################
# ###################################################
#           plot spectral fit input validity
# ###################################################


def plot_input_validity(coll=None, key_data=None):

    # ---------------
    # check

    add_models(coll)
    add_fit(coll, key_data=key_data)

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

    # close
    plt.close('all')
    return


# ###################################################
# ###################################################
#               spectral fit - compute
# ###################################################


def compute_fit(coll=None, key_data=None):

    # ---------------
    # check

    add_models(coll)
    add_fit(coll, key_data=key_data)

    # ---------------
    # select data

    lk = [
        k0 for k0, v0 in coll.dobj['spect_fit'].items()
        if v0['key_data'] == key_data
    ]

    # ---------------
    # compute

    for k0 in lk:
        coll.compute_spectral_fit(
            key=k0,
            verb=None,
            timing=None,
        )

    return