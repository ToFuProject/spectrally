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
        'sm-1': {
            'bck0': 'linear',
            'l00': 'gauss',
            'l01': 'gauss',
            'l02': 'lorentz',
        },
        'sm00': {
            'bck0': 'linear',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'l01': {'type': 'gauss', 'lamb0': 3.95e-10},
            'l02': {'type': 'lorentz', 'lamb0': 3.97e-10},
        },
        'sm01': {
            'bck0': 'exp',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'sl00': 'lorentz',
            'l02': {'type': 'pvoigt', 'lamb0': 3.97e-10},
        },
        'sm02': {
            'bck0': 'exp',
            'l00': {'type': 'gauss', 'lamb0': 3.92e-10},
            'l01': {'type': 'lorentz', 'lamb0': 3.95e-10},
            'l02': {'type': 'voigt', 'lamb0': 3.97e-10},
        },
        'sm03': {
            'bck0': 'linear',
            'l00': {'type': 'pulse1'},
            'l01': {'type': 'pulse2'},
            'l02': {'type': 'lognorm'},
        },
    }

    # ---------------
    # add models

    # check err
    try:
        coll.add_spectral_model(
            key='sm-1',
            dmodel=dmodel['sm-1'],
            dconstraints=None,
        )
        raise Exception('sucess')
    except Exception as err:
        if "For model" not in str(err):
            raise Exception("Wrong error raised!")

    # no constraints
    coll.add_spectral_model(
        key='sm00',
        dmodel=dmodel['sm00'],
        dconstraints=None,
    )

    # with constraints
    coll.add_spectral_model(
        key='sm01',
        dmodel=dmodel['sm01'],
        dconstraints={
            'g00': {'ref': 'l00_amp', 'sl00_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_width', 'sl00_gamma': [0, 1, 0]},
        },
    )

    # with voigt
    coll.add_spectral_model(
        key='sm02',
        dmodel=dmodel['sm02'],
        dconstraints={
            'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
            'g01': {'ref': 'l00_width', 'l01_gamma': [0, 1, 0]},
        },
    )

    # with pulses
    coll.add_spectral_model(
        key='sm03',
        dmodel=dmodel['sm03'],
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

        for ff in ['sum', 'cost']: # , 'details', 'jac']:

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

    # --------------
    # prepare xfree

    # sm00
    # 'bck0_a0', 'bck0_a1',
    # 'l00_amp', 'l00_shift', 'l00_width',
    # 'l01_amp', 'l01_shift', 'l01_width',
    # 'l02_amp', 'l02_shift', 'l02_gamma'

    # sm01
    # 'bck0_amp', 'bck0_rate',
    # 'l00_amp', 'l00_shift', 'l00_width',
    # 'l02_amp', 'l02_shift', 'l02_width', 'l02_t', 'l02_gamma',
    # 'sl00_shift'

    # sm02
    # 'bck0_amp', 'bck0_rate',
    # 'l00_amp', 'l00_shift', 'l00_width',
    # 'l01_shift',
    # 'l02_amp', 'l02_shift', 'l02_width', 'l02_gamma'

    # sm03
    # 'bck0_a0', 'bck0_a1',
    # 'l00_amp', 'l00_t0', 'l00_t_up', 'l00_t_down',
    # 'l01_amp', 'l01_t0', 'l01_t_up', 'l01_t_down',
    # 'l02_amp', 'l02_t0', 'l02_mu', 'l02_sigma'

    t = coll.ddata['t']['data']

    dxfree = {
        'sm00': np.r_[
            0.2, 0.,
            1, 0.01e-10, 0.005e-10,
            0.8, -0.01e-10, 0.001e-10,
            1.2e-12, 0.01e-10, 0.,
        ],
        'sm01': np.r_[
            0.1, 0.01,
            1, 0.01e-10, 0.005e-10,
            0.8, -0.01e-10, 0.001e-10, 1.2, 0.,
            0.01e-10,
        ][None, :] * np.exp(-(t[:, None] - np.mean(t))**2 / 2**2),
        'sm02': np.r_[
            0.1, -0.01,
            1, 0.01e-10, 0.002e-10,
            -0.01e-10,
            1, 0.1e-10, 0.05e-10, 0.1e-10,
        ],
        'sm03': np.r_[
            0.1, 0.,
            2, 3.91e-10, 0.001e-10, 0.004e-10,
            1, 3.94e-10, 0.001e-10, 0.007e-10,
            np.sqrt(2*np.pi)*3.98e-10*0.001, 3.8999e-10, np.log(0.08e-10), 0.05,
        ],
    }

    # ---------------
    # add model data

    wsm = coll._which_model
    lkstore = []
    for ii, kmodel in enumerate(coll.dobj[wsm].keys()):

        # get nx, nf, ref_nx
        nf, nx = coll.dobj[wsm][kmodel]['dconstraints']['c1'].shape
        ref_nx = coll.dobj[wsm][kmodel]['ref_nx']

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

        # interpolate
        for jj, details in enumerate([False, True]):
            for kk, store in enumerate([False, True]):

                store_key = f'interp_{kmodel}_{jj}_{kk}'
                lambi = ('lamb' if store else lamb)

                _ = coll.interpolate_spectral_model(
                    key_model=kmodel,
                    key_data=kdata,
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

        # remove data (but not model ref)
        del coll._ddata[kdata]

    # remove stored output
    coll.remove_data(lkstore)

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
                key_model='sm00',
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
