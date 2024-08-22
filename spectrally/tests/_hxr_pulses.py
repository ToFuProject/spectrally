# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:11:23 2024

@author: dvezinet
"""

import os


import numpy as np


from .._class02_SpectralFit import SpectralFit as Collection


# ############################################################
# ############################################################
#                DEFAULTS
# ############################################################


_PATH_HERE = os.path.dirname(__file__)
_PFE = os.path.join(_PATH_HERE, 'input', 'HXR_pulses.npz')


# ############################################################
# ############################################################
#                main
# ############################################################


def main(
    coll=None,
    pfe=_PFE,
    # spectral fits to be run
    lspect_fit=['exp', 'gauss', 'lognorm'],
    strict=True,
    verb=1,
):

    # --------------
    # initialize
    # --------------

    if coll is None:
        coll = Collection()

    # --------------
    # load file
    # --------------

    dout = dict(np.load(pfe, allow_pickle=True))

    # --------------
    # populate
    # --------------

    # -------------
    # ref

    coll.add_ref('npulse', size=dout['pulse'].size)
    coll.add_ref('nsamp', size=dout['data'].shape[0])

    # -------------
    # data

    # pulses
    coll.add_data(
        key='pulses',
        data=dout['pulse'],
        ref='npulse',
        units='index',
    )

    # indices
    coll.add_data(
        key='sample',
        data=np.arange(0, dout['data'].shape[0]),
        ref='nsamp',
        units='index',
    )

    # data
    coll.add_data(
        key='current',
        data=dout['data'],
        ref=('nsamp', 'npulse'),
        units='A',
    )

    # --------------
    # spectral models
    # --------------

    # spectral constrainst
    dconstraints = {
        'gbck': {
            'ref': 'bck_a0',
            'bck_a1': [0, 0, 0],
            'bck_a2': [0, 0, 0],
        },
    }

    # single exponential pulse
    coll.add_spectral_model(
        key='sm_exp',
        dmodel={
            'bck': 'poly',
            'pulse': 'pulse_exp',
        },
        dconstraints=dconstraints,
    )

    # single gaussian pulse
    coll.add_spectral_model(
        key='sm_gauss',
        dmodel={
            'bck': 'poly',
            'pulse': 'pulse_gauss',
        },
        dconstraints=dconstraints,
    )

    # single lognorm pulse
    coll.add_spectral_model(
        key='sm_lognorm',
        dmodel={
            'bck': 'poly',
            'pulse': 'lognorm',
        },
        dconstraints=dconstraints,
    )

    # --------------
    # define spectral fits
    # --------------

    # single exponential pulse
    for k0 in lspect_fit:
        coll.add_spectral_fit(
            key=f'sf_{k0}',
            key_model=f'sm_{k0}',
            key_data='current',
            key_sigma=None,
            key_lamb='sample',
            # params
            dparams=None,
            dvalid={
                'mask': None,
                'domain': None,
                'focus': None,
                'focus_logic': None,
            },
        )

    # ---------------------
    # compute spectral fits
    # ---------------------

    for k0 in lspect_fit:

        coll.compute_spectral_fit(
            key=f"sf_{k0}",
            strict=strict,
            verb=verb,
            timing=None,
        )

    return coll


# ############################################################
# ############################################################
#                __main__
# ############################################################


if __name__ == "__main__":
    coll = main()