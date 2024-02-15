# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import copy


import numpy as np
from bsplines2d import BSplines2D as Previous
import datastock as ds


from ._class00_SpectralLines import SpectralLine as Previous
from . import _class01_check as _check


__all__ = ['SpectralFit']


#############################################
#############################################
#       DEFAULT VALUES
#############################################




#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralFit(Previous):

    _which_model = 'spect_model'

    _ddef = copy.deepcopy(Previous._ddef)

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _dshow = dict(Previous._dshow)
    _dshow[_which_model] = ['']

    _ddef['params']['dobj'] = {
        'spect_model': {
            'lambda0': {'cls': float, 'def': 0.},
            'source': {'cls': str, 'def': 'unknown'},
        },
    }

    # -------------------
    # add spectral model
    # -------------------

    def add_spectral_model(
        self,
        key=None,
        dmodel=None,
    ):

        return _check._model(
            coll=self,
            key=key,
            dmodel=dmodel,
        )

    # -------------------
    # show spectral model
    # -------------------



    # -------------------
    # plot spectral model
    # -------------------