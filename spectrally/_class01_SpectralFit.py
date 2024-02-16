# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import copy


import numpy as np
from bsplines2d import BSplines2D as Previous
import datastock as ds


from ._class00_SpectralLines import SpectralLines as Previous
from . import _class01_check_model as _check_model
from . import _class01_check_constraints as _check_constraints


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
    }

    # ###################
    # -------------------
    # Spectral models
    # -------------------

    # -------------------
    # add spectral model
    # -------------------

    def add_spectral_model(
        self,
        key=None,
        dmodel=None,
        dconstraints=None,
    ):
        """ Add a spectral model for future fitting

        Defined by a set of functions and constraints.
            - dmodel: dict of (key, function type) pairs
            - dconstraints: dict of (key, {'ref': k0, k1: [c0, c1], k2: [c0, c1]})

        Available function types for dmodel are:
            - 'linear': typically a linear background
            - 'exp': typically an exponential background
            - 'gauss': typically a doppler-broadened line
            - 'lorentz': ypically a natural-broadened line
            - 'pvoigt': typically a doppler-and-natural broadened line

        dconstraints holds a dict of constraints groups.
        Each group is a dict with a 'ref' variable
        Other variables (keys) are compued as linear functions of 'ref'

        Parameters
        ----------
        key : str, optional
            DESCRIPTION. The default is None.
        dmodel : dict, optional
            DESCRIPTION. The default is None.
        dconstraints : dict, optional
            DESCRIPTION. The default is None.

        """

        # check and store the model
        _check_model._dmodel(
            coll=self,
            key=key,
            dmodel=dmodel,
            dconstraints=dconstraints,
        )

        # check and store the constraints
        _check_constraints._dconstraints(
            coll=self,
            key=key,
            dconstraints=dconstraints,
        )

    # -------------------
    # show spectral model
    # -------------------

    def _get_show_obj(self, which=None):
        if which == self._which_model:
            return _check_model._show
        else:
            return super()._get_show_obj()

    # -------------------
    # get spectral model
    # -------------------

    def get_spectral_model_variables(self, key=None):
        """ Get ordered list of individual variable names """
        return _check_model._get_var(self, key=key)


    # ###################
    # -------------------
    # Spectral fits
    # -------------------

    # -------------------
    # add spectral fit
    # -------------------

    def add_spectral_fit(
        self,
        key_model=None,
        key_data=None,
        key_lamb=None,
        dinitial=None,
        dscales=None,
        dconstants=None,
        domain=None,
        # optional 2d fit
        key_bs=None,
    ):

        # _check_fit._fit(
        #     coll=self,
        #     key=key,
        #     dmodel=dmodel,
        #     dconstraints=dconstraints,
        # )

        return