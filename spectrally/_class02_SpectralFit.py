# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import copy


from ._class01_SpectralModel import SpectralModel as Previous
from . import _class02_check_fit as _check_fit
from . import _class01_fit_func as _fit_func
from . import _class02_compute_fit as _compute_fit
from . import _class02_plot_valid as _plot_valid
from . import _class02_plot_fit as _plot_fit


__all__ = ['SpectralFit']


#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralFit(Previous):

    _ddef = copy.deepcopy(Previous._ddef)

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _dshow = dict(Previous._dshow)

    _ddef['params']['dobj'] = {
    }

    # ###################
    # -------------------
    # Spectral models
    # -------------------

    def _get_show_obj(self, which=None):
        if which == self._which_fit:
            return _check_fit._show
        else:
            return super()._get_show_obj(which)

    # ###################
    # -------------------
    # Spectral fits
    # -------------------

    # -------------------
    # add spectral fit
    # -------------------

    def add_spectral_fit(
        self,
        # keys
        key=None,
        key_model=None,
        key_data=None,
        key_sigma=None,
        # wavelength
        key_lamb=None,
        # optional 2d fit
        key_bs=None,
        key_bs_vect=None,
        # fit parameters
        dparams=None,
        dvalid=None,
        # compute options
        chain=None,
    ):

        _check_fit._check(
            coll=self,
            # keys
            key=key,
            key_model=key_model,
            key_data=key_data,
            key_sigma=key_sigma,
            # wavelength
            key_lamb=key_lamb,
            # optional 2d fit
            key_bs=key_bs,
            key_bs_vect=key_bs_vect,
            # fit parameters
            dparams=dparams,
            dvalid=dvalid,
            # compute options
            chain=chain,
        )

        return

    # -------------------
    # get func details, cost, jac
    # -------------------

    def get_spectral_fit_func(
        self,
        key=None,
        func=None,
    ):
        """ Return the fitting functions for a given model
        """

        return _fit_func.main(
            coll=self,
            key=key,
            func=func,
        )

    # -------------------
    # compute spectral fit
    # -------------------

    def compute_spectral_fit(
        self,
        key=None,
        # binning
        binning=None,
        # solver options
        solver=None,
        dsolver_options=None,
        # storing
        store=None,
        # options
        strict=None,
        verb=None,
        timing=None,
    ):

        return _compute_fit.main(
            coll=self,
            key=key,
            # binning
            binning=None,
            # solver options
            solver=solver,
            dsolver_options=dsolver_options,
            # storing
            store=store,
            # options
            strict=strict,
            verb=verb,
            timing=timing,
        )

    # ----------------------------------
    # plot spectral fit data validity
    # ----------------------------------

    def plot_spectral_fit_input_validity(
        self,
        key=None,
        # options
        dprop=None,
        vmin=None,
        vmax=None,
        # figure
        dax=None,
        fs=None,
        dmargin=None,
        tit=None,
    ):

        return _plot_valid.plot(
            coll=self,
            key=key,
            # options
            dprop=dprop,
            vmin=vmin,
            vmax=vmax,
            # figure
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            tit=tit,
        )

    # ----------------------------
    # plot spectral fit
    # ----------------------------

    def plot_spectral_fit(
        self,
        key=None,
        keyY=None,
        # options
        details=None,
        # plotting
        dprop=None,
        vmin=None,
        vmax=None,
        # figure
        dax=None,
        fs=None,
        dmargin=None,
        tit=None,
    ):
        """ Plot a spectral model using specified data

        lamb can be:
            - a key to an existing vector
            - a user-provided vector (1d np.ndarray)

        """

        return _plot_fit.main(
            coll=self,
            key=key,
            keyY=keyY,
            # options
            details=details,
            # plotting
            dprop=dprop,
            vmin=vmin,
            vmax=vmax,
            # figure
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            tit=tit,
        )
