# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import copy


from ._class00_SpectralLines import SpectralLines as Previous
from . import _class01_check_model as _check_model
from . import _class01_show as _show
from . import _class01_check_constraints as _check_constraints
from . import _class01_fit_func as _fit_func
from . import _class01_interpolate as _interpolate
from . import _class01_moments as _moments
from . import _class01_plot as _plot


__all__ = ['SpectralModel']


#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralModel(Previous):

    _which_model = 'spect_model'
    _which_fit = 'spect_fit'

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
            - dconstraints: dict of:
                'key': {'ref': k0, k1: [c0, c1], k2: [c0, c1]}

        Available function types for dmodel are:
            - 'poly': typically a polynomial background
            - 'exp': typically an exponential background
            - 'gauss': typically a doppler-broadened line
            - 'lorentz': ypically a natural-broadened line
            - 'pvoigt': typically a doppler-and-natural broadened line

        dconstraints holds a dict of constraints groups.
        Each group is a dict with a 'ref' variable
        Other variables (keys) are compued as poly functions of 'ref'

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
            return _show._show
        else:
            return super()._get_show_obj(which)

    def _get_show_details(self, which=None):
        if which == self._which_model:
            return _show._show_details
        else:
            super()._get_show_details(which)

    # -------------------
    # get spectral model
    # -------------------

    def get_spectral_model_variables(
        self,
        key=None,
        returnas=None,
        concatenate=None,
    ):
        """ Get ordered list of individual variable names """

        return _check_model._get_var(
            coll=self,
            key=key,
            returnas=returnas,
            concatenate=concatenate,
        )

    def get_spectral_model_variables_dind(
        self,
        key=None,
    ):
        """ Get ordered list of individual variable names """

        return _check_model._get_var_dind(
            coll=self,
            key=key,
        )

    # ----------------------
    # interpolate spectral model
    # ----------------------

    def interpolate_spectral_model(
        self,
        key_model=None,
        key_data=None,
        lamb=None,
        # options
        details=None,
        binning=None,
        # others
        returnas=None,
        store=None,
        store_key=None,
    ):
        """ Interpolate the spectral model at lamb using key_data


        Parameters
        ----------
        key_model : str, optional
            key to the desired spectral model
        key_data : str, optional
            key to the data to be used for the model's free variables
                - has to have the model's ref in its own references
        lamb : str/np.ndarray
            DESCRIPTION. The default is None.

        Returns
        -------
        dout : dict
            output dict of interpolated data, with units and ref

        """

        return _interpolate.main(
            coll=self,
            key_model=key_model,
            key_data=key_data,
            lamb=lamb,
            # options
            details=details,
            binning=binning,
            # others
            returnas=returnas,
            store=store,
            store_key=store_key,
        )

    # ----------------------
    # extract spectral model moments
    # ----------------------

    def get_spectral_model_moments(
        self,
        key=None,
        key_data=None,
        lamb=None,
        returnas=None,
    ):
        """

        Given an array of solution for the model's free variables (key_data)
        Returns, a dict with, for each function in the model:

            - all variable values (claasified by type)
            - some moments of interest:
                - integral
                - physics-derived (Te from exp, Ti from line width...)

        Parameters
        ----------
        key_model : str
            key to the desired spectral model (or spectral fit)
        key_data : str
            key to the data to be used as input for the model's free variables
        lamb:str or 1d np.ndarray, optional
            wavelenth vector to be used for computing limited integrals
        returnas: str
            Flag indicating whether to return dout as:
                - 'dict_arrays': nested dict of {'ktype': {'kvar': array}}
                - 'dict_varnames': dict of {'kfunc_var': values}

        Returns
        -------
        dout : dict
            dict of all moments, classified per type and function

        """

        dout = _moments.main(
            coll=self,
            key=key,
            key_data=key_data,
            lamb=lamb,
            returnas=returnas,
        )

        return dout

    # ###################
    # -------------------
    # Spectral fits
    # -------------------

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

    # ----------------------------
    # plot spectral modela and fit
    # ----------------------------

    def plot_spectral_model(
        self,
        key_model=None,
        key_data=None,
        lamb=None,
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

        return _plot.main(
            coll=self,
            key_model=key_model,
            key_data=key_data,
            lamb=lamb,
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