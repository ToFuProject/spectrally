# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:57:00 2024

@author: dvezinet
"""


import numpy as np
import scipy.constants as scpct


from . import _class01_interpolate as _interpolate


#############################################
#############################################
#       main
#############################################


def main(
    coll=None,
    key_model=None,
    key_data=None,
    lamb=None,
    dmz=None,
):

    # ------------
    # check inputs
    # ------------

    (
        key_model, ref_nx, ref_nf,
        key_data,
        key_lamb, lamb, ref_lamb,
        details,
        returnas, store, store_key,
    ) = _interpolate._check(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
        # others
        returnas=None,
        store=None,
        store_key=None,
    )

    # -------------------------
    # prepare model parameters
    # -------------------------

    # dconstraints
    wsm = coll._which_model
    dconstraints = coll.dobj[wsm][key_model]['dconstraints']

    # coefs
    c0 = dconstraints['c0']
    c1 = dconstraints['c1']
    c2 = dconstraints['c2']

    # param_val
    param_val = coll.get_spectral_model_variables(
        key_model,
        returnas='param_value',
        concatenate=True,
    )['param_value']

    # dind
    dind = coll.get_spectral_model_variables_dind(key_model)

    # ------------
    # mz
    # ------------

    dind = _check_mz(dmz, dind=dind)

    # ------------
    # get func
    # ------------

    func = _get_func_moments(
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        param_val=param_val,
    )

    # ------------
    # compute
    # ------------

    dout = func(
        x_free=coll.ddata[key_data]['data'],
        lamb=lamb,
        scale=None,
    )

    # -------------
    # format output
    # -------------

    return dout


#############################################
#############################################
#       check
#############################################


def _check_mz(
    dmz=None,
    dind=None,
):

    # -----------
    # trivial
    # ----------

    # add mz if user-provided
    if dmz is not None:

        for kfunc in ['gauss', 'pvoigt', 'voigt']:

            if dind.get(kfunc) is None:
                continue

            dind[kfunc]['mz']

    return dind



#############################################
#############################################
#       moments
#############################################


def _get_func_moments(
    c0=None,
    c1=None,
    c2=None,
    dind=None,
    param_val=None,
):

    # --------------
    # prepare
    # --------------

    def func(
        x_free=None,
        lamb=None,
        param_val=param_val,
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        scale=None,
    ):

        # ----------
        # initialize

        dout = {k0: {} for k0 in dind.keys() if k0 not in ['func', 'nfunc']}

        # ----------------------------
        # get x_full from constraints

        if c0 is None:
            x_full = x_free
        else:
            x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0

        # -------------------
        # rescale

        if scale is not None:
            pass

        # ------------------
        # sum all linear

        kfunc = 'linear'
        if dind.get(kfunc) is not None:

            a0 = x_full[dind[kfunc]['a0']['ind']]
            a1 = x_full[dind[kfunc]['a1']['ind']]

            # variables
            dout[kfunc]['a0'] = a0
            dout[kfunc]['a1'] = a1

            # integral
            if lamb is not None:
                dout[kfunc]['integ'] = (
                    a0 * (lamb[-1] - lamb[0])
                    + a1 * (lamb[-1]**2 - lamb[0]**2)/2
                )

        # --------------------
        # sum all exponentials

        kfunc = 'exp_lamb'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            rate = x_full[dind[kfunc]['rate']['ind']]

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['rate'] = rate

            # physics
            dout[kfunc]['Te'] = (scpct.h * scpct.c / rate) / scpct.e

            # integral
            if lamb is not None:
                dout[kfunc]['integ'] = (
                    (amp / rate)
                    * (np.exp(lamb[-1] * rate) - np.exp(lamb[0] * rate))
                )

        # -----------------
        # sum all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            sigma = x_full[dind[kfunc]['sigma']['ind']]
            vccos = x_full[dind[kfunc]['vccos']['ind']]
            lamb0 = param_val[dind[kfunc]['lamb0']]

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['sigma'] = sigma
            dout[kfunc]['vccos'] = vccos

            # physics
            if dind[kfunc].get('mz') is not None:
                mz = param_val[dind[kfunc]['mz']]
                dout[kfunc]['Ti'] = (
                    (sigma / lamb0)**2 * mz * scpct.c**2 * scpct.e
                )

            # integral
            dout[kfunc]['integ'] = amp * sigma * np.sqrt(2 * np.pi)

        # -------------------
        # sum all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            gam = x_full[dind[kfunc]['gam']['ind']]
            vccos = x_full[dind[kfunc]['vccos']['ind']]
            lamb0 = param_val[dind[kfunc]['lamb0']]

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['gam'] = gam
            dout[kfunc]['vccos'] = vccos

            # integral
            dout[kfunc]['integ'] = amp * np.pi * gam


        # --------------------
        # sum all pseudo-voigt

        kfunc = 'pvoigt'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            sigma = x_full[dind[kfunc]['sigma']['ind']]
            gam = x_full[dind[kfunc]['gam']['ind']]
            vccos = x_full[dind[kfunc]['vccos']['ind']]
            lamb0 = param_val[dind[kfunc]['lamb0']]

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['sigma'] = sigma
            dout[kfunc]['gam'] = gam
            dout[kfunc]['vccos'] = vccos

            # physics
            if dind[kfunc].get('mz') is not None:
                mz = param_val[dind[kfunc]['mz']]
                dout[kfunc]['Ti'] = (
                    (sigma / lamb0)**2 * mz * scpct.c**2 * scpct.e
                )

            # integral
            dout[kfunc]['integ'] = None

        # ------------------
        # sum all pulse1

        kfunc = 'pulse1'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            t0 = x_full[dind[kfunc]['t0']['ind']]
            tup = x_full[dind[kfunc]['t_up']['ind']]
            tdown = x_full[dind[kfunc]['t_down']['ind']]

            ind0 = lamb > t0

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['t0'] = t0
            dout[kfunc]['tup'] = tup
            dout[kfunc]['tdown'] = tdown

            # integral
            dout[kfunc]['integ'] = None

        # ------------------
        # sum all pulse2

        kfunc = 'pulse2'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            t0 = x_full[dind[kfunc]['t0']['ind']]
            tup = x_full[dind[kfunc]['t_up']['ind']]
            tdown = x_full[dind[kfunc]['t_down']['ind']]

            indup = (lamb < t0)
            inddown = (lamb >= t0)

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['t0'] = t0
            dout[kfunc]['tup'] = tup
            dout[kfunc]['tdown'] = tdown

            # integral
            dout[kfunc]['integ'] = None

        # ------------------
        # sum all lognorm

        kfunc = 'lognorm'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            t0 = x_full[dind[kfunc]['t0']['ind']]
            sigma = x_full[dind[kfunc]['sigma']['ind']]
            mu = x_full[dind[kfunc]['mu']['ind']]

            # max at lamb - t0 = exp(sigma/2 * (mu-1))
            # max = amp / mu

            # variables
            dout[kfunc]['amp'] = amp
            dout[kfunc]['t0'] = t0
            dout[kfunc]['sigma'] = sigma
            dout[kfunc]['mu'] = mu

            # integral
            dout[kfunc]['integ'] = None

        return dout

    return func