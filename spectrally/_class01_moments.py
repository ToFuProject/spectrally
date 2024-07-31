# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:57:00 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import scipy.constants as scpct
import datastock as ds


from . import _class01_model_dict as _model_dict
from . import _class01_interpolate as _interpolate


#############################################
#############################################
#       main
#############################################


def main(
    coll=None,
    key=None,
    key_data=None,
    lamb=None,
    dmz=None,
):

    # ------------
    # check inputs
    # ------------

    # key_model vs key_fit
    key_model, key_data, lamb = _check(
        coll=coll,
        key=key,
        key_data=key_data,
        lamb=lamb,
    )

    # all other variables
    (
        key_model, ref_nx, ref_nf,
        key_data,
        key_lamb, lamb, ref_lamb,
        details, binning,
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
        axis=coll.ddata[key_data]['ref'].index(ref_nx),
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


def _check(coll=None, key=None, key_data=None, lamb=None):

    # ---------------------
    # key_model vs key_fit
    # ---------------------

    wsm = coll._which_model
    wsf = coll._which_fit

    lokm = list(coll.dobj.get(wsm, {}).keys())
    lokf = list(coll.dobj.get(wsf, {}).keys())

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lokm + lokf,
    )

    if key in lokf:
        key_fit = key
        key = coll.dobj[wsf][key_fit]['key_model']

        if key_data is None:
            key_data = coll.dobj[wsf][key_fit]['key_sol']

        if lamb is None:
            lamb = coll.dobj[wsf][key_fit]['key_lamb']

    return key, key_data, lamb


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
    axis=None,
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
        axis=axis,
    ):

        # ----------
        # prepare

        lambD = lamb[-1] - lamb[0]
        lambm = 0.5*(lamb[0] + lamb[-1])

        # ----------
        # initialize

        dout = {
            k0: {} for k0 in dind.keys()
            if k0 not in ['func', 'nfunc', 'jac']
        }

        # ----------------------------
        # get x_full from constraints

        if c0 is None:
            x_full = x_free
        else:
            if x_free.ndim > 1:
                shape = list(x_free.shape)
                shape[axis] = c0.size
                x_full = np.full(shape, np.nan)
                sli = list(shape)
                sli[axis] = slice(None)
                sli = np.array(sli)
                ich = np.array([ii for ii in range(len(shape)) if ii != axis])
                linds = [range(shape[ii]) for ii in ich]
                for ind in itt.product(*linds):
                    sli[ich] = ind
                    slii = tuple(sli)
                    x_full[slii] = (
                        c2.dot(x_free[slii]**2) + c1.dot(x_free[slii]) + c0
                    )

            else:
                x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0

        sli = [None if ii == axis else slice(None) for ii in range(x_free.ndim)]
        extract = _get_var_extract_func(x_full, dind, axis, sli)

        # -------------------
        # rescale

        if scale is not None:
            pass

        # ---------------------
        # extract all variables

        for kfunc, v0 in _model_dict._DMODEL.items():
            if dind.get(kfunc) is not None:
                for kvar in v0['var']:
                    dout[kfunc][kvar] = extract(kfunc, kvar)

        # ------------------
        # sum all poly

        kfunc = 'poly'
        if dind.get(kfunc) is not None:

            a0 = dout[kfunc]['a0']
            a1 = dout[kfunc]['a1']
            a2 = dout[kfunc]['a2']

            # integral
            if lamb is not None:
                dout[kfunc]['integ'] = (
                    a0 * (lamb[-1] - lamb[0])
                    + a1 * (lamb[-1]**2 - lamb[0]**2)/2
                )

            # argmax, max
            dout[kfunc]['argmax'] = np.full(a0.shape, np.nan)
            dout[kfunc]['max'] = np.full(a1.shape, np.nan)
            iok = a2 != 0
            dout[kfunc]['argmax'][iok] = lambm - lambD * a1[iok]/(2*a2[iok])
            dout[kfunc]['max'][iok] = a0[iok] - a1[iok]**2 / (4*a2[iok])

        # --------------------
        # sum all exponentials

        kfunc = 'exp_lamb'
        if dind.get(kfunc) is not None:

            # physics
            rate = dout[kfunc]['rate']
            dout[kfunc]['Te'] = (scpct.h * scpct.c / rate) / scpct.e

            # integral
            if lamb is not None:
                amp = dout[kfunc]['rate']
                dout[kfunc]['integ'] = (
                    (amp / rate)
                    * (np.exp(lamb[-1] * rate) - np.exp(lamb[0] * rate))
                )

        # -----------------
        # sum all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            sigma = dout[kfunc]['sigma']
            vccos = dout[kfunc]['vccos']

            # argmax
            dout[kfunc]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, kfunc, amp.shape, axis,
            )

            # integral
            dout[kfunc]['integ'] = amp * sigma * np.sqrt(2 * np.pi)

            # physics
            if dind[kfunc].get('mz') is not None:
                dout[kfunc]['Ti'] = _get_Ti(
                    sigma,
                    param_val,
                    dind,
                    kfunc,
                    sigma.shape,
                    axis,
                )

        # -------------------
        # sum all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            gam = dout[kfunc]['gam']
            vccos = dout[kfunc]['vccos']

            # argmax
            dout[kfunc]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, kfunc, amp.shape, axis,
            )

            # integral
            dout[kfunc]['integ'] = amp * np.pi * gam

        # --------------------
        # sum all pseudo-voigt

        kfunc = 'pvoigt'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            sigma = dout[kfunc]['sigma']
            vccos = dout[kfunc]['vccos']

            # argmax
            dout[kfunc]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, kfunc, amp.shape, axis,
            )

            # integral
            dout[kfunc]['integ'] = np.full(sigma.shape, np.nan)

            # physics
            if dind[kfunc].get('mz') is not None:
                dout[kfunc]['Ti'] = _get_Ti(
                    sigma,
                    param_val,
                    dind,
                    kfunc,
                    sigma.shape,
                    axis,
                )

        # --------------------
        # sum all voigt

        kfunc = 'voigt'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            sigma = dout[kfunc]['sigma']
            vccos = dout[kfunc]['vccos']

            # argmax
            dout[kfunc]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, kfunc, amp.shape, axis,
            )

            # integral
            dout[kfunc]['integ'] = amp

            # physics
            if dind[kfunc].get('mz') is not None:
                dout[kfunc]['Ti'] = _get_Ti(
                    sigma,
                    param_val,
                    dind,
                    kfunc,
                    sigma.shape,
                    axis,
                )

        # ------------------
        # sum all pulse_exp

        kfunc = 'pulse_exp'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            tau = dout[kfunc]['tau']
            t_down = dout[kfunc]['t_down']
            t_up = dout[kfunc]['t_up']

            # integral
            dout[kfunc]['integ'] = amp * (t_down - t_up)

            # prepare
            t0 = lamb[0] + lambD * tau
            dtdu = t_down - t_up
            lntdu = np.log(t_down / t_up)

            # position of max
            dout[kfunc]['t0'] = t0
            dout[kfunc]['argmax'] = t0 + lntdu * t_down*t_up / dtdu

            # value at max
            dout[kfunc]['max'] = amp * (
                np.exp(-lntdu * t_up / dtdu)
                - np.exp(-lntdu * t_down / dtdu)
            )

        # ------------------
        # sum all pulse_gauss

        kfunc = 'pulse_gauss'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            tau = dout[kfunc]['tau']
            t_down = dout[kfunc]['t_down']
            t_up = dout[kfunc]['t_up']

            # integral
            dout[kfunc]['integ'] = amp/2 * np.sqrt(np.pi) * (t_up + t_down)

            # prepare
            t0 = lamb[0] + lambD * tau

            # position of max
            dout[kfunc]['t0'] = t0
            dout[kfunc]['argmax'] = t0

            # value at max
            dout[kfunc]['max'] = amp

        # ------------------
        # sum all lognorm

        kfunc = 'lognorm'
        if dind.get(kfunc) is not None:

            amp = dout[kfunc]['amp']
            tau = dout[kfunc]['tau']
            sigma = dout[kfunc]['sigma']
            mu = dout[kfunc]['mu']

            # integral
            dout[kfunc]['integ'] = np.full(mu.shape, np.nan)

            # prepare
            t0 = lamb[0] + lambD * tau

            # position of max
            dout[kfunc]['t0'] = t0
            dout[kfunc]['argmax'] = t0 + np.exp(mu - sigma**2)

            # value at max
            dout[kfunc]['max'] = amp * np.exp(0.5*sigma**2 - mu)

        return dout

    return func


# #####################################################################
# #####################################################################
#                   Mutualizing
# #####################################################################


def _get_var_extract_func(x_full, dind, axis, sli):
    def func(kfunc, kvar, dind=dind, axis=axis, sli=sli, x_full=x_full):
        sli[axis] = dind[kfunc][kvar]['ind']
        return x_full[tuple(sli)]
    return func

def _get_line_argmax(vccos, param_val, dind, kfunc, shape, axis):

    # extract lamb0
    lamb0 = param_val[dind[kfunc]['lamb0']]

    # reshape lamb0
    reshape = [1 for ii in shape]
    reshape[axis] = lamb0.size
    lamb0 = lamb0.reshape(tuple(reshape))

    return lamb0 * (1 + vccos)

def _get_Ti(sigma, param_val, dind, kfunc, shape, axis):

    # extract lamb0, mz
    lamb0 = param_val[dind[kfunc]['lamb0']]
    mz = param_val[dind[kfunc]['mz']]

    # reshape lamb0 and mz
    reshape = [1 for ii in shape]
    reshape[axis] = lamb0.size
    reshape = tuple(reshape)
    lamb0 = lamb0.reshape(reshape)
    mz = mz.reshape(reshape)

    return (sigma / lamb0)**2 * mz * scpct.c**2 / scpct.e