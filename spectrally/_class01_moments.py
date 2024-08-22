# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:57:00 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import scipy.constants as scpct
import astropy.units as asunits
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
    # return
    returnas=None,
):

    # ------------
    # check inputs
    # ------------

    # key_model vs key_fit
    returnas = _check(
        returnas=returnas,
    )

    # all other variables
    (
        key_model, ref_nx, ref_nf,
        key_data, key_std,
        key_lamb, lamb, ref_lamb,
        binning, details,
        _, store, store_key,
    ) = _interpolate._check(
        coll=coll,
        key_model=key,
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
    axis = coll.ddata[key_data]['ref'].index(ref_nx)

    # ------------
    # get func
    # ------------

    func = _get_func_moments(
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        param_val=param_val,
        axis=axis,
    )

    # -------------------
    # extract / compute
    # -------------------

    dout = func(
        x_free=coll.ddata[key_data]['data'],
        x_std=None if key_std is None else coll.ddata[key_std]['data'],
        lamb=lamb,
    )

    # -------------
    # format output
    # -------------

    if returnas == 'dict_varnames':

        ref = tuple([rr for rr in coll.ddata[key_data]['ref'] if rr != ref_nx])

        dout = _format(
            coll=coll,
            key_model=key_model,
            key_data=key_data,
            key_lamb=key_lamb,
            din=dout,
            dind=dind,
            axis=axis,
            ref=ref,
            binning=binning,
        )

    return dout


#############################################
#############################################
#       check
#############################################


def _check(
    returnas=None,
):

    # -------------
    # returnas
    # -------------

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='dict_varnames',
        allowed=['dict_ftypes', 'dict_varnames'],
    )

    return returnas


def _check_mz(
    dmz=None,
    dind=None,
):

    # -----------
    # trivial
    # ----------

    # add mz if user-provided
    if dmz is not None:

        for ktype in ['gauss', 'pvoigt', 'voigt']:

            if dind.get(ktype) is None:
                continue

            dind[ktype]['mz']

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

    # --------------
    # prepare
    # --------------

    def func(
        x_free=None,
        x_std=None,
        lamb=None,
        param_val=param_val,
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
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
            if x_std is not None:
                xf_min = x_full - x_std
                xf_max = x_full + x_std

        else:

            if x_free.ndim > 1:
                shape = list(x_free.shape)
                shape[axis] = c0.size
                x_full = np.full(shape, np.nan)
                if x_std is not None:
                    xf_min = np.full(shape, np.nan)
                    xf_max = np.full(shape, np.nan)

                sli = list(shape)
                sli[axis] = slice(None)
                sli = np.array(sli)
                ich = np.array([ii for ii in range(len(shape)) if ii != axis])
                linds = [range(shape[ii]) for ii in ich]

                for ii, ind in enumerate(itt.product(*linds)):
                    sli[ich] = ind
                    slii = tuple(sli)
                    x_full[slii] = (
                        c2.dot(x_free[slii]**2) + c1.dot(x_free[slii]) + c0
                    )

                    if x_std is not None:
                        xf_min[slii] = (
                            c2.dot((x_free[slii] - x_std[slii])**2)
                            + c1.dot(x_free[slii] - x_std[slii]) + c0
                        )
                        xf_max[slii] = (
                            c2.dot((x_free[slii] + x_std[slii])**2)
                            + c1.dot(x_free[slii] + x_std[slii]) + c0
                        )

            else:
                x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0
                if x_std is not None:
                    xf_min = c2.dot((x_free - x_std)**2) + c1.dot(x_free - x_std) + c0
                    xf_max = c2.dot((x_free + x_std)**2) + c1.dot(x_free + x_std) + c0

        sli = [None if ii == axis else slice(None) for ii in range(x_free.ndim)]
        extract = _get_var_extract_func(dind, axis, sli)

        # ---------------------
        # extract all variables

        for ktype, v0 in _model_dict._DMODEL.items():
            if dind.get(ktype) is not None:
                for kvar in v0['var']:
                    dout[ktype][kvar] = extract(ktype, kvar, x_full)
                    if x_std is not None:
                        dout[ktype][f"{kvar}_min"] = extract(ktype, kvar, xf_min)
                        dout[ktype][f"{kvar}_max"] = extract(ktype, kvar, xf_max)

        # ------------------
        # sum all poly

        ktype = 'poly'
        if dind.get(ktype) is not None:

            a0 = dout[ktype]['a0']
            a1 = dout[ktype]['a1']
            a2 = dout[ktype]['a2']

            # integral
            if lamb is not None:
                dout[ktype]['integ'] = (
                    a0 * (lamb[-1] - lamb[0])
                    + a1 * (lamb[-1]**2 - lamb[0]**2)/2
                    + a2 * (lamb[-1]**3 - lamb[0]**3)/3
                )

            # argmax, max
            dout[ktype]['argmax'] = np.full(a0.shape, np.nan)
            dout[ktype]['max'] = np.full(a1.shape, np.nan)
            iok = a2 != 0
            dout[ktype]['argmax'][iok] = lambm - lambD * a1[iok]/(2*a2[iok])
            dout[ktype]['max'][iok] = a0[iok] - a1[iok]**2 / (4*a2[iok])

        # --------------------
        # sum all exponentials

        ktype = 'exp_lamb'
        if dind.get(ktype) is not None:

            # physics
            rate = dout[ktype]['rate']
            dout[ktype]['Te'] = (scpct.h * scpct.c / rate) / scpct.e

            # integral
            if lamb is not None:
                amp = dout[ktype]['rate']
                dout[ktype]['integ'] = (
                    (amp / rate)
                    * (np.exp(lamb[-1] * rate) - np.exp(lamb[0] * rate))
                )

        # -----------------
        # sum all gaussians

        ktype = 'gauss'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            sigma = dout[ktype]['sigma']
            vccos = dout[ktype]['vccos']

            # argmax
            dout[ktype]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, ktype, amp.shape, axis,
            )

            # integral
            dout[ktype]['integ'] = amp * sigma * np.sqrt(2 * np.pi)

            # physics
            if dind[ktype].get('mz') is not None:
                dout[ktype]['Ti'] = _get_Ti(
                    sigma,
                    param_val,
                    dind,
                    ktype,
                    sigma.shape,
                    axis,
                )

        # -------------------
        # sum all Lorentzians

        ktype = 'lorentz'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            gam = dout[ktype]['gam']
            vccos = dout[ktype]['vccos']

            # argmax
            dout[ktype]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, ktype, amp.shape, axis,
            )

            # integral
            dout[ktype]['integ'] = amp * np.pi * gam

        # --------------------
        # sum all pseudo-voigt

        ktype = 'pvoigt'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            sigma = dout[ktype]['sigma']
            vccos = dout[ktype]['vccos']

            # argmax
            dout[ktype]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, ktype, amp.shape, axis,
            )

            # integral
            dout[ktype]['integ'] = np.full(sigma.shape, np.nan)

            # physics
            if dind[ktype].get('mz') is not None:
                dout[ktype]['Ti'] = _get_Ti(
                    sigma,
                    param_val,
                    dind,
                    ktype,
                    sigma.shape,
                    axis,
                )

        # --------------------
        # sum all voigt

        ktype = 'voigt'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            sigma = dout[ktype]['sigma']
            vccos = dout[ktype]['vccos']

            # argmax
            dout[ktype]['argmax'] = _get_line_argmax(
                vccos, param_val, dind, ktype, amp.shape, axis,
            )

            # integral
            dout[ktype]['integ'] = amp

            # physics
            if dind[ktype].get('mz') is not None:
                dout[ktype]['Ti'] = _get_Ti(
                    sigma,
                    param_val,
                    dind,
                    ktype,
                    sigma.shape,
                    axis,
                )

        # ------------------
        # sum all pulse_exp

        ktype = 'pulse_exp'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            tau = dout[ktype]['tau']
            t_down = dout[ktype]['t_down']
            t_up = dout[ktype]['t_up']

            # integral
            dout[ktype]['integ'] = amp * (t_down - t_up)

            # prepare
            t0 = lamb[0] + lambD * tau
            dtdu = t_down - t_up
            lntdu = np.log(t_down / t_up)

            # position of max
            dout[ktype]['t0'] = t0
            dout[ktype]['argmax'] = t0 + lntdu * t_down*t_up / dtdu

            # value at max
            dout[ktype]['max'] = amp * (
                np.exp(-lntdu * t_up / dtdu)
                - np.exp(-lntdu * t_down / dtdu)
            )

        # ------------------
        # sum all pulse_gauss

        ktype = 'pulse_gauss'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            tau = dout[ktype]['tau']
            t_down = dout[ktype]['t_down']
            t_up = dout[ktype]['t_up']

            # integral
            dout[ktype]['integ'] = amp/2 * np.sqrt(np.pi) * (t_up + t_down)

            # prepare
            t0 = lamb[0] + lambD * tau

            # position of max
            dout[ktype]['t0'] = t0
            dout[ktype]['argmax'] = t0

            # value at max
            dout[ktype]['max'] = amp

        # ------------------
        # sum all lognorm

        ktype = 'lognorm'
        if dind.get(ktype) is not None:

            amp = dout[ktype]['amp']
            tau = dout[ktype]['tau']
            sigma = dout[ktype]['sigma']
            mu = dout[ktype]['mu']

            # integral
            dout[ktype]['integ'] = np.full(mu.shape, np.nan)

            # prepare
            t0 = lamb[0] + lambD * tau

            # position of max
            dout[ktype]['t0'] = t0
            dout[ktype]['argmax'] = t0 + np.exp(mu - sigma**2)

            # value at max
            dout[ktype]['max'] = amp * np.exp(0.5*sigma**2 - mu)

        return dout

    return func


# #####################################################################
# #####################################################################
#                   Mutualizing
# #####################################################################


def _get_var_extract_func(dind, axis, sli):
    def func(ktype, kvar, x_full, dind=dind, axis=axis, sli=sli):
        sli[axis] = dind[ktype][kvar]['ind']
        return x_full[tuple(sli)]
    return func


def _get_line_argmax(vccos, param_val, dind, ktype, shape, axis):

    # lamb0
    lamb0 = param_val[dind[ktype]['lamb0']]

    # reshape lamb0
    reshape = [1 for ii in shape]
    reshape[axis] = lamb0.size
    lamb0 = lamb0.reshape(tuple(reshape))

    return lamb0 * (1 + vccos)


def _get_Ti(sigma, param_val, dind, ktype, shape, axis):

    # lamb0, mz
    lamb0 = param_val[dind[ktype]['lamb0']]
    mz = param_val[dind[ktype]['mz']]

    # reshape lamb0 and mz
    reshape = [1 for ii in shape]
    reshape[axis] = lamb0.size
    reshape = tuple(reshape)
    lamb0 = lamb0.reshape(reshape)
    mz = mz.reshape(reshape)

    return (sigma / lamb0)**2 * mz * scpct.c**2 / scpct.e


#############################################
#############################################
#       format
#############################################


def _format(
    coll=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    din=None,
    dind=None,
    axis=None,
    ref=None,
    binning=None,
):

    # ---------------
    # prepare
    # ---------------

    dout = {}
    wsm = coll._which_model
    sli = [slice(None) for ii in range(len(ref)+1)]

    # -------------
    # units
    # -------------

    # data
    try:
        units_data = asunits.Unit(coll.ddata[key_data]['units'])
    except Exception as err:
        units_data = coll.ddata[key_data]['units']

    # lamb
    try:
        units_lamb = asunits.Unit(coll.ddata[key_lamb]['units'])
    except Exception as err:
        units_lamb = coll.ddata[key_lamb]['units']

    # --------------
    # loop
    # --------------

    for ktype, vtype in din.items():

        lkfunc = [
            kfunc for kfunc in coll.dobj[wsm][key_model]['keys']
            if coll.dobj[wsm][key_model]['dmodel'][kfunc]['type'] == ktype
        ]

        for kvar in vtype.keys():

            for ii, kfunc in enumerate(lkfunc):

                key = f"{kfunc}_{kvar}"
                sli[axis] = ii

                # get units
                units = _units(
                    coll, ktype, kvar,
                    units_data=units_data,
                    units_lamb=units_lamb,
                    binning=binning,
                )

                # store
                dout[key] = {
                    'data': din[ktype][kvar][tuple(sli)],
                    'ref': ref,
                    'units': units,
                }

    return dout


#############################################
#############################################
#       units
#############################################


def _units(coll, ktype, kvar, units_data=None, units_lamb=None, binning=None):

    if kvar.endswith('_min') or kvar.endswith('_max'):
        kvar = kvar[:-4]
    units = None

    # ------------
    # trivial
    # ------------

    if kvar == 'integ':
        if binning is False:
            units = _try_units_mult(units_data, units_lamb)
        else:
            units = units_data

    elif kvar == 'max':
        if binning is False:
            units = units_data
        else:
            units = _try_units_divide(units_data, units_lamb)

    elif kvar == 'argmax':
        units = units_lamb

    # ----------------------------------
    # non-trivial but common to several
    # ----------------------------------

    elif kvar == 'Ti':

        # assuming mz is provided in kg
        units = 'eV'

    # ----------------
    # specific to each
    # ----------------
    else:

        if ktype == 'poly':
            if binning is False:
                units = units_data
            else:
                units = _try_units_divide(units_data, units_lamb)

        elif ktype == 'exp_lamb':

            if kvar == 'amp':
                if binning is False:
                    units = _try_units_mult(units_data, units_lamb)
                else:
                    units = units_data

            elif kvar == 'rate':
                units = units_lamb

            elif kvar == 'Te':
                if units_lamb == 'm':
                    units = 'eV'
                else:
                    units = asunits('eV') * (asunits.Units('m') / units_lamb)

        elif ktype in ['gauss', 'lorentz', 'pvoigt', 'voigt']:

            if kvar == 'amp':
                if binning is False:
                    units = units_data
                else:
                    units = _try_units_divide(units_data, units_lamb)

            elif kvar == 'sigma':
                units = units_lamb

            elif kvar == 'gam':
                units = units_lamb

            elif kvar == 'vccos':
                units = ''

        elif ktype in ['pulse_exp', 'pulse_gauss']:

            if kvar == 'amp':
                if binning is False:
                    units = units_data
                else:
                    units = _try_units_divide(units_data, units_lamb)

            elif kvar == 'tau':
                units = ''

            elif kvar == 't0':
                units = units_lamb

            elif kvar == 't_up':
                units = units_lamb

            elif kvar == 't_down':
                units = units_lamb

        elif ktype == 'lognorm':

            if kvar == 'amp':
                if binning is False:
                    units = _try_units_mult(units_data, units_lamb)
                else:
                    units = units_data

            elif kvar == 'tau':
                units = ''

            elif kvar == 't0':
                units = units_lamb

            elif kvar == 'mu':
                units = 'TBD'

            elif kvar == 'sigma':
                units = ''

    # --------------
    # safety check
    # --------------

    if units is None:
        msg = f"units not reckognized for {ktype} {kvar}"
        raise Exception(msg)

    try:
        return asunits.Unit(units)
    except Exception as err:
        return units


# ###########################
# ###########################
#      try units
# ###########################


def _try_units_mult(u0, u1):
    try:
        units = u0 * u1
    except Exception as err:
        units = f"{u0} x {u1}"
    return units


def _try_units_divide(u0, u1):
    try:
        units = u0 / u1
    except Exception as err:
        units = f"{u0} / {u1}"
    return units