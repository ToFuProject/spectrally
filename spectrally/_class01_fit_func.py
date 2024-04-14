# -*- coding: utf-8 -*-
"""
created on tue feb 20 14:44:51 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import datastock as ds


# local


#############################################
#############################################
#       defaults
#############################################


_DFUNC = {
    '1d': {
        'cost': None,
    },
}


#############################################
#############################################
#       get func 1d
#############################################


def fit1d(
    coll=None,
    key=None,
    func=None,
):

    # ----------------
    # check inputs
    # ----------------

    key, key_bs, func = _check(
        coll=coll,
        key=key,
        func=func,
    )

    # ----------------
    # get func
    # ----------------

    if key_bs is None:
        dout = {
            k0: _DFUNC['1d'][k0](
                coll=coll,
                key=key,
            )
            for k0 in func
        }

    else:
        dout = {
            k0: _DFUNC['bs'][k0](
                coll=coll,
                key=key,
            )
            for k0 in func
        }

    return dout


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key=None,
    func=None,
):

    # -------------
    # key
    # -------------

    wsf = coll._which_fit
    lok = list(coll.dobj.get(wsf, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # key_bs
    key_bs = coll.dobj[wsf][key]['key_bs']

    # -------------
    # func
    # -------------

    if isinstance(func, str):
        func = [func]
    lok = ['cost', 'details', 'jac']
    func = ds._generic_check._check_var_iter(
        func, 'func',
        types=list,
        types_iter=str,
        allowed=lok,
    )

    return key, key_bs, func


#############################################
#############################################
#       model 1d
#############################################


def _get_func_cost_1d():

    def func(
        x,
        xscale=xscale,
        indx=indx,
        lambrel=lambrel,
        lambnorm=lambnorm,
        ibckax=ibckax,
        ibckrx=ibckrx,
        ial=ial,
        iwl=iwl,
        ishl=ishl,
        idratiox=idratiox,
        idshx=idshx,
        scales=None,
        coefsal=coefsal[None, :],
        coefswl=coefswl[None, :],
        coefssl=coefssl[None, :],
        offsetal=offsetal[None, :],
        offsetwl=offsetwl[None, :],
        offsetsl=offsetsl[None, :],
        double=dinput['double'],
        indok=None,
        const=None,
        data=np.zeros((x.size,)),
    ):

        if indok is None:
            indok = np.ones(lambrel.shape, dtype=bool)

        # xscale = x*scales   !!! scales ??? !!! TBC
        xscale[indx] = x*scales[indx]
        xscale[~indx] = const

        # make sure iwl is 2D to get all lines at once
        amp = xscale[ial] * coefsal + offsetal
        inv_2wi2 = 1./(2.*(xscale[iwl] * coefswl + offsetwl))
        shift = xscale[ishl] * coefssl + offsetsl

        # ----------
        # sum

        y = (
            np.nansum(
                amp * np.exp(-(lambnorm[indok, :]-(1 + shift))**2 * inv_2wi2),
                axis=1,
            )
            + xscale[ibckax] * np.exp(xscale[ibckrx] * lambrel[indok])
        )

        # ----------
        # return

        return y - data[iok]

    # -----------
    # return func

    return func
