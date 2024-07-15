# #!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
created on tue feb 20 14:44:51 2024

@author: dvezinet
"""


import numpy as np
import datastock as ds


# local
from . import _class01_fit_func_1d as _1d


#############################################
#############################################
#       defaults
#############################################


_DFUNC = {
    '1d': {
        'sum': _1d._get_func_sum,
        'cost': _1d._get_func_cost,
        'details': _1d._get_func_details,
        'jac': _1d._get_func_jacob,
    },
}


#############################################
#############################################
#       main
#############################################


def main(
    coll=None,
    key=None,
    func=None,
):

    # ----------------
    # check inputs
    # ----------------

    key, key_bs, key_model, func = _check(
        coll=coll,
        key=key,
        func=func,
    )

    # ----------------
    # prepare
    # ----------------

    # n_all
    n_all = len(coll.get_spectral_model_variables(
        key=key_model,
        returnas='all',
        concatenate=True,
    )['all'])

    # param_val
    param_val = coll.get_spectral_model_variables(
        key_model,
        returnas='param_value',
        concatenate=True,
    )['param_value']

    # dind
    dind = coll.get_spectral_model_variables_dind(key_model)

    # ----------------
    # dconstraints
    # ----------------

    # dconstraints
    wsm = coll._which_model
    dconstraints = coll.dobj[wsm][key_model]['dconstraints']

    # coefs
    c0 = dconstraints['c0']
    c1 = dconstraints['c1']
    c2 = dconstraints['c2']

    # check is constraints
    no_constraints = (
        np.allclose(c0, 0)
        and c1.shape == (n_all, n_all)
        and np.allclose(c1, np.eye(n_all))
        and c2.shape == (n_all, n_all)
        and np.allclose(c2, np.eye(n_all))
    )
    if no_constraints:
        c0, c1, c2 = None, None, None

    # ----------------
    # get func
    # ----------------

    if key_bs is None:
        dout = {
            k0: _DFUNC['1d'][k0](
                c0=c0,
                c1=c1,
                c2=c2,
                dind=dind,
                param_val=param_val,
            )
            for k0 in func
        }

    else:
        dout = {
            k0: _DFUNC['2d'][k0](
                coll=coll,
                key=key,
            )
            for k0 in func
        }

    # ----------------
    # return
    # ----------------

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

    wsm = coll._which_model
    wsf = coll._which_fit
    lok_m = list(coll.dobj.get(wsm, {}).keys())
    lok_1d = [
        k0 for k0, v0 in coll.dobj.get(wsf, {}).items()
        if v0['key_bs'] is None
    ]
    lok_2d = [
        k0 for k0, v0 in coll.dobj.get(wsf, {}).items()
        if v0['key_bs'] is not None
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok_m + lok_1d + lok_2d,
    )

    # key_bs
    key_bs = None
    if key in lok_2d:
        key_bs = coll.dobj[wsf][key]['key_bs']

    # key_model
    if key in lok_m:
        key_model = key
    else:
        key_model = coll.dobj[wsf][key]['key_model']

    # -------------
    # func
    # -------------

    lok = list(_DFUNC['1d'].keys())
    if func is None:
        func = lok[0]
    if isinstance(func, str):
        func = [func]

    func = ds._generic_check._check_var_iter(
        func, 'func',
        types=list,
        types_iter=str,
        allowed=lok,
    )

    return key, key_bs, key_model, func
