# #!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
created on tue feb 20 14:44:51 2024

@author: dvezinet
"""


import datastock as ds


# local
from . import _class01_fit_func_1d as _1d


#############################################
#############################################
#       defaults
#############################################


_DFUNC = {
    '1d': {
        'cost': _1d._get_func_val,
        'cost': _1d._get_func_cost,
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

    key, key_bs, func = _check(
        coll=coll,
        key=key,
        func=func,
    )

    xall= coll.get_spectral_model_variables(key_model, all_free_tied='all')


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

    # -------------
    # func
    # -------------

    if func is None:
        func = ['val']
    if isinstance(func, str):
        func = [func]
    lok = ['val', 'cost', 'details', 'jac']
    func = ds._generic_check._check_var_iter(
        func, 'func',
        types=list,
        types_iter=str,
        allowed=lok,
    )

    return key, key_bs, func
