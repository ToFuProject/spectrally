# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:33:27 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import datastock as ds


#############################################
#############################################
#       main
#############################################


def main(
    coll=None,
    key_model=None,
    key_data=None,
    lamb=None,
    # others
    details=None,
):

    # -----------------
    # check
    # -----------------

    (
        key_model, ref_nx, ref_nf,
        key_data,
        key_lamb, lamb, ref_lamb,
        details,
    ) = _check(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
        # others
        details=details,
    )

    # ----------------
    # prepare
    # ----------------

    # ----------
    # data_in

    data_in = coll.ddata[key_data]['data']
    ref_in = coll.ddata[key_data]['ref']

    iref_nx = ref_in.index(ref_nx)
    if details is True:
        iref_nf = -1

    # -----------------------
    # prepare loop on indices

    key_bs = None
    if key_bs is None:
        lind = [
            range(ss) for ii, ss in enumerate(data_in.shape)
            if ii != iref_nx
        ]
    else:
        raise NotImplementedError()

    # -------------
    # initialize

    # shape_out, ref_out
    shape_out = list(data_in.shape)
    ref_out = list(ref_in)

    shape_out[iref_nx] = lamb.size
    ref_out[iref_nx] = ref_lamb
    if details is True:
        shape_out.append(coll.dref[ref_nf]['size'])
        ref_out.append(ref_nf)

    # data_out
    data_out = np.full(tuple(shape_out), np.nan)

    # ----------------
    # get func
    # ----------------

    if details is True:
        func = coll.get_spectral_fit_func(
            key=key_model,
            func='details',
        )

    else:
        func = coll.get_spectral_fit_func(
            key=key_model,
            func='val',
        )

    # ----------------
    # compute
    # ----------------

    # --------------
    # prepare slices

    # slices
    if details is True:
        sli_in = tuple([])
        sli_out = tuple([])
    else:
        sli_in = tuple([])
        sli_out = tuple([])

    # -------
    # loop

    for ind in itt.product(*lind):

        # update slices
        sli_in[] = ind
        sli_out[] = ind

        # call func
        data_out[tuple(sli_out)] = func(
            x_free=data_in[tuple(sli_in)],
            lamb=lamb,
        )

    # --------------
    # return
    # --------------

    dout = {
        'data': data_out,
        'ref': tuple(ref_out),
        'units': None,
    }

    return dout


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key_model=None,
    key_data=None,
    lamb=None,
    # others
    details=None,
):

    # ----------
    # key_model
    # ----------

    wsm = coll._which_model
    key_model = ds._generic_check._check_var(
        key_model, 'key_model',
        types=str,
        allowed=list(coll.dobj.get(wsm, {}).keys()),
    )

    # derive ref_model
    ref_nf = coll.dobj[wsm][key_model]['ref_nf']
    ref_nx = coll.dobj[wsm][key_model]['ref_nx']

    # ----------
    # key_data
    # ----------

    # list of acceptable values
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if ref_nx in v0['ref']
    ]

    # check
    key_data = ds._generic_check._check_var(
        key_data, 'key_data',
        types=str,
        allowed=lok,
    )

    # -----------------
    # lamb
    # -----------------

    if isinstance(lamb, np.ndarray):
        c0 = (
            lamb.ndim == 1
            and np.all(np.isifinite(lamb))
        )
        if not c0:
            _err_lamb(lamb)

        key_lamb = None
        ref_lamb = None

    elif isinstance(lamb, str):
        c0 = (
            lamb in coll.ddata.keys()
            and coll.ddata[lamb]['data'].ndim == 1
            and np.all(np.isifinite(coll.ddata[lamb]['data']))
        )
        if not c0:
            _err_lamb(lamb)

        key_lamb = lamb
        lamb = coll.ddata[key_lamb]['data']
        ref_lamb = coll.ddata[key_lamb]['ref'][0]

    else:
        _err_lamb(lamb)

    # -----------------
    # details
    # -----------------

    details = ds._generic_check._check_var(
        details, 'details',
        types=bool,
        default=False,
    )

    return (
        key_model, ref_nx, ref_nf,
        key_data,
        key_lamb, lamb, ref_lamb,
        details,
    )


def _err_lamb(lamb):
    msg = (
        "Arg lamb nust be either:\n"
        "\t- 1d np.ndarray with finite values only\n"
        "\t- str: a key to an existing 1d vector with finite values only\n"
        "Provided:\n{lamb}"
    )
    raise Exception(msg)