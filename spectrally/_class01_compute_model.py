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
    # others
    returnas=None,
    store=None,
    store_key=None,
):

    # -----------------
    # check
    # -----------------

    (
        key_model, ref_nx, ref_nf,
        key_data,
        key_lamb, lamb, ref_lamb,
        details,
        returnas, store, store_key,
    ) = _check(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
        # others
        details=details,
        # others
        returnas=returnas,
        store=store,
        store_key=store_key,
    )

    # ----------------
    # prepare
    # ----------------

    # ----------
    # data_in

    data_in = coll.ddata[key_data]['data']
    ref_in = coll.ddata[key_data]['ref']
    ndim_in = data_in.ndim

    iref_nx = ref_in.index(ref_nx)
    if details is True:
        iref_nf = 0
        iref_nx_out = iref_nx + 1
    else:
        iref_nx_out = iref_nx

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
    shape_in = data_in.shape
    shape_out = list(shape_in)
    ref_out = list(ref_in)

    shape_out[iref_nx] = lamb.size
    ref_out[iref_nx] = ref_lamb
    if details is True:
        shape_out.insert(0, coll.dref[ref_nf]['size'])
        ref_out.insert(0, ref_nf)

    # data_out
    data_out = np.full(tuple(shape_out), np.nan)

    # ----------------
    # get func
    # ----------------

    if details is True:
        func = coll.get_spectral_fit_func(
            key=key_model,
            func='details',
        )['details']

    else:
        func = coll.get_spectral_fit_func(
            key=key_model,
            func='sum',
        )['sum']

    # ----------------
    # compute
    # ----------------

    # --------------
    # prepare slices

    if ndim_in > 1:

        # slices
        sli_in = list(shape_in)
        sli_out = list(shape_out)

        sli_in[iref_nx] = slice(None)
        sli_out[iref_nx_out] = slice(None)
        if details is True:
            sli_out[0] = slice(None)

        # as array
        sli_in = np.array(sli_in)
        sli_out = np.array(sli_out)

        # indices to change
        ind0 = np.array(
            [ii for ii in range(len(shape_in)) if ii != iref_nx],
            dtype=int,
        )

        # adjust for details
        if details is True:
            ind0_out = ind0 + 1
        else:
            ind0_out = ind0

    else:
        ind0 = None

    # -------
    # loop

    if ind0 is None:

        # call func
        data_out = func(
            x_free=data_in,
            lamb=lamb,
        )

    else:

        for ind in itt.product(*lind):

            # update slices
            sli_in[ind0] = ind
            sli_out[ind0_out] = ind

            # call func
            data_out[tuple(sli_out)] = func(
                x_free=data_in[tuple(sli_in)],
                lamb=lamb,
            )

    # --------------
    # return
    # --------------

    dout = {
        'key': store_key,
        'data': data_out,
        'ref': tuple(ref_out),
        'dim': coll.ddata[key_data]['dim'],
        'quant': coll.ddata[key_data]['quant'],
        'units': coll.ddata[key_data]['units'],
    }

    # --------------
    # store
    # --------------

    if store is True:

        coll.add_data(
            **dout,
        )

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
    # others
    returnas=None,
    store=None,
    store_key=None,
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
            and np.all(np.isfinite(lamb))
        )
        if not c0:
            _err_lamb(lamb)

        key_lamb = None
        ref_lamb = None

    elif isinstance(lamb, str):
        c0 = (
            lamb in coll.ddata.keys()
            and coll.ddata[lamb]['data'].ndim == 1
            and np.all(np.isfinite(coll.ddata[lamb]['data']))
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

    # -----------------
    # store
    # -----------------

    lok = [False]
    if key_lamb is not None:
        lok.append(True)
    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
        allowed=lok,
    )

    # -----------------
    # returnas
    # -----------------

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=(not store),
        allowed=[False, True, dict],
    )

    # -----------------
    # store_key
    # -----------------

    if store is True:
        lout = list(coll.ddata.keys())
        store_key = ds._generic_check._check_var(
            store_key, 'store_key',
            types=str,
            excluded=lout,
        )
    else:
        store_key = None

    return (
        key_model, ref_nx, ref_nf,
        key_data,
        key_lamb, lamb, ref_lamb,
        details,
        returnas, store, store_key,
    )


def _err_lamb(lamb):
    msg = (
        "Arg lamb nust be either:\n"
        "\t- 1d np.ndarray with finite values only\n"
        "\t- str: a key to an existing 1d vector with finite values only\n"
        "Provided:\n{lamb}"
    )
    raise Exception(msg)
