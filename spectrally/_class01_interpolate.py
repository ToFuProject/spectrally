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
    # options
    details=None,
    binning=None,
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
        key_data, key_std,
        key_lamb, lamb, ref_lamb,
        binning, details,
        returnas, store, store_key,
    ) = _check(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
        # others
        binning=binning,
        details=details,
        # others
        returnas=returnas,
        store=store,
        store_key=store_key,
    )

    # -----------------
    # optionnal binning
    # -----------------

    dbinning = coll.get_spectral_fit_binning_dict(
        binning=binning,
        lamb=lamb,
        iok=None,
        axis=None,
    )

    # ----------------
    # prepare
    # ----------------

    # ----------
    # data_in

    data_in = coll.ddata[key_data]['data']
    ref_in = coll.ddata[key_data]['ref']
    ndim_in = data_in.ndim

    # ----------
    # details

    iref_nx = ref_in.index(ref_nx)
    if details is True:
        iref_nx_out = iref_nx + 1
    else:
        iref_nx_out = iref_nx

    # ----------
    # std_in

    if key_std is not None:
        std_in = coll.ddata[key_std]['data']
        nx = coll.dref[ref_nx]['size']

    # -----------------------
    # prepare loop on indices
    # -----------------------

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
    # -------------

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

    # --------------
    # prepare slices
    # --------------

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
    if ndim_in > 1:
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

    # -------------------------
    # loop to compute data_out
    # -------------------------

    if dbinning is False:
        bin_ind = False
        bin_dlamb = None
    else:
        bin_ind = dbinning['ind']
        bin_dlamb = dbinning['dlamb']

    for ind in itt.product(*lind):

        # update slices
        if ind0 is not None:
            sli_in[ind0] = ind
            sli_out[ind0_out] = ind

        # call func
        data_out[tuple(sli_out)] = func(
            x_free=data_in[tuple(sli_in)],
            lamb=lamb if dbinning is False else dbinning['lamb'],
            bin_ind=bin_ind,
            bin_dlamb=bin_dlamb,
        )

    # -----------------------------
    # loop on std to get error bar
    # -----------------------------

    data_min = None
    data_max = None
    if key_std is not None:

        data_min = np.full(data_out.shape, np.inf)
        data_max = np.full(data_out.shape, -np.inf)
        inc = np.r_[-1, 0, 1]
        lind_std = [inc for ii in range(nx)]

        for ind in itt.product(*lind):

            # update slices
            if ind0 is not None:
                sli_in[ind0] = ind
                sli_out[ind0_out] = ind

            datain = data_in[tuple(sli_in)]

            for stdi in itt.product(*lind_std):

                # data = data_in + std * (-1, 0, 1)
                datain = (
                    data_in[tuple(sli_in)]
                    + np.r_[stdi] * std_in[tuple(sli_in)]
                )

                # call func
                datai = func(
                    x_free=datain,
                    lamb=lamb if dbinning is False else dbinning['lamb'],
                    bin_ind=bin_ind,
                    bin_dlamb=bin_dlamb,
                )

                # update min, max
                data_min[tuple(sli_out)] = np.minimum(
                    data_min[tuple(sli_out)], datai,
                )
                data_max[tuple(sli_out)] = np.maximum(
                    data_max[tuple(sli_out)],
                    datai,
                )

    # --------------
    # return
    # --------------

    dout = {
        'key': store_key,
        'key_data': key_data,
        'key_model': key_model,
        'key_lamb': key_lamb,
        'lamb': lamb,
        'details': details,
        'data': data_out,
        'data_min': data_min,
        'data_max': data_max,
        'ref': tuple(ref_out),
        'dim': coll.ddata[key_data]['dim'],
        'quant': coll.ddata[key_data]['quant'],
        'units': coll.ddata[key_data]['units'],
    }

    # --------------
    # store
    # --------------

    if store is True:

        lout = [
            'key_data', 'key_model', 'key_lamb',
            'lamb', 'details',
            'data_min', 'data_max',
        ]
        coll.add_data(
            **{
                k0: v0 for k0, v0 in dout.items()
                if k0 not in lout
            },
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
    binning=None,
    details=None,
    # others
    returnas=None,
    store=None,
    store_key=None,
):

    # ---------------------
    # key_model, key_data
    # ---------------------

    key_model, key_data, key_std, lamb, binning = _check_keys(
        coll=coll,
        key=key_model,
        key_data=key_data,
        lamb=lamb,
        binning=binning,
    )

    # derive ref_model
    wsm = coll._which_model
    ref_nf = coll.dobj[wsm][key_model]['ref_nf']
    ref_nx = coll.dobj[wsm][key_model]['ref_nx']

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
        key_data, key_std,
        key_lamb, lamb, ref_lamb,
        binning, details,
        returnas, store, store_key,
    )


def _check_keys(coll=None, key=None, key_data=None, lamb=None, binning=None):

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

    # ---------------
    # if key_fit
    # ---------------

    key_std = None
    if key in lokf:
        key_fit = key
        key_model = coll.dobj[wsf][key_fit]['key_model']

        if key_data is None:
            key_data = coll.dobj[wsf][key_fit]['key_sol']
            key_std = coll.dobj[wsf][key_fit]['key_std']

        if lamb is None:
            lamb = coll.dobj[wsf][key_fit]['key_lamb']

        binning = coll.dobj[wsf][key_fit]['dinternal']['binning']

    else:
        key_model = key

    # ----------
    # key_data
    # ----------

    # derive ref_model
    ref_nx = coll.dobj[wsm][key_model]['ref_nx']

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

    return key_model, key_data, key_std, lamb, binning


def _err_lamb(lamb):
    msg = (
        "Arg 'lamb' nust be either:\n"
        "\t- 1d np.ndarray with finite values only\n"
        "\t- str: a key to an existing 1d vector with finite values only\n"
        f"Provided:\n{lamb}"
    )
    raise Exception(msg)