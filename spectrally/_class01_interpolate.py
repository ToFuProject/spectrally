# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:33:27 2024

@author: dvezinet
"""


import datetime as dtm
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
    # uncertainty propagation
    uncertainty_method=None,
    # others
    returnas=None,
    store=None,
    store_key=None,
    # timing
    timing=None,
):

    # -----------------
    # check
    # -----------------

    (
        key_model, ref_nx, ref_nf,
        key_data, key_cov,
        key_lamb, lamb, ref_lamb,
        binning, details,
        uncertainty_method,
        returnas, store, store_key,
        timing,
    ) = _check(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
        # others
        binning=binning,
        details=details,
        # uncertainty propagation
        uncertainty_method=uncertainty_method,
        # others
        returnas=returnas,
        store=store,
        store_key=store_key,
        # timing
        timing=timing,
    )

    if timing is True:
        fname = 'interpolate_spectral_model()'
        t0 = dtm.datetime.now()  # DB

    # -----------------
    # optionnal binning
    # -----------------

    dbinning = coll.get_spectral_fit_binning_dict(
        binning=binning,
        lamb=lamb,
        iok=None,
        axis=None,
    )

    if timing is True:
        t1 = dtm.datetime.now()  # DB
        print(f'... timing {fname}: binning dict {(t1-t0).total_seconds()} s')

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
    # coraviance

    if key_cov is not None:
        cov_in = coll.ddata[key_cov]['data']
        nx = coll.dref[ref_nx]['size']

    if timing is True:
        t2 = dtm.datetime.now()  # DB
        print(f'... timing {fname}: prepare data {(t2-t1).total_seconds()} s')

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

    if timing is True:
        t3 = dtm.datetime.now()  # DB
        print(f'... timing {fname}: initialize {(t3-t2).total_seconds()} s')

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

    if timing is True:
        t4 = dtm.datetime.now()  # DB
        print(f'... timing {fname}: get func {(t4-t3).total_seconds()} s')

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

    if timing is True:
        t5 = dtm.datetime.now()  # DB
        print(f'... timing {fname}: prepare slices {(t5-t4).total_seconds()} s')

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

    if timing is True:
        t6 = dtm.datetime.now()  # DB
        print(f'... timing {fname}: data_out {(t6-t5).total_seconds()} s')

    # -----------------------------
    # loop on cov to get error bar
    # -----------------------------

    if key_cov is not None:

        data_min, data_max = _uncertainty_propagation(
            # resources
            coll=coll,
            key_model=key_model,
            lamb=lamb,
            data_in=data_in,
            data_out=data_out,
            nx=nx,
            lind=lind,
            func=func,
            # method
            method=uncertainty_method,
            # slicing
            ind0=ind0,
            ind0_out=ind0_out,
            sli_in=sli_in,
            sli_out=sli_out,
            # uncertainty input
            cov=cov_in,
            # binning
            dbinning=dbinning,
            bin_ind=bin_ind,
            bin_dlamb=bin_dlamb,
            # timing
            timing=timing,
        )

        if timing is True:
            t7 = dtm.datetime.now()  # DB
            print(f'... timing {fname}: uncertainty {(t7-t6).total_seconds()} s')

    else:
        data_min = None
        data_max = None

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
    # uncertainty propagation
    uncertainty_method=None,
    # others
    returnas=None,
    store=None,
    store_key=None,
    # timing
    timing=None,
):

    # ---------------------
    # key_model, key_data
    # ---------------------

    key_model, key_data, key_cov, lamb, binning = _check_keys(
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
    # uncertainty_method
    # -----------------

    uncertainty_method = ds._generic_check._check_var(
        uncertainty_method, 'uncertainty_method',
        types=str,
        default='old',
        allowed=['new', 'old'],
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

    # -----------------
    # timing
    # -----------------

    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
    )

    return (
        key_model, ref_nx, ref_nf,
        key_data, key_cov,
        key_lamb, lamb, ref_lamb,
        binning, details,
        uncertainty_method,
        returnas, store, store_key,
        timing,
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

    key_cov = None
    if key in lokf:
        key_fit = key
        key_model = coll.dobj[wsf][key_fit]['key_model']

        if key_data is None:
            key_data = coll.dobj[wsf][key_fit]['key_sol']
            key_cov = coll.dobj[wsf][key_fit]['key_cov']

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

    return key_model, key_data, key_cov, lamb, binning


def _err_lamb(lamb):
    msg = (
        "Arg 'lamb' nust be either:\n"
        "\t- 1d np.ndarray with finite values only\n"
        "\t- str: a key to an existing 1d vector with finite values only\n"
        f"Provided:\n{lamb}"
    )
    raise Exception(msg)


#############################################
#############################################
#       uncertainty propagation
#############################################


def _uncertainty_propagation(
    # resources
    coll=None,
    key_model=None,
    lamb=None,
    data_in=None,
    data_out=None,
    nx=None,
    lind=None,
    func=None,
    # method
    method=None,
    # slicing
    ind0=None,
    ind0_out=None,
    sli_in=None,
    sli_out=None,
    # uncertainty input
    cov=None,
    # binning
    dbinning=None,
    bin_ind=None,
    bin_dlamb=None,
    # timing
    timing=None,
):

    # ------------------
    # prepare
    # ------------------

    if timing is True:
        i0 = 0  # DB
        dt0 = 0
        dt1 = 0

    # prepare output
    data_min = np.full(data_out.shape, np.inf)
    data_max = np.full(data_out.shape, -np.inf)

    # sli_cov
    sli_cov = np.r_[
        [0 for ii in range(data_in.ndim-1)] + [slice(None), slice(None)]
    ]
    ind_cov = np.arange(data_in.ndim-1)

    # -------------------------------------------------------
    # Proper uncertainty propagation using covariance matrix
    # -------------------------------------------------------

    # ref: https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    if method == 'new':

        # ------------------------
        # get jacobian at solution

        # function
        func_jac = coll.get_spectral_fit_func(
            key=key_model,
            func='jac',
        )['jac']

        # ------------------------
        # loop on all spectra

        for ind in itt.product(*lind):

            # update slices
            if ind0 is not None:
                sli_in[ind0] = ind
                sli_out[ind0_out] = ind
                sli_cov[ind_cov] = ind

            datain = data_in[tuple(sli_in)]

            # values
            jac = func_jac(
                x_free=datain,
                lamb=lamb if dbinning is False else dbinning['lamb'],
                bin_ind=bin_ind,
                bin_dlamb=bin_dlamb,
            )

            # derive std of sum
            std_sum = np.sqrt((jac.T).dot(cov[tuple(sli_cov)].dot(jac)))

            # ------------------------
            # derive min, max

            sliout = tuple(sli_out)
            data_min[sliout] = data_out[sliout] - std_sum
            data_max[sliout] = data_out[sliout] + std_sum

    # ------------------------------------------------------
    # brute force uncertainty propagation using min / max of all variables
    # ------------------------------------------------------

    else:

        # prepare
        inc = np.r_[-1, 0, 1]
        lind_std = [inc for ii in range(nx)]

        # std_in
        sli_in

        # -------------------
        # loop on spectra

        for ind in itt.product(*lind):

            # update slices
            if ind0 is not None:
                sli_in[ind0] = ind
                sli_out[ind0_out] = ind
                sli_cov[ind_cov] = ind

            # std
            std = np.sqrt(np.diag(cov[tuple(sli_cov)]))

            # loop on [-1, 0, 1] combinations
            for stdi in itt.product(*lind_std):

                # data = data_in + std * (-1, 0, 1)
                datain = (
                    data_in[tuple(sli_in)]
                    + np.r_[stdi] * std
                )

                if timing is True:
                    t00 = dtm.datetime.now()

                # call func
                datai = func(
                    x_free=datain,
                    lamb=lamb if dbinning is False else dbinning['lamb'],
                    bin_ind=bin_ind,
                    bin_dlamb=bin_dlamb,
                )

                if timing is True:
                    t01 = dtm.datetime.now()
                    dt0 += (t01 - t00).total_seconds()

                # update min, max
                data_min[tuple(sli_out)] = np.minimum(
                    data_min[tuple(sli_out)], datai,
                )
                data_max[tuple(sli_out)] = np.maximum(
                    data_max[tuple(sli_out)],
                    datai,
                )

                if timing is True:
                    t02 = dtm.datetime.now()
                    dt1 += (t02 - t01).total_seconds()
                    i0 += 1

        if timing is True:
            msg = (
                f"\t... timing std: func {dt0} s\n"
                f"\t... timing std: max/min {dt1} s\n"
                f"\t... timing std: ntot = {i0}\n"
            )
            print(msg)

    return data_min, data_max