# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import itertools as itt
import datetime as dtm


import numpy as np
import scipy.optimize as scpopt


# local
from . import _class02_x0_scale_bounds as _x0_scale_bounds


#############################################
#############################################
#       DEFAULTS
#############################################


#############################################
#############################################
#       Main
#############################################


def main(
    coll=None,
    # keys
    key=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    # lamb, data, axis
    lamb=None,
    data=None,
    axis=None,
    ravel=None,
    # binning
    binning=None,
    # options
    chain=None,
    dscales=None,
    dbounds_low=None,
    dbounds_up=None,
    dx0=None,
    # solver options
    solver=None,
    dsolver_options=None,
    # options
    strict=None,
    verb=None,
    timing=None,
):
    """ Fit 1d spectra

    """

    # ------------
    # iok
    # ------------

    # iok_all
    iok_all, iok_reduced = get_iok_all(
        coll=coll,
        key=key,
        axis=axis,
    )

    # trivial
    if not np.any(iok_reduced):
        return

    # ravel
    if ravel is True:
        iok_all = iok_all[:, None]
        iok_reduced = np.array([iok_reduced])

    # ------------
    # lamb, data, dind
    # ------------

    dind = coll.get_spectral_model_variables_dind(key_model)
    lk_xfree = coll.get_spectral_model_variables(key_model, 'free')['free']

    wsm = coll._which_model
    dmodel = coll.dobj[wsm][key_model]['dmodel']

    # ------------
    # get scale, bounds
    # ------------

    # check dscales
    dscales = _x0_scale_bounds._get_dict(
        lx_free_keys=lk_xfree,
        dmodel=dmodel,
        din=dscales,
        din_name='dscales',
    )

    # check dbounds_low
    dbounds_low = _x0_scale_bounds._get_dict(
        lx_free_keys=lk_xfree,
        dmodel=dmodel,
        din=dbounds_low,
        din_name='dbounds_low',
    )

    # check dbounds_up
    dbounds_up = _x0_scale_bounds._get_dict(
        lx_free_keys=lk_xfree,
        dmodel=dmodel,
        din=dbounds_up,
        din_name='dbounds_up',
    )

    # get scales and bounds
    scales, bounds0, bounds1 = _x0_scale_bounds._get_scales_bounds(
        nxfree=len(lk_xfree),
        lamb=lamb,
        data=data,
        iok_all=iok_all,
        dind=dind,
        dscales=dscales,
        dbounds_low=dbounds_low,
        dbounds_up=dbounds_up,
    )

    # ------------
    # get x0
    # ------------

    # check dx0
    dx0 = _x0_scale_bounds._get_dict(
        lx_free_keys=lk_xfree,
        dmodel=dmodel,
        din=dx0,
        din_name='dx0',
    )

    # ------------
    # get functions
    # ------------

    # func_cost, func_jac
    dfunc = coll.get_spectral_fit_func(
        key=key_model,
        func=['cost', 'jac'],
    )

    # -----------------
    # optionnal binning
    # -----------------

    if binning is not False:
        lambd = lamb[1] - lamb[0]

        lamb_edges = np.r_[
            lamb[0] - lambd,
            0.5*(lamb[1:] + lamb[:-1]),
        ]

        nlamb = lamb.size
        lamb_inc = np.linspace(0, 1, binning+2)[1:-1] * lambd
        lamb = (lamb_edges[:, None] + lamb_inc[None, :]).ravel()

        # safety check
        assert lamb.size == coll.ddata[key_lamb]['data'].size * binning

        binning = np.arange(0, nlamb, binning)

    # ------------
    # Main loop
    # ------------

    dout = _loop(
        coll=coll,
        key=key,
        # lamb, data, axis
        lamb=lamb,
        data=data,
        axis=axis,
        # iok
        dind=dind,
        iok_all=iok_all,
        iok_reduced=iok_reduced,
        # x0, bounds, scale
        scales=scales,
        bounds0=bounds0,
        bounds1=bounds1,
        dx0=dx0,
        # options
        chain=chain,
        binning=binning,
        # func
        func_cost=dfunc['cost'],
        func_jac=dfunc['jac'],
        # solver options
        solver=solver,
        dsolver_options=dsolver_options,
        # options
        lk_xfree=lk_xfree,
        strict=strict,
        verb=verb,
        timing=timing,
    )

    return dout


#############################################
#############################################
#       get iok
#############################################


def get_iok_all(
    coll=None,
    key=None,
    axis=None,
):

    # ----------
    # prepare
    # ----------

    wsf = coll._which_fit
    kiok = coll.dobj[wsf][key]['dvalid']['iok']
    iok = coll.ddata[kiok]['data']
    meaning = coll.dobj[wsf][key]['dvalid']['meaning']

    # ----------
    # iok_all
    # ----------

    # get list of valid indices
    lind_valid = [
        k0 for k0, v0 in meaning.items()
        if any([ss in v0 for ss in ['ok', 'incl']])
    ]

    # iok_all
    iok_all = (iok == lind_valid[0])
    for k0 in lind_valid[1:]:
        iok_all = np.logical_or(iok_all, (iok == k0))

    iok_reduced = np.any(iok_all, axis=axis)

    return iok_all, iok_reduced


#############################################
#############################################
#       Main loop
#############################################


def _loop(
    coll=None,
    key=None,
    # lamb, data, axis
    lamb=None,
    data=None,
    axis=None,
    # iok
    dind=None,
    iok_all=None,
    iok_reduced=None,
    # x0, bounds, scale
    scales=None,
    bounds0=None,
    bounds1=None,
    dx0=None,
    # options
    chain=None,
    binning=None,
    # func
    func_cost=None,
    func_jac=None,
    # solver options
    solver=None,
    dsolver_options=None,
    # options
    lk_xfree=None,
    strict=None,
    verb=None,
    timing=None,
):

    # -----------------
    # prepare
    # -----------------

    # shape_reduced
    shape_reduced = tuple([
        ss for ii, ss in enumerate(data.shape)
        if ii != axis
    ])

    # shape_sol
    nxfree = len(lk_xfree)
    shape_sol = list(data.shape)
    shape_sol[axis] = nxfree

    # lind
    lind = [range(ss) for ss in shape_reduced]

    # nspect
    nspect = int(np.prod(shape_reduced))

    # verb init
    if verb is not False:
        end = '\r'

    # timing init
    if timing is True:
        t0 = dtm.datetime.now()

    # -----------------
    # initialize
    # -----------------

    validity = np.zeros(shape_reduced, dtype=int)
    success = np.full(shape_reduced, np.nan)
    cost = np.full(shape_reduced, np.nan)
    chi2n = np.full(shape_reduced, np.nan)
    nfev = np.full(shape_reduced, np.nan)
    time = np.full(shape_reduced, np.nan)
    sol = np.full(shape_sol, np.nan)

    message = []
    errmsg = []

    # ----------
    # slice_sol

    sli_sol = np.array([
        slice(None) if ii == axis else 0
        for ii, ss in enumerate(data.shape)
    ])
    ind_ind = np.array([ii for ii in range(data.ndim) if ii != axis])

    # -----------------
    # main loop
    # -----------------

    for ii, ind in enumerate(itt.product(*lind)):

        # -------------
        # check iok_all

        if not iok_reduced[ind]:
            validity[ind] = -1
            continue

        # -------
        # slices

        sli_sol[ind_ind] = ind
        slii = tuple(sli_sol)

        # ---------------
        # x0

        # get x0
        if chain is False or ii == 0:
            x0 = _x0_scale_bounds._get_x0(
                nxfree=nxfree,
                lamb=lamb,
                data=data[slii],
                iok=iok_all[slii],
                dind=dind,
                dx0=dx0,
                scales=scales,
            )

        # ------
        # verb

        if verb == 3:
            msg = f"\nspect {ii+1} / {nspect}"
            print(msg)

        # -----------
        # try solving

        try:
            dti = None
            t0i = dtm.datetime.now()     # DB

            # optimization
            res = scpopt.least_squares(
                func_cost,
                x0,
                jac=func_jac,
                bounds=(bounds0, bounds1),
                x_scale='jac',
                f_scale=1.0,
                jac_sparsity=None,
                args=(),
                kwargs={
                    'data': data[slii],
                    'scales': scales,
                    'lamb': lamb,
                    # 'const': const[ii, :],
                    'iok': iok_all[slii],
                    'binning': binning,
                },
                **dsolver_options,
            )
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True:
                x0 = res.x

            # time
            time[ind] = round(dti, ndigits=6)

            # other outputs
            success[ind] = res.success
            cost[ind] = res.cost
            nfev[ind] = res.nfev

            chi2n[ind] = np.sqrt(cost[ind] * 2) / iok_all[slii].sum()

            sol[slii] = res.x * scales
            # sol_x[ii, ~indx] = const[ii, :] / scales[ii, ~indx]

            # message
            message.append(res.message)
            errmsg.append('')

        # ---------------
        # manage failures

        except Exception as err:

            message.append('')
            msg = str(err)
            lerr = [
                'is infeasible',
                'Each lower bound must be strictly less than',
            ]
            if any([ee in msg for ee in lerr]):
                msg += _add_err_bounds(
                    key=key,
                    lk_xfree=lk_xfree,
                    scales=scales,
                    x0=x0,
                    bounds0=bounds0,
                    bounds1=bounds1,
                )

            if strict:
                raise Exception(msg) from err
            else:
                errmsg[ii] = msg
                validity[ii] = -2

    # --------------
    # prepare output
    # --------------

    dout = {
        'validity': validity,
        'sol': sol,
        'msg': np.reshape(message, shape_reduced),
        'nfev': nfev,
        'cost': cost,
        'chi2n': chi2n,
        'success': success,
        'time': time,
        'errmsg': np.reshape(errmsg, shape_reduced),
        'scales': scales,
        'bounds0': bounds0,
        'bounds1': bounds1,
        'x0': x0,
        # solver
        'solver': solver,
        'dsolver_options': dsolver_options,
    }

    return dout


#############################################
#############################################
#       Errors
#############################################


def _add_err_bounds(
    key=None,
    lk_xfree=None,
    scales=None,
    x0=None,
    bounds0=None,
    bounds1=None,
):

    # -------------
    # is_out
    # -------------

    is_out = np.nonzero(
        (x0 < bounds0) | (x0 > bounds1) | (bounds0 > bounds1)
    )[0]

    # -------------
    # dout, din
    # -------------

    dout = {
        k0: {
            'scale': f"{scales[ii]:.3e}",
            'x0': f"{x0[ii]:.3e}",
            'bounds0': f"{bounds0[ii]:.3e}",
            'bounds1': f"{bounds1[ii]:.3e}",
        }
        for ii, k0 in enumerate(lk_xfree) if ii in is_out
    }

    # -------------
    # msg arrays
    # -------------

    head = ['var', 'scale', 'x0', 'bounds0', 'bounds1']

    arr_out = [
        (k0, v0['scale'], v0['x0'], v0['bounds0'], v0['bounds1'])
        for k0, v0 in dout.items()
    ]

    arr_in = [
        (k0, v0['scale'], v0['x0'], v0['bounds0'], v0['bounds1'])
        for k0, v0 in dout.items()
    ]

    # max_just
    max_just = np.max([
        [len(ss) for ss in head]
        + [np.max([len(ss) for ss in arr]) for arr in arr_out]
        + [np.max([len(ss) for ss in arr]) for arr in arr_in]
    ])

    # -------------
    # msg
    # -------------

    lstr = [
        " ".join([ss.ljust(max_just) for ss in head]),
        " ".join(['-'*max_just for ss in head]),
    ]
    for arr in arr_out:
        lstr.append(" ".join([ss.ljust(max_just) for ss in arr]))

    msg = f" for spectral_fit '{key}'\n\n" + "\n".join(lstr)

    return msg
