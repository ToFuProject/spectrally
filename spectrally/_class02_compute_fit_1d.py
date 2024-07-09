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
    key=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    ref_data=None,
    ref_lamb=None,
    shape_data=None,
    axis=None,
    # options
    chain=None,
    dscales=None,
    dbounds_low=None,
    dbounds_up=None,
    dx0=None,
    # solver options
    method=None,
    xtol=None,
    ftol=None,
    gtol=None,
    tr_solver=None,
    tr_options=None,
    max_nfev=None,
    loss=None,
    verbscp=None,
    # options
    strict=None,
    verb=None,
    timing=None,
):
    """ Fit 1d spectra

    """

    # ------------
    # ind_ok
    # ------------

    # iok_all
    iok_all = get_iok_all(
        coll=coll,
        key=key,
        axis=axis,
    )

    lamb = coll.ddata[key_lamb]['data']

    dind = coll.get_spectral_model_variables_dind(key_model)

    data = coll.ddata[key_data]['data']

    # ------------
    # get x0, scale, bounds
    # ------------

    wsm = coll._which_model
    lk_xfree = coll.get_spectral_model_variables(key_model, 'free')['free']
    dmodel = coll.dobj[wsm][key_model]['dmodel']

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


    # check dx0
    dx0 = None

    # get scales and bounds
    scale, bounds = _x0_scale_bounds._get_scales_bounds(
        nxfree=None,
        lamb=lamb,
        data=None,
        iok_all=iok_all,
        dind=dind,
        dscales=dscales,
        dbounds_low=dbounds_low,
        dbounds_up=dbounds_up,
    )

    # get x0
    x0 = _x0_scale_bounds._get_x0()

    # ------------
    # get functions
    # ------------

    # func_cost, func_jac
    dfunc = coll.get_spectral_fit_func(
        key=key_model,
        func=['cost', 'jac'],
    )

    # ------------
    # Main loop
    # ------------

    dout = _loop(
        coll=coll,
        key=key,
        shape_data=shape_data,
        axis=axis,
        iok_all=iok_all,
        # data
        data=data,
        # x0, bounds, scale
        scale=scale,
        bounds=bounds,
        x0=x0,
        # options
        chain=chain,
        # func
        fun_cost=dfunc['cost'],
        func_jac=dfunc['jac'],
        # solver options
        method=method,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        tr_solver=tr_solver,
        tr_options=tr_options,
        max_nfev=max_nfev,
        loss=loss,
        verbscp=verbscp,
        # options
        strict=None,
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

    # -------------
    # safety check
    # -------------

    c0 = np.all(iok_all, axis=axis)

    return iok_all


#############################################
#############################################
#       Main loop
#############################################


def _loop(
    coll=None,
    key=None,
    shape_data=None,
    axis=None,
    iok_all=None,
    # data
    data=None,
    # x0, bounds, scale
    scale=None,
    bounds=None,
    x0=None,
    # options
    chain=None,
    # func
    func_cost=None,
    func_jac=None,
    # solver options
    method=None,
    xtol=None,
    ftol=None,
    gtol=None,
    tr_solver=None,
    tr_options=None,
    max_nfev=None,
    loss=None,
    verbscp=None,
    # options
    strict=None,
    verb=None,
    timing=None,
):

    # -----------------
    # prepare
    # -----------------

    # shape_reduced
    shape_reduced = tuple([
        ss for ii, ss in enumerate(shape_data)
        if ii != axis
    ])

    # shape_sol
    shape_sol = []

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
    nfev = np.full(shape_reduced, np.nan)
    time = np.full(shape_reduced, np.nan)
    sol = np.full(shape_sol, np.nan)

    message = ['' for ss in range(nspect)]
    errmsg = ['' for ss in range(nspect)]

    # ----------
    # slice_sol

    sli_sol = np.array([
        slice(None) if ii == axis else 0
        for ii, ss in enumerate(shape_data)
    ])
    ind_ind = np.array([ii for ii in range(len(shape_data)) if ii != axis])

    # -----------------
    # main loop
    # -----------------

    for ii, ind in enumerate(itt.product(*lind)):

        # -------------
        # check iok_all

        if not iok_all(ind):
            validity[ind] = -1
            continue

        # -------
        # slices

        sli_sol[ind_ind] = ind
        slii = tuple(sli_sol)

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
                bounds=bounds,
                method=method,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                x_scale='jac',
                f_scale=1.0,
                loss=loss,
                diff_step=None,
                tr_solver=tr_solver,
                tr_options=tr_options,
                jac_sparsity=None,
                max_nfev=max_nfev,
                verbose=verbscp,
                args=(),
                kwargs={
                    'data': data[slii],
                    'scales': None,
                    # 'const': const[ii, :],
                    # 'indok': iok_all,
                },
            )
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True:
                x0 = res.x
            else:
                x0 = _x0_scale_bounds._get_x0()

            # cost, message, time
            success[ind] = res.success
            cost[ind] = res.cost
            nfev[ind] = res.nfev
            message[ii] = res.message
            time[ind] = round(
                (dtm.datetime.now()-t0i).total_seconds(),
                ndigits=3,
            )
            sol[slii] = res.x
            # sol_x[ii, ~indx] = const[ii, :] / scales[ii, ~indx]

        # ---------------
        # manage failures

        except Exception as err:
            if strict:
                raise err
            else:
                errmsg[ii] = str(err)
                validity[ii] = -2

    # --------------
    # prepare output
    # --------------

    dout = {
        'validity': validity,
        'sol': sol,
        'msg': message,
        'nfev': nfev,
        'cost': cost,
        'success': success,
        'time': time,
    }

    return dout