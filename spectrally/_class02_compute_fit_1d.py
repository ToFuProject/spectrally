# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import datetime as dtm


import numpy as np


# local


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
    verb=None,
    timing=None,
):
    """ Fit 1d spectra

    """

    # ------------
    # prepare
    # ------------

    # iok_all
    iok_all = get_iok_all(
        coll=coll,
        key=key,
        axis=axis,
    )

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
        # func
        fun_cost=func_cost,
        func_jac=func_jac,
        # options
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
    # func
    func_cost=None,
    func_jac=None,
    # options
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
    cost = np.full(shape_reduced, np.nan)
    nfev = np.full(shape_reduced, np.nan)
    time = np.full(shape_reduced, np.nan)
    sol = np.full(shape_sol, np.nan)

    # ----------
    # slice_sol

    sli_sol = [
        slice(None) if ii == axis else 0
        for ii, ss in enumerate(shape_data)
    ]

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


        sli_sol = tuple([])

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
                x0[ii, indx],
                jac=func_jac,
                bounds=bounds[:, indx],
                method=method,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                x_scale=1.0,
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
                    'data': datacost[ii, :],
                    'scales': scales[ii, :],
                    'const': const[ii, :],
                    'indok': dprepare['indok_bool'][ii, :],
                },
            )
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True and ii < nspect-1:
                x0[ii+1, indx] = res.x

            # cost, message, time
            success[ind] = res.success
            cost[ind] = res.cost
            nfev[ind] = res.nfev
            message[ii] = res.message
            time[ind] = round(
                (dtm.datetime.now()-t0i).total_seconds(),
                ndigits=3,
            )
            sol_x[ii, indx] = res.x
            sol_x[ii, ~indx] = const[ii, :] / scales[ii, ~indx]

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
