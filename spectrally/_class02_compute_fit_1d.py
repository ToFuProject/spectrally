# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import itertools as itt
import functools
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
    verb_scp=None,
    timing=None,
):
    """ Fit 1d spectra

    """

    # -----------
    # verb

    if verb >= 2:
        msg = (
            "Preparing input data (scales, bounds, x0...)\n"
        )
        print(msg)

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

    # -----------------
    # optionnal binning
    # -----------------

    dbinning = coll.get_spectral_fit_binning_dict(
        binning=binning,
        lamb=lamb,
        iok=iok_all,
        axis=axis,
    )

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
        iok=iok_all,
        axis=axis,
        dind=dind,
        dscales=dscales,
        dbounds_low=dbounds_low,
        dbounds_up=dbounds_up,
        # binning
        dbinning=dbinning,
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
        func=['sum', 'cost', 'jac'],
    )

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
        dbinning=dbinning,
        # func
        func_sum=dfunc.get('sum'),
        func_cost=dfunc.get('cost'),
        func_jac=dfunc['jac'],
        # solver options
        solver=solver,
        dsolver_options=dsolver_options,
        # options
        lk_xfree=lk_xfree,
        strict=strict,
        verb=verb,
        verb_scp=verb_scp,
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
    dbinning=None,
    # func
    func_sum=None,
    func_cost=None,
    func_jac=None,
    # solver options
    solver=None,
    dsolver_options=None,
    # options
    lk_xfree=None,
    strict=None,
    verb=None,
    verb_scp=None,
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

    # timing init
    if timing is True:
        t0 = dtm.datetime.now()

    # -----------------
    # prepare verb
    # -----------------

    # verb init
    if verb == 1:
        end = '\r'
    elif verb in [2, 3]:
        end = '\n'

    # iterations
    if verb >= 1:
        sep = '  '
        ditems = {
            'spectrum ind': {
                'just': max(13, len(str(shape_reduced))*2+3),
                'val': None,
            },
            'nfev': {'just': 6, 'val': None},
            'cost_final': {'just': 10, 'val': None},
            'status': {'just': 6, 'val': None},
            # 'termination': {'just': 12, 'val': None},
        }
        litems = ['spectrum ind', 'nfev', 'cost_final', 'status']

        if verb in [1, 2]:
            msg = (
                sep.join([k0.ljust(ditems[k0]['just']) for k0 in litems])
                + '\n'
                + sep.join(['-'*ditems[k0]['just'] for k0 in litems])
            )
            print(msg)

    # -----------------
    # initialize
    # -----------------

    validity = np.zeros(shape_reduced, dtype=int)
    status = np.full(shape_reduced, np.nan)
    cost = np.full(shape_reduced, np.nan)
    chi2n = np.full(shape_reduced, np.nan)
    nfev = np.full(shape_reduced, np.nan)
    time = np.full(shape_reduced, np.nan)
    sol = np.full(shape_sol, np.nan)
    if solver == 'scipy.curve_fit':
        std = np.full(shape_sol, np.nan)
    else:
        std = None

    message = ['' for ii in range(np.prod(shape_reduced))]
    errmsg = ['' for ii in range(np.prod(shape_reduced))]

    # ----------
    # slice_sol

    sli_sol = np.array([
        slice(None) if ii == axis else 0
        for ii, ss in enumerate(data.shape)
    ])
    ind_ind = np.array([ii for ii in range(data.ndim) if ii != axis])

    # -----------------
    # initialize parameter dict
    # -----------------

    dparams = {
        'scales': scales,
        'iok': None,
        'bin_ind': False if dbinning is False else dbinning['ind'],
        'bin_dlamb': None if dbinning is False else dbinning['dlamb'],
    }
    if solver == 'scipy.least_squares':
        dparams['lamb'] = lamb if dbinning is False else dbinning['lamb']
        dparams['data'] = None
    else:
        pass


    # -----------------
    # main loop
    # -----------------

    for ii, ind in enumerate(itt.product(*lind)):

        # -------------
        # check iok_all

        if not iok_reduced[ind]:
            message[ii] = 'no valid data'
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
                axis=axis,
                dind=dind,
                dx0=dx0,
                scales=scales,
                dbinning=dbinning,
            )

        # ------
        # verb

        if verb == 3:
            msg = f"\nspectrum {ind} / {shape_reduced}"
            print(msg)

        # -----------
        # parameters

        dparams['iok'] = iok_all[slii]
        if dbinning is False:
            dparams['bin_iok'] = iok_all[slii]
        else:
            dparams['bin_iok'] = dbinning['iok'][slii]

        # -----------
        # try solving

        try:
            dti = None
            t0i = dtm.datetime.now()     # DB

            if solver == 'scipy.least_squares':

                # update data
                dparams['data'] = data[slii]

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
                    kwargs=dparams,
                    **dsolver_options,
                )
                dti = (dtm.datetime.now() - t0i).total_seconds()

                # time
                time[ind] = round(dti, ndigits=6)

                # x0 if chain
                if chain is True:
                    x0 = res.x

                # other outputs
                status[ind] = res.status
                cost[ind] = res.cost
                nfev[ind] = res.nfev

                chi2n[ind] = np.sqrt(cost[ind] * 2) / iok_all[slii].sum()

                # store scaled solution
                sol[slii] = res.x * scales
                # sol_x[ii, ~indx] = const[ii, :] / scales[ii, ~indx]

                # message
                message[ii] = res.message
                errmsg[ii] = ''

            elif solver == 'scipy.curve_fit':

                def func_sum2(xdata, *xfree, dparams=dparams):
                    return func_sum(np.r_[xfree], lamb=xdata, **dparams)

                def func_jac2(xdata, *xfree, dparams=dparams):
                    return func_jac(np.r_[xfree], lamb=xdata, **dparams)

                popt, pcov, infodict, mesg, ier = scpopt.curve_fit(
                    func_sum2,
                    lamb if dbinning is False else dbinning['lamb'],
                    data[slii][iok_all[slii]],
                    p0=x0,
                    sigma=None,
                    absolute_sigma=False,
                    check_finite=True,    # to be updated
                    bounds=(bounds0, bounds1),
                    jac=func_jac2,
                    **dsolver_options,
                )
                dti = (dtm.datetime.now() - t0i).total_seconds()

                # time
                time[ind] = round(dti, ndigits=6)

                # x0 if chain
                if chain is True:
                    x0 = popt

                # other outputs
                status[ind] = ier
                cost[ind] = np.sum(infodict['fvec']**2)
                nfev[ind] = infodict['nfev']

                chi2n[ind] = np.sqrt(cost[ind]) / iok_all[slii].sum()

                # store scaled solution
                sol[slii] = popt * scales
                std[slii] = np.sqrt(np.diag(pcov)) * scales

                # message
                message[ii] = mesg
                errmsg[ii] = ''

        # ---------------
        # manage failures

        except Exception as err:

            msg = (
                f"\nError for spect_fit '{key}' with solver = '{solver}':\n"
                + str(err)
            )
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

        # -------------
        # verb

        finally:

            if verb in [1, 2]:
                ditems['spectrum ind']['val'] = f"{ind} / {shape_reduced}"
                ditems['status']['val'] = f"{status[ind]:.0f}"
                ditems['cost_final']['val'] = f"{cost[ind]:.4e}"
                ditems['nfev']['val'] = f"{nfev[ind]:.0f}"

                msg = (
                    sep.join([
                        ditems[k0]['val'].ljust(ditems[k0]['just'])
                        for k0 in litems
                    ])
                )
                print(msg, end=end)

    # -------------
    # adjust verb
    # -------------

    if verb == 2:
        print()

    # --------------
    # prepare output
    # --------------

    dout = {
        'validity': validity,
        'sol': sol,
        'std': std,
        'msg': np.reshape(message, shape_reduced),
        'nfev': nfev,
        'cost': cost,
        'chi2n': chi2n,
        'status': status,
        'time': time,
        'errmsg': np.reshape(errmsg, shape_reduced),
        'scales': scales,
        'bounds0': bounds0,
        'bounds1': bounds1,
        'x0': x0,
        # binning:
        'dbinning': dbinning,
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