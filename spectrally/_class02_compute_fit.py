# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:44:51 2024

@author: dvezinet
"""

# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import datastock as ds


# local
from . import _class02_compute_fit_1d as _compute_fit_1d


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
    # solver options
    solver=None,
    dsolver_options=None,
    # storing
    store=None,
    # options
    strict=None,
    verb=None,
    timing=None,
):
    """ Compute the fit of any previously added spectral fit

    """

    # ------------
    # check inputs
    # ------------

    (
        key, is1d,
        key_model,
        key_data, key_lamb,
        ref_data, ref_lamb,
        lamb, data, axis,
        store,
        strict, verb, timing,
    ) = _check(
        coll=coll,
        key=key,
        # storing
        store=store,
        # options
        strict=strict,
        verb=verb,
        timing=timing,
    )

    # ----------------
    # particular case

    ravel = False
    if data.ndim == 1:
        # don't change axis !
        data = data[:, None]
        ravel = True

    # ------------
    # solver_options
    # ------------

    dsolver_options = _get_solver_options(
        solver=solver,
        dsolver_options=dsolver_options,
    )

    # ------------
    # fit
    # ------------

    if is1d is True:
        compute = _compute_fit_1d.main

    else:
        pass

    dout = compute(
        coll=coll,
        # keys
        key=key,
        key_model=key_model,
        key_data=key_data,
        key_lamb=key_lamb,
        # lamb, data, axis
        lamb=lamb,
        data=data,
        axis=axis,
        ravel=ravel,
        # options
        chain=None,
        dscales=None,
        dbounds_low=None,
        dbounds_up=None,
        dx0=None,
        # solver options
        solver=solver,
        dsolver_options=dsolver_options,
        # options
        strict=strict,
        verb=verb,
        timing=timing,
    )

    if dout is None:
        # TP BE dealt with
        print()
        print("dout is None, 127")
        print()
        pass

    # --------------
    # format output
    # --------------

    if ravel is True:
        pass

    # ------------
    # store
    # ------------

    if store is True:
        _store(
            coll=coll,
            # keys
            key=key,
            key_model=key_model,
            key_data=key_data,
            key_lamb=key_lamb,
            axis=axis,
            # dout
            dout=dout,
        )

    return dout


#############################################
#############################################
#       Check
#############################################


def _check(
    coll=None,
    key=None,
    # storing
    store=None,
    # options
    strict=None,
    verb=None,
    timing=None,
):

    # --------------
    # key
    # --------------

    wsf = coll._which_fit
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
        allowed=lok_1d + lok_2d,
    )

    # is1d
    is1d = key in lok_1d

    if not is1d:
        msg = "2d fit not implemented yet"
        raise NotImplementedError(msg)

    # -----------
    # derive keys

    key_model = coll.dobj[wsf][key]['key_model']
    key_data = coll.dobj[wsf][key]['key_data']
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    lamb = coll.ddata[key_lamb]['data']
    data = coll.ddata[key_data]['data']
    ref_data = coll.ddata[key_data]['ref']
    ref_lamb = coll.ddata[key_lamb]['ref']
    axis = ref_data.index(ref_lamb[0])

    # --------------
    # store
    # --------------

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # --------------
    # strict
    # --------------

    strict = ds._generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=False,
    )

    # --------------
    # verb
    # --------------

    def_verb = 2
    verb = ds._generic_check._check_var(
        verb, 'verb',
        default=def_verb,
        allowed=[False, True, 0, 1, 2, 3],
    )
    if verb is True:
        verb = def_verb

    # --------------
    # timing
    # --------------

    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
    )

    return (
        key, is1d,
        key_model,
        key_data, key_lamb,
        ref_data, ref_lamb,
        lamb, data, axis,
        store,
        strict, verb, timing,
    )


#############################################
#############################################
#       Solver options
#############################################


def _get_solver_options(
    solver=None,
    dsolver_options=None,
):

    # -------------------
    # available solvers
    # -------------------

    lok = ['scipy.least_squares']
    solver = ds._generic_check._check_var(
        solver, 'solver',
        types=str,
        default='scipy.least_squares',
        allowed=lok,
    )

    # -------------------
    # get default
    # -------------------

    if solver == 'scipy.least_squares':

        ddef = dict(
            # solver options
            method='trf',
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            tr_solver='exact',
            tr_options={},
            diff_step=None,
            max_nfev=None,
            loss='linear',
            verbose=2,
        )

    else:
        raise NotImplementedError()

    # -------------------
    # implement
    # -------------------

    if dsolver_options is None:
        dsolver_options = {}

    if not isinstance(dsolver_options, dict):
        msg = (
        )
        raise Exception(msg)

    # add default values
    for k0, v0 in ddef.items():
        if dsolver_options.get(k0) is None:
            dsolver_options[k0] = v0

    # clear irrelevant keys
    lkout = [k0 for k0 in dsolver_options.keys() if k0 not in ddef.keys()]
    for k0 in lkout:
        del dsolver_options[k0]

    return dsolver_options


#############################################
#############################################
#       store
#############################################


def _store(
    coll=None,
    # keys
    key=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    axis=None,
    # dout
    dout=None,
):

    # ------------
    # add ref
    # ------------

    wsm = coll._which_model
    refx_free = coll.dobj[wsm][key_model]['ref_nx']
    ref = list(coll.ddata[key_data]['ref'])
    ref[axis] = refx_free

    ref_reduced = tuple([rr for ii, rr in ref if ii != axis])

    # ------------
    # add data
    # ------------

    # solution
    ksol = f"{key}_sol"
    coll.add_data(
        key=ksol,
        data=dout['sol'],
        ref=tuple(ref),
        units=coll.ddata[key_data]['units'],
        dim='fit_sol',
    )

    # other outputs
    lk = ['cost', 'time', 'success', 'nfev', 'time', 'msg', 'validity', 'errmsg']
    dk_out = {k0: f"{key}_k0" for k0 in lk}

    for k0, k1 in dk_out.items():
        coll.add_data(
            key=k1,
            data=dout[k0],
            ref=ref_reduced,
            units='',
            dim='fit_out',
        )

    # ------------
    # store in fit
    # ------------

    wsf = coll._which_fit
    coll._dobj[wsf][key]['key_sol'] = ksol

    # scale, bounds, x0
    coll._dobj[wsf][key]['dinternal'] = {
        'scale': dout['scale'],
        'bounds0': dout['bounds0'],
        'bounds1': dout['bounds1'],
        'x0': dout['x0'],
    }

    # solver output
    coll._dobj[wsf][key]['dsolver'] = {
        'solver': dout['solver'],
        'dsolver_options': dout['dsolver_options'],
    }

    for k0, k1 in dk_out.items():
        coll._dobj[wsf][key]['dsolver'][k0] = k1

    return