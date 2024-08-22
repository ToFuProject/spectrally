# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:44:51 2024

@author: dvezinet
"""

# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import warnings


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
    # binning
    binning=None,
    # solver options
    solver=None,
    dsolver_options=None,
    # options
    chain=None,
    dscales=None,
    dbounds_low=None,
    dbounds_up=None,
    dx0=None,
    # storing
    store=None,
    overwrite=None,
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
        chain,
        store, overwrite,
        strict, verb, verb_scp, timing,
    ) = _check(
        coll=coll,
        key=key,
        # options
        chain=chain,
        dscales=dscales,
        dbounds_low=dbounds_low,
        dbounds_up=dbounds_up,
        dx0=dx0,
        # storing
        store=store,
        overwrite=overwrite,
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

    solver, dsolver_options = _get_solver_options(
        solver=solver,
        dsolver_options=dsolver_options,
        verb_scp=verb_scp,
    )

    # ------------
    # fit
    # ------------

    if is1d is True:
        compute = _compute_fit_1d.main

    else:
        pass

    # -----------
    # verb

    if verb >= 1:
        msg = (
            "\n\n-------------------------------------------\n"
            f"\tComputing spectral fit '{key}'\n"
        )
        print(msg)

    # --------
    # run

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
        # binning
        binning=binning,
        # options
        chain=chain,
        dscales=dscales,
        dbounds_low=dbounds_low,
        dbounds_up=dbounds_up,
        dx0=dx0,
        # solver options
        solver=solver,
        dsolver_options=dsolver_options,
        # options
        strict=strict,
        verb=verb,
        verb_scp=verb_scp,
        timing=timing,
    )

    if dout is None:
        msg = (
            "No valid spectrum in chosen data:\n"
            f"\t- spectral_fit: {key}\n"
            f"\t- key_data: {key_data}\n"
        )
        warnings.warn(msg)
        return

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
            overwrite=overwrite,
            # keys
            key=key,
            key_model=key_model,
            key_data=key_data,
            key_lamb=key_lamb,
            # flags
            axis=axis,
            ravel=ravel,
            # dout
            dout=dout,
            # verb
            verb=verb,
        )

        dout = None

    return dout


#############################################
#############################################
#       Check
#############################################


def _check(
    coll=None,
    key=None,
    # options
    chain=None,
    dscales=None,
    dbounds_low=None,
    dbounds_up=None,
    dx0=None,
    # storing
    store=None,
    overwrite=None,
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

    # ---------------------------------
    # voigt not handled for fitting yet
    # ---------------------------------

    wsm = coll._which_model
    lvoigt = [
        k0 for k0, v0 in coll.dobj[wsm][key_model]['dmodel'].items()
        if v0['type'] == 'voigt'
    ]
    if len(lvoigt) > 0:
        msg = (
            "Fitting is not implemented for voigt profiles yet\n"
            f"\t- key_model: `{key_model}`\n"
            f"\t- voigt functions: {lvoigt}\n"
            "Consider using another spectral model with pvoigt instead"
        )
        raise Exception(msg)

    # --------------
    # chain
    # --------------

    chain = ds._generic_check._check_var(
        chain, 'chain',
        types=bool,
        default=False,
    )

    # --------------
    # store
    # --------------

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # --------------
    # overwrite
    # --------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    if overwrite is True:
        ksol = coll.dobj[wsf][key]['key_sol']
        overwrite &= (ksol is not None and ksol in coll.ddata.keys())

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
    elif verb is False:
        verb = 0

    # verb_scp
    verb_scp = {
        0: 0,
        1: 0,
        2: 0,
        3: 2,
    }[verb]

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
        chain,
        store, overwrite,
        strict, verb, verb_scp, timing,
    )


#############################################
#############################################
#       Solver options
#############################################


def _get_solver_options(
    solver=None,
    dsolver_options=None,
    verb_scp=None,
):

    # -------------------
    # available solvers
    # -------------------

    lok = ['scipy.least_squares', 'scipy.curve_fit']
    solver = ds._generic_check._check_var(
        solver, 'solver',
        types=str,
        # default='scipy.least_squares',
        default='scipy.curve_fit',
        allowed=lok,
    )

    # -------------------
    # get default
    # -------------------

    if solver == 'scipy.least_squares':

        ddef = dict(
            # solver options
            method='trf',
            xtol=None,
            ftol=1e-10,
            gtol=None,
            tr_solver='exact',
            tr_options={},
            diff_step=None,
            max_nfev=None,
            loss='linear',
            verbose=verb_scp,
        )

    elif solver == 'scipy.curve_fit':

        ddef = dict(
            # solver options
            full_output=True,
            # nan_policy='raise',  # not available on MacOS ?
            # common with least_squares
            method='trf',
            xtol=None,
            ftol=1e-10,
            gtol=None,
            tr_solver='exact',
            tr_options={},
            diff_step=None,
            max_nfev=1000,
            loss='linear',
            verbose=verb_scp,
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

    return solver, dsolver_options


#############################################
#############################################
#       store
#############################################


def _store(
    coll=None,
    overwrite=None,
    # keys
    key=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    # flags
    axis=None,
    ravel=None,
    # dout
    dout=None,
    # verb
    verb=None,
):

    # ------------
    # add ref
    # ------------

    wsm = coll._which_model
    wsf = coll._which_fit
    refx_free = coll.dobj[wsm][key_model]['ref_nx']
    ref = list(coll.ddata[key_data]['ref'])
    ref[axis] = refx_free

    ref_reduced = tuple([rr for ii, rr in enumerate(ref) if ii != axis])

    # ------------
    # add data
    # ------------

    # solution names
    ksol = f"{key}_sol"

    # solution arrays
    sol = dout['sol'].ravel() if ravel is True else dout['sol']
    if dout['std'] is None:
        kstd = None
        std = None
    else:
        kstd = f"{key}_std"
        std = dout['std'].ravel() if ravel is True else dout['std']

    # add
    if ksol in coll.ddata.keys():
        if overwrite is True:
            c0 = (
                coll.ddata[ksol]['ref'] == tuple(ref)
                and coll.dobj[wsf][key]['key_sol'] == ksol
            )
            if c0 is True:
                coll._ddata[ksol]['data'] = sol
                if std is not None:
                    coll._ddata[kstd]['data'] = std

            else:
                msg = (
                    "Fit sol '{ksol}' exists in ddata but not in spect_fit!"
                )
                raise Exception(msg)

        else:
            msg = (
                f"Aborting storing, spect_fit '{key}' was already computed\n"
                "Use overwrite=True to overwrite the former solution\n"
            )
            raise Exception(msg)

    else:
        # solution
        coll.add_data(
            key=ksol,
            data=sol,
            ref=tuple(ref),
            units=coll.ddata[key_data]['units'],
            dim='fit_sol',
        )

        # std
        if std is not None:
            coll.add_data(
                key=kstd,
                data=std,
                ref=tuple(ref),
                units=coll.ddata[key_data]['units'],
                dim='fit_std',
            )

    # -------------
    # other outputs

    lk = [
        'cost', 'chi2n', 'time', 'status', 'nfev',
        'msg', 'validity', 'errmsg',
    ]
    dk_out = {k0: f"{key}_{k0}" for k0 in lk}

    if ravel is False:

        if overwrite is True:
            for k0, k1 in dk_out.items():
                coll._ddata[k1]['data'] = dout[k0]

        else:
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
    coll._dobj[wsf][key]['key_std'] = kstd

    # scale, bounds, x0
    coll._dobj[wsf][key]['dinternal'] = {
        'scales': dout['scales'],
        'bounds0': dout['bounds0'],
        'bounds1': dout['bounds1'],
        'x0': dout['x0'],
        'binning': (
            False if dout['dbinning'] is False
            else dout['dbinning']['binning']
        ),
    }

    # solver output
    coll._dobj[wsf][key]['dsolver'] = {
        'solver': dout['solver'],
        'dsolver_options': dout['dsolver_options'],
    }

    if ravel is True:
        for k0, k1 in dk_out.items():
            coll._dobj[wsf][key]['dsolver'][k0] = dout[k0]

    else:
        for k0, k1 in dk_out.items():
            coll._dobj[wsf][key]['dsolver'][k0] = k1

    # ---------
    # verb
    # ----------

    if verb >= 2:
        msg = f"Storing under key '{ksol}'"
        print(msg)

    return