# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import datastock as ds


#############################################
#############################################
#       DEFAULTS
#############################################


_DMODEL = {
    'linear': {'var': ['c0', 'c1']},
    'exp': {'var': ['amp', 'rate']},
    'gauss': {'var': ['amp', 'x0', 'width']},
    'lorentz': {'var': ['amp', 'x0', 'gamma']},
    'pvoigt': {'var': ['amp', 'x0', 'width', 't', 'gamma']},
    'voigt': {'var': ['amp', 'x0', 'width', 'gamma']},
}
_LMODEL_ORDER = ['linear', 'exp', 'gauss', 'lorentz', 'pvoigt', 'voigt']


#############################################
#############################################
#       MODEL CHECK
#############################################


def _dmodel(
    coll=None,
    key=None,
    dmodel=None,
):

    # ----------
    # key
    # ----------

    wsm = coll._which_model
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get(wsm, {}),
        short='sm',
        key=key,
        ndigits=2,
    )

    # --------------
    # check dmodel
    # --------------

    dmodel = _check_dmodel(
        coll=coll,
        key=key,
        dmodel=dmodel,
    )

    # --------------
    # store
    # --------------

    dobj = {
        wsm: {
            key: {
                'keys': sorted(dmodel.keys()),
                'dmodel': dmodel,
                'dconstraints': None,
            },
        },
    }

    coll.update(dobj=dobj)

    return


#############################################
#############################################
#       check dmodel
#############################################


def _dmodel_err(key, dmodel):
    return (
        f"For model '{key}' dmodel must be  dict of the form:\n"
        "\t'bck': 'exp',\n"
        "\t'l0': 'gauss',\n"
        "\t'l1': 'lorentz',\n"
        "\t...: ...,\n"
        "\t'ln': 'pvoigt',\n"
        f"Provided:\n{dmodel}"
    )


def _check_dmodel(
    coll=None,
    key=None,
    dmodel=None,
):

    # -------------
    # model
    # -------------

    if isinstance(dmodel, str):
        dmodel = [dmodel]

    if isinstance(dmodel, (tuple, list)):
        dmodel = {ii: mm for ii, mm in enumerate(dmodel)}

    if not isinstance(dmodel, dict):
        raise Exception(_dmodel_err(key, dmodel))

    # ------------
    # check dict
    # ------------

    dmod2 = {}
    dout = {}
    ibck, il = 0, 0
    for k0, v0 in dmodel.items():

        # ----------
        # check str vs dict

        if isinstance(v0, dict):
            if isinstance(v0.get('type'), str):
                typ = v0['type']
            else:
                dout[k0] = v0
                continue
        elif isinstance(v0, str):
            typ = v0
        else:
            dout[k0] = v0
            continue

        # ----------
        # check type

        if typ not in _DMODEL.keys():
            dout[k0] = v0
            continue

        # ----------
        # check key

        if isinstance(k0, int):
            if typ in ['linear', 'exp']:
                k1 = f'bck{ibck}'
                ibck += 1
            else:
                k1 = f"l{il}"
                il += 1
        else:
            k1 = k0

        dmod2[k1] = {'type': typ, 'var': _DMODEL[typ]['var']}

    # ---------------
    # raise error
    # ---------------

    if len(dout) > 0:
        raise Exception(_dmodel_err(key, dout))

    return dmod2


#############################################
#############################################
#       Get variables
#############################################


def _get_var(
    coll=None,
    key=None,
    all_free_tied=None,
    concatenate=None,
):

    # --------------
    # check key
    # -------------

    # key
    wsm = coll._which_model
    lok = list(coll.dobj.get(wsm, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # keys and dmodel
    keys = coll.dobj[wsm][key]['keys']
    dmodel = coll.dobj[wsm][key]['dmodel']

    # free_only
    all_free_tied = ds._generic_check._check_var(
        all_free_tied, 'all_free_tied',
        types=str,
        default='all',
        allowed=['all', 'free', 'tied'],
    )

    # concatenate
    concatenate = ds._generic_check._check_var(
        concatenate, 'concatenate',
        types=bool,
        default=True,
    )

    # -------------
    # get lvar
    # -------------

    # all variables
    if all_free_tied == 'all':
        lvar = [
            [f"{k0}_{k1}" for k1 in dmodel[k0]['var']]
            for k0 in keys
        ]

    # only free or only tied variables
    else:
        dconstraints = coll.dobj[wsm][key]['dconstraints']
        lref = [v0['ref'] for v0 in dconstraints['dconst'].values()]

        # lvar
        lvar = [
            [
                f"{k0}_{k1}" for k1 in dmodel[k0]['var']
                if (
                        all_free_tied == 'free'
                        and f"{k0}_{k1}" in lref
                    )
                or (
                    all_free_tied == 'tied'
                    and f"{k0}_{k1}" not in lref
                )
            ]
            for k0 in keys
        ]

    # ----------------
    # concatenate
    # ----------------

    if concatenate is True:
        lvar = list(itt.chain.from_iterable(lvar))

    return lvar


#############################################
#############################################
#       Get variables dind
#############################################


def _get_var_dind(
    coll=None,
    key=None,
):

    # --------------
    # check key
    # -------------

    # key
    wsm = coll._which_model
    lok = list(coll.dobj.get(wsm, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # keys and dmodel
    keys = coll.dobj[wsm][key]['keys']
    dmodel = coll.dobj[wsm][key]['dmodel']

    # -------------
    # get lvar
    # -------------

    x_all = coll.get_spectral_model_variables(key, all_free_tied='all')

    # ---------------
    # derive dind
    # ---------------

    types = sorted(set([v0['type'] for v0 in dmodel.values()]))

    dind = {}
    for ktype in types:

        # list functions with corresponding model type
        lf = [k0 for k0 in keys if dmodel[k0]['type'] == ktype]

        # populate
        dind[ktype] = {
            k1: {
                'ind': np.array([x_all.index(f"{kf}_{k1}") for kf in lf]),
                'keys': [f"{kf}_{k1}" for kf in lf],
            }
            for k1 in dmodel[lf[0]]['var']
        }

    # ---------------
    # safety checks
    # ---------------

    # aggregate all variables
    lvar = tuple(itt.chain.from_iterable([
        list(itt.chain.from_iterable([
            v1['keys']
            for k1, v1 in vtype.items()
        ]))
        for ktype, vtype in dind.items()
    ]))

    # check all indices are unique
    lind = tuple(itt.chain.from_iterable([
        list(itt.chain.from_iterable([
            v1['ind']
            for k1, v1 in vtype.items()
        ]))
        for ktype, vtype in dind.items()
    ]))

    # check all variable are represented
    nn = len(x_all)
    c0 = (
        (tuple(sorted(lvar)) == tuple(sorted(x_all)))
        and np.allclose(sorted(lind), np.arange(nn))
        and (tuple([lvar[ii] for ii in np.argsort(lind)]) == tuple(x_all))
    )
    if not c0:
        msg = (
            "dind corrupted!\n"
            f"\t- x_all: {x_all}\n"
            f"\t- lvar: {lvar}\n"
            f"\t- lind: {lind}\n"
            f"\ndind:\n{dind}\n"
        )
        raise Exception(msg)

    return dind


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([which] + _LMODEL_ORDER + ['constraints'])

    # ---------------------------
    # data
    # ---------------------------

    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    lar0 = []
    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # add nb of func of each type
        dmod = coll.dobj[which][k0]['dmodel']
        for k1 in _LMODEL_ORDER:
            nn = str(len([k2 for k2, v2 in dmod.items() if v2['type'] == k1]))
            arr.append(nn)

        # add nb of constraints
        dconst = coll.dobj[which][k0]['dconstraints']['dconst']
        nn = str(len([k1 for k1, v1 in dconst.items() if len(v1) > 1]))
        arr.append(nn)

        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar


# =============================================================================
# def _initial_from_from_model(
#     coll=None,
#     key=None,
#     dmodel=None,
# ):
#
#     # -----------
#     # initialize
#     # ----------
#
#     wsl = coll._which_lines
#
#     dinit = {}
#     for k0, v0 in dmodel.items():
#
#         # -----
#         # bck
#
#         if v0['type'] == 'linear':
#             dinit[k0] = {'c0': 0, 'c1': 0}
#
#         elif v0['type'] == 'exp':
#             dinit[k0] = {'c0': 0, 'c1': 0}
#
#         else:
#
#             # if from spectral lines
#             if k0 in coll.dobj.get(wsl, {}).keys():
#                 lamb0 = coll.dobj[wsl][k0]['lamb0']
#             else:
#                 lamb0 = 0
#
#             if v0['type'] == 'gauss':
#                 dinit[k0] = {
#                     'amp': 1,
#                     'shift': lamb0,
#                     'width': 1,
#                 }
#
#             elif v0['type'] == 'lorentz':
#                 dinit[k0] = np.r_[0, lamb0, 0]
#
#             elif v0['type'] == 'pvoigt':
#                 dinit[k0] = np.r_[0, lamb0, 0]
#
#
#     # -------------
#     # lines
#     # -------------
#
#
#
#
#
#     coll.add_spectral_model(key=key, dmodel=dmodel)
#     coll.set_spectral_model_initial(key=key, initial=initial)
#
#     return
# =============================================================================