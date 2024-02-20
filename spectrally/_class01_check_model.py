# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import itertools as itt
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
}
_LMODEL_ORDER = ['linear', 'exp', 'gauss', 'lorentz', 'pvoigt']


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
        coll._which_model: {
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
    concatenate=None,
):

    # --------------
    # check key
    # -------------

    # key
    lok = list(coll.dobj.get(coll._which_model, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # keys and dmodel
    keys = coll.dobj[coll._which_model][key]['keys']
    dmodel = coll.dobj[coll._which_model][key]['dmodel']

    # concatenate
    concatenate = ds._generic_check._check_var(
        concatenate, 'concatenate',
        types=bool,
        default=True,
    )

    # -------------
    # get lvar
    # -------------

    lvar = [
        [f"{k0}_{k1}" for k1 in dmodel[k0]['var']]
        for k0 in keys
    ]

    if concatenate is True:
        lvar = list(itt.chain.from_iterable(lvar))

    return lvar


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

    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # add nb of func of each type
        dmod = coll.dobj[which][k0]['dmodel']
        for k1 in _LMODEL_ORDER:
            nn = len([k2 for k2, v2 in dmod.items() if v2['type'] == k1])
            arr.append(nn)

        # add nb of constraints
        dconst = coll.dobj[which][k0]['dconstraints']['dconst']
        nn = len([k1 for k1, v1 in dconst.items() if len(v1) > 1])
        arr.append(nn)

        lar.append(arr)

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
