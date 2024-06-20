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

    # ----------
    # background
    'linear': {'var': ['a0', 'a1']},
    'exp': {'var': ['amp', 'rate']},

    # --------------
    # spectral lines
    'gauss': {
        'var': ['amp', 'shift', 'width'],
        'param': [('lamb0', float)],
    },
    'lorentz': {
        'var': ['amp', 'shift', 'gamma'],
        'param': [('lamb0', float)],
    },
    'pvoigt': {
        'var': ['amp', 'shift', 'width', 't', 'gamma'],
        'param': [('lamb0', float)],
    },
    'voigt': {
        'var': ['amp', 'shift', 'width', 'gamma'],
        'param': [('lamb0', float)],
    },

    # -----------
    # pulse shape
    'pulse1': {'var': ['amp', 't0', 't_up', 't_down']},
    'pulse2': {'var': ['amp', 't0', 't_up', 't_down']},
    'lognorm': {'var': ['amp', 't0', 'mu', 'sigma']},
}


_LMODEL_ORDER = [
    # background
    'linear', 'exp',
    # spectral lines
    'gauss', 'lorentz', 'pvoigt', 'voigt',
    # pulse shape
    'pulse1', 'pulse2', 'lognorm',
]


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

    # add ref_nfun
    knfunc = f"nf_{key}"
    coll.add_ref(knfunc, size=len(dmodel))

    # dmodel
    dobj = {
        wsm: {
            key: {
                'keys': sorted(dmodel.keys()),
                'ref_nx': None,
                'ref_nf': knfunc,
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

    # prepare list of str
    lstr = []
    for ii, (k0, v0) in enumerate(_DMODEL.items()):
        if v0.get('param') is None:
            stri = f"\t- 'f{ii}': '{k0}'"
        else:
            lpar = v0['param']
            pstr = ", ".join([f"'{k1}': {v1}" for (k1, v1) in lpar])
            stri = f"\t- 'f{ii}': " + "{" + f"'type': '{k0}', {pstr}" + "}"

        if k0 == 'linear':
            lstr.append("\t# background-oriented")
        elif k0 == 'gauss':
            lstr.append("\t# spectral lines-oriented")
        elif k0 == 'pulse1':
            lstr.append("\t# pulse-oriented")
        lstr.append(stri)

    # concatenate msg
    return (
        f"For model '{key}' dmodel must be a dict of the form:\n"
         + "\n".join(lstr)
         + f"\n\nProvided:\n{dmodel}"
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

    # prepare for extracting lamb0
    wsl = coll._which_lines

    # ------------
    # check dict
    # ------------

    dmod2 = {}
    dout = {}
    ibck, il = 0, 0
    for k0, v0 in dmodel.items():

        # -----------------
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

        # ---------------------------
        # check parameter (if needed)

        haspar = _DMODEL[typ].get('param') is not None
        if haspar is True:

            lpar = _DMODEL[typ]['param']
            c0 = (
                isinstance(v0, dict)
                and all([
                    isinstance(v0.get(kpar), vpar)
                    for (kpar, vpar) in lpar
                ])
            )

            # all parameters properly defined
            if c0:
                dpar = {kpar: v0[kpar] for (kpar, vpar) in lpar}

            else:

                # check if lamb0 can be extracted from existing lines
                c1 = (
                    typ in ['gauss', 'lorentz', 'pvoigt', 'voigt']
                    and len(lpar) == 1
                    and k1 in coll.dobj.get(wsl, {}).keys()
                )
                if c1:
                    dpar = {'lamb0': coll.dobj[wsl][k1]['lamb0']}

                else:
                    dout[k0] = v0
                    continue

        # ----------------
        # assemble

        dmod2[k1] = {'type': typ, 'var': _DMODEL[typ]['var']}

        # add parameter
        if haspar is True:
            dmod2[k1]['param'] = dpar

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
    returnas=None,
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

    # returnas
    if isinstance(returnas, str):
        returnas = [returnas]
    returnas = ds._generic_check._check_var_iter(
        returnas, 'returnas',
        types=(list, tuple),
        types_iter=str,
        default=['all', 'param'],
        allowed=['all', 'free', 'tied', 'param_key', 'param_value'],
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

    dout = {}

    # ---------------
    # all variables

    if 'all' in returnas:
        dout['all'] = [
            [f"{k0}_{k1}" for k1 in dmodel[k0]['var']]
            for k0 in keys
        ]

    # -----------------------
    # free or tied variables

    if 'free' in returnas or 'tied' in returnas:
        dconstraints = coll.dobj[wsm][key]['dconstraints']
        lref = [v0['ref'] for v0 in dconstraints['dconst'].values()]

        # lvar
        if 'free' in returnas:
            dout['free'] = [
                [
                    f"{k0}_{k1}" for k1 in dmodel[k0]['var']
                    if f"{k0}_{k1}" in lref
                ]
                for k0 in keys
            ]

        if 'tied' in returnas:
            dout['tied'] = [
                [
                    f"{k0}_{k1}" for k1 in dmodel[k0]['var']
                    if f"{k0}_{k1}" not in lref
                ]
                for k0 in keys
            ]

    # ---------------
    # parameters

    if 'param_key' in returnas:
        dout['param_key'] = [
            [f"{k0}_{k1}" for (k1, v1) in _DMODEL[dmodel[k0]['type']]['param']]
            for k0 in keys if dmodel[k0].get('param') is not None
        ]

    if 'param_value' in returnas:
        dout['param_value'] = [
            [dmodel[k0]['param'][k1] for (k1, v1) in _DMODEL[dmodel[k0]['type']]['param']]
            for k0 in keys if dmodel[k0].get('param') is not None
        ]

    # ----------------
    # concatenate
    # ----------------

    if concatenate is True:
        for k0, v0 in dout.items():
            dout[k0] = list(itt.chain.from_iterable(v0))

            if k0 == 'param_value':
                dout[k0] = np.array(dout[k0])

    # ----------------
    # return
    # ----------------

    return dout


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
    # get lvar and param
    # -------------

    dout = coll.get_spectral_model_variables(
        key,
        returnas=['all', 'param_key', 'param_value'],
        concatenate=True,
    )
    x_all = dout['all']
    param_key = dout['param_key']

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

        # add param
        if dmodel[lf[0]].get('param') is not None:
            for kpar in dmodel[lf[0]]['param'].keys():
                dind[ktype][kpar] = [
                    param_key.index(f"{kf}_{kpar}")
                    for kf in lf
                ]

    # ---------------
    # safety checks
    # ---------------

    # aggregate all variables
    lvar = tuple(itt.chain.from_iterable([
        list(itt.chain.from_iterable([
            v1['keys']
            for k1, v1 in vtype.items()
            if not isinstance(v1, list)
        ]))
        for ktype, vtype in dind.items()
    ]))

    # check all indices are unique
    lind = tuple(itt.chain.from_iterable([
        list(itt.chain.from_iterable([
            v1['ind']
            for k1, v1 in vtype.items()
            if not isinstance(v1, list)
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


#############################################
#############################################
#       Show single model
#############################################


def _show_details(coll=None, key=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # get dmodel
    # ---------------------------

    wsm = coll._which_model
    dmodel = coll.dobj[wsm][key]['dmodel']
    dconst = coll.dobj[wsm][key]['dconstraints']['dconst']

    lkeys = coll.dobj[wsm][key]['keys']
    llvar = [dmodel[kf]['var'] for kf in lkeys]

    nvarmax = np.max([len(lvar) for lvar in llvar])
    lfree = coll.get_spectral_model_variables(key, returnas='free')['free']

    # ---------------------------
    # column names
    # ---------------------------

    lvar = [f"var{ii}" for ii in range(nvarmax)]
    lcol.append(['func', 'type', ' '] + lvar)

    # ---------------------------
    # data
    # ---------------------------

    lar0 = []
    for kf in lkeys:

        # initialize with key, type
        arr = [kf, dmodel[kf]['type'], '|']

        # add nb of func of each type
        for ii, k1 in enumerate(dmodel[kf]['var']):
            key = f"{kf}_{k1}"
            if key in lfree:
                nn = key
            else:
                gg = [kg for kg, vg in dconst.items() if key in vg.keys()][0]
                nn = f"{key}({dconst[gg]['ref']})"
            arr.append(nn)

        # complement
        arr += ['' for ii in range(nvarmax - ii - 1)]

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
