# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import scipy.constants as scpct
import datastock as ds


# local
from . import _class01_model_dict as _model_dict


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
    if knfunc not in coll.dref.keys():
        coll.add_ref(knfunc, size=len(dmodel))
    elif coll.dref[knfunc]['size'] != len(dmodel):
        msg = (
            f"add_model('{key}') requires adding ref knfunc = '{knfunc}'"
            " but it already exists and doesn't have the correct value!\n"
            f"\t- existing: {coll.dref[knfunc]['size']}\n"
            f"\t- requested: {len(dmodel)}\n"
        )
        raise Exception(msg)

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
    for ii, (k0, v0) in enumerate(_model_dict._DMODEL.items()):
        if v0.get('param') is None:
            stri = f"\t- 'f{ii}': '{k0}'"
        else:
            lpar = v0['param']
            pstr = ", ".join([f"'{tpar[0]}': {tpar[1]}" for tpar in lpar])
            stri = f"\t- 'f{ii}': " + "{" + f"'type': '{k0}', {pstr}" + "}"

        if k0 == 'poly':
            lstr.append("\t# background-oriented")
        elif k0 == 'gauss':
            lstr.append("\t# spectral lines-oriented")
        elif k0 == 'pulse_exp':
            lstr.append("\t# pulse-oriented")
        lstr.append(stri)

    # Provided
    if isinstance(dmodel, dict):
        prov = "\n".join([f"\t'{k0}': {v0}," for k0, v0 in dmodel.items()])
        prov = "{\n" + prov + "\n}"
    else:
        prov = str(dmodel)

    # concatenate msg
    return (
        f"For model '{key}' dmodel must be a dict of the form:\n"
         + "\n".join(lstr)
         + f"\n\nProvided:\n{prov}"
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

        if typ not in _model_dict._DMODEL.keys():
            dout[k0] = v0
            continue

        # ----------
        # check key

        if isinstance(k0, int):
            if typ in ['poly', 'exp']:
                k1 = f'bck{ibck}'
                ibck += 1
            else:
                k1 = f"l{il}"
                il += 1
        else:
            k1 = k0

        # ---------------------------
        # check parameter (if needed)

        haspar = _model_dict._DMODEL[typ].get('param') is not None
        if haspar is True:

            lpar = _model_dict._DMODEL[typ]['param']

            # loop on parameters
            dpar = {}
            for tpar in lpar:

                # provided
                c0 = (
                    isinstance(v0, dict)
                    and isinstance(v0.get(tpar[0]), tpar[1])
                )
                if c0:
                    dpar[tpar[0]] = v0[tpar[0]]

                elif tpar[0] in ('lamb0', 'mz'):

                    # check if lamb0 can be extracted from existing lines
                    c1 = (
                        typ in ['gauss', 'lorentz', 'pvoigt', 'voigt']
                        and k1 in coll.dobj.get(wsl, {}).keys()
                    )
                    if c1 and tpar[0] == 'lamb0':
                        dpar[tpar[0]] = coll.dobj[wsl][k1]['lamb0']

                    elif c1 and tpar[0] == 'mz':
                        kion = coll.dobj[wsl][k1]['ion']
                        dpar[tpar[0]] = coll.dobj['ion'][kion]['A'] * scpct.m_u

                    elif len(tpar) == 3:
                        dpar[tpar[0]] = tpar[2]
                    else:
                        dout[k0] = v0
                        continue

                elif len(tpar) == 3:
                    dpar[tpar[0]] = tpar[2]

                else:
                    dout[k0] = v0
                    continue

        # ----------------
        # assemble

        dmod2[k1] = {
            'type': typ,
            'var': _model_dict._DMODEL[typ]['var'],
        }

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
            [
                f"{k0}_{tpar[0]}"
                for tpar in _model_dict._DMODEL[dmodel[k0]['type']]['param']
            ]
            for k0 in keys if dmodel[k0].get('param') is not None
        ]

    if 'param_value' in returnas:
        dout['param_value'] = [
            [
                dmodel[k0]['param'][tpar[0]]
                for tpar in _model_dict._DMODEL[dmodel[k0]['type']]['param']
            ]
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
        returnas=['all', 'free', 'param_key', 'param_value'],
        concatenate=True,
    )
    x_all = dout['all']
    x_free = dout['free']
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

    # ----------------
    # add func
    # ----------------

    dind['func'] = {}
    for ktype in types:

        # list functions with corresponding model type
        lf = [k0 for k0 in keys if dmodel[k0]['type'] == ktype]

        # get indices
        ind = [keys.index(ff) for ff in lf]

        # store
        dind['func'][ktype] = {
            'keys': lf,
            'ind': np.array(ind, dtype=int),
        }

    #-------------------------
    # add total number of func

    dind['nfunc'] = len(keys)

    # ----------------
    # add jac
    # ----------------

    dind['jac'] = {ktype: {} for ktype in types}
    for ktype in types:
        lf = [k0 for k0 in keys if dmodel[k0]['type'] == ktype]
        for kvar in _model_dict._DMODEL[ktype]['var']:
            lvar = [f"{kf}_{kvar}" for kf in lf]
            lind = [ii for ii, vv in enumerate(lvar) if vv in x_free]

            inds = np.array(
                [x_free.index(lvar[ii]) for ii in lind],
                dtype=int,
            )

            if inds.size > 0:
                dind['jac'][ktype][kvar] = {
                    'val': inds,
                    'var': np.array(lind, dtype=int),
                }

    # ---------------
    # safety checks
    # ---------------

    lind = sorted(itt.chain.from_iterable([
        list(itt.chain.from_iterable([v1['val'].tolist() for v1 in v0.values()]))
        for v0 in dind['jac'].values()
    ]))
    if not np.allclose(lind, np.arange(len(x_free))):
        msg = (
            "Something wrong with dind['jac'] !\n"
            f"\t- lind = {lind}\n"
            f"\t- dind['jac']:\n{dind['jac']}\n"
        )
        raise Exception(msg)

    return dind