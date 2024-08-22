# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:36:24 2024

@author: dvezinet
"""


# built-in
import itertools as itt


# common
import numpy as np
import datastock as ds


# local
from . import _class01_model_dict as _model_dict


__all__ = ['get_available_spectral_model_functions']


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # list of functions
    # ---------------------------

    # list of models
    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    # list of relevant functions
    lfunc = []
    for k0 in lkey:
        dmod = coll.dobj[which][k0]['dmodel']
        for k1 in _model_dict._LMODEL_ORDER:
            lk2 = [k2 for k2, v2 in dmod.items() if v2['type'] == k1]
            if len(lk2) > 0 and k1 not in lfunc:
                lfunc.append(k1)

    # reorder
    lfunc = [k1 for k1 in _model_dict._LMODEL_ORDER if k1 in lfunc]

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([which] + lfunc + ['constraints', 'free var'])

    # ---------------------------
    # data array
    # ---------------------------

    lar0 = []
    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # add nb of func of each type
        dmod = coll.dobj[which][k0]['dmodel']
        for k1 in lfunc:
            nn = str(len([k2 for k2, v2 in dmod.items() if v2['type'] == k1]))
            arr.append(nn)

        # add nb of constraints
        dconst = coll.dobj[which][k0]['dconstraints']['dconst']
        nn = str(len([k1 for k1, v1 in dconst.items() if len(v1) > 1]))
        arr.append(nn)

        # add number of free variables
        lfree = coll.get_spectral_model_variables(k0, returnas='free')['free']
        lall = coll.get_spectral_model_variables(k0, returnas='all')['all']
        arr.append(f"{len(lfree)} / {len(lall)}")

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

    lpar = sorted(set(itt.chain.from_iterable([
        v0.get('param', {}).keys() for v0 in dmodel.values()
    ])))

    # ---------------------------
    # column names
    # ---------------------------

    lvar = [f"var{ii}" for ii in range(nvarmax)]
    lcol.append(['func', 'type', ' '] + lvar + [' '] + lpar)

    # ---------------------------
    # data
    # ---------------------------

    lar0 = []
    for kf in lkeys:

        # initialize with key, type
        arr = [kf, dmodel[kf]['type'], '|']

        # add variables of each func
        for ii, k1 in enumerate(dmodel[kf]['var']):
            key = f"{kf}_{k1}"
            if key in lfree:
                nn = key
            else:
                gg = [kg for kg, vg in dconst.items() if key in vg.keys()][0]
                nn = f"{key}({dconst[gg]['ref']})"

            arr.append(nn)

        # complement
        arr += ['' for ii in range(nvarmax - ii - 1)] + ['|']

        # add parameters of each func
        for k1 in lpar:
            nn = dmodel[kf].get('param', {}).get(k1, '')
            if not isinstance(nn, str):
                nn = f"{nn:.6e}"
            arr.append(nn)

        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar


#############################################
#############################################
#       Show available models
#############################################


def get_available_spectral_model_functions(
    # print parameters
    sep=None,
    line=None,
    justify=None,
    table_sep=None,
    # bool options
    verb=None,
    returnas=None,
):
    """ Print all te available functions types for spectral fitting

    All arguments are optional
    >>> get_available_spectral_model_functions()


    Parameters
    ----------
    sep : str, optional
        seperator character
    line : str, optional
        line character
    justify : str, optional
        'left' or 'right'
    table_sep : str, optional
        separator character between tables
    verb : bool, optional
        Whether to print
    returnas : bool, optional
        Whether to return

    Returns
    -------

    May return the printed msg as str if requested

    """

    # ---------------------------
    # get dmodel
    # ---------------------------

    dmodel = _model_dict._DMODEL

    lkeys = _model_dict._LMODEL_ORDER
    llvar = [dmodel[kf]['var'] for kf in lkeys]

    nvarmax = np.max([len(lvar) for lvar in llvar])
    lpar = sorted(set(itt.chain.from_iterable([
        [kk[0] for kk in v0.get('param', [])] for v0 in dmodel.values()
    ])))

    # ---------------------------
    # column names
    # ---------------------------

    lvar = [f"var{ii}" for ii in range(nvarmax)]
    lcol = [
        ['use case', ' ', 'func type', ' ', 'description', ' ']
        + lvar + [' '] + lpar
    ]

    # ---------------------------
    # data
    # ---------------------------

    lar = []
    lar0 = []
    for kf in lkeys:

        # use case
        if kf == 'poly':
            use = 'bck'
        elif kf == 'gauss':
            use = 'lines'
        elif kf == 'pulse_exp':
            use = 'pulses'
        else:
            use = ''

        # initialize with key, type
        arr = [use, '|', kf, '|', dmodel[kf]['description'], '|']

        # add variables of each func
        for ii, k1 in enumerate(dmodel[kf]['var']):
            arr.append(k1)

        # complement
        arr += ['' for ii in range(nvarmax - ii - 1)] + ['|']

        # add parameters of each func
        for k1 in lpar:
            ln = [
                vv[1] for vv in dmodel[kf].get('param', [])
                if vv[0] == k1
            ]
            if len(ln) > 0:
                if hasattr(ln[0], '__name__'):
                    nn = ln[0].__name__
                else:
                    nn = f"{ln[0]}"
            else:
                nn = ""
            arr.append(nn)

        lar0.append(arr)

    lar.append(lar0)

    # ---------------------------
    # pretty print
    # ---------------------------

    return ds._generic_utils.pretty_print(
        headers=lcol,
        content=lar,
        sep=sep,
        line=line,
        justify=justify,
        table_sep=table_sep,
        verb=verb,
        returnas=returnas,
    )