# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:09:08 2024

@author: dvezinet
"""


import matplotlib.pyplot as plt
from matplotlib import gridspec
import datastock as ds


# local


#############################################
#############################################
#       DEFAULTS
#############################################


_DPROP = {
    0: {'color': 'k', 'ls': 'None', 'marker': 'o'},
}



#############################################
#############################################
#       main
#############################################


def plot(
    coll=None,
    key=None,
    # options
    dprop=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
):

    # -----------------
    # check
    # -----------------

    key = _check(
        coll=coll,
        key=key,
        # options
        dprop=dprop,
    )

    wsf = coll._which_fit
    data = coll.ddata[coll.dobj[wsf][key]['key_data']]['data']
    ndim = data.ndim
    key_bs = coll.dobj[wsf][key]['key_bs']

    # -----------------
    # check
    # -----------------

    if ndim == 1:
        assert key_bs is None

        _plot_1d(
            coll=coll,
            key=key,
            # options
            dprop=dprop,
            # figure
            dax=dax,
            fs=fs,
            dmargin=dmargin,
        )

    else:
        raise NotImplementedError()


    return


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key=None,
    # options
    dprop=None,
):

    # -----------------
    # key
    # -----------------

    wsf = coll._which_fit
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        lok=list(coll.dobj.get(wsf, {}).keys()),
    )

    # -----------------
    # dprop
    # -----------------

    if dprop is None:
        dprop = {}

    if not isinstance(dprop, dict):
        msg = "Arg dprop must be a dict"
        raise Exception(msg)

    lk = sorted()
    for k0 in lk:

        if dprop.get(k0) is None:
            dprop[k0] = {}

        for k1, v1 in _DPROP.items():
            if dprop[k0].get(k1) is None:
                dprop[k0][k1] = _DPROP[k0][k1]

    return key, dprop


#############################################
#############################################
#       plot 1d
#############################################


def _plot_1d(
    coll=None,
    key=None,
    # options
    dprop=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
):

    # -----------------
    # prepare
    # -----------------

    wsf = coll._which_fit
    lamb = coll.ddata[coll.dobj[wsf][key]['key_lamb']]['data']
    data = coll.ddata[coll.dobj[wsf][key]['key_data']]['data']

    dvalid = coll.dobj[wsf]['dvalid']
    iok = dvalid['iok']

    # -----------------
    # prepare figure
    # -----------------

    if dax is None:
        dax = _get_dax_1d(
            fs=fs,
            dmargin=dmargin,
        )

    dax = ds._generic_check._check_dax(dax=dax, main='data')

    # -----------------
    # plot
    # -----------------

    kax = 'data'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # validity
        for k0, v0 in dvalid['meaning'].items():

            ind = (iok == k0)
            ax.plot(
                lamb[ind],
                data[ind],
                label=v0,
                **dprop[k0],
            )

        # legend
        ax.legend(loc='upper left', bbx_to_anchor=(1, 1))

        # nsigma
        ax.axhline(dvalid['nsigma'], ls='--', c='k')

    return dax


def _get_dax_1d(
    fs=None,
    dmargin=None,
):


    # ---------------
    # check
    # ---------------

    if fs is None:
        fs = (8, 6)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.90,
            'bottom': 0.05, 'top': 0.90,
            'wspace': 0.10, 'hspace': 0.10,
        }

    # ---------------
    # figure
    # ---------------

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)

    # ---------------
    # axes
    # ---------------

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel()
    ax.set_ylabel()

    dax = {'data': {'handle': ax}}

    return dax


#############################################
#############################################
#       plot 2d
#############################################