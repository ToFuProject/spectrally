# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:09:08 2024

@author: dvezinet
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as transforms
import datastock as ds


# local


#############################################
#############################################
#       DEFAULTS
#############################################


_MK = '.'
_MS = 6
_CM = plt.cm.Set2((np.arange(1, 9) - 1)/7)
_DPROP = {
    0: {'color': 'k', 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -1: {'color': _CM[0], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -2: {'color': _CM[1], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -3: {'color': _CM[2], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -4: {'color': _CM[3], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -5: {'color': _CM[4], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -6: {'color': _CM[5], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -7: {'color': _CM[6], 'ls': 'None', 'marker': _MK, 'ms': _MS},
    -8: {'color': _CM[7], 'ls': 'None', 'marker': _MK, 'ms': _MS},
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
    vmin=None,
    vmax=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
):

    # -----------------
    # check
    # -----------------

    key, dprop, vmin, vmax, tit = _check(
        coll=coll,
        key=key,
        # options
        dprop=dprop,
        vmin=vmin,
        vmax=vmax,
        # figure
        tit=tit,
    )

    wsf = coll._which_fit
    data = coll.ddata[coll.dobj[wsf][key]['key_data']]['data']
    ndim = data.ndim
    key_bs = coll.dobj[wsf][key]['key_bs']

    # -----------------
    # plot
    # -----------------

    if ndim == 1:
        assert key_bs is None

        dax = _plot_1d(
            coll=coll,
            key=key,
            # options
            dprop=dprop,
            vmin=vmin,
            vmax=vmax,
            # figure
            dax=dax,
            fs=fs,
            dmargin=dmargin,
        )

    else:
        raise NotImplementedError()

    # -----------------
    # Complenent
    # -----------------

    fig = list(dax.values())[0]['handle'].figure
    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    return dax


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key=None,
    # options
    dprop=None,
    vmin=None,
    vmax=None,
    # figure
    tit=None,
):

    # -----------------
    # key
    # -----------------

    wsf = coll._which_fit
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=list(coll.dobj.get(wsf, {}).keys()),
    )

    # -----------------
    # dprop
    # -----------------

    if dprop is None:
        dprop = {}

    if not isinstance(dprop, dict):
        msg = "Arg dprop must be a dict"
        raise Exception(msg)

    lk = sorted(coll.dobj[wsf][key]['dvalid']['meaning'].keys())
    for k0 in lk:

        if dprop.get(k0) is None:
            dprop[k0] = {}

        for k1, v1 in _DPROP[k0].items():
            if dprop[k0].get(k1) is None:
                dprop[k0][k1] = _DPROP[k0][k1]

    # -----------------
    # vmin, vmax
    # -----------------

    # vmin
    vmin = float(ds._generic_check._check_var(
        vmin, 'vmin',
        types=(int, float),
        default=0,
    ))

    # vmax
    key_data = coll.dobj[wsf][key]['key_data']
    vmax_def = np.nanmax(coll.ddata[key_data]['data']) * 1.05
    vmax = float(ds._generic_check._check_var(
        vmax, 'vmax',
        types=(int, float),
        default=vmax_def,
    ))

    # -----------------
    # figure
    # -----------------

    tit_def = f"input data validity for {wsf} '{key}'"
    tit = ds._generic_check._check_var(
        tit_def, 'tit_def',
        types=(str, bool),
        default=tit_def,
    )

    return key, dprop, vmin, vmax, tit


#############################################
#############################################
#       plot 1d
#############################################


def _plot_1d(
    coll=None,
    key=None,
    # options
    dprop=None,
    vmin=None,
    vmax=None,
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

    dvalid = coll.dobj[wsf][key]['dvalid']
    iok = coll.ddata[dvalid['iok']]['data']
    frac = dvalid['frac'][0]

    # -----------------
    # prepare figure
    # -----------------

    if dax is None:
        dax = _get_dax_1d(
            fs=fs,
            dmargin=dmargin,
            # labelling
            coll=coll,
            key=key,
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
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

        # frac
        ax.set_title(
            f"frac = {frac:.3f} vs {dvalid['fraction']}",
            size=12,
            fontweight='bold',
        )

        # nsigma
        ax.axhline(dvalid['nsigma']**2, ls='--', c='k')

        trans = transforms.blended_transform_factory(
            ax.transAxes,
            ax.transData,
        )
        ax.text(
            1.02,
            dvalid['nsigma']**2,
            r'$n_{\sigma}^2$',
            size=12,
            fontweight='normal',
            transform=trans,
        )

        # vmin vmax
        if vmin is not None:
            ax.set_ylim(bottom=vmin)
        if vmax is not None:
            ax.set_ylim(top=vmax)

    return dax


# ---------------------
# create axes
# ---------------------


def _get_dax_1d(
    fs=None,
    dmargin=None,
    # labelling
    coll=None,
    key=None,
):

    # ---------------
    # check
    # ---------------

    if fs is None:
        fs = (11, 6)

    if dmargin is None:
        dmargin = {
            'left': 0.07, 'right': 0.78,
            'bottom': 0.08, 'top': 0.90,
            'wspace': 0.10, 'hspace': 0.10,
        }

    # ---------------
    # prepare labels
    # ---------------

    wsf = coll._which_fit
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    key_data = coll.dobj[wsf][key]['key_data']
    xlab = f"{key_lamb} ({coll.ddata[key_lamb]['units']})"
    ylab = f"{key_data} ({coll.ddata[key_data]['units']})"

    # ---------------
    # figure
    # ---------------

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)

    # ---------------
    # axes
    # ---------------

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(xlab, size=12, fontweight='bold')
    ax.set_ylabel(ylab, size=12, fontweight='bold')

    dax = {'data': {'handle': ax}}

    return dax


#############################################
#############################################
#       plot 2d
#############################################
