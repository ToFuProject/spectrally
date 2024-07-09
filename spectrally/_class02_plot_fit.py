# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:09:08 2024

@author: dvezinet
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import datastock as ds


# local


#############################################
#############################################
#       DEFAULTS
#############################################





#############################################
#############################################
#       main
#############################################


def main(
    coll=None,
    key_model=None,
    key_data=None,
    lamb=None,
    # other dimensions
    keyY=None,
    dref_vectorY=None,
    # others
    details=None,
    # options
    dprop=None,
    vmin=None,
    vmax=None,
    cmap=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    # interactivity
    connect=True,
    dinc=None,
    show_commands=None,
):

    # -----------------
    # check
    # -----------------

    (
        key_model, ref_model, key_data, lamb,
        keyY, keyZ,
        details,
        dprop, vmin, vmax, tit,
    ) = _check(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
        # keyY
        keyY=keyY,
        dref_vectorY=dref_vectorY,
        # others
        details=details,
        # options
        dprop=dprop,
        vmin=vmin,
        vmax=vmax,
        # figure
        tit=tit,
    )

    data = coll.ddata[key_data]['data']
    ndim = data.ndim

    # -----------------
    # compute
    # -----------------


    # -----------------
    # plot
    # -----------------

    if ndim == 1:

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

    elif ndim == 2:

        dax, dgroup = _plot_2d(
            coll=coll,
            key=key,
            # keyY
            keyY=keyY,
            # bsplines
            key_bs=key_bs,
            key_bs_vect=key_bs_vect,
            # options
            dprop=dprop,
            dvminmax=None,
            cmap=cmap,
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

    # get figure handle
    if isinstance(dax, dict):
        fig = list(dax.values())[0]['handle'].figure
    else:
        fig = list(dax.dax.values())[0]['handle'].figure

    # add title
    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    # ----------------------
    # connect interactivity
    # ----------------------

    # ndim = 1 => no interactivity
    if ndim == 1:
        return dax

    # connect interactivity
    elif connect is True:
        dax.setup_interactivity(
            kinter='inter0', dgroup=dgroup, dinc=dinc,
        )
        dax.disconnect_old()
        dax.connect()

        dax.show_commands(verb=show_commands)
        return dax

    # interactivity not connected
    else:
        return dax, dgroup


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key_model=None,
    key_data=None,
    lamb=None,
    # keyY
    keyY=None,
    dref_vectorY=None,
    # keyZ
    keyZ=None,
    dref_vectorZ=None,
    # others
    details=None,
    # options
    dprop=None,
    vmin=None,
    vmax=None,
    # figure
    tit=None,
):

    # ----------
    # key_model
    # ----------

    wsm = coll._which_model
    key_model = ds._generic_check._check_var(
        key_model, 'key_model',
        types=str,
        allowed=list(coll.dobj.get(wsm, {}).keys()),
    )

    # derive x_free
    x_free = coll.get_spectral_model_variables(
        key_model,
        returnas='free',
        concatenate=True,
    )['free']

    # derive ref_model
    ref_model = coll.dobj[wsm][key_model]['ref']

    # ----------
    # key_data
    # ----------

    # list of acceptable values
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if ref_model in v0['ref']
    ]

    # check
    key_data = ds._generic_check._check_var(
        key_data, 'key_data',
        types=str,
        allowed=lok,
    )

    # derive ndim
    ndim = coll.ddata[key_data]['data'].ndim
    if ndim > 1:
        msg = "plot_spectral_model() not implemented for ndim > 1"
        raise NotImplementedError(msg)

    # -----------------
    # lamb
    # -----------------

    if isinstance(lamb, np.ndarray):
        c0 = (
            lamb.ndim == 1
            and np.all(np.isifinite(lamb))
        )
        if not c0:
            _err_lamb(lamb)

    elif isinstance(lamb, str):
        c0 = (
            lamb in coll.ddata.keys()
            and coll.ddata[lamb]['data'].ndim == 1
            and np.all(np.isifinite(coll.ddata[lamb]['data']))
        )
        if not c0:
            _err_lamb(lamb)

    else:
        _err_lamb(lamb)

    # -----------------
    # details
    # -----------------

    details = ds._generic_check._check_var(
        details, 'details',
        types=bool,
        default=True,
    )

    return (
        key_model, ref_model, key_data, lamb,
        keyY, keyZ,
        details,
        dprop, vmin, vmax, tit,
    )


def _err_lamb(lamb):
    msg = (
    )
    raise Exception(msg)


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

    wsm = coll._which_model
    if isinstance(lamb, str):
        lamb = coll.ddata[lamb]['data']
    data = coll.ddata[key_data]['data']

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
            r'$n_{\sigma}^2$' + f" = {dvalid['nsigma']}" + r"$^2$",
            size=12,
            fontweight='normal',
            transform=trans,
        )

        # focus
        if dvalid.get('focus') is not None:
            for ff in dvalid['focus']:
                ax.axvspan(ff[0], ff[1], fc=(0.8, 0.8, 0.8, 0.5))

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