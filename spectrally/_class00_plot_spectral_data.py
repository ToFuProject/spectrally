# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:09:08 2024

@author: dvezinet
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
import datastock as ds


from . import _class01_plot


# ###############################################################
# ###############################################################
#               Main
# ###############################################################


def main(
    coll=None,
    key_data=None,
    key_lamb=None,
    keyY=None,
    key_lines=None,
    # plotting
    dprop=None,
    dvminmax=None,
    # lines labels
    lines_labels=True,
    lines_labels_color=None,
    lines_labels_rotation=None,
    lines_labels_horizontalalignment=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    # interactivity
    nmax=None,
    connect=None,
    dinc=None,
    show_commands=None,
):

    # -------------------
    # check
    # -------------------

    (
        key_data, key_lines,
        lines_labels, dlabels, lines_labels_rotation,
        lines_labels_horizontalalignment,
        connect,
    ) = _check(
        coll=coll,
        key_data=key_data,
        key_lines=key_lines,
        # plotting
        lines_labels=lines_labels,
        lines_labels_color=lines_labels_color,
        lines_labels_rotation=lines_labels_rotation,
        lines_labels_horizontalalignment=lines_labels_horizontalalignment,
    )

    # -------------------
    # prepare figure
    # -------------------

    ndim = coll.ddata[key_data]['data'].ndim
    if dax is None:
        dax = _get_dax(
            ndim=ndim,# resources
            coll=coll,
            key_data=key_data,
            key_lamb=key_lamb,
            fs=fs,
            dmargin=dmargin,
            tit=tit,
        )

    dax = ds._generic_check._check_dax(dax)

    # -------------------
    # plot
    # -------------------

    if ndim == 1:
        dax = _plot_1d(
            coll=coll,
            key_data=key_data,
            key_lamb=key_lamb,
            dax=dax,
            # lines labels
            dlabels=dlabels,
            lines_labels=lines_labels,
            lines_labels_color=lines_labels_color,
            lines_labels_rotation=lines_labels_rotation,
            lines_labels_horizontalalignment=lines_labels_horizontalalignment,
        )

    elif ndim == 2:
        dax, dgroup = _plot_2d(
            coll=coll,
            key_data=key_data,
            key_lamb=key_lamb,
            keyY=keyY,
            dax=dax,
            dvminmax=dvminmax,
            # lines labels
            dlabels=dlabels,
            lines_labels=lines_labels,
            lines_labels_color=lines_labels_color,
            lines_labels_rotation=lines_labels_rotation,
            lines_labels_horizontalalignment=lines_labels_horizontalalignment,
        )

    # -------------------
    # finalize
    # -------------------

    _finalize_figure(
        dax=dax,
        key_data=key_data,
        tit=tit,
    )

    # ---------------------
    # connect interactivity
    # ---------------------

    if isinstance(dax, dict):
        return dax
    else:
        if connect is True:
            dax.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
            dax.disconnect_old()
            dax.connect()

            dax.show_commands(verb=show_commands)
            return dax
        else:
            return dax, dgroup


# ###############################################################
# ###############################################################
#               check
# ###############################################################


def _check(
    coll=None,
    key_data=None,
    key_lines=None,
    # lines labels
    lines_labels=None,
    lines_labels_color=None,
    lines_labels_rotation=None,
    lines_labels_horizontalalignment=None,
    # interactivity
    nmax=None,
    connect=None,
):

    # -------------
    # key_data
    # -------------

    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['data'].ndim in [1, 2]
    ]
    key_data = ds._generic_check._check_var(
        key_data, 'key_data',
        types=str,
        allowed=lok,
        extra_msg="plot_spectral_data() only implemented for 1d and 2d data!"
    )

    # -------------
    # key lines
    # -------------

    if isinstance(key_lines, str):
        key_lines = [key_lines]

    wsl = coll._which_lines
    lok = list(coll.dobj.get(wsl, {}).keys())
    key_lines = ds._generic_check._check_var_iter(
        key_lines, 'key_lines',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
        extra_msg="Arg key_lines must be a list of known spectral lines!"
    )

    # -------------
    # lines_labels
    # -------------

    if lines_labels in [False, True]:
        labs = None
    else:
        labs = lines_labels

    dlabels = coll.get_spectral_lines_labels(
        keys=key_lines,
        labels=labs,
        colors=lines_labels_color,
    )

    # ---------------------
    # lines_labels_rotation
    # ---------------------

    lines_labels_rotation = float(ds._generic_check._check_var(
        lines_labels_rotation, 'lines_labels_rotation',
        types=(float, int),
        default=45,
    ))

    # ---------------------
    # lines_labels_horizontalalignment
    # ---------------------

    lines_labels_horizontalalignment = ds._generic_check._check_var(
        lines_labels_horizontalalignment, 'lines_labels_horizontalalignment',
        types=str,
        default='left',
        allowed=['left', 'center', 'right'],
    )

    # -------------
    # connect
    # -------------

    connect = ds._generic_check._check_var(
        connect, 'connect',
        types=bool,
        default=True,
    )

    return (
        key_data, key_lines,
        lines_labels, dlabels, lines_labels_rotation,
        lines_labels_horizontalalignment,
        connect,
    )


# ###############################################################
# ###############################################################
#               plot 1d
# ###############################################################


def _plot_1d(
    coll2=None,
    dout=None,
    dkeys=None,
    dax=None,
    details=None,
    # lines labels
    dlabels=None,
    lines_labels=None,
    lines_labels_color=None,
    lines_labels_rotation=None,
    lines_labels_horizontalalignment=None,
):

    # ------------
    # plot spectrum
    # -----------

    axtype = 'spectrum'
    lax = [kax for kax, vax in dax.items() if axtype in vax['type']]
    for kax in lax:
        ax = dax[kax]['handle']

        # plot fit
        ll, = ax.plot(
            coll2.ddata[dkeys['lamb']]['data'],
            coll2.ddata[dkeys['sum']]['data'],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
        )

        # plot data
        ax.plot(
            coll2.ddata[dkeys['lamb']]['data'],
            coll2.ddata[dkeys['data']]['data'],
            ls='None',
            marker='.',
            lw=1.,
            color='k',
        )

        # details
        if details is True:
            for ff in dkeys['details']:
                ax.plot(
                    coll2.ddata[dkeys['lamb']]['data'],
                    coll2.ddata[ff]['data'],
                    ls='-',
                    marker='None',
                    lw=1.,
                )

        # ------------
        # plot std
        # -----------

        if dkeys.get('sum_min') is not None:
            # plot fit
            ax.fill_between(
                coll2.ddata[dkeys['lamb']]['data'],
                coll2.ddata[dkeys['sum_min']]['data'],
                coll2.ddata[dkeys['sum_max']]['data'],
                ec='None',
                lw=0.,
                fc=ll.get_color(),
                alpha=0.3,
            )

    # ------------
    # plot diff
    # -----------

    axtype = 'diff'
    lax = [kax for kax, vax in dax.items() if axtype in vax['type']]
    for kax in lax:
        ax = dax[kax]['handle']

        # plot fit
        ax.plot(
            coll2.ddata[dkeys['lamb']]['data'],
            coll2.ddata[dkeys['data']]['data'] - coll2.ddata[dkeys['sum']]['data'],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
        )

    # ----------------
    # add lines_labels
    # ----------------

    if lines_labels is not False:
        raise NotImplementedError()

    return dax


# ###############################################################
# ###############################################################
#               plot 2d
# ###############################################################


def _plot_2d(
    coll=None,
    key_data=None,
    key_lamb=None,
    keyY=None,
    dax=None,
    dvminmax=None,
    # lines labels
    dlabels=None,
    lines_labels=None,
    lines_labels_color=None,
    lines_labels_rotation=None,
    lines_labels_horizontalalignment=None,
):

    # --------------
    # plot data
    # --------------

    lax = ['2d_data', 'spectrum', 'vert']
    coll2, dgroup0 = coll.plot_as_array(
        key=key_data,
        keyX=key_lamb,
        keyY=keyY,
        dax={k0: v0 for k0, v0 in dax.items() if k0 in lax},
        aspect='auto',
        dvminmax=dvminmax,
        connect=False,
        inplace=False,
    )

    # ----------------------------------
    # needed bot for details and labels
    # ----------------------------------

    reflamb = coll2.ddata[key_lamb]['ref'][0]
    refs = coll2.ddata[key_data]['ref']
    axis = refs.index(reflamb)

    # ----------------
    # add lines_labels
    # ----------------

    if lines_labels is not False:

        _add_labels(
            # resources
            coll=coll,
            coll2=coll2,
            axis=axis,
            # parameters
            kax='spectrum',
            kax_labels='labels',
            dax=dax,
            dlabels=dlabels,
            lines_labels_color=lines_labels_color,
            lines_labels_rotation=lines_labels_rotation,
            lines_labels_horizontalalignment=lines_labels_horizontalalignment,
        )

    return coll2, dgroup0


# ###############################################################
# ###############################################################
#               labels
# ###############################################################


def _add_labels(
    # resources
    coll=None,
    coll2=None,
    axis=None,
    # parameters
    kax=None,
    kax_labels=None,
    dax=None,
    dlabels=None,
    lines_labels_color=None,
    lines_labels_rotation=None,
    lines_labels_horizontalalignment=None,
):

    # -----------------------
    # add ax_labels to coll2

    coll2.add_axes(key=kax_labels, harmonize=False, **dax[kax_labels])
    ax_labels = coll2.dax[kax_labels]['handle']
    ax = coll2.dax[kax]['handle']

    # -------------
    # prepare data

    # blended transform
    trans = mtransforms.blended_transform_factory(
        ax_labels.transData,
        ax_labels.transAxes,
    )

    # loop on functions / spectral lines
    wsl = coll._which_lines
    for k0, v0 in dlabels.items():

        # argmax
        lamb0 = coll.dobj[wsl][k0]['lamb0']

        # ------------------
        # add vertical lines

        # plot
        ll = ax.axvline(
            lamb0,
            ls='--',
            marker='None',
            lw=1.,
            c=v0['color'],
        )

        # ---------------------
        # add labels above axes

        ll = ax_labels.text(
            lamb0,
            0.05,
            v0['label'],
            size=12,
            color=ll.get_color(),
            rotation=lines_labels_rotation,
            horizontalalignment=lines_labels_horizontalalignment,
            verticalalignment='bottom',
            transform=trans,
        )

    return


# ###############################################################
# ###############################################################
#               dax
# ###############################################################


def _get_dax(
    ndim=None,
    # resources
    coll=None,
    key_data=None,
    key_fit=None,
    key_lamb=None,
    # figure
    fs=None,
    dmargin=None,
    tit=None,
):

    if ndim == 1:
        return _get_dax_1d(
            fs=fs,
            dmargin=dmargin,
            tit=tit,
        )

    if ndim == 2:
        return _get_dax_2d(
            coll=coll,
            key_data=key_data,
            key_fit=key_fit,
            key_lamb=key_lamb,
            # options
            fs=fs,
            dmargin=dmargin,
            tit=tit,
        )


def _get_dax_1d(
    fs=None,
    dmargin=None,
    tit=None,
):

    # ---------------
    # check inputs
    # ---------------

    if fs is None:
        fs = (10, 6)

    if dmargin is None:
        dmargin = {
            'left': 0.10, 'right': 0.90,
            'bottom': 0.1, 'top': 0.90,
            'wspace': 0.1, 'hspace': 0.1,
        }

    # ---------------
    # prepare figure
    # ---------------

    dax = {}
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(3, 1, **dmargin)

    # ------------
    # add axes
    # ------------

    # spectrum
    ax = fig.add_subplot(gs[:2, 0])
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['1d'] = {'handle': ax, 'type': 'spectrum'}

    ax = fig.add_subplot(gs[2, 0])
    dax['diff'] = {'handle': ax, 'type': 'diff'}

    return dax


def _get_dax_2d(
    coll=None,
    key_data=None,
    key_fit=None,
    key_lamb=None,
    # options
    fs=None,
    dmargin=None,
    tit=None,
):

    # ---------------
    # check inputs
    # ---------------

    if fs is None:
        fs = (19, 10)

    if dmargin is None:
        dmargin = {
            'left': 0.03, 'right': 0.99,
            'bottom': 0.06, 'top': 0.90,
            'wspace': 0.50, 'hspace': 0.30,
        }

    # ---------------
    # prepare figure
    # ---------------

    data_lab = f"{key_data} ({coll.ddata[key_data]['units']})"
    lamb_lab = f"{key_lamb} ({coll.ddata[key_lamb]['units']})"

    # ---------------
    # prepare figure
    # ---------------

    dax = {}
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(6, 9, **dmargin)

    # ------------
    # add axes
    # ------------

    # --------
    # images

    ax = fig.add_subplot(gs[:, 1:5])
    ax.set_title(f'data\n{key_data}', size=12, fontweight='bold')
    # ax.set_ylabel()
    ax0 = ax
    dax['2d_data'] = {'handle': ax, 'type': 'matrix'}

    # --------
    # vertical

    ax = fig.add_subplot(gs[:, 0], sharey=ax0)
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['vert'] = {'handle': ax, 'type': 'vertical'}

    # ----------
    # spectrum

    ax = fig.add_subplot(gs[1:-1, 5:], sharex=ax0)
    ax.set_xlabel(lamb_lab, size=12, fontweight='bold')
    ax.set_ylabel(data_lab, size=12, fontweight='bold')
    dax['spectrum'] = {'handle': ax, 'type': 'horizontal'}

    # ----------
    # spectrum labels

    pos_1d = dax['spectrum']['handle'].get_position()
    pos_2d = ax0.get_position()
    y0 = pos_1d.y0 + pos_1d.height
    height = pos_2d.y1 - y0

    ax = fig.add_axes([pos_1d.x0, y0, pos_1d.width, height], sharex=ax0)
    ax.set_ylim(0, 1)
    ax.axis('off')
    dax['labels'] = {'handle': ax, 'type': 'text'}

    return dax



# ###############################################################
# ###############################################################
#               Finalize figure
# ###############################################################


def _finalize_figure(dax=None, key_data=None, tit=None):

    # -------------
    # tit
    # -------------

    titdef = (
        f"Spectral data '{key_data}'"
    )
    tit = ds._generic_check._check_var(
        tit, 'tit',
        types=str,
        default=titdef,
    )

    # -------------
    # tit
    # -------------

    if isinstance(dax, dict):
        fig = list(dax.values())[0]['handle'].figure
    else:
        fig = list(dax.dax.values())[0]['handle'].figure

    if tit is not None:
        fig.suptitle(tit, size=12, fontweight='bold')

    return