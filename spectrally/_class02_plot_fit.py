# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:09:08 2024

@author: dvezinet
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datastock as ds


from . import _class01_plot


# ###############################################################
# ###############################################################
#               Main
# ###############################################################


def main(
    coll=None,
    key=None,
    keyY=None,
    # options
    details=None,
    # plotting
    dprop=None,
    vmin=None,
    vmax=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    connect=None,
    dinc=None,
    show_commands=None,
):

    # -------------------
    # check
    # -------------------

    (
        key_fit, key_model, key_sol, key_data, key_lamb,
        details, binning, connect,
    ) = _check(
        coll=coll,
        key=key,
        details=details,
        connect=connect,
    )

    # -------------------
    # interpolate
    # -------------------

    dout = coll.interpolate_spectral_model(
        key_model=key_fit,
        # options
        details=details,
        # others
        returnas=dict,
    )

    # -------------------
    # extract coll2
    # -------------------

    coll2, dkeys, ndim = _extract_coll2(
        coll=coll,
        key_model=key_model,
        key_data=key_data,
        dout=dout,
        details=details,
        keyY=keyY,
    )

    # -------------------
    # prepare figure
    # -------------------

    if dax is None:
        dax = _get_dax(
            ndim=ndim,# resources
            coll=coll,
            key_data=key_data,
            key_fit=key,
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
            coll2,
            dout=dout,
            dkeys=dkeys,
            dax=dax,
            details=details
        )

    elif ndim == 2:
        dax, dgroup = _plot_2d(
            coll=coll,
            key_model=key_model,
            coll2=coll2,
            dout=dout,
            keyY=keyY,
            dkeys=dkeys,
            dax=dax,
            details=details,
        )

    # -------------------
    # finalize
    # -------------------

    _finalize_figure(
        dax=dax,
        dout=dout,
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
    key=None,
    # options
    details=None,
    connect=None,
):

    # -------------
    # key
    # -------------

    wsf = coll._which_fit
    lok = [
        k0 for k0, v0 in coll.dobj.get(wsf, {}).items()
        if v0['key_sol'] is not None
    ]

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # key model, data, lamb
    key_model = coll.dobj[wsf][key]['key_model']
    key_sol = coll.dobj[wsf][key]['key_sol']
    key_data = coll.dobj[wsf][key]['key_data']
    key_lamb = coll.dobj[wsf][key]['key_lamb']

    binning = coll.dobj[wsf][key]['dinternal']['binning']

    # -------------
    # details
    # -------------

    details = ds._generic_check._check_var(
        details, 'details',
        types=bool,
        default=True,
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
        key, key_model, key_sol, key_data, key_lamb,
        details, binning, connect,
    )


# ###############################################################
# ###############################################################
#               extract coll2
# ###############################################################


def _extract_coll2(
    coll=None,
    key_model=None,
    key_data=None,
    dout=None,
    details=None,
    keyY=None,
):

    coll2, dkeys, ndim = _class01_plot._extract_coll2(
        coll=coll,
        key_model=key_model,
        dout=dout,
        details=details,
        keyY=keyY,
    )

    # ----------
    # add data

    lk = ['data', 'units', 'dim', 'quant', 'ref']
    coll2.add_data(
        key=key_data,
        **{k0: coll._ddata[key_data][k0] for k0 in lk}
    )

    dkeys['data'] = key_data

    # ----------
    # add error

    kerr = 'error'
    lk = ['units', 'dim', 'quant']
    coll2.add_data(
        key=kerr,
        data=coll.ddata[key_data]['data'] - coll2.ddata[dkeys['sum']]['data'],
        **{k0: coll._ddata[key_data][k0] for k0 in lk}
    )

    dkeys['error'] = kerr

    return coll2, dkeys, ndim


# ###############################################################
# ###############################################################
#               plot 1d
# ###############################################################


def _plot_1d(coll2=None, dout=None, dkeys=None, dax=None, details=None):

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

    return dax


# ###############################################################
# ###############################################################
#               plot 2d
# ###############################################################


def _plot_2d(
    coll=None,
    key_model=None,
    coll2=None,
    dout=None,
    dkeys=None,
    keyY=None,
    dax=None,
    details=None,
):

    # --------------
    # dvminmax
    # --------------

    vmin = np.nanmin(coll2.ddata[dkeys['data']]['data'])
    vmax = np.nanmax(coll2.ddata[dkeys['data']]['data'])

    dvminmax = {
        'data': {'min': vmin, 'max': vmax}
    }

    errmax = np.nanmax(np.abs(coll2.ddata[dkeys['error']]['data']))
    dvminmax_err = {
        'data': {'min': -errmax, 'max': errmax}
    }

    # --------------
    # plot data
    # --------------

    coll2, dgroup0 = coll2.plot_as_array(
        key=dkeys['data'],
        keyX=dkeys['lamb'],
        keyY=keyY,
        dax={k0: dax[k0] for k0 in ['vert', '2d_data', 'spectrum']},
        aspect='auto',
        dvminmax=dvminmax,
        connect=False,
        inplace=True,
    )

    # --------------
    # plot fit
    # --------------

    coll2, dgroup1 = coll2.plot_as_array(
        key=dkeys['sum'],
        keyX=dkeys['lamb'],
        keyY=keyY,
        dax={k0: dax[k0] for k0 in ['vert', '2d_fit', 'spectrum']},
        aspect='auto',
        dvminmax=dvminmax,
        connect=False,
        inplace=True,
    )

    # --------------
    # plot error
    # --------------

    coll2, dgroup0 = coll2.plot_as_array(
        key=dkeys['error'],
        keyX=dkeys['lamb'],
        keyY=keyY,
        dax={k0: dax[k0] for k0 in ['vert', '2d_err', 'error']},
        aspect='auto',
        dvminmax=dvminmax_err,
        cmap=plt.cm.seismic,
        connect=False,
        inplace=True,
    )

    # -----------------
    # plot std
    # -----------------

    # if dkeys.get('sum_min') is not None:
    #     # plot fit
    #     ax.fill(
    #         coll2.ddata[dkeys['lamb']]['data'],
    #         coll2.ddata[dkeys['sum_min']]['data'],
    #         coll2.ddata[dkeys['sum_max']]['data'],
    #         ec='None',
    #         lw=0.,
    #         fc=ll.get_color(),
    #         alpha=0.5,
    #     )

    # --------------
    # plot spectrum
    # --------------

    if details is True:

        reflamb = coll2.ddata[dkeys['lamb']]['ref'][0]
        lamb = coll2.ddata[dkeys['lamb']]['data']
        nmax = dgroup0['X']['nmax']
        wsm = coll._which_model
        lfunc = coll.dobj[wsm][key_model]['keys']

        refs = coll2.ddata[dkeys['sum']]['ref']
        axis = refs.index(reflamb)
        refs = (refs[axis - 1],)
        nan = np.full(lamb.shape, np.nan)

        kax = 'spectrum'
        ax = dax[kax]['handle']
        for ii, ff in enumerate(lfunc):

            for jj in range(nmax):

                ll, = ax.plot(
                    lamb,
                    nan,
                    ls='-',
                    marker='None',
                    lw=1.,
                )

                xydata = 'ydata'
                km = f'{lfunc[ii]}_{jj}'

                coll2.add_mobile(
                    key=km,
                    handle=ll,
                    refs=(refs,),
                    data=(lfunc[ii],),
                    dtype=[xydata],
                    group_vis='Y',  # 'X' <-> 'Y'
                    axes=kax,
                    ind=jj,
                )

    return coll2, dgroup0


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
            'left': 0.05, 'right': 0.98,
            'bottom': 0.08, 'top': 0.90,
            'wspace': 0.30, 'hspace': 0.20,
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
    gs = gridspec.GridSpec(4, 14, **dmargin)

    # ------------
    # add axes
    # ------------

    # --------
    # images

    ax = fig.add_subplot(gs[:, 1:3])
    ax.set_title(f'data\n{key_data}', size=12, fontweight='bold')
    # ax.set_ylabel()
    ax0 = ax
    dax['2d_data'] = {'handle': ax, 'type': 'matrix'}

    ax = fig.add_subplot(gs[:, 3:5], sharex=ax0, sharey=ax0)
    ax.set_title(f'fit\n{key_fit}', size=12, fontweight='bold')
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['2d_fit'] = {'handle': ax, 'type': 'matrix'}

    ax = fig.add_subplot(gs[:, 5:7], sharex=ax0, sharey=ax0)
    ax.set_title('error', size=12, fontweight='bold')
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['2d_err'] = {'handle': ax, 'type': 'matrix'}

    # --------
    # vertical

    ax = fig.add_subplot(gs[:, 0], sharey=ax0)
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['vert'] = {'handle': ax, 'type': 'vertical'}

    # ----------
    # spectrum

    ax = fig.add_subplot(gs[:2, 8:], sharex=ax0)
    ax.set_ylabel(data_lab, size=12, fontweight='bold')
    dax['spectrum'] = {'handle': ax, 'type': 'horizontal'}

    ax = fig.add_subplot(gs[2, 8:], sharex=ax0)
    ax.set_xlabel(lamb_lab, size=12, fontweight='bold')
    ax.set_ylabel('error', size=12, fontweight='bold')
    dax['error'] = {'handle': ax, 'type': 'horizontal'}

    return dax



# ###############################################################
# ###############################################################
#               Finalize figure
# ###############################################################


def _finalize_figure(dax=None, dout=None, tit=None):

    # -------------
    # tit
    # -------------

    titdef = (
        f"Spectral model '{dout['key_model']}'\n"
        f"using data '{dout['key_data']}'"
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