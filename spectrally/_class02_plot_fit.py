# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:09:08 2024

@author: dvezinet
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
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
    # lines labels
    lines_labels=True,
    lines_labels_color='k',
    # figure
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    # interactivity
    connect=None,
    dinc=None,
    show_commands=None,
):

    # -------------------
    # check
    # -------------------

    (
        key_fit, key_model, key_sol, key_data, key_lamb,
        details, binning,
        lines_labels, lines_labels_color,
        connect,
    ) = _check(
        coll=coll,
        key=key,
        # options
        details=details,
        # plotting
        lines_labels=lines_labels,
        lines_labels_color=lines_labels_color,
        # interactivity
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
            details=details,
            lines_labels=lines_labels,
        )

    elif ndim == 2:
        dax, dgroup = _plot_2d(
            coll=coll,
            key_model=key_model,
            key_fit=key_fit,
            coll2=coll2,
            dout=dout,
            keyY=keyY,
            dkeys=dkeys,
            dax=dax,
            details=details,
            # lines labels
            lines_labels=lines_labels,
            lines_labels_color=lines_labels_color,
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
    # lines labels
    lines_labels=None,
    lines_labels_color=None,
    # interactivity
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
    # lines_labels
    # -------------

    if lines_labels is True:
        lines_labels = None

    if lines_labels is not False:
        lines_labels = coll.get_spectral_lines_labels(
            keys=key_model,
            labels=lines_labels,
        )

    # -------------
    # lines_labels_color
    # -------------

    if lines_labels is False:
        lines_labels_color = None

    else:

        if lines_labels_color is None:
            lines_labels_color = 'sum'

        if lines_labels_color in ['sum', 'details']:
            pass

        elif mcolors.is_color_like(lines_labels_color):
            pass

        else:
            msg = (
                "Arg 'lines_labels_color' must be either:\n"
                "\t- 'sum': labels color is the common color of sum curve\n"
                "\t- 'details': labels color is the color of each func\n"
                "\t- color-like: labels color is set to the provided color\n"
                f"Provided:\n{lines_labels_color}"
            )
            raise Exception(msg)

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
        details, binning,
        lines_labels, lines_labels_color,
        connect,
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
    key_fit=None,
    coll2=None,
    dout=None,
    dkeys=None,
    keyY=None,
    dax=None,
    details=None,
    # lines labels
    lines_labels=None,
    lines_labels_color=None,
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

    dcolor = None
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

                # plot
                ll, = ax.plot(
                    lamb,
                    nan,
                    ls='-',
                    marker='None',
                    lw=1.,
                )

                # extract color
                if lines_labels_color == 'details':
                    dcolor[ff][jj] = ll.get_color()

                # add mobile
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

    # ----------------
    # add lines_labels
    # ----------------

    if lines_labels is not False:
        _add_labels(
            # resources
            coll=coll,
            coll2=coll2,
            key_fit=key_fit,
            dkeys=dkeys,
            axis=axis,
            nmax=nmax,
            dcolor=dcolor,
            # parameters
            kax='spectrum',
            lines_labels=lines_labels,
            lines_labels_color=lines_labels_color,
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
    key_fit=None,
    dkeys=None,
    axis=None,
    nmax=None,
    dcolor=None,
    # parameters
    kax=None,
    lines_labels=None,
    lines_labels_color=None,
):

    # ----------------
    # get kax

    ax = coll2.dax[kax]['handle']

    # -------------
    # prepare data

    # extract moments
    dmom = coll.get_spectral_model_moments(key_fit)

    # prepare
    data = coll2.ddata[dkeys['sum']]['data']
    refs = list(coll2.ddata[dkeys['sum']]['ref'])

    # lamb
    lamb = coll2.ddata[dkeys['lamb']]['data']
    sh_lamb = [lamb.size if ii == axis else 1 for ii in range(data.ndim)]
    sh_argmax = [1 if ii == axis else ss for ii, ss in enumerate(data.shape)]
    lambf = lamb.reshape(sh_lamb)

    # custom ref for lines
    rpxy = 'peaks_n2'
    refs[axis] = rpxy
    refm = tuple([rr for ii, rr in enumerate(refs) if ii != axis])
    # add ref
    coll2.add_ref(rpxy, size=2)

    # vmax
    vmax = coll2.dax[kax]['handle'].get_ylim()[1]
    vmaxf = np.full(sh_argmax, vmax)

    # loop on functions / spectral lines
    nan2 = np.r_[np.nan, np.nan]
    for k0, v0 in lines_labels.items():

        # argmax
        argmax = dmom[f'{k0}_argmax']['data']
        argmaxf = argmax.reshape(sh_argmax)

        # get ind
        ind = np.abs(argmaxf - lambf).argmin(axis=axis, keepdims=True)

        # dmax
        dmaxf = np.take_along_axis(data, ind, axis=axis)

        peaks_x = np.concatenate((argmaxf, argmaxf), axis=axis)
        peaks_y = np.concatenate((dmaxf, vmaxf), axis=axis)

        # ---------------
        # add data

        # add argmax (for text)
        key_argmax = f'peaks_argmax_{k0}'
        coll2.add_data(
            key=key_argmax,
            data=argmax,
            units=coll2.ddata[dkeys['lamb']]['units'],
            ref=refm,
        )

        # add peaks_x
        keyx = f'peaks_x_{k0}'
        coll2.add_data(
            key=keyx,
            data=peaks_x,
            units=coll2.ddata[dkeys['lamb']]['units'],
            ref=tuple(refs),
        )

        keyy = f'peaks_y_{k0}'
        coll2.add_data(
            key=keyy,
            data=peaks_y,
            units=coll2.ddata[dkeys['sum']]['units'],
            ref=tuple(refs),
        )

        # ------------------
        # add vertical lines

        for jj in range(nmax):

            # set color
            if lines_labels_color == 'sum':
                color = dcolor[jj]
            elif lines_labels_color == 'details':
                color = dcolor[k0][jj]
            else:
                color = lines_labels_color

            # plot
            ll, = ax.plot(
                nan2,
                nan2,
                ls='--',
                marker='None',
                lw=1.,
                c=color,
            )

            # add mobile
            km = f'peaks_dashed_{k0}_{jj}'
            coll2.add_mobile(
                key=km,
                handle=ll,
                refs=(refm, refm),
                data=(keyx, keyy),
                dtype=['xdata', 'ydata'],
                group_vis='Y',  # 'X' <-> 'Y'
                axes=kax,
                ind=jj,
            )

            # ---------------------
            # add labels above axes

            # ll = ax.text(
            #     np.nan,
            #     1,
            #     v0,
            #     color=color,
            #     rotation=45,
            #     horizontalalignment='left',
            #     verticalalignment='bottom',
            #     transform=trans,
            # )

            # # add mobile
            # km = f'peaks_text_{k0}_{jj}'
            # coll2.add_mobile(
            #     key=km,
            #     handle=ll,
            #     refs=(refm,),
            #     data=(key_argmax,),
            #     dtype=['xdata'],
            #     group_vis='Y',  # 'X' <-> 'Y'
            #     axes=kax,
            #     ind=jj,
            # )

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