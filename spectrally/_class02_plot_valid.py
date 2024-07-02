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


_MK = '.'
_MS = 6
_DPROP = {
    0:  {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -1: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -2: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -3: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -4: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -5: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -6: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -7: {'ls': 'None', 'marker': _MK, 'ms': _MS},
    -8: {'ls': 'None', 'marker': _MK, 'ms': _MS},
}


def set_dprop():
    lk = sorted(_DPROP.keys())
    vmin = np.min(lk)
    vmax = np.max(lk)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.cm.Set1
    for k0 in _DPROP.keys():
        _DPROP[k0]['color'] = cm(norm(k0))

    return cm, vmin, vmax

_CM, _VMIN, _VMAX = set_dprop()


#############################################
#############################################
#       main
#############################################


def plot(
    coll=None,
    key=None,
    keyY=None,
    dref_vectorY=None,
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

    key, keyY, keyZ, dprop, vmin, vmax, tit = _check(
        coll=coll,
        key=key,
        # keyY
        keyY=keyY,
        dref_vectorY=dref_vectorY,
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
    key_bs_vect = coll.dobj[wsf][key]['key_bs_vect']

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

    if isinstance(dax, dict):
        fig = list(dax.values())[0]['handle'].figure
    else:
        fig = list(dax.dax.values())[0]['handle'].figure

    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    # ----------------------
    # connect interactivity
    # ----------------------

    if ndim == 1:
        return dax

    elif connect is True:
        dax.setup_interactivity(
            kinter='inter0', dgroup=dgroup, dinc=dinc,
        )
        dax.disconnect_old()
        dax.connect()

        dax.show_commands(verb=show_commands)
        return dax

    else:
        return dax, dgroup


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key=None,
    # keyY
    keyY=None,
    dref_vectorY=None,
    # keyZ
    keyZ=None,
    dref_vectorZ=None,
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
    key_data = coll.dobj[wsf][key]['key_data']
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    ref_lamb = coll.ddata[key_lamb]['ref'][0]

    # -----------------
    # keyY
    # -----------------

    ndim = coll.ddata[coll.dobj[wsf][key]['key_data']]['data'].ndim
    key_bs_vect = coll.dobj[wsf][key]['key_bs_vect']
    if ndim > 1 and key_bs_vect is None:
        if dref_vectorY is None:
            dref_vectorY = {}
        if dref_vectorY.get('key0') is None:
            dref_vectorY['key0'] = keyY

        if ndim == 2 and dref_vectorY.get('ref') is None:
            ref = [
                rr for rr in coll.ddata[key_data]['ref']
                if rr != ref_lamb
            ][0]
            dref_vectorY['ref'] = ref

        keyY = coll.get_ref_vector(**dref_vectorY)[3]
        if keyY is None:
            msg = f"No ref vector for ref '{ref}'"
            raise Exception(msg)

    elif ndim > 1:
        keyY = key_bs_vect
    else:
        keyY = None

    # -----------------
    # keyZ
    # -----------------

    if ndim > 2:
        if dref_vectorZ is None:
            dref_vectorZ = {}
        if dref_vectorZ.get('key0') is None:
            dref_vectorZ['key0'] = keyZ

        if dref_vectorZ.get('ref') is None:
            refY = coll.ddata[keyY]['ref'][0]
            ref = [
                rr for rr in coll.ddata[key_data]['ref']
                if rr not in [ref_lamb, refY]
            ][0]
            dref_vectorZ['ref'] = ref

        keyZ = coll.get_ref_vector(**dref_vectorZ)[3]
        if keyZ is None:
            msg = f"No ref vector for ref '{ref}'"
            raise Exception(msg)

    else:
        keyZ = None

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

    return key, keyY, keyZ, dprop, vmin, vmax, tit


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


def _plot_2d(
    coll=None,
    key=None,
    # keyY
    keyY=None,
    # bsplines
    key_bs=None,
    key_bs_vect=None,
    # options
    dprop=None,
    dvminmax=None,
    cmap=None,
    # figure
    dax=None,
    fs=None,
    dmargin=None,
):

    # -----------------
    # prepare
    # -----------------

    wsf = coll._which_fit
    key_data = coll.dobj[wsf][key]['key_data']
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    lamb = coll.ddata[key_lamb]['data']
    yy = coll.ddata[keyY]['data']
    data = coll.ddata[coll.dobj[wsf][key]['key_data']]['data']

    dvalid = coll.dobj[wsf][key]['dvalid']
    iok = coll.ddata[dvalid['iok']]['data']
    frac = dvalid['frac'][0]

    # -----------------
    # prepare figure
    # -----------------

    if dax is None:
        dax = _get_dax_2d(
            fs=fs,
            dmargin=dmargin,
            # bsplines
            keyY=keyY,
            key_bs=key_bs,
            key_bs_vect=key_bs_vect,
            dmeaning=dvalid['meaning'],
            # labelling
            coll=coll,
            key=key,
        )

    dax = ds._generic_check._check_dax(dax=dax, main='data')

    # -----------------
    # prepare - extract relevant data
    # -----------------

    collax, key_data, key_iok, key_lamb, key_frac = _prepare(
        coll=coll,
        key=key,
        keyY=keyY,
    )

    # -----------------
    # plot as array
    # -----------------

    collax, dgroup = collax.plot_as_array(
        key=key_data,
        keyX=key_lamb,
        keyY=keyY,
        aspect='auto',
        dvminmax=dvminmax,
        cmap=cmap,
        dax={
            k0: v0 for k0, v0 in dax.items()
            if k0 in ['data_img', 'data_vert', 'spectrum']
        },
        inplace=True,
        connect=False,
    )

    # dgroup = {
        # 'X': {
            # 'ref': [dkeys['X']['ref']],
            # 'data': ['index'],
            # 'nmax': nmax,
        # },
        # 'Y': {
            # 'ref': [dkeys['Y']['ref']],
            # 'data': ['index'],
            # 'nmax': nmax,
        # },
    # }

    # -----------------
    # plot valid
    # -----------------

    collax, dgroup2 = collax.plot_as_array(
        key=key_iok,
        keyX=key_lamb,
        keyY=keyY,
        aspect='auto',
        dvminmax={'data': {'min': _VMIN, 'max': _VMAX}},
        cmap=_CM,
        dax={
            k0: v0 for k0, v0 in dax.items()
            if k0 in ['valid_img', 'valid_1d']
        },
        inplace=True,
        connect=False,
    )

    # ----------------
    # add nsigma and focus
    # ---------------

    kax = 'spectrum'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # hline
        ax.axhline(dvalid['nsigma']**2, ls='--', c='k')

        # text
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

    # -----------------
    # plot frac
    # -----------------

    kax = 'frac'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        dcol = {
            int(k0.split('_')[-1][1:]): v0['handle'].get_color()
            for k0, v0 in collax.dobj['mobile'].items()
            if k0.startswith(f'{key_iok}_h')
        }

        # ind0, ind1
        for ii in range(dgroup['Y']['nmax']):

            lh = ax.axhline(
                collax.ddata[keyY]['data'][0],
                c=dcol[ii],
                lw=1.,
                ls='-',
            )

            # update collax
            kh = f'frac_h{ii}'
            collax.add_mobile(
                key=kh,
                handle=lh,
                refs=collax.ddata[keyY]['ref'],
                data=(keyY,),
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

        # fixed
        ax.axvline(
            dvalid['fraction'],
            c='k',
            ls='--',
            lw=1,
        )

        # fixed
        ax.plot(
            collax.ddata[dvalid['frac']]['data'],
            collax.ddata[keyY]['data'],
            c='k',
            ls='-',
            lw=1,
        )

        # axes interactivity
        # add axes
        collax.add_axes(
            handle=ax,
            key=kax,
            refy=[collax.ddata[keyY]['ref'][0]],
            datay=[keyY],
            harmonize=True,
        )

    return collax, dgroup


# ---------------------
# prepare data extraction
# ---------------------


def _prepare(
    coll=None,
    key=None,
    keyY=None,
    keyZ=None,
):

    # ----------
    # prepare
    # ----------

    # data, lamb
    wsf = coll._which_fit
    key_data = coll.dobj[wsf][key]['key_data']
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    ndim = coll.ddata[key_data]['data'].ndim

    # iok, frac
    dvalid = coll.dobj[wsf][key]['dvalid']
    key_iok = dvalid['iok']
    key_frac = dvalid['frac']

    # keyY

    # ----------
    # list keys
    # ----------

    keys = [key_data, key_lamb, keyY, key_iok, key_frac]

    if ndim == 3:
        keys.append(keyZ)

    # ---------
    # extract
    # ---------

    collax = coll.extract(
        keys=keys,
        inc_monot=True,
        inc_vectors=True,
        inc_allrefs=False,
        return_keys=False,
    )[0]

    return collax, key_data, key_iok, key_lamb, key_frac


# ---------------------
# create axes
# ---------------------


def _get_dax_2d(
    fs=None,
    dmargin=None,
    keyY=None,
    # bsplines
    key_bs=None,
    key_bs_vect=None,
    dmeaning=None,
    # labelling
    coll=None,
    key=None,
):

    # ---------------
    # check
    # ---------------

    if fs is None:
        fs = (14, 10)

    if dmargin is None:
        dmargin = {
            'left': 0.06, 'right': 0.98,
            'bottom': 0.06, 'top': 0.92,
            'wspace': 0.9, 'hspace': 0.50,
        }

    # ---------------
    # prepare labels
    # ---------------

    wsf = coll._which_fit
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    key_data = coll.dobj[wsf][key]['key_data']
    xlab = f"{key_lamb} ({coll.ddata[key_lamb]['units']})"
    dlab = f"{key_data} ({coll.ddata[key_data]['units']})"
    ylab = f"{keyY} ({coll.ddata[keyY]['units']})"

    # ---------------
    # figure
    # ---------------

    fig = plt.figure(figsize=fs)

    nh = (1, 3, 9)
    ncols = (2 * nh[0] + 2 * nh[1] + 2 * nh[2])
    gs = gridspec.GridSpec(ncols=ncols, nrows=6, **dmargin)

    # ---------------
    # axes
    # ---------------

    dax = {}

    # data_img
    n0 = nh[0]
    ax = fig.add_subplot(gs[:3, n0:(n0 + nh[2])])
    ax.set_ylabel(ylab, size=12, fontweight='bold')
    ax.set_title('data', size=14, fontweight='bold')
    dax['data_img'] = {'handle': ax, 'type': 'matrix'}
    ax0 = ax

    # data_cbar
    ax = fig.add_subplot(gs[:3, :nh[0]])
    dax['data_cbar'] = ax

    # data_vert
    n0 = nh[0] + nh[2]
    ax = fig.add_subplot(gs[:3, n0:(n0+nh[1])], sharey=ax0)
    ax.set_xlabel(dlab, size=12, fontweight='bold')
    ax.set_ylabel(ylab, size=12, fontweight='bold')
    dax['data_vert'] = {'handle': ax, 'type': 'vertical'}

    # bs_knots
    if key_bs_vect is not None:
        n0 = (nh[0] + nh[1] + nh[2])
        ax = fig.add_subplot(gs[:3, n0:(n0+nh[0])], sharey=ax0)
        dax['bs_knots'] = ax

    # valid_img
    n0 = (2*nh[0] + 2*nh[1] + nh[2])
    ax = fig.add_subplot(gs[:3, n0:(n0+nh[2])], sharex=ax0, sharey=ax0)
    ax.set_xlabel(xlab, size=12, fontweight='bold')
    ax.set_ylabel(ylab, size=12, fontweight='bold')
    ax.set_title('validity', size=14, fontweight='bold')
    dax['valid_img'] = {'handle': ax, 'type': 'matrix'}

    # frac
    n0 = (2*nh[0] + nh[1] + nh[2])
    ax = fig.add_subplot(gs[:3, n0:(n0+nh[1])], sharey=ax0)
    ax.set_title('frac', size=14, fontweight='bold')
    dax['frac'] = {'handle': ax, 'type': 'frac'}

    # spectrum
    n0 = nh[0]
    ax = fig.add_subplot(gs[3:5, n0:(n0+nh[2])], sharex=ax0)
    ax.set_ylabel(dlab, size=12, fontweight='bold')
    ax.set_title('spectrum', size=14, fontweight='bold')
    dax['spectrum'] = {'handle': ax, 'type': 'horizontal'}

    # valid_1d
    n0 = nh[0]
    ax = fig.add_subplot(gs[5, n0:(n0+nh[2])], sharex=ax0)
    ax.set_xlabel(xlab, size=12, fontweight='bold')
    ax.set_ylabel('validity', size=12, fontweight='bold')
    trans = transforms.blended_transform_factory(
        ax.transAxes,
        ax.transData,
    )
    for k0, v0 in dmeaning.items():
        ax.text(
            1.02,
            k0,
            v0,
            color=_DPROP[k0]['color'],
            transform=trans,
            horizontalalignment='left',
            verticalalignment='center',
        )
    dax['valid_1d'] = {'handle': ax, 'type': 'horizontal'}

    # -----------------
    # adjust visibility
    # -----------------

    plt.setp(dax['data_img']['handle'].get_xticklabels(), visible=False)
    plt.setp(dax['spectrum']['handle'].get_xticklabels(), visible=False)
    plt.setp(dax['spectrum']['handle'].get_xlabel(), visible=False)

    return dax


#############################################
#############################################
#       plot 2d
#############################################
