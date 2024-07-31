# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datastock as ds


# ###############################################################
# ###############################################################
#               Main
# ###############################################################


def main(
    coll=None,
    key_model=None,
    key_data=None,
    lamb=None,
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

    details, connect = _check(
        details=details,
        connect=connect,
    )

    # -------------------
    # interpolate
    # -------------------

    dout = coll.interpolate_spectral_model(
        key_model=key_model,
        key_data=key_data,
        lamb=lamb,
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
        dout=dout,
        details=details,
        keyY=keyY,
    )

    # -------------------
    # prepare figure
    # -------------------

    if dax is None:
        dax = _get_dax(
            ndim=ndim,
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
    details=None,
    connect=None,
):

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

    return details, connect


# ###############################################################
# ###############################################################
#               extract coll2
# ###############################################################


def _extract_coll2(
    coll=None,
    key_model=None,
    dout=None,
    details=None,
    keyY=None,
):

    # ------------
    # initialize
    # -------------

    wsm = coll._which_model
    key = 'data'
    key_lamb = dout['key_lamb']
    if key_lamb is None:
        key_lamb = 'lamb'

    coll2 = coll.__class__()

    # --------------
    # add all refs
    # --------------

    # ref_nfunc
    ref = dout['ref']
    if details is True:
        ref = list(ref)
        ref[0] = 'nfunc'
        coll2.add_ref(ref[0], size=dout['data'].shape[0])

    # kre_lamb
    if dout['key_lamb'] is None:
        ilamb = ref.index(None)
        kref_lamb = 'nlamb'
        ref = list(ref)
        ref[ilamb] = kref_lamb
        coll2.add_ref(kref_lamb, size=dout['data'].shape[ilamb])

    # all other refs
    for rr in ref:
        if rr not in coll2.dref.keys():
            coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

    # --------------
    # add all data
    # --------------

    lk = ['data', 'units', 'dim', 'quant']

    # -------
    # lamb

    if dout['key_lamb'] is None:
        coll2.add_data(key_lamb, data=dout['lamb'], ref=kref_lamb)
    else:
        coll2.add_data(
            dout['key_lamb'],
            **{k0: coll.ddata[dout['key_lamb']][k0] for k0 in lk + ['ref']},
        )

    # ---------------
    # sum + details

    if details is True:

        # data
        lfunc = coll.dobj[wsm][key_model]['keys']
        for ii, ff in enumerate(lfunc):
            coll2.add_data(
                ff,
                data=dout[key][ii, ...],
                ref=tuple(ref[1:]),
                **{k0: dout[k0] for k0 in ['dim', 'quant', 'units']},
            )

        # sum
        ksum = f"{key}_sum"
        coll2.add_data(
            ksum,
            data=np.sum(dout[key], axis=0),
            ref=tuple(ref[1:]),
            **{k0: dout[k0] for k0 in ['dim', 'quant', 'units']}
        )

    else:
        ksum = f"{key}_sum"
        coll2.add_data(
            ksum,
            ref=tuple(ref),
            **{k0: dout[k0] for k0 in lk}
        )

    # -------------------------
    # all other vectors if any

    if keyY is not None:
        coll2.add_data(
            keyY,
            **{k0: coll.ddata[keyY][k0] for k0 in lk + ['ref']},
        )

    # --------------
    # add std
    # --------------

    if dout['data_min'] is not None:

        ksmin = f"{key}_sum_min"
        coll2.add_data(
            ksmin,
            data=dout['data_min'],
            ref=tuple(ref),
            **{k0: dout[k0] for k0 in lk[1:]}
        )

        ksmax = f"{key}_sum_max"
        coll2.add_data(
            ksmax,
            data=dout['data_max'],
            ref=tuple(ref),
            **{k0: dout[k0] for k0 in lk[1:]}
        )

    # --------------
    # dkeys
    # --------------

    dkeys = {
        'lamb': key_lamb,
        'sum': ksum,
    }
    if details is True:
        dkeys['details'] = lfunc
    if dout['data_min'] is not None:
        dkeys['sum_min'] = ksmin
        dkeys['sum_max'] = ksmax

    # --------------
    # prepare data
    # --------------

    # get ndim
    ndim = coll2.ddata[dkeys['sum']]['data'].ndim
    if ndim > 2:
        raise NotImplementedError()

    return coll2, dkeys, ndim


# ###############################################################
# ###############################################################
#               plot 1d
# ###############################################################


def _plot_1d(coll2=None, dout=None, dkeys=None, dax=None, details=None):

    # ------------
    # plot
    # -----------

    kax = 'spectrum'
    lax = [vax['handle'] for vax in dax.values() if kax in vax['type']]
    for ax in lax:

        ax.plot(
            coll2.ddata[dkeys['lamb']]['data'],
            coll2.ddata[dkeys['sum']]['data'],
            ls='-',
            marker='None',
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
    # plot fixed 2d
    # --------------

    coll2, dgroup = coll2.plot_as_array(
        key=dkeys['sum'],
        keyX=dkeys['lamb'],
        keyY=keyY,
        dax=dax,
        aspect='auto',
        connect=False,
        inplace=True,
    )

    # --------------
    # plot spectrum
    # --------------

    if details is True:

        lamb = coll2.ddata[dkeys['lamb']]['data']
        nmax = dgroup['X']['nmax']
        wsm = coll._which_model
        lfunc = coll.dobj[wsm][key_model]['keys']
        refs = (coll2.ddata[dkeys['sum']]['ref'][0],)
        nan = np.full(lamb.shape, np.nan)

        axtype = 'horizontal'
        lax = [kax for kax, vax in dax.items() if axtype in vax['type']]
        for kax in lax:
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

    return coll2, dgroup


# ###############################################################
# ###############################################################
#               dax
# ###############################################################


def _get_dax(
    ndim=None,
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

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(1, 1, **dmargin)

    # ------------
    # add axes
    # ------------

    ax = fig.add_subplot(gs[0, 0])
    # ax.set_xlabel()
    # ax.set_ylabel()

    # ------------
    # populate dax
    # ------------

    dax = {'1d': {'handle': ax, 'type': 'spectrum'}}

    return dax


def _get_dax_2d(
    fs=None,
    dmargin=None,
    tit=None,
):

    # ---------------
    # check inputs
    # ---------------

    if fs is None:
        fs = (18, 10)

    if dmargin is None:
        dmargin = {
            'left': 0.07, 'right': 0.98,
            'bottom': 0.08, 'top': 0.90,
            'wspace': 0.50, 'hspace': 0.10,
        }

    # ---------------
    # prepare figure
    # ---------------

    dax = {}
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 7, **dmargin)

    # ------------
    # add axes
    # ------------

    ax = fig.add_subplot(gs[:, 0])
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['vert'] = {'handle': ax, 'type': 'vertical'}

    ax = fig.add_subplot(gs[:, 1:3])
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['2d'] = {'handle': ax, 'type': 'matrix'}

    ax = fig.add_subplot(gs[0, 3:])
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['hor'] = {'handle': ax, 'type': 'horizontal'}

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