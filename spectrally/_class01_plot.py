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
):

    # -------------------
    # check
    # -------------------

    details = _check(
        details=details,
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
        dout=dout,
        details=details,
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
        _plot_1d(coll2, dout=dout, dkeys=dkeys, dax=dax, details=details)
    elif ndim == 2:
        _plot_2d(coll2, dout=dout, dkeys=dkeys, dax=dax, details=details)

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

    if ndim > 1:
        pass

    return dax


# ###############################################################
# ###############################################################
#               check
# ###############################################################


def _check(
    details=None,
):

    # -------------
    # details
    # -------------

    details = ds._generic_check._check_var(
        details, 'details',
        types=bool,
        default=True,
    )

    return details


# ###############################################################
# ###############################################################
#               extract coll2
# ###############################################################


def _extract_coll2(
    coll=None,
    dout=None,
    details=None,
):

    # -------------
    # extract
    # -------------

    # ------------
    # initialize

    key = 'data'
    key_lamb = dout['key_lamb']
    if key_lamb is None:
        key_lamb = 'lamb'

    coll2 = coll.__class__()

    # --------------
    # add all refs

    ref = dout['ref']
    if details is True:
        ref = list(ref)
        ref[0] = 'nfunc'
        coll2.add_ref(ref[0], size=dout['data'].shape[0])

    if dout['key_lamb'] is None:
        ilamb = ref.index(None)
        kref_lamb = 'nlamb'
        ref = list(ref)
        ref[ilamb] = kref_lamb
        coll2.add_ref(kref_lamb, size=dout['data'].shape[ilamb])

    for rr in ref:
        if rr not in coll2.dref.keys():
            coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

    # --------------
    # add all data

    # data
    lk = ['data', 'units', 'dim', 'quant']
    coll2.add_data(
        key,
        ref=tuple(ref),
        **{k0: dout[k0] for k0 in lk},
    )

    # lamb
    if dout['key_lamb'] is None:
        coll2.add_data(key_lamb, data=dout['lamb'], ref=kref_lamb)
    else:
        coll2.add_data(
            dout['key_lamb'],
            **{k0: coll.ddata[dout['key_lamb']][k0] for k0 in lk + ['ref']},
        )

    # sum if details
    if details is True:
        ksum = f"{key}_sum"
        coll2.add_data(
            ksum,
            data=np.sum(dout['data'], axis=0),
            ref=tuple(ref[1:]),
            **{k0: dout[k0] for k0 in ['dim', 'quant', 'units']}
        )

    # all other vectrs if any

    # --------------
    # dkeys
    # --------------

    dkeys = {
        'lamb': key_lamb,
        'sum': ksum if details is True else key,
    }
    if details is True:
        dkeys['details'] = key

    # --------------
    # prepare data
    # --------------

    # get ndim
    ndim = coll2.ddata[dkeys['sum']]['data'].ndim
    if ndim > 1:
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
            for ii in range(coll2.ddata[dkeys['details']]['data'].shape[0]):
                ax.plot(
                    coll2.ddata[dkeys['lamb']]['data'],
                    coll2.ddata[dkeys['details']]['data'][ii, ...],
                    ls='-',
                    marker='None',
                    lw=1.,
                )

    return


# ###############################################################
# ###############################################################
#               plot 2d
# ###############################################################


def _plot_2d(coll2=None, dout=None, dkeys=None, dax=None, details=None):

    # --------------
    # plot fixed 2d
    # --------------

    dax2, dgroup = coll2.plot_as_array(
        key=None,
        keyX=None,
        keyY=None,
        dax=dax,
        connect=False,
    )

    # --------------
    # plot spectrum
    # --------------

    if details is True:
        kax = 'spectrum'
        lax = [vax['handle'] for vax in dax.values() if kax in vax['type']]
        for ax in lax:

            for ii in range(coll2.ddata[dkeys['details']]['data'].shape[0]):
                ax.plot(
                    coll2.ddata[dkeys['lamb']]['data'],
                    coll2.ddata[dkeys['details']]['data'][ii, ...],
                    ls='-',
                    marker='None',
                    lw=1.,
                )

    return


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
    gs = gridspec.GridSpec(2, 1, **dmargin)

    # ------------
    # add axes
    # ------------

    ax = fig.add_subplot(gs[:, 0])
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax = {'2d': {'handle': ax, 'type': '2d'}}

    ax = fig.add_subplot(gs[0, 1])
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax = {'1d': {'handle': ax, 'type': 'spectrum'}}

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

    if tit is not None:
        fig.suptitle(tit, size=12, fontweight='bold')

    return