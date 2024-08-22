# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:32:47 2024

@author: dvezinet
"""


# common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datastock as ds


# local
from . import _class01_model_dict as _model_dict


__all__ = ['display_spectral_model_function']


# #######################################################
# #######################################################
#            Main
# #######################################################


def display_spectral_model_function(
    # func type
    ftype=None,
    # plot vector
    xx=None,
    binning=None,
    # plotting
    dax=None,
    # resources
    dmodel=_model_dict._DMODEL,
):

    plt.rcParams['text.usetex'] = True

    # -------------
    # check inputs
    # -------------

    ftype = ds._generic_check._check_var(
        ftype, 'ftype',
        types=str,
        allowed=list(dmodel.keys()),
    )

    dfunc = dmodel[ftype]

    # --------------
    # prepare data
    # --------------

    # get plot vector
    xx = _get_xx(xx, ftype)

    # get dparam and free variables from plot vector
    dpar, xfree = _get_dpar_xfree(ftype, xx)

    # -----------------
    # prepare Collection
    # -----------------

    # initialize
    from ._class02_SpectralFit import SpectralFit as Collection
    coll = Collection()

    # add model
    coll.add_spectral_model(
        key='sm00',
        dmodel={
            'func': {'type': ftype, **dpar},
        },
        dconstraints=None,
    )

    # add ref
    coll.add_ref('n0', size=xx.size)

    # add data
    coll.add_data('x', data=xx)

    # add xfree
    coll.add_data('var', data=xfree)

    # --------------
    # compute model
    # --------------

    coll.interpolate_spectral_model(
        key_model='sm00',
        key_data='var',
        lamb='x',
        store=True,
        store_key='y',
    )

    # --------------
    # prepare figure
    # --------------

    if dax is None:
        dax = _create_dax(
            binning=binning,
            fs=None,
            dmargin=None,
            tit=None,
        )

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot
    # --------------

    _plot(
        coll=coll,
        ftype=ftype,
        dfunc=dfunc,
        dax=dax,
    )

    return coll, dax


# #######################################################
# #######################################################
#            Prepare x, y
# #######################################################


def _get_xx(xx, ftype):

    # -------------
    # set if None
    # -------------

    lbck = ['poly', 'exp_lamb']
    llines = ['gauss', 'lorentz', 'pvoigt', 'voigt']
    lpulse = ['pulse_exp', 'pulse_gauss', 'lognorm']

    if xx is None:
        if ftype in lbck + llines:
            xx = np.linspace(3.94, 4, 501) * 1e-10

        elif ftype in lpulse:
            xx = np.linspace(100, 300, 501) * 1e-9

        else:
            raise NotImplementedError()

    # -------------
    # check
    # -------------

    try:
        xx = np.unique(np.atleast_1d(xx).ravel().astype(float))
    except Exception as err:
        msg = "Arg xx must be a 1d increasing array of floats"
        raise Exception(msg) from err

    return xx


# #######################################################
# #######################################################
#            Get dpar and xfree
# #######################################################


def _get_dpar_xfree(ftype, xx):

    # ---------------
    # prepare from xx
    # ---------------

    x0 = xx[0]
    x1 = xx[-1]
    dx = xx[1] - x0
    Dx = xx[-1] - x0
    xm = np.mean(xx)
    xmin = np.min(xx)
    xmax = np.max(xx)

    # ---------------
    # dpar
    # ---------------

    llines = ['gauss', 'lorentz', 'pvoigt', 'voigt']
    if ftype in llines:
        dpar = {'lamb0': xm}
    else:
        dpar = {}

    # ---------------
    # xfree
    # ---------------

    if ftype == 'poly':

        a0 = 1
        a1 = -0.2
        a2 = -1

        xfree = np.r_[a0, a1, a2]

    elif ftype == 'exp_lamb':

        rate = np.nanmax([
            np.abs(np.log(xmax * x0 / (xmin * x1)) / (1/x1 - 1./x0)),
            np.abs(np.log(xmin * x0 / (xmax * x1)) / (1/x1 - 1./x0)),
        ])
        amp = 1 * np.exp(rate/xm) * xm

        xfree = np.r_[amp, rate]

    elif ftype == 'gauss':

        amp = 1
        sigma = Dx / 10
        vccos = 0.

        xfree = np.r_[amp, vccos, sigma]

    elif ftype == 'lorentz':

        amp = 1
        gam = Dx / 10
        vccos = 0.

        xfree = np.r_[amp, vccos, gam]

    elif ftype == 'pvoigt':

        amp = 1
        sigma = Dx / 10
        gam = Dx / 10
        vccos = 0.

        xfree = np.r_[amp, vccos, sigma, gam]

    elif ftype == 'voigt':

        amp = 1
        sigma = Dx / 10
        gam = Dx / 10
        vccos = 0.

        xfree = np.r_[amp, vccos, sigma, gam]

    elif ftype == 'pulse_exp':

        amp = 1
        t0 = 0.3
        t_up = Dx / 30
        t_down = Dx / 5

        xfree = np.r_[amp, t0, t_up, t_down]

    elif ftype == 'pulse_gauss':

        amp = 1
        t0 = 0.3
        t_up = Dx / 30
        t_down = Dx / 5

        xfree = np.r_[amp, t0, t_up, t_down]

    elif ftype == 'lognorm':

        t0 = 0.3
        sigma = 1
        std = Dx / 5
        mu = 0.5 * (np.log(std**2/(np.exp(sigma**2) - 1)) - sigma**2)
        amp = (xmax - xmin) * np.exp(mu - 0.5*sigma**2)

        xfree = np.r_[amp, t0, mu, sigma]

    return dpar, xfree


# #######################################################
# #######################################################
#            plot
# #######################################################


def _plot(
    coll=None,
    ftype=None,
    dfunc=None,
    dax=None,
):

    # --------------
    # plot data
    # --------------

    axtype = 'plot'
    lax = [kax for kax, vax in dax.items() if axtype in vax['type']]
    for kax in lax:
        ax = dax[kax]['handle']

        ax.plot(
            coll.ddata['x']['data'],
            coll.ddata['y']['data'],
            ls='-',
            marker='None',
            lw=1,
            c='k',
        )

    # ----------------
    # plot expression
    # ----------------

    axtype = 'text'
    lax = [kax for kax, vax in dax.items() if axtype in vax['type']]
    for kax in lax:
        ax = dax[kax]['handle']

        # main expression
        ax.text(
            0.,
            0.5,
            dfunc['expressions']['main'],
            size=28,
            fontweight='bold',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='center',
        )

        # reference
        ref = dfunc['expressions'].get('ref')
        if ref is not None:
            ax.text(
                0.,
                0.1,
                ref,
                size=16,
                fontweight='bold',
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='center',
            )

        # others
        nexp = len(dfunc['expressions']) - 1
        for ii, (k0, v0) in enumerate(dfunc['expressions'].items()):

            if k0 in ['main', 'ref']:
                continue

            ax.text(
                0.6,
                ii / (nexp+1),
                v0,
                size=18,
                fontweight='bold',
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='center',
            )

    # ----------------
    # Figure title
    # ----------------

    ax.figure.suptitle(ftype, size=18, fontweight='bold')

    return


# #######################################################
# #######################################################
#            Figure
# #######################################################


def _create_dax(
    binning=None,
    fs=None,
    dmargin=None,
    tit=None,
):

    # ---------------
    # check inputs
    # ---------------

    if fs is None:
        fs = (14, 8)

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
    gs = gridspec.GridSpec(2, 1, **dmargin)

    # ------------
    # add axes
    # ------------

    # plot
    ax = fig.add_subplot(gs[0, 0], frameon=True)
    # ax.set_xlabel()
    # ax.set_ylabel()
    dax['plot'] = {'handle': ax, 'type': 'plot'}

    # text
    ax = fig.add_subplot(gs[1, 0], sharex=ax, frameon=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([])
    plt.setp(ax.get_xticklabels(), visible=False)
    dax['text'] = {'handle': ax, 'type': 'text'}

    return dax