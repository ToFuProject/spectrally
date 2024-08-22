# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:23:58 2024

@author: dvezinet
"""


import numpy as np
import datastock as ds


# #######################################################
# #######################################################
#            Main
# #######################################################


def main(
    coll=None,
    lamb=None,
    binning=None,
    # other fields
    iok=None,
    axis=None,
):

    # ------------------
    # check inputs
    # ------------------

    binning, axis, lamb = _check(coll, binning=binning, lamb=lamb, axis=axis)

    # ------------------
    # trivial
    # ------------------

    if binning is False:
        return False

    # ------------------
    # compute
    # ------------------

    # -----------
    # lamb

    lamb_edges = np.r_[
        lamb[0] - 0.5*(lamb[1] - lamb[0]),
        0.5*(lamb[1:] + lamb[:-1]),
        lamb[-1] + 0.5*(lamb[-1] - lamb[-2]),
    ]

    # in case of non-uniform lamb
    lambd = np.diff(lamb_edges)

    # increments
    lamb_inc = np.linspace(0, 1, binning+1)
    lamb_inc = 0.5*(lamb_inc[1:] + lamb_inc[:-1])
    lamb_inc = lamb_inc[None, :] * lambd[:, None]

    # for later integration
    bin_dlamb = np.repeat(
        lambd / binning,
        binning,
    )

    # get new lamb
    nlamb = lamb.size
    lamb = (lamb_edges[:-1, None] + lamb_inc).ravel()

    # safety check
    assert lamb.size == nlamb * binning

    # --------------
    # update iok_all

    if iok is not None:

        if axis is None:
            if iok.shape == (nlamb,):
                axis = 0
            else:
                msg = "Arg axis must be provided for binning of iok!"
                raise Exception(msg)

        bin_iok = np.repeat(iok, binning, axis=axis)

    else:
        bin_iok = None

    # --------------------------
    # update binning to indices

    ind = np.arange(0, lamb.size, binning)

    # ---------------
    # output
    # ---------------

    dout = {
        'binning': binning,
        'ind': ind,
        'lamb': lamb,
        'dlamb': np.abs(bin_dlamb),
        'iok': bin_iok,
        'edges': lamb_edges,
        'axis': axis,
    }

    return dout


# #######################################################
# #######################################################
#            check
# #######################################################


def _check(
    coll=None,
    binning=None,
    lamb=None,
    axis=None,
):

    # --------------
    # binning
    # --------------

    binning = ds._generic_check._check_var(
        binning, 'binning',
        types=(bool, int),
        default=False,
    )

    # default to 10 if True
    if binning is True:
        binning = 10

    # safety check
    if (binning is not False) and binning <= 0:
        msg = (
            "Arg 'binning' must be a > 0 int\n"
            f"Provided: {binning}"
        )
        raise Exception(msg)

    # ---------------
    # lamb
    # ---------------

    if isinstance(lamb, (list, tuple, np.ndarray)):
        lamb = np.asarray(lamb).ravel()

    elif isinstance(lamb, str):
        lamb = coll.ddata[lamb]['data']

    # -------------
    # axis
    # -------------

    if axis is not None:
        axis = int(ds._generic_check._check_var(
            axis, 'axis',
            types=(int, float),
            sign='>=0',
        ))

    return binning, axis, lamb