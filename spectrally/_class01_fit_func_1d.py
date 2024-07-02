# #!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
created on tue feb 20 14:44:51 2024

@author: dvezinet
"""


import numpy as np


# ############################################
# ############################################
#       sum
# ############################################


def _get_func_sum(
    c0=None,
    c1=None,
    c2=None,
    dind=None,
    param_val=None,
):

    # --------------
    # prepare
    # --------------

    def func(
        x_free=None,
        lamb=None,
        param_val=param_val,
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        scale=None,
    ):

        # ----------
        # initialize

        val = np.zeros(lamb.shape, dtype=float)

        # ----------------------------
        # get x_full from constraints

        if c0 is None:
            x_full = x_free
        else:
            x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0

        # -------------------
        # rescale

        if scale is not None:
            pass

        # ------------------
        # sum all linear

        kfunc = 'linear'
        if dind.get(kfunc) is not None:

            a0 = x_full[dind[kfunc]['a0']['ind']][:, None]
            a1 = x_full[dind[kfunc]['a1']['ind']][:, None]

            val += np.sum(a0 + lamb * a1, axis=0)

        # --------------------
        # sum all exponentials

        kfunc = 'exp'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            rate = x_full[dind[kfunc]['rate']['ind']][:, None]

            val += np.sum(amp * np.exp(lamb * rate), axis=0)

        # -----------------
        # sum all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            width = x_full[dind[kfunc]['width']['ind']][:, None]
            shift = x_full[dind[kfunc]['shift']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            val += np.sum(
                amp * np.exp(
                    -(lamb - lamb0*(1 + shift))**2/width**2
                ),
                axis=0,
            )

        # -------------------
        # sum all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            gam = x_full[dind[kfunc]['gamma']['ind']][:, None]
            shift = x_full[dind[kfunc]['shift']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            # https://en.wikipedia.org/wiki/Cauchy_distribution
            # value at lamb0 = amp / (pi * gam)

            val += np.sum(
                amp * gam
                / (np.pi * ((lamb - lamb0*(1 + shift))**2 + gam**2)),
                axis=0,
            )

        # --------------------
        # sum all pseudo-voigt

        kfunc = 'pvoigt'
        if dind.get(kfunc) is not None:

            pass

        # ------------------
        # sum all pulse1

        kfunc = 'pulse1'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
            tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
            tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]

            ind0 = lamb > t0

            val += np.sum(
                amp * ind0 * (
                    np.exp(-(lamb - t0)/tdown) - np.exp(-(lamb - t0)/tup)
                ),
                axis=0,
            )

        # ------------------
        # sum all pulse2

        kfunc = 'pulse2'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
            tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
            tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]

            indup = (lamb < t0)
            inddown = (lamb >= t0)

            val += np.sum(
                amp * (
                    indup * np.exp(-(lamb - t0)**2/tup**2)
                    + inddown * np.exp(-(lamb - t0)**2/tdown**2)
                ),
                axis=0,
            )

        # ------------------
        # sum all lognorm

        kfunc = 'lognorm'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
            sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
            mu = x_full[dind[kfunc]['mu']['ind']][:, None]

            val += np.sum(
                (amp / ((lamb - t0) * sigma * np.sqrt(2*np.pi)))
                * np.exp(-(np.log(lamb - t0) - mu)**2 / (2.*sigma**2)),
                axis=0,
            )

        return val

    return func


# ############################################
# ############################################
#       cost
# ############################################


def _get_func_cost(
    c0=None,
    c1=None,
    c2=None,
    dind=None,
    param_val=None,
):

    # --------------
    # prepare
    # --------------

    func_sum = _get_func_sum(
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        param_val=param_val,
    )

    # ------------
    # cost
    # ------------

    def func(
        x_free=None,
        lamb=None,
        scale=None,
        data=None,
        # sum
        func_sum=func_sum,
    ):

        return func_sum(x_free, lamb=lamb, scale=scale) - data

    return func


# ############################################
# ############################################
#       details
# ############################################


def _get_func_details(
    c0=None,
    c1=None,
    c2=None,
    dind=None,
    param_val=None,
):

    # --------------
    # prepare
    # --------------

    def func(
        x_free=None,
        lamb=None,
        param_val=param_val,
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        scale=None,
    ):

        # ----------
        # initialize

        shape = tuple([dind['nfunc']] + list(lamb.shape))
        val = np.zeros(shape, dtype=float)

        # ----------------------------
        # get x_full from constraints

        if c0 is None:
            x_full = x_free
        else:
            x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0

        # -------------------
        # rescale

        if scale is not None:
            pass

        # ------------------
        # sum all linear

        kfunc = 'linear'
        if dind.get(kfunc) is not None:

            a0 = x_full[dind[kfunc]['a0']['ind']][:, None]
            a1 = x_full[dind[kfunc]['a1']['ind']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = a0 + lamb * a1

        # --------------------
        # sum all exponentials

        kfunc = 'exp'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            rate = x_full[dind[kfunc]['rate']['ind']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = amp * np.exp(lamb * rate)

        # -----------------
        # sum all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            width = x_full[dind[kfunc]['width']['ind']][:, None]
            shift = x_full[dind[kfunc]['shift']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp * np.exp(-(lamb - lamb0*(1 + shift))**2/width**2)
            )

        # -------------------
        # sum all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            gam = x_full[dind[kfunc]['gamma']['ind']][:, None]
            shift = x_full[dind[kfunc]['shift']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            # https://en.wikipedia.org/wiki/Cauchy_distribution
            # value at lamb0 = amp / (pi * gam)

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp * gam
                / (np.pi * ((lamb - lamb0*(1 + shift))**2 + gam**2))
            )

        # --------------------
        # sum all pseudo-voigt

        kfunc = 'pvoigt'
        if dind.get(kfunc) is not None:

            pass

        # ------------------
        # sum all pulse1

        kfunc = 'pulse1'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
            tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
            tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]

            ind0 = lamb > t0

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp * ind0 * (
                    np.exp(-(lamb - t0)/tdown) - np.exp(-(lamb - t0)/tup)
                )
            )

        # ------------------
        # sum all pulse2

        kfunc = 'pulse2'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
            tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
            tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]

            indup = (lamb < t0)
            inddown = (lamb >= t0)

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp * (
                    indup * np.exp(-(lamb - t0)**2/tup**2)
                    + inddown * np.exp(-(lamb - t0)**2/tdown**2)
                )
            )

        # ------------------
        # sum all lognorm

        kfunc = 'lognorm'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
            sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
            mu = x_full[dind[kfunc]['mu']['ind']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                (amp / ((lamb - t0) * sigma * np.sqrt(2*np.pi)))
                * np.exp(-(np.log(lamb - t0) - mu)**2 / (2.*sigma**2))
            )

        return val

    return func


# ################################################
# ################################################
#               Back up
# ################################################


def _get_func_cost_1d_old():

    def func(
        x,
        xscale=xscale,
    ):

        if indok is None:
            indok = np.ones(lambrel.shape, dtype=bool)

        # xscale = x*scales   !!! scales ??? !!! TBC
        xscale[indx] = x*scales[indx]
        xscale[~indx] = const

        # make sure iwl is 2D to get all lines at once
        amp = xscale[ial] * coefsal + offsetal
        inv_2wi2 = 1./(2.*(xscale[iwl] * coefswl + offsetwl))
        shift = xscale[ishl] * coefssl + offsetsl

        # ----------
        # sum

        y = (
            np.nansum(
                amp * np.exp(-(lambnorm[indok, :]-(1 + shift))**2 * inv_2wi2),
                axis=1,
            )
            + xscale[ibckax] * np.exp(xscale[ibckrx] * lambrel[indok])
        )

        # ----------
        # return

        return y - data[iok]

    # -----------
    # return func

    return func


