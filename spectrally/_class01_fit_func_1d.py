# #!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
created on tue feb 20 14:44:51 2024

@author: dvezinet
"""


import numpy as np
import scipy.special as scpsp



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
        scales=None,
        iok=None,
    ):

        # ---------------
        # iok

        if iok is not None:
            lamb = lamb[iok]

        # ----------
        # initialize

        shape = tuple([dind['nfunc']] + list(lamb.shape))
        val = np.zeros(shape, dtype=float)

        # -------------------
        # rescale

        if scales is not None:
            x_free = x_free * scales

        # ----------------------------
        # get x_full from constraints

        if c0 is None:
            x_full = x_free
        else:
            x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0

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

        kfunc = 'exp_lamb'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            rate = x_full[dind[kfunc]['rate']['ind']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = amp * np.exp(- rate / lamb) / lamb

        # -----------------
        # sum all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
            vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp * np.exp(-(lamb - lamb0*(1 + vccos))**2/(2*sigma**2))
            )

        # -------------------
        # sum all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            gam = x_full[dind[kfunc]['gam']['ind']][:, None]
            vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            # https://en.wikipedia.org/wiki/Cauchy_distribution
            # value at lamb0 = amp / (pi * gam)

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp / (1 + ((lamb - lamb0*(1 + vccos)) / gam)**2)
            )

        # --------------------
        # sum all pseudo-voigt

        kfunc = 'pvoigt'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
            gam = x_full[dind[kfunc]['gam']['ind']][:, None]
            vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            # https://en.wikipedia.org/wiki/Voigt_profile

            fg = 2 * np.sqrt(2*np.log(2)) * sigma
            fl = 2 * gam
            ftot = (
                fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
                + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5
            ) ** (1./5.)
            ratio = fl / ftot

            # eta
            eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3

            # update widths of gauss and Lorentz
            sigma = ftot / (2 * np.sqrt(2*np.log(2)))
            gam = ftot / 2.

            # weighted sum
            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = amp * (
                eta / (1 + ((lamb - lamb0*(1 + vccos)) / gam)**2)
                + (1-eta) * np.exp(-(lamb - lamb0*(1 + vccos))**2/(2*sigma**2))
            )

        # ------------

        kfunc = 'voigt'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
            gam = x_full[dind[kfunc]['gam']['ind']][:, None]
            vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = amp * scpsp.voigt_profile(
                lamb - lamb0*(1 + vccos),
                sigma,
                gam,
            )

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

            amp = x_full[dind[kfunc]['amp']['ind']]
            t0 = x_full[dind[kfunc]['t0']['ind']]
            sigma = x_full[dind[kfunc]['sigma']['ind']]
            mu = x_full[dind[kfunc]['mu']['ind']]

            # max at t - t0 = exp(mu - sigma**2)
            # max = amp * exp(sigma**2/2 - mu)
            # variance = (exp(sigma**2) - 1) * exp(2mu + sigma**2)
            # skewness = (exp(sigma**2) + 2) * sqrt(exp(sigma**2) - 1)

            ind = dind['func'][kfunc]['ind']
            for ii, i0 in enumerate(ind):
                iok = lamb > t0[ii]
                val[i0, iok] = (
                    (amp[ii] / (lamb[iok] - t0[ii]))
                    * np.exp(
                        -(np.log(lamb[iok] - t0[ii]) - mu[ii])**2
                        / (2.*sigma[ii]**2)
                    )
                )

        return val

    return func


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

    func_details = _get_func_details(
        c0=c0,
        c1=c1,
        c2=c2,
        dind=dind,
        param_val=param_val,
    )

    # --------------
    # prepare
    # --------------

    def func(
        x_free=None,
        lamb=None,
        # scales, iok
        scales=None,
        iok=None,
    ):

        return np.sum(
            func_details(x_free, lamb=lamb, scales=scales, iok=iok),
            axis=0,
        )

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
        # scales, iok
        scales=None,
        iok=None,
        # data
        data=None,
        # sum
        func_sum=func_sum,
    ):
        if iok is not None:
            data = data[iok]

        return func_sum(x_free, lamb=lamb, scales=scales, iok=iok) - data

    return func


# ############################################
# ############################################
#       Jacobian
# ############################################


def _get_func_jacob(
    c0=None,
    c1=None,
    c2=None,
    dindj=None,
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
        dindj=dindj,
        dind=dind,
        # scales, iok
        scales=None,
        iok=None,
        # unused
        **kwdargs,
    ):

        # ---------------
        # iok

        if iok is not None:
            lamb = lamb[iok]

        # ----------
        # initialize

        shape = tuple(list(lamb.shape) + [x_free.size])
        val = np.zeros(shape, dtype=float)
        lamb = lamb[:, None]

        # -------------------
        # rescale

        if scales is not None:
            x_free = x_free * scales

        # ----------------------------
        # get x_full from constraints

        if c0 is None:
            x_full = x_free
        else:
            x_full = c2.dot(x_free**2) + c1.dot(x_free) + c0

        # -------
        # linear

        kfunc = 'linear'
        if dind.get(kfunc) is not None:

            ind = dind['jac'][kfunc].get('a0')
            if ind is not None:
                val[:, ind] = 1.

            ind = dind['jac'][kfunc].get('a1')
            if ind is not None:
                val[:, ind] = lamb

        # --------
        # exp_lamb

        kfunc = 'exp_lamb'
        if dind.get(kfunc) is not None:
            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            rate = x_full[dind[kfunc]['rate']['ind']][None, :]

            exp_on_lamb = np.exp(- rate / lamb) / lamb

            ind = dind['jac'][kfunc].get('amp')
            if ind is not None:
                val[:, ind] = exp_on_lamb

            ind = dind['jac'][kfunc].get('rate')
            if ind is not None:
                val[:, ind] = - amp * exp_on_lamb / lamb

        # -----------------
        # sum all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
            vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            dlamb = lamb - lamb0*(1 + vccos)
            exp = np.exp(-dlamb**2/(2*sigma**2))

            ind = dind['jac'][kfunc].get('amp')
            if ind is not None:
                val[:, ind] = exp

            ind = dind['jac'][kfunc].get('vccos')
            if ind is not None:
                val[:, ind] = amp * exp * lamb0 * dlamb / sigma**2

            ind = dind['jac'][kfunc].get('sigma')
            if ind is not None:
                val[:, ind] = amp * exp * dlamb**2 / sigma**3

        # -------------------
        # sum all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            gam = x_full[dind[kfunc]['gam']['ind']][:, None]
            vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
            lamb0 = param_val[dind[kfunc]['lamb0']][:, None]

            # https://en.wikipedia.org/wiki/Cauchy_distribution
            # value at lamb0 = amp / (pi * gam)

            lamb_on_gam = (lamb - lamb0*(1 + vccos)) / gam

            ind = dind['jac'][kfunc].get('amp')
            if ind is not None:
                val[:, ind] = 1. / (1 + lamb_on_gam**2)

            ind = dind['jac'][kfunc].get('vccos')
            if ind is not None:
                val[:, ind] = (
                    (amp * lamb0 / gam)
                    * 2 * lamb_on_gam / (1 + lamb_on_gam**2)**2
                )

            ind = dind['jac'][kfunc].get('gam')
            if ind is not None:
                val[:, ind] = (
                    amp * 2 * lamb_on_gam**2 / (1 + lamb_on_gam**2)**2
                    / gam
                )

        # ----------- TO BE FINISHED ------------







# =============================================================================
#         # --------------------
#         # sum all pseudo-voigt
#
#         kfunc = 'pvoigt'
#         if dind.get(kfunc) is not None:
#
#             amp = x_full[dind[kfunc]['amp']['ind']][:, None]
#             sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
#             gam = x_full[dind[kfunc]['gam']['ind']][:, None]
#             vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
#             lamb0 = param_val[dind[kfunc]['lamb0']][:, None]
#
#             # https://en.wikipedia.org/wiki/Voigt_profile
#
#             fg = 2 * np.sqrt(2*np.log(2)) * sigma
#             fl = 2 * gam
#             ftot = (
#                 fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
#                 + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5
#             ) ** (1./5.)
#             ratio = fl / ftot
#
#             # eta
#             eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
#
#             # update widths of gauss and Lorentz
#             sigma = ftot / (2 * np.sqrt(2*np.log(2)))
#             gam = ftot / 2.
#
#             # weighted sum
#             ind = dind['func'][kfunc]['ind']
#             val[ind, ...] = amp * (
#                 eta / (1 + ((lamb - lamb0*(1 + vccos)) / gam)**2)
#                 + (1-eta) * np.exp(-(lamb - lamb0*(1 + vccos))**2/(2*sigma**2))
#             )
#
#         # ------------
#
#         kfunc = 'voigt'
#         if dind.get(kfunc) is not None:
#
#             amp = x_full[dind[kfunc]['amp']['ind']][:, None]
#             sigma = x_full[dind[kfunc]['sigma']['ind']][:, None]
#             gam = x_full[dind[kfunc]['gam']['ind']][:, None]
#             vccos = x_full[dind[kfunc]['vccos']['ind']][:, None]
#             lamb0 = param_val[dind[kfunc]['lamb0']][:, None]
#
#             ind = dind['func'][kfunc]['ind']
#             val[ind, ...] = amp * scpsp.voigt_profile(
#                 lamb - lamb0*(1 + vccos),
#                 sigma,
#                 gam,
#             )
#
#         # ------------------
#         # sum all pulse1
#
#         kfunc = 'pulse1'
#         if dind.get(kfunc) is not None:
#
#             amp = x_full[dind[kfunc]['amp']['ind']][:, None]
#             t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
#             tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
#             tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]
#
#             ind0 = lamb > t0
#
#             ind = dind['func'][kfunc]['ind']
#             val[ind, ...] = (
#                 amp * ind0 * (
#                     np.exp(-(lamb - t0)/tdown) - np.exp(-(lamb - t0)/tup)
#                 )
#             )
#
#         # ------------------
#         # sum all pulse2
#
#         kfunc = 'pulse2'
#         if dind.get(kfunc) is not None:
#
#             amp = x_full[dind[kfunc]['amp']['ind']][:, None]
#             t0 = x_full[dind[kfunc]['t0']['ind']][:, None]
#             tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
#             tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]
#
#             indup = (lamb < t0)
#             inddown = (lamb >= t0)
#
#             ind = dind['func'][kfunc]['ind']
#             val[ind, ...] = (
#                 amp * (
#                     indup * np.exp(-(lamb - t0)**2/tup**2)
#                     + inddown * np.exp(-(lamb - t0)**2/tdown**2)
#                 )
#             )
#
#         # ------------------
#         # sum all lognorm
#
#         kfunc = 'lognorm'
#         if dind.get(kfunc) is not None:
#
#             amp = x_full[dind[kfunc]['amp']['ind']]
#             t0 = x_full[dind[kfunc]['t0']['ind']]
#             sigma = x_full[dind[kfunc]['sigma']['ind']]
#             mu = x_full[dind[kfunc]['mu']['ind']]
#
#             # max at t - t0 = exp(mu - sigma**2)
#             # max = amp * exp(sigma**2/2 - mu)
#             # variance = (exp(sigma**2) - 1) * exp(2mu + sigma**2)
#             # skewness = (exp(sigma**2) + 2) * sqrt(exp(sigma**2) - 1)
#
#             ind = dind['func'][kfunc]['ind']
#             for ii, i0 in enumerate(ind):
#                 iok = lamb > t0[ii]
#                 val[i0, iok] = (
#                     (amp[ii] / (lamb[iok] - t0[ii]))
#                     * np.exp(
#                         -(np.log(lamb[iok] - t0[ii]) - mu[ii])**2
#                         / (2.*sigma[ii]**2)
#                     )
#                 )
# =============================================================================

        return val

    return func







# ################################################
# ################################################
# ################################################
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