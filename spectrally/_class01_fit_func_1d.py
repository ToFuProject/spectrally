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
        bin_iok=None,
        # binning
        bin_ind=None,
        bin_dlamb=None,
    ):

        # ---------------------
        # get lamb limits + iok

        # for pulses
        lamb00 = lamb[0]
        lambD = lamb[-1] - lamb[0]
        lambm = 0.5*(lamb[-1] + lamb[0])

        # iok
        if bin_iok is not None and bin_ind is False:
            lamb = lamb[bin_iok]

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
        # sum all poly

        kfunc = 'poly'
        if dind.get(kfunc) is not None:

            a0 = x_full[dind[kfunc]['a0']['ind']][:, None]
            a1 = x_full[dind[kfunc]['a1']['ind']][:, None]
            a2 = x_full[dind[kfunc]['a2']['ind']][:, None]

            ind = dind['func'][kfunc]['ind']
            lamb_rel = (lamb - lambm) / lambD
            val[ind, ...] = (
                a0
                + a1 * lamb_rel
                + a2 * lamb_rel**2
            )

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
            sigma2 = ftot / (2 * np.sqrt(2*np.log(2)))
            gam2 = ftot / 2.

            # weighted sum
            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = amp * (
                eta / (1 + ((lamb - lamb0*(1 + vccos)) / gam2)**2)
                + (1-eta) * np.exp(
                    -(lamb - lamb0*(1 + vccos))**2
                    / (2*sigma2**2)
                )
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
        # sum all pulse_exp

        kfunc = 'pulse_exp'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            tau = x_full[dind[kfunc]['tau']['ind']][:, None]
            tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
            tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]

            ind0 = lamb > (lamb00 + lambD * tau)
            dlamb = lamb - (lamb00 + lambD * tau)

            ind = dind['func'][kfunc]['ind']
            val[ind, ...] = (
                amp * ind0 * (
                    np.exp(-dlamb/tdown)
                    - np.exp(-dlamb/tup)
                )
            )

        # ------------------
        # sum all pulse_gauss

        kfunc = 'pulse_gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][:, None]
            tau = x_full[dind[kfunc]['tau']['ind']][:, None]
            tup = x_full[dind[kfunc]['t_up']['ind']][:, None]
            tdown = x_full[dind[kfunc]['t_down']['ind']][:, None]

            indup = (lamb < (lamb00 + lambD * tau))
            inddown = (lamb >= (lamb00 + lambD * tau))

            ind = dind['func'][kfunc]['ind']
            dlamb = lamb - (lamb00 + lambD * tau)

            val[ind, ...] = (
                amp * (
                    indup * np.exp(-dlamb**2/tup**2)
                    + inddown * np.exp(-dlamb**2/tdown**2)
                )
            )

        # ------------------
        # sum all lognorm

        kfunc = 'lognorm'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            tau = x_full[dind[kfunc]['tau']['ind']]
            sigma = x_full[dind[kfunc]['sigma']['ind']]
            mu = x_full[dind[kfunc]['mu']['ind']]

            # max at t - t0 = exp(mu - sigma**2)
            # max = amp * exp(sigma**2/2 - mu)
            # variance = (exp(sigma**2) - 1) * exp(2mu + sigma**2)
            # skewness = (exp(sigma**2) + 2) * sqrt(exp(sigma**2) - 1)

            ind = dind['func'][kfunc]['ind']
            for ii, i0 in enumerate(ind):
                ioki = lamb > (lamb00 + lambD * tau[ii])

                dlamb = lamb[ioki] - (lamb00 + lambD * tau[ii])

                val[i0, ioki] = (
                    (amp[ii] / dlamb)
                    * np.exp(-(np.log(dlamb) - mu[ii])**2 / (2.*sigma[ii]**2))
                )

        # -------
        # binning

        if bin_ind is not False:

            val = np.add.reduceat(val * bin_dlamb[None, :], bin_ind, axis=1)
            if iok is not None:
                val = val[:, iok]

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
        bin_iok=None,
        bin_ind=None,
        bin_dlamb=None,
        # unused (data)
        **kwdargs,
    ):

        return np.sum(
            func_details(
                x_free,
                lamb=lamb,
                scales=scales,
                iok=iok,
                bin_iok=bin_iok,
                bin_ind=bin_ind,
                bin_dlamb=bin_dlamb,
            ),
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
        # binning
        bin_ind=None,
        bin_iok=None,
        bin_dlamb=None,
        # data
        data=None,
        # sum
        func_sum=func_sum,
    ):
        if iok is not None:
            data = data[iok]

        return func_sum(
            x_free,
            lamb=lamb,
            scales=scales,
            iok=iok,
            bin_iok=bin_iok,
            bin_ind=bin_ind,
            bin_dlamb=bin_dlamb,
        ) - data

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
        bin_iok=None,
        # binning
        bin_ind=None,
        bin_dlamb=None,
        # unused (data)
        **kwdargs,
    ):

        # ---------------------
        # get lamb limits + iok

        # for pulses
        lamb00 = lamb[0]
        lambD = lamb[-1] - lamb[0]
        lambm = 0.5*(lamb[-1] + lamb[0])

        # iok
        if bin_iok is not None and bin_ind is False:
            lamb = lamb[bin_iok]

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
        # poly

        kfunc = 'poly'
        if dind.get(kfunc) is not None:

            lamb_rel = (lamb - lambm) / lambD

            vind = dind['jac'][kfunc].get('a0')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = 1. * scales[None, ival]

            vind = dind['jac'][kfunc].get('a1')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = lamb_rel * scales[None, ival]

            vind = dind['jac'][kfunc].get('a2')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = lamb_rel**2 * scales[None, ival]

        # --------
        # exp_lamb

        kfunc = 'exp_lamb'
        if dind.get(kfunc) is not None:
            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            rate = x_full[dind[kfunc]['rate']['ind']][None, :]

            exp_on_lamb = np.exp(- rate / lamb) / lamb

            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = exp_on_lamb[None, ivar] * scales[None, ival]

            vind = dind['jac'][kfunc].get('rate')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    - amp[:, ivar] * exp_on_lamb[:, ivar]
                    * scales[None, ival] / lamb
                )

        # -----------------
        # all gaussians

        kfunc = 'gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            sigma = x_full[dind[kfunc]['sigma']['ind']][None, :]
            vccos = x_full[dind[kfunc]['vccos']['ind']][None, :]
            lamb0 = param_val[dind[kfunc]['lamb0']][None, :]

            dlamb = lamb - lamb0*(1 + vccos)
            exp = np.exp(-dlamb**2/(2*sigma**2))

            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = exp[:, ivar] * scales[None, ival]

            vind = dind['jac'][kfunc].get('vccos')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar] * exp[:, ivar]
                    * (dlamb[:, ivar] / sigma[:, ivar]**2) * lamb0[:, ivar]
                    * scales[None, ival]
                )

            vind = dind['jac'][kfunc].get('sigma')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar] * exp[:, ivar]
                    * (dlamb[:, ivar]**2 / sigma[:, ivar]**3)
                    * scales[None, ival]
                )

        # -------------------
        # all Lorentzians

        kfunc = 'lorentz'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            gam = x_full[dind[kfunc]['gam']['ind']][None, :]
            vccos = x_full[dind[kfunc]['vccos']['ind']][None, :]
            lamb0 = param_val[dind[kfunc]['lamb0']][None, :]

            # https://en.wikipedia.org/wiki/Cauchy_distribution
            # value at lamb0 = amp / (pi * gam)

            lamb_on_gam = (lamb - lamb0*(1 + vccos)) / gam

            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    scales[None, ival] / (1 + lamb_on_gam[:, ivar]**2)
                )

            vind = dind['jac'][kfunc].get('vccos')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    (amp[:, ivar] * lamb0[:, ivar] / gam[:, ivar])
                    * scales[None, ival]
                    * 2 * lamb_on_gam[:, ivar]
                    / (1 + lamb_on_gam[:, ivar]**2)**2
                )

            vind = dind['jac'][kfunc].get('gam')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar] * 2 * lamb_on_gam[:, ivar]**2
                    / (1 + lamb_on_gam[:, ivar]**2)**2
                    * scales[None, ival] / gam[:, ivar]
                )

        # -------------------
        # all pvoigt

        kfunc = 'pvoigt'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            sigma = x_full[dind[kfunc]['sigma']['ind']][None, :]
            gam = x_full[dind[kfunc]['gam']['ind']][None, :]
            vccos = x_full[dind[kfunc]['vccos']['ind']][None, :]
            lamb0 = param_val[dind[kfunc]['lamb0']][None, :]

            fg = 2 * np.sqrt(2*np.log(2)) * sigma
            fl = 2 * gam

            ftot_norm = (
                fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
                + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5
            )
            ftot = ftot_norm ** (1./5.)
            ratio = fl / ftot

            # eta
            eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3

            # update widths of gauss and Lorentz
            sigma2 = ftot / (2 * np.sqrt(2*np.log(2)))
            gam2 = ftot / 2.

            dlamb = lamb - lamb0*(1 + vccos)
            exp = np.exp(-dlamb**2/(2*sigma2**2))
            lamb_on_gam = dlamb / gam2
            lorentz_norm = 1. / (1 + lamb_on_gam**2)

            # weighted sum
            # ind = dind['func'][kfunc]['ind']
            # val[ind, ...] = amp * (
            #     eta / (1 + ((lamb - lamb0*(1 + vccos)) / gam)**2)
            #     + (1-eta) * np.exp(-(lamb - lamb0*(1 + vccos))**2/(2*sigma**2))
            # )

            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    (
                        eta[:, ivar] * lorentz_norm[:, ivar]
                        + (1-eta[:, ivar])*exp[:, ivar]) * scales[None, ival]
                )

            vind = dind['jac'][kfunc].get('vccos')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = amp[:, ivar] * scales[None, ival] * (
                    eta[:, ivar] * (lamb0[:, ivar] / gam2[:, ivar]) *
                    2 * lamb_on_gam[:, ivar] / (1 + lamb_on_gam[:, ivar]**2)**2
                    + (1 - eta[:, ivar]) * exp[:, ivar]
                    * (dlamb[:, ivar] / sigma2[:, ivar]**2) * lamb0[:, ivar]
                )

            # --------------
            # widths

            # sigma
            vind = dind['jac'][kfunc].get('sigma')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                ds_fg = 2 * np.sqrt(2*np.log(2))
                ds_ftot = (1/5) * ftot_norm**(-4./5.) * ds_fg * (
                    5*fg**4 + 2.69269*4*fg**3*fl + 2.42843*3*fg**2*fl**2
                    + 4.47163*2*fg*fl**3 + 0.07842*fl**4
                )
                ds_ratio = (-fl/ftot**2) * ds_ftot
                ds_eta = ds_ratio * (
                    1.36603 - 0.47719 * 2 * ratio + 0.11116 * 3 * ratio**2
                )
                ds_sigma2 = ds_ftot / (2 * np.sqrt(2*np.log(2)))
                ds_gam2 = ds_ftot / 2.

                ds_lamb_on_gam = - ds_gam2 * dlamb / gam2**2
                ds_exp = ds_sigma2 * (dlamb**2/sigma2**3) * exp

                val[:, ival] = amp[:, ivar] * scales[None, ival] * (
                    ds_eta[:, ivar] / (1 + lamb_on_gam[:, ivar]**2)
                    + eta[:, ivar] * ds_lamb_on_gam[:, ivar]
                    * (-2*lamb_on_gam[:, ivar])
                    / (1 + lamb_on_gam[:, ivar]**2)**2
                    - ds_eta[:, ivar] * exp[:, ivar]
                    + (1 - eta[:, ivar]) * ds_exp[:, ivar]
                )

            vind = dind['jac'][kfunc].get('gam')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                dg_fl = 2
                dg_ftot = (1/5) * ftot_norm**(-4./5.) * dg_fl * (
                    2.69269*fg**4 + 2.42843*fg**3*2*fl
                    + 4.47163*fg**2*3*fl**2 + 0.07842*fg*4*fl**3 + 5*fl**4
                )
                dg_ratio = dg_fl/ftot + (-fl/ftot**2) * dg_ftot
                dg_eta = dg_ratio * (
                    1.36603 - 0.47719 * 2 * ratio + 0.11116 * 3 * ratio**2
                )
                dg_sigma2 = dg_ftot / (2 * np.sqrt(2*np.log(2)))
                dg_gam2 = dg_ftot / 2.

                dg_lamb_on_gam = - dg_gam2 * dlamb / gam2**2
                dg_exp = dg_sigma2 * (dlamb**2/sigma2**3) * exp

                val[:, ival] = amp[:, ivar] * scales[None, ival] * (
                    dg_eta[:, ivar] / (1 + lamb_on_gam[:, ivar]**2)
                    + eta[:, ivar] * dg_lamb_on_gam[:, ivar]
                    * (-2*lamb_on_gam[:, ivar])
                    / (1 + lamb_on_gam[:, ivar]**2)**2
                    - dg_eta[:, ivar] * exp[:, ivar]
                    + (1 - eta[:, ivar]) * dg_exp[:, ivar]
                )

        # -------------------
        # all pulse_exp

        kfunc = 'pulse_exp'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            tau = x_full[dind[kfunc]['tau']['ind']][None, :]
            tup = x_full[dind[kfunc]['t_up']['ind']][None, :]
            tdown = x_full[dind[kfunc]['t_down']['ind']][None, :]

            ind0 = lamb >= (lamb00 + lambD * tau)
            dlamb = lamb - (lamb00 + lambD * tau)
            exp_up = np.exp(-dlamb/tup)
            exp_down = np.exp(-dlamb/tdown)

            # amp
            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    scales[None, ival]
                    * ind0[:, ivar] * (exp_down[:, ivar] - exp_up[:, ivar])
                )

            # tau
            vind = dind['jac'][kfunc].get('tau')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                dtau_exp_up = (
                    scales[ival]
                    * exp_up[:, ivar]
                    * (lambD/tup[:, ivar])
                )
                dtau_exp_down = (
                    scales[None, ival]
                    * exp_down[:, ivar]
                    * (lambD/tdown[:, ivar])
                )
                # dtau_ind0 = ind0_t / lambd
                val[:, ival] = amp[:, ivar] * (
                    ind0[:, ivar] * (dtau_exp_down - dtau_exp_up)
                    # + dtau_ind0 * (exp_down - exp_up)
                )

            # tup
            vind = dind['jac'][kfunc].get('t_up')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar]
                    * ind0[:, ivar] * scales[None, ival] * (
                        - exp_up[:, ivar] * (dlamb[:, ivar]/tup[:, ivar]**2)
                    )
                )

            # tdown
            vind = dind['jac'][kfunc].get('t_down')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar]
                    * ind0[:, ivar] * scales[None, ival] * (
                        exp_down[:, ivar] * (dlamb[:, ivar]/tdown[:, ivar]**2)
                    )
                )

        # -------------------
        # all pulse_gauss

        kfunc = 'pulse_gauss'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']][None, :]
            tau = x_full[dind[kfunc]['tau']['ind']][None, :]
            tup = x_full[dind[kfunc]['t_up']['ind']][None, :]
            tdown = x_full[dind[kfunc]['t_down']['ind']][None, :]

            indup = (lamb < (lamb00 + lambD * tau))
            inddown = (lamb >= (lamb00 + lambD * tau))

            dlamb = lamb - (lamb00 + lambD * tau)
            exp_up = np.exp(-dlamb**2/tup**2)
            exp_down = np.exp(-dlamb**2/tdown**2)

            # val[ind, ...] = (
            #     amp * (
            #         indup * np.exp(-(lamb - t0)**2/tup**2)
            #         + inddown * np.exp(-(lamb - t0)**2/tdown**2)
            #     )
            # )

            # amp
            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = scales[None, ival] * (
                    indup[:, ivar] * exp_up[:, ivar]
                    + inddown[:, ivar] * exp_down[:, ivar]
                )

            # tau
            vind = dind['jac'][kfunc].get('t0')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                dtau_exp_up = (
                    scales[None, ival]
                    * exp_up[:, ivar]
                    * (-1/tup[:, ivar]**2) * (-2*lambD*dlamb[:, ivar])
                )
                dtau_exp_down = (
                    scales[None, ival]
                    * exp_down[:, ivar]
                    * (-1/tdown[:, ivar]**2) * (-2*lambD*dlamb[:, ivar])
                )
                # dtau_ind0 = ind0_t / lambd
                val[:, ival] = amp[:, ivar] * (
                    indup[:, ivar] * dtau_exp_up
                    + inddown[:, ivar] * dtau_exp_down
                )

            # tup
            vind = dind['jac'][kfunc].get('t_up')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar] * indup[:, ivar] * scales[None, ival] * (
                        exp_up[:, ivar] * (2*dlamb[:, ivar]**2/tup[:, ivar]**3)
                    )
                )

            # tdown
            vind = dind['jac'][kfunc].get('t_down')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                val[:, ival] = (
                    amp[:, ivar] * inddown[:, ivar] * scales[None, ival] * (
                        exp_down[:, ivar] * (2*dlamb[:, ivar]**2/tdown[:, ivar]**3)
                    )
                )

        # -------------------
        # all lognorm

        kfunc = 'lognorm'
        if dind.get(kfunc) is not None:

            amp = x_full[dind[kfunc]['amp']['ind']]
            tau = x_full[dind[kfunc]['tau']['ind']]
            sigma = x_full[dind[kfunc]['sigma']['ind']]
            mu = x_full[dind[kfunc]['mu']['ind']]

            # max at t - t0 = exp(mu - sigma**2)
            # max = amp * exp(sigma**2/2 - mu)
            # variance = (exp(sigma**2) - 1) * exp(2mu + sigma**2)
            # skewness = (exp(sigma**2) + 2) * sqrt(exp(sigma**2) - 1)

            # amp
            vind = dind['jac'][kfunc].get('amp')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                for ii, i0 in enumerate(ival):
                    ioki = lamb[:, 0] > (lamb00 + lambD * tau[ivar[ii]])
                    dlamb = lamb[ioki, 0] - (lamb00 + lambD * tau[ivar[ii]])

                    log_mu = np.log(dlamb) - mu[ivar[ii]]
                    exp = np.exp(-(log_mu)**2 / (2.*sigma[ivar[ii]]**2))

                    val[ioki, i0] = (scales[i0] / dlamb) * exp

            # tau
            vind = dind['jac'][kfunc].get('tau')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                for ii, i0 in enumerate(ival):
                    ioki = lamb[:, 0] > (lamb00 + lambD * tau[ivar[ii]])
                    dlamb = lamb[ioki, 0] - (lamb00 + lambD * tau[ivar[ii]])

                    log_mu = np.log(dlamb) - mu[ivar[ii]]
                    exp = np.exp(-(log_mu)**2 / (2.*sigma[ivar[ii]]**2))

                    dtau_inv_dlamb = scales[i0] * lambD/dlamb**2
                    dtau_logmu = scales[i0] * (-lambD) / dlamb
                    dtau_exp = (
                        exp * dtau_logmu * (-2*log_mu / (2.*sigma[ivar[ii]]**2))
                    )

                    val[ioki, i0] = amp[ivar[ii]] * (
                        dtau_inv_dlamb * exp
                        + (1/dlamb) * dtau_exp
                    )

            # sigma
            vind = dind['jac'][kfunc].get('sigma')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                for ii, i0 in enumerate(ival):
                    ioki = lamb[:, 0] > (lamb00 + lambD * tau[ivar[ii]])
                    dlamb = lamb[ioki, 0] - (lamb00 + lambD * tau[ivar[ii]])

                    log_mu = np.log(dlamb) - mu[ivar[ii]]
                    exp = np.exp(-(log_mu)**2 / (2.*sigma[ivar[ii]]**2))

                    val[ioki, i0] = (
                        (amp[ivar[ii]] / dlamb) * exp * scales[i0]
                        * (log_mu**2/sigma[ivar[ii]]**3)
                    )

            # mu
            vind = dind['jac'][kfunc].get('mu')
            if vind is not None:
                ival, ivar = vind['val'], vind['var']
                for ii, i0 in enumerate(ival):
                    ioki = lamb[:, 0] > (lamb00 + lambD * tau[ivar[ii]])
                    dlamb = lamb[ioki, 0] - (lamb00 + lambD * tau[ivar[ii]])

                    log_mu = np.log(dlamb) - mu[ivar[ii]]
                    exp = np.exp(-(log_mu)**2 / (2.*sigma[ivar[ii]]**2))

                    val[ioki, i0] = 0.1 * (
                        (amp[ivar[ii]] / dlamb) * exp * scales[i0]
                        * (-1/(2.*sigma[ivar[ii]]**2)) * (-2*log_mu)
                    )

                # val[iok, i0] = (
                #     (amp[ii] / dlamb)
                #     * np.exp(-(np.log(dlamb) - mu[ii])**2 / (2.*sigma[ii]**2))
                # )

        # -------
        # binning

        if bin_ind is not False:
            val = np.add.reduceat(val * bin_dlamb[:, None], bin_ind, axis=0)
            if iok is not None:
                val = val[iok, :]

        return val

    return func