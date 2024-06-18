# #!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
created on tue feb 20 14:44:51 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import datastock as ds


# local


#############################################
#############################################
#       cost
#############################################


def _get_func_val(
    x_all=None,
    x_free=None,
    dconstraints=None,
    dind=None,
    param=None,
):

    # --------------
    # prepare
    # --------------

    # ------------
    # constraints

    c0 = dconstraints['c0']
    c1 = dconstraints['c1']
    c2 = dconstraints['c2']

    n_all = len(x_all)
    if np.allclose(c0, 0) and np.allclose(c1, 0.) and np.allclose(c2, 0.):
        assert c0.shape == (n_all,), c0.shape
        assert c1.shape == (n_all, n_all), c1.shape
        assert c2.shape == (n_all, n_all), c2.shape
        c0, c1, c2 = None, None, None

    # --------------
    # prepare
    # --------------

    for k0, v0 in dmodel.items():

        def func(
            x_free=None,
            lamb=None,
            param=param,
            n_all=n_all,
            c0=c0,
            c1=c1,
            c2=c2,
            dind=dind,
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



            # ------------------
            # sum all linear

            kfunc = 'linear'
            if dind.get(kfunc) is not None:

                a0 = x_full[dind[kfunc]['a0']]
                a1 = x_full[dind[kfunc]['a1']]

                val += a0 * lamb + a1

            # --------------------
            # sum all exponentials

            kfunc = 'exp'
            if dind.get(kfunc) is not None:

                amp = x_full[dind[kfunc]['amp']]
                rate = x_full[dind[kfunc]['rate']]

                val += amp * np.exp(lamb * rate)

            # -----------------
            # sum all gaussians

            kfunc = 'gauss'
            if dind.get(kfunc) is not None:

                amp = x_full[dind[kfunc]['amp']]
                width = x_full[dind[kfunc]['width']]
                shift = x_full[dind[kfunc]['shift']]
                lamb0 = param[dind[kfunc]['lamb0']]

                val += amp * np.exp(-(lamb - lamb0*(1 + shift)**2)/width**2)

            # -------------------
            # sum all Lorentzians

            kfunc = 'lorentz'
            if dind.get(kfunc) is not None:

                amp = x_full[dind[kfunc]['amp']]
                gam = x_full[dind[kfunc]['gamma']]
                shift = x_full[dind[kfunc]['shift']]
                lamb0 = param[dind[kfunc]['lamb0']]

                val += (
                    amp * (0.5/np.pi) * gam
                    / ((lamb - lamb0*(1 + shift))**2 + (0.5*gam)**2)
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

                pass

            # ------------------
            # sum all pulse2

            kfunc = 'pulse2'
            if dind.get(kfunc) is not None:

                pass

            # ------------------
            # sum all lognorm

            kfunc = 'lognorm'
            if dind.get(kfunc) is not None:

                pass



            val = (
                np.nansum(
                    amp * np.exp(-(lambnorm[indok, :]-(1 + shift))**2 * inv_2wi2),
                    axis=1,
                )
                + xscale[ibckax] * np.exp(xscale[ibckrx] * lambrel[indok])
            )

        return val

        dfunc[k0] = None

    return func

















def _get_func_cost_1d_old():

    def func(
        x,
        xscale=xscale,
        indx=indx,
        lambrel=lambrel,
        lambnorm=lambnorm,
        ibckax=ibckax,
        ibckrx=ibckrx,
        ial=ial,
        iwl=iwl,
        ishl=ishl,
        idratiox=idratiox,
        idshx=idshx,
        scales=None,
        coefsal=coefsal[None, :],
        coefswl=coefswl[None, :],
        coefssl=coefssl[None, :],
        offsetal=offsetal[None, :],
        offsetwl=offsetwl[None, :],
        offsetsl=offsetsl[None, :],
        double=dinput['double'],
        indok=None,
        const=None,
        data=np.zeros((x.size,)),
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