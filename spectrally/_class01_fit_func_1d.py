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
    dmodel=None,
    dconstraints=None,
    dconst=None,
):

    # --------------
    # prepare
    # --------------

    dfunc = _get_dfunc()

    # --------------
    # prepare
    # --------------

    for k0, v0 in dmodel.items():



        dfunc[k0] = None


        def func(
            x=None,
            lamb=None,
            xall=xall,
        ):

            # ----------
            # initialize

            val = np.zeros(lamb.shape, dtype=float)
            x_full = np.zeros((len(x_all),), dtype=float)

            # ----------------------------
            # get x_full from constraints



            # ----------
            # sum

            for k0, v0 in dmodel.items():

                if v0['type'] == 'gauss':
                    amp = x_full[ind]
                    w2 = x_full[ind]
                    shift = x_full[ind]

                    val += amp * np.exp(-(lambnorm - (1 + shift)**2)/w2)




            val = (
                np.nansum(
                    amp * np.exp(-(lambnorm[indok, :]-(1 + shift))**2 * inv_2wi2),
                    axis=1,
                )
                + xscale[ibckax] * np.exp(xscale[ibckrx] * lambrel[indok])
            )

        return val

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
