# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:31:49 2024

@author: dvezinet
"""


import numpy as np


#############################################
#############################################
#    harmonize dict for scale, bounds, x0
#############################################


def _get_dict(
    lx_free_keys=None,
    dmodel=None,
    din=None,
    din_name=None,
):

    # --------------
    # trivial
    # --------------

    if din is None:
        return {}

    # --------------
    # non-trivial
    # --------------

    c0 = (
        isinstance(din, dict)
        and all([isinstance(k0, str) for k0 in din.keys()])
    )
    if not c0:
        msg = (
            f"Arg '{din_name}' must be a dict of the form:\n"
            "\t- key_of_free_variable: value (float)\n"
            "Provided:\n{din}"
        )
        raise Exception(msg)

    # --------------
    # check keys
    # --------------

    derr = {
        k0: v0 for k0, v0 in din.items()
        if k0 not in lx_free_keys or not np.isscalar(v0)
    }
    if len(derr) > 0:
        lstr = [f"\t- '{k0}': {v0}" for k0, v0 in derr.items()]
        msg = (
            "The following key / values are non-conform from '{din_name}':/n"
            + "\n".join(lstr)
            + "\nAll keys must be natching free variable names!\n"
            "Available keys:\n{lk_free_keys}"
        )
        raise Exception(msg)

    # --------------
    # convert to ind / val format
    # --------------

    dout = {}
    for k0, v0 in din.items():

        # get func name and type + variable name
        ftype = [k1 for k1, v1 in dmodel.items() if k0 in v1.keys()][0]
        var = k0.split('_')[-1]

        if ftype not in dout.keys():
            dout[ftype] = {}

        if var not in dout[ftype].keys():
            dout[ftype][var] = {'ind': [], 'val': []}

        dout[ftype][var]['ind'].append(lx_free_keys.index(k0))
        dout[ftype][var]['val'].append(v0)

     # --------------
     # sort
     # --------------

    lktypes = list(dout.items())
    for k0 in lktypes:
        lkvar = list(dout[k0].keys())
        for k1 in lkvar:
            inds = np.argsort(dout[k0][k1]['ind'])
            dout[k0][var]['ind'] = np.array(dout[k0][k1]['ind'])[inds]
            dout[k0][var]['val'] = np.array(dout[k0][k1]['val'])[inds]

    return dout


#############################################
#############################################
#       get scales, bounds
#############################################


def _get_scales_bounds(
    nxfree=None,
    lamb=None,
    data=None,
    iok_all=None,
    dind=None,
    dscales=None,
    dbounds_low=None,
    dbounds_up=None,
):

    # ------------------
    # initialize
    # ------------------

    scale = np.zeros((nxfree,), dtype=float)
    bounds0 = np.zeros((nxfree,), dtype=float)
    bounds1 = np.zeros((nxfree,), dtype=float)

    # ------------------
    # prepare
    # ------------------

    lambD = lamb[-1] - lamb[0]
    lambd = lamb[1] - lamb[0]
    lambm = np.mean(lamb)

    data_max = np.nanmax(data[iok_all])
    data_min = np.nanmin(data[iok_all])
    data_mean = np.nanmean(data[iok_all])

    ldins = [(dscales, scale), (dbounds_low, bounds0, dbounds_up, bounds1)]

    # ------------------
    # all linear
    # ------------------

    kfunc = 'linear'
    if dind.get(kfunc) is not None:

        a1max = (data_max - data_min) / lambD

        # -------
        # a1

        kvar = 'a1'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = a1max
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # -------
        # a0

        kvar = 'a0'
        ind = dind['jac'][kfunc].get(kvar)
        if ind is not None:
            scale[ind] = max(
                np.abs(data_min - a1max*lamb[0]),
                np.abs(data_max + a1max*lamb[0]),
            )
            bounds0[ind] = -10.
            bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )


    # --------------------
    # all exponentials
    # ------------------

    kfunc = 'exp_lamb'
    if dind.get(kfunc) is not None:

        rate = np.nanmax([
            np.abs(
                np.log(data_max * lamb[0] / (data_min * lamb[-1]))
                / (1/lamb[-1] - 1./lamb[0])
            ),
            np.abs(
                np.log(data_min * lamb[0] / (data_max * lamb[-1]))
                / (1/lamb[-1] - 1./lamb[0])
            ),
        ])

        # rate
        kvar = 'rate'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = rate
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_mean * np.exp(rate/lambm) * lambm
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # -----------------
    # all gaussians
    # -----------------

    kfunc = 'gauss'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # -----------------
    # all lorentz
    # -----------------

    kfunc = 'lorentz'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # -----------------
    # all pvoigt
    # -----------------

    kfunc = 'pvoigt'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # -----------------
    # all voigt
    # -----------------

    kfunc = 'voigt'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambD / 5
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # ------------------
    # all pulse1
    # ------------------

    kfunc = 'pulse1'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # t0
        kvar = 't0'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambm
        bounds0[ind] = (lamb[0] - lambd) / lambm
        bounds1[ind] = (lamb[1] + lambd) / lambm

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # tup
        kvar = 'tup'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambd
        bounds0[ind] = 1.e-2
        bounds1[ind] = lambD/lambd

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # tdown
        kvar = 'tdown'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = 0.5 * lambD
        bounds0[ind] = 1.e-2
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # ------------------
    # all pulse2
    # ------------------

    kfunc = 'pulse2'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # t0
        kvar = 't0'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambm
        bounds0[ind] = (lamb[0] - lambd) / lambm
        bounds1[ind] = (lamb[1] + lambd) / lambm

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # tup
        kvar = 'tup'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambd
        bounds0[ind] = 1.e-2
        bounds1[ind] = lambD/lambd

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # tdown
        kvar = 'tdown'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = 0.5 * lambD
        bounds0[ind] = 1.e-2
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    # ------------------
    # all lognorm
    # ------------------

    kfunc = 'lognorm'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = data_max - data_min
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # t0
        kvar ='t0'
        ind = dind['jac'][kfunc].get('t0')
        scale[ind] = lambm
        bounds0[ind] = (lamb[0] - lambd) / lambm
        bounds1[ind] = (lamb[1] + lambd) / lambm

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = lambd
        bounds0[ind] = 1.e-2
        bounds1[ind] = lambD/lambd

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

        # mu
        kvar = 'mu'
        ind = dind['jac'][kfunc].get(kvar)
        scale[ind] = 0.5 * lambD
        bounds0[ind] = 1.e-2
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scale=None if ii == 0 else scale
            )

    return scale


# #######################
# generic pdate from dict
# #######################


def _update_din_from_user(din, kfunc, kvar, val, scale=None):

    if din.get(kfunc, {}).get(kvar) is not None:
        ind = din[kfunc][kvar]['ind']
        if scale is None:
            val[ind] = din[kfunc][kvar]['val']
        else:
            val[ind] = din[kfunc][kvar]['val'] / scale[ind]

    return


#############################################
#############################################
#       get x0
#############################################


def _get_x0(
    x_free=None,
    iok=None,
    lambD=None,
    lambd=None,
    dind=None,
):



    return