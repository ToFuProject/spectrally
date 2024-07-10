# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:31:49 2024

@author: dvezinet
"""


import numpy as np



#############################################
#############################################
#    DEFAULTS
#############################################


_DEF_SCALES_FACTORS = {
    'width': 2,   # lambD / 2
    'shift': 2,   # lambD / 2
}

_DEF_X0_FACTORS = {
    'width': 10,  # lambD / 10
    'shift': 10,  # lambD / 10
}


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

    scales = np.zeros((nxfree,), dtype=float)
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

    ldins = [(dscales, scales), (dbounds_low, bounds0), (dbounds_up, bounds1)]

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
        scales[ind] = a1max
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # -------
        # a0

        kvar = 'a0'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = max(
            np.abs(data_min - a1max*lamb[0]),
            np.abs(data_max + a1max*lamb[0]),
        )
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
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
        scales[ind] = rate
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_mean * np.exp(rate/lambm) * lambm
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # -----------------
    # all gaussians
    # -----------------

    kfunc = 'gauss'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = (lambD/lambm) / _DEF_SCALES_FACTORS['shift']
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = lambD / _DEF_SCALES_FACTORS['width']
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # -----------------
    # all lorentz
    # -----------------

    kfunc = 'lorentz'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = (lambD/lambm) / _DEF_SCALES_FACTORS['shift']
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = lambD / _DEF_SCALES_FACTORS['width']
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # -----------------
    # all pvoigt
    # -----------------

    kfunc = 'pvoigt'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = (lambD/lambm) / _DEF_SCALES_FACTORS['shift']
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = lambD / _DEF_SCALES_FACTORS['width']
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = lambD / _DEF_SCALES_FACTORS['width']
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # -----------------
    # all voigt
    # -----------------

    kfunc = 'voigt'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_max - data_min
        bounds0[ind] = 0.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = (lambD/lambm) / _DEF_SCALES_FACTORS['shift']
        bounds0[ind] = -1.
        bounds1[ind] = 1.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = lambD / _DEF_SCALES_FACTORS['width']
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = lambD / _DEF_SCALES_FACTORS['width']
        bounds0[ind] = 1e-3
        bounds1[ind] = 2

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # ------------------
    # all pulse1
    # ------------------

    kfunc = 'pulse1'
    if dind.get(kfunc) is not None:

        _get_scales_bounds_pulse(
            dind=dind,
            kfunc=kfunc,
            ldins=ldins,
            # data
            data_min=data_min,
            data_max=data_max,
            lamb=lamb,
            lambd=lambd,
            lambm=lambm,
            lambD=lambD,
            # arrays to fill
            scales=scales,
            bounds0=bounds0,
            bounds1=bounds1,
        )

    # ------------------
    # all pulse2
    # ------------------

    kfunc = 'pulse2'
    if dind.get(kfunc) is not None:

        _get_scales_bounds_pulse(
            dind=dind,
            kfunc=kfunc,
            ldins=ldins,
            # data
            data_min=data_min,
            data_max=data_max,
            lamb=lamb,
            lambd=lambd,
            lambm=lambm,
            lambD=lambD,
            # arrays to fill
            scales=scales,
            bounds0=bounds0,
            bounds1=bounds1,
        )

    # ------------------
    # all lognorm
    # ------------------

    kfunc = 'lognorm'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_max - data_min
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # t0
        kvar ='t0'
        ind = dind['jac'][kfunc].get('t0')
        scales[ind] = 1
        bounds0[ind] = 0
        bounds1[ind] = 1

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = 0.5
        bounds0[ind] = 0.1
        bounds1[ind] = 10

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

        # mu
        kvar = 'mu'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = 0.5 * (np.log((lambD / 10)**2/(np.exp(0.5**2) - 1)) - 0.5**2)
        bounds0[ind] = 1.e-2
        bounds1[ind] = 10

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    return scales, bounds0, bounds1


#############################################
#############################################
#       get x0
#############################################


def _get_x0(
    nxfree=None,
    lamb=None,
    data=None,
    iok=None,
    dind=None,
    dx0=None,
    scales=None,
):

    # ------------------
    # initialize
    # ------------------

    x0 = np.zeros((nxfree,), dtype=float)

    # ------------------
    # prepare
    # ------------------

    lambD = lamb[-1] - lamb[0]
    lambd = lamb[1] - lamb[0]
    lambm = np.mean(lamb)

    data_max = np.nanmax(data[iok])
    data_min = np.nanmin(data[iok])
    data_mean = np.nanmean(data[iok])
    data_median = np.median(data[iok])
    data_pulse_sign = np.sign(data_mean - data_median)

    ldins = [(dx0, x0)]

    # ------------------
    # all linear
    # ------------------

    kfunc = 'linear'
    if dind.get(kfunc) is not None:

        a1max = ((data_max - data_min) / lambD) / 10

        # -------
        # a1

        kvar = 'a1'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = a1max / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # -------
        # a0

        kvar = 'a0'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = np.abs(data_min - a1max*lamb[0]) / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
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
        ]) / 10.

        # rate
        kvar = 'rate'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = rate / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        scales[ind] = data_mean * np.exp(rate/lambm) * lambm / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # -----------------
    # all gaussians
    # -----------------

    kfunc = 'gauss'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = (data_max - data_min) / 2 / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['shift'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['width'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # -----------------
    # all lorentz
    # -----------------

    kfunc = 'lorentz'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = (data_max - data_min) / 2 / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['shift'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['width'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # -----------------
    # all pvoigt
    # -----------------

    kfunc = 'pvoigt'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = (data_max - data_min) / 2. / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['shift'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['width'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['width'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # -----------------
    # all voigt
    # -----------------

    kfunc = 'voigt'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = (data_max - data_min) / 2 / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # vccos
        kvar = 'vccos'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['shift'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['width'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # gam
        kvar = 'gam'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = lambD / _DEF_X0_FACTORS['width'] / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # ------------------
    # all pulse1
    # ------------------

    kfunc = 'pulse1'
    if dind.get(kfunc) is not None:

        _get_x0_pulse(
            dind=dind,
            kfunc=kfunc,
            ldins=ldins,
            # data
            data_min=data_min,
            data_max=data_max,
            data_pulse_sign=data_pulse_sign,
            lambd=lambd,
            lambm=lambm,
            lambD=lambD,
            # arrays to fill
            x0=x0,
            scales=scales,
        )

    # ------------------
    # all pulse2
    # ------------------

    kfunc = 'pulse2'
    if dind.get(kfunc) is not None:

        _get_x0_pulse(
            dind=dind,
            kfunc=kfunc,
            ldins=ldins,
            # data
            data_min=data_min,
            data_max=data_max,
            data_pulse_sign=data_pulse_sign,
            lambd=lambd,
            lambm=lambm,
            lambD=lambD,
            # arrays to fill
            x0=x0,
            scales=scales,
        )

    # ------------------
    # all lognorm
    # ------------------

    kfunc = 'lognorm'
    if dind.get(kfunc) is not None:

        # amp
        kvar = 'amp'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = (data_max - data_min) / 2 / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # t0
        kvar ='t0'
        ind = dind['jac'][kfunc].get('t0')
        x0[ind] = 0.3

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # sigma
        kvar = 'sigma'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = 0.5

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

        # mu
        kvar = 'mu'
        ind = dind['jac'][kfunc].get(kvar)
        x0[ind] = 1

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    return x0


# #######################
# generic pdate from dict
# #######################


def _update_din_from_user(din, kfunc, kvar, val, scales=None):

    if din.get(kfunc, {}).get(kvar) is not None:
        ind = din[kfunc][kvar]['ind']
        if scales is None:
            val[ind] = din[kfunc][kvar]['val']
        else:
            val[ind] = din[kfunc][kvar]['val'] / scales[ind]

    return


# ###########################
# generic pulse scales bounds
# ###########################


def _get_scales_bounds_pulse(
    dind=None,
    kfunc=None,
    ldins=None,
    # data
    data_min=None,
    data_max=None,
    data_pulse_sign=None,
    lamb=None,
    lambd=None,
    lambm=None,
    lambD=None,
    # arrays to fill
    scales=None,
    bounds0=None,
    bounds1=None,
):


    # amp
    kvar = 'amp'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        scales[ind] = data_max - data_min
        bounds0[ind] = -10.
        bounds1[ind] = 10.

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # t0
    kvar = 't0'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        scales[ind] = 1
        bounds0[ind] = 0
        bounds1[ind] = 1

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # tup
    kvar = 't_up'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        scales[ind] = 0.05 * lambD
        bounds0[ind] = 1.e-3
        bounds1[ind] = 20

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    # tdown
    kvar = 't_down'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        scales[ind] = 0.2 * lambD
        bounds0[ind] = 1.e-3
        bounds1[ind] = 20

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=None if ii == 0 else scales
            )

    return


# #######################
# generic pulse x0
# #######################


def _get_x0_pulse(
    dind=None,
    kfunc=None,
    ldins=None,
    # data
    data_min=None,
    data_max=None,
    data_pulse_sign=None,
    lambd=None,
    lambm=None,
    lambD=None,
    # arrays to fill
    x0=None,
    scales=None,
):

    # amp
    kvar = 'amp'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        x0[ind] = (data_max - data_min) * data_pulse_sign / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # t0
    kvar = 't0'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        x0[ind] = 0.3

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # tup
    kvar = 't_up'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        x0[ind] = 0.05 * lambD / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    # tdown
    kvar = 't_down'
    ind = dind['jac'][kfunc].get(kvar)
    if ind is not None:
        x0[ind] = 0.2 * lambD / scales[ind]

        for ii, (din, val) in enumerate(ldins):
            _update_din_from_user(
                din, kfunc, kvar, val,
                scales=scales,
            )

    return