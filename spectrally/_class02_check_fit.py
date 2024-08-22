# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:44:51 2024

@author: dvezinet
"""


import datastock as ds


import numpy as np


# local
from . import _class01_valid as _valid


#############################################
#############################################
#       DEFAULTS
#############################################


_LFIT_ORDER = [
    'sol',
    'model', 'data', 'sigma', 'lamb', 'bs',
    'positive', 'nsigma', 'fraction',
    'mask', 'domain', 'focus',
]


#############################################
#############################################
#       fit CHECK
#############################################


def _check(
    coll=None,
    # key
    key=None,
    # model
    key_model=None,
    # data and noise
    key_data=None,
    key_sigma=None,
    # wavelength and phi
    key_lamb=None,
    key_bs_vect=None,
    key_bs=None,
    # dict
    dparams=None,
    dvalid=None,
    # compute options
    chain=None,
):

    # ---------------------
    # basic check on inputs
    # ---------------------

    (
        key,
        key_model,
        key_data,
        key_sigma,
        key_lamb,
        key_bs_vect,
        key_bs,
        # derived
        ref,
        ref0,
        shape,
        shape0,
        axis_lamb,
        axis_bs,
    ) = _check_keys(**locals())

    # ---------------------
    # domain
    # ---------------------

    # prepare ddata
    ddata = {
        key_lamb: {
            'data': coll.ddata[key_lamb]['data'],
            'ref': coll.ddata[key_lamb]['ref'][0],
        }
    }
    if key_bs is not None:
        ddata.update({
            key_bs_vect: {
                'data': coll.ddata[key_bs_vect]['data'],
                'ref': coll.ddata[key_bs_vect]['ref'][0],
            },
        })

    # ---------------------
    # mask & domain
    # ---------------------

    dvalid = _valid.mask_domain(
        # resources
        coll=coll,
        key_data=key_data,
        key_lamb=key_lamb,
        key_bs_vect=key_bs_vect,
        # options
        dvalid=dvalid,
        ref=ref,
        ref0=ref0,
        shape0=shape0,
    )

    # ---------------------
    # validity
    # ---------------------

    dvalid = _valid.valid(
        coll=coll,
        key=key,
        key_data=key_data,
        key_lamb=key_lamb,
        key_bs=key_bs,
        dvalid=dvalid,
        ref=ref,
        ref0=ref0,
    )

    # ---------------------
    # dparams
    # ---------------------

    # if dparams is None:
        # dparams = _dparams.main()

    # # -------- BACKUP ------------
    # # Add dscales, dx0 and dbounds

    # dinput['dscales'] = fit12d_dscales(dscales=dscales, dinput=dinput)
    # dinput['dbounds'] = fit12d_dbounds(dbounds=dbounds, dinput=dinput)
    # dinput['dx0'] = fit12d_dx0(dx0=dx0, dinput=dinput)
    # dinput['dconstants'] = fit12d_dconstants(
        # dconstants=dconstants,
        # dinput=dinput,
    # )

    # ---------------------
    # store
    # ---------------------

    wsf = coll._which_fit
    dobj = {
        wsf: {
            key: {
                'key_model': key_model,
                'key_data': key_data,
                'key_sigma': key_sigma,
                'key_lamb': key_lamb,
                'key_bs': key_bs,
                'key_bs_vect': key_bs_vect,
                'key_sol': None,
                'key_std': None,
                'dparams': dparams,
                'dvalid': dvalid,
            },
        },
    }

    coll.update(dobj=dobj)

    return


#############################################
#############################################
#        check keys
#############################################


def _check_keys(
    coll=None,
    # keys
    key=None,
    key_model=None,
    key_data=None,
    key_sigma=None,
    key_lamb=None,
    key_bs_vect=None,
    key_bs=None,
    # unused
    **kwdargs,
):

    # -------------
    # key
    # -------------

    wsf = coll._which_fit
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get(wsf, {}),
        short='sf',
        key=key,
        ndigits=2,
    )

    # -------------
    # key_model
    # -------------

    wsm = coll._which_model
    lok = list(coll.dobj.get(wsm, {}).keys())
    key_model = ds._generic_check._check_var(
        key_model, 'key_model',
        types=str,
        allowed=lok,
    )

    # -------------
    # key_data
    # -------------

    # key_data
    lok = list(coll.ddata.keys())
    key_data = ds._generic_check._check_var(
        key_data, 'key_data',
        types=str,
        allowed=lok,
    )

    # derive refs
    ref = coll.ddata[key_data]['ref']
    shape = coll.ddata[key_data]['shape']

    # -------------
    # key_lamb
    # -------------

    # key_lamb
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['monot'] == (True,)
        and v0['ref'][0] in ref
    ]
    key_lamb = ds._generic_check._check_var(
        key_lamb, 'key_lamb',
        types=str,
        allowed=lok,
        extra_msg="Should be a 1d vector, strictly monotonous",
    )

    # axis_lamb
    ref_lamb = coll.ddata[key_lamb]['ref'][0]
    axis_lamb = ref.index(ref_lamb)

    # -------------
    # key_bs
    # -------------

    c0 = (
        len(ref) >= 2
        and key_bs_vect is not None
    )
    if c0:

        # key_bs_vect
        lok = [
            k0 for k0, v0 in coll.ddata.items()
            if v0['monot'] == (True,)
            and v0['ref'][0] in ref
            and v0['ref'][0] != coll.ddata[key_lamb]['ref'][0]
        ]
        key_bs_vect = ds._generic_check._check_var(
            key_bs_vect, 'key_bs_vect',
            types=str,
            allowed=lok,
        )

        axis_bs = ref.index(coll.ddata[key_bs_vect]['ref'][0])

        # units, dim
        units = coll.ddata[key_bs_vect]['units']
        quant = coll.ddata[key_bs_vect]['quant']
        dim = coll.ddata[key_bs_vect]['dim']

        # key_bs
        wbs = coll._which_bsplines
        lok = [
            k0 for k0, v0 in coll.dobj.get(wbs, {}).items()
            if len(v0['shape']) == 1
            and (
                coll.ddata[v0['apex'][0]]['units'] == units
                or coll.ddata[v0['apex'][0]]['quant'] == quant
                or coll.ddata[v0['apex'][0]]['dim'] == dim
            )
        ]
        key_bs = ds._generic_check._check_var(
            key_bs, 'key_bs',
            types=str,
            allowed=lok,
        )

        # ref0
        ref0 = tuple([
            rr for ii, rr in enumerate(ref)
            if ii in [axis_lamb, axis_bs]
        ])

        # shape0
        shape0 = tuple([
            ss for ii, ss in enumerate(shape)
            if ii in [axis_lamb, axis_bs]
        ])

    else:
        key_bs_vect = None
        key_bs = None
        axis_bs = None
        shape0 = (shape[axis_lamb],)
        ref0 = (ref_lamb,)

    return (
        key,
        key_model,
        key_data,
        key_sigma,
        key_lamb,
        key_bs_vect,
        key_bs,
        # derived
        ref,
        ref0,
        shape,
        shape0,
        axis_lamb,
        axis_bs,
    )


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([which] + _LFIT_ORDER)

    # ---------------------------
    # data
    # ---------------------------

    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    lar0 = []
    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # add nb of func of each type
        dfit = coll.dobj[which][k0]
        for k1 in _LFIT_ORDER:

            if k1 in ['model', 'data', 'sigma', 'lamb', 'bs', 'sol']:
                nn = '' if dfit[f"key_{k1}"] is None else dfit[f"key_{k1}"]

            elif k1 in ['nsigma', 'fraction', 'positive']:
                nn = str(dfit['dvalid'][k1])

            elif k1 in ['mask']:
                nn = str(dfit['dvalid']['mask']['key'] is not None)

            elif k1 in ['domain']:
                c0 = all([
                    len(v0['spec']) == 1
                    and np.allclose(v0['spec'][0], np.inf*np.r_[-1, 1])
                    for k0, v0 in dfit['dvalid']['domain'].items()
                ])
                if c0:
                    nn = ''
                else:
                    lk = list(dfit['dvalid']['domain'].keys())
                    if len(lk) == 2 and lk[0] != dfit['key_lamb']:
                        lk = [lk[1], lk[0]]

                    nn = ', '.join([
                        str(len(dfit['dvalid']['domain'][k0]['spec']))
                        for k0 in lk
                    ])

            elif k1 in ['focus']:
                if dfit['dvalid'].get('focus') is None:
                    nn = ''
                else:
                    nn = len(dfit['dvalid']['focus'])
                    nn = f"{nn} / {dfit['dvalid']['focus_logic']}"

            arr.append(nn)

        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar


# ###########################################
# ###########################################
#        check dinitial
# ###########################################


# def _check_dscales(
    # coll=None,
    # key_model=None,
    # key_data=None,
    # key_lamb=None,
    # dscales=None,
# ):