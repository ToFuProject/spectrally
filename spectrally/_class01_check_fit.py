# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:44:51 2024

@author: dvezinet
"""


import itertools as itt
import datastock as ds


# local
from . import _class01_valid as _valid



#############################################
#############################################
#       DEFAULTS
#############################################




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
        dvalid=dvalid,
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
                'dparams': dparams,
                'dvalid': dvalid,
                'sol': None,
            },
        },
    }

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


# ###########################################
# ###########################################
#        check domain
# ###########################################


def _check_domain(
    coll=None,
    key_lamb=None,
    domain=None,
):

    indok = np.zeros(data.shape, dtype=np.int8)
    if mask is not None:
        indok[:, ~mask] = -1

    inddomain, domain = apply_domain(lamb, domain=domain)
    if mask is not None:
        indok[:, (~inddomain) & mask] = -2
    else:
        indok[:, ~inddomain] = -2

    return ind_domain


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