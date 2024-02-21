# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:44:51 2024

@author: dvezinet
"""


import itertools as itt
import datastock as ds


# local



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
    # keys
    key=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    # dict
    dinitial=None,
    dscales=None,
    dconstants=None,
    domain=None,
    # optional 2d fit
    key_bs=None,
):

    # ---------------------
    # basic check on inputs
    # ---------------------

    (
         key,
         key_model,
         key_data,
         key_lamb,
         key_bs,
         key_bs_vect,
         # derived
         ref,
         axis_lamb,
    ) = _check_keys(**locals())

    # ---------------------
    # domain
    # ---------------------

    # prepare ddata
    ddata = {'lamb': coll.ddata[key]['data']}
    if key_bs is not None:
        ddata.update({key_bs_vect: coll.ddata[key_bs_vect]['data']})

    # check domain
    domain = _domain.main(
        ddata=ddata,
        domain=domain,
    )
    del ddata

    # ---------------------
    # dscales
    # ---------------------

    if dscales is None:
        dscales = _dscales.main()

    # ---------------------
    # dinitial
    # ---------------------

    if dinitial is None:
        dinitial = _get_dinital_from_data(
        )


    # ---------------------
    # dconstants
    # ---------------------

    if dconstants is None:
        dconstants = _get_dconstants_from_data()

    # -------- BACKUP ------------
    # Add dscales, dx0 and dbounds

    dinput['dscales'] = fit12d_dscales(dscales=dscales, dinput=dinput)
    dinput['dbounds'] = fit12d_dbounds(dbounds=dbounds, dinput=dinput)
    dinput['dx0'] = fit12d_dx0(dx0=dx0, dinput=dinput)
    dinput['dconstants'] = fit12d_dconstants(
        dconstants=dconstants, dinput=dinput,
    )

    # ---------------------
    # store
    # ---------------------

    wsf = coll._which_fit
    dobj = {
        wsf: {
            key: {
                'key_model': key_model,
                'key_data': key_data,
                'key_lamb': key_lamb,
                'key_bs': key_bs,
                'domain': domain,
                'dinitial': dinitial,
                'dscales': dscales,
                'dconstants': dconstants,
                'dindok': {'ind': indok, 'meaning': dindok},
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
    key_lamb=None,
    key_bs=None,
    # unused
    **kwdargs,
):

    # -------------
    # key
    # -------------

    wsf = coll._which_fit
    key = ds._generic_check._obj_key(
        d0=coll.get(wsf, {}),
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

    # -------------
    # key_lamb
    # -------------

    # key_lamb
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['monot'] == (True,)
        and v0['ref'] in ref
    ]
    key_lamb = ds._generic_check._check_var(
        key_lamb, 'key_lamb',
        types=str,
        allowed=lok,
    )

    # axis_lamb
    axis_lamb = ref.index(coll.ddata[key_lamb]['ref'][0])

    # -------------
    # key_bs
    # -------------

    if key_bs is not None:

        wbs = coll._which_bsplines
        lok = [
            k0 for k0, v0 in coll.dobj.get(wbs, {}).items()
            if len(v0['shape']) == 1
        ]
        key_bs = ds._generic_check._check_var(
            key_bs, 'key_bs',
            types=str,
            allowed=lok,
        )

    # -------------
    # dict
    # -------------

    return (
        key,
        key_model,
        key_data,
        key_lamb,
        key_bs,
        # derived
        ref,
        axis_lamb,
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


def _check_dscales(
    coll=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    dscales=None,
):