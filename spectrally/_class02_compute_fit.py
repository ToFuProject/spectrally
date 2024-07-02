# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:44:51 2024

@author: dvezinet
"""

# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import datastock as ds


# local
from . import _class02_compute_fit_1d as _compute_fit_1d


#############################################
#############################################
#       DEFAULTS
#############################################


#############################################
#############################################
#       Main
#############################################


def main(
    coll=None,
    key=None,
    # options
    verb=None,
    timing=None,
):
    """ Compute the fit of any previously added spectral fit

    """

    # ------------
    # check inputs
    # ------------

    (
        key, is1d,
        key_model,
        key_data, key_lamb,
        ref_data, ref_lamb,
        shape_data, axis,
        verb, timing,
    ) = _check(
        coll=coll,
        key=key,
        # options
        verb=verb,
        timing=timing,
    )

    # ------------
    # fit
    # ------------

    if is1d is True:
        compute = _compute_fit_1d.main

    else:
        pass

    dout = compute(
        coll=coll,
        key=key,
        key_model=key_model,
        key_data=key_data,
        key_lamb=key_lamb,
        ref_data=ref_data,
        ref_lamb=ref_lamb,
        shape_data=shape_data,
        axis=axis,
        verb=verb,
        timing=timing,
    )

    # ------------
    # store
    # ------------

    _store(
        coll=coll,
        dout=dout,
    )

    return


#############################################
#############################################
#       Check
#############################################


def _check(
    coll=None,
    key=None,
    # options
    verb=None,
    timing=None,
):

    # --------------
    # key
    # --------------

    wsf = coll._which_fit
    lok_1d = [
        k0 for k0, v0 in coll.dobj.get(wsf, {}).items()
        if v0['key_bs'] is None
    ]
    lok_2d = [
        k0 for k0, v0 in coll.dobj.get(wsf, {}).items()
        if v0['key_bs'] is not None
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok_1d + lok_2d,
    )

    # is1d
    is1d = key in lok_1d

    if not is1d:
        msg = "2d fit not implemented yet"
        raise NotImplementedError(msg)

    # -----------
    # derive keys

    key_model = coll.dobj[wsf][key]['key_model']
    key_data = coll.dobj[wsf][key]['key_data']
    key_lamb = coll.dobj[wsf][key]['key_lamb']
    shape_data = coll.ddata[key_data]['data'].shape
    ref_data = coll.ddata[key_data]['ref']
    ref_lamb = coll.ddata[key_lamb]['ref']
    axis = ref_data.index(ref_lamb[0])

    # --------------
    # verb
    # --------------

    def_verb = 2
    verb = ds._generic_check._check_var(
        verb, 'verb',
        default=def_verb,
        allowed=[False, True, 0, 1, 2, 3],
    )
    if verb is True:
        verb = def_verb

    # --------------
    # timing
    # --------------

    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
    )

    return (
        key, is1d,
        key_model,
        key_data, key_lamb,
        ref_data, ref_lamb,
        shape_data, axis,
        verb, timing,
    )


#############################################
#############################################
#       store
#############################################


def _store(
    coll=None,
    dout=None,
):

    return
