# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import datastock as ds


#############################################
#############################################
#       add
#############################################


def add_lines(
    coll=None,
    key=None,
    ion=None,
    lamb0=None,
    transition=None,
    pec=None,
    source=None,
    symbol=None,
    **kwdargs,
):

    # -----------
    # check

    ion, lamb0, transition, source, symbol = _check(
        coll=coll,
        key=key,
        ion=ion,
        lamb0=lamb0,
        transition=transition,
        pec=pec,
        source=source,
        symbol=symbol,
    )

    # -----------
    # add ion

    if ion is not None and ion not in coll.dobj.get('ion', {}).keys():
        coll.add_ion(ion)

    # --------
    # add line

    coll.add_obj(
        which=coll._which_lines,
        key=key,
        ion=ion,
        lamb0=lamb0,
        transition=transition,
        pec=pec,
        source=source,
        symbol=symbol,
        **kwdargs,
    )

    return


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    key=None,
    ion=None,
    lamb0=None,
    transition=None,
    pec=None,
    source=None,
    symbol=None,
):

    # -----------------
    # check consistency
    # -----------------




    # ---------------
    # check items
    # ---------------


    # ------------
    # ion

    if ion is not None:
        ion = ds._generic_check._check_var(
            ion, 'ion',
            types=str,
        )

    # ------------
    # lamb0

    lamb0 = float(ds._generic_check._check_var(
        lamb0, 'lamb0',
        types=(int, float),
        sign=">0",
    ))

    # ------------
    # transition

    if transition is not None:
        transition = ds._generic_check._check_var(
            transition, 'transition',
            types=str,
        )

    # ------------
    # source

    if source is not None:
        source = ds._generic_check._check_var(
            source, 'source',
            types=str,
        )

    # ------------
    # symbol

    symbol = ds._generic_check._obj_key(
        {
            v0.get('symbol'): None
            for v0 in coll.dobj.get(coll._which_lines, {}).values()
        },
        short='l',
        key=symbol,
        ndigits=3,
    )

    return ion, lamb0, transition, source, symbol
