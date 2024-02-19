# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import json
import itertools as itt


import numpy as np
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


#############################################
#############################################
#       add ions from lines dobj
#############################################


def _add_lines_from_dobj(coll=None, dref=None, ddata=None, dobj=None):

    # ---------
    # add ions

    lions = sorted(dobj.get('ion', {}).keys())
    if len(lions) > 0:
        for ion in lions:
            coll.add_ion(ion)
        del dobj['ion']

    # -------
    # update

    coll.update(ddata=ddata, dref=dref, dobj=dobj)

    return


#############################################
#############################################
#      remove lines
#############################################


def remove_lines(
    coll=None,
    keys=None,
    propagate=None,
):

    # ----------------
    # check inputs
    # ----------------

    # --------
    # keys

    if isinstance(keys, str):
        keys = [keys]

    wsl = coll._which_lines
    lok = list(coll.dobj.get(wsl, {}).keys())
    keys = ds._generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
        default=lok,
    )

    # --------------
    # propagate

    propagate = ds._generic_check._check_var(
        propagate, 'propagate',
        types=bool,
        default=True,
    )

    # -----------------
    # propagate
    # -----------------

    if propagate is True:

        dlines = coll.dobj.get(wsl)
        lout = [k0 for k0 in dlines.keys() if k0 not in keys]

        # ions
        lion_in = set([
            dlines[k0]['ion'] for k0 in keys
            if dlines[k0].get('ion') is not None
        ])
        lion_out = set([
            dlines[k0]['ion'] for k0 in lout
            if dlines[k0].get('ion') is not None
        ])
        lions = list(lion_in.difference(lion_out))

        # sources
        ls_in = set([
            dlines[k0]['source'] for k0 in keys
            if dlines[k0].get('source') is not None
        ])
        ls_out = set([
            dlines[k0]['source'] for k0 in lout
            if dlines[k0].get('source') is not None
        ])
        lsources = list(ls_in.difference(ls_out))

        # pec
        lpec_in = set([
            dlines[k0]['pec'] for k0 in keys
            if dlines[k0].get('pec') is not None
        ])
        lpec_out = set([
            dlines[k0]['pec'] for k0 in lout
            if dlines[k0].get('pec') is not None
        ])
        lpec = list(lpec_in.difference(lpec_out))

        # lref
        lref = list(set(itt.chain.from_iterable([
            coll.ddata[k0]['ref'] for k0 in lpec
        ])))

    # -----------------
    # remove
    # -----------------

    coll.remove_obj(which=coll._which_lines, key=keys)

    if propagate is True:
        if len(lions) > 0:
            coll.remove_obj(which='ion', key=lions)
        if len(lref) > 0:
            coll.remove_ref(lref, propagate=True)
        if len(lpec) > 0:
            coll.remove_data(lpec, propagate=True)
        if len(lsources) > 0:
            coll.remove_obj(which='source', key=lsources)

    return


#############################################
#############################################
#       save lines to file
#############################################


def _save_to_file(
    coll=None,
    keys=None,
    path=None,
    name=None,
    overwrite=None,
):

    # ---------------
    # check inputs
    # ---------------

    keys, pfe, overwrite = _check_to_file(
        coll=coll,
        keys=keys,
        path=path,
        name=name,
        overwrite=overwrite,
    )

    # --------------
    # extract lines
    # ---------------

    lparam = ['ion', 'source', 'lamb0', 'symbol', 'type', 'transition']
    dlines = coll.dobj.get(coll._which_lines, {})
    dout = {
        k0: {k1: dlines[k0][k1] for k1 in lparam}
        for k0 in keys
    }

    # --------------
    # save to file
    # --------------

    if pfe.endswith('json'):

        with open(pfe, 'w+') as fn:
            json.dump(
                dout,
                fn,
                indent=4,
                sort_keys=True,
                ensure_ascii=False,
                check_circular=True,
                allow_nan=True,
                separators=(', ', ': '),
            )

    else:
        np.savez(pfe, **dout)

    # ---------------
    # verb

    msg = f"Saved to:\n\t{pfe}"
    print(msg)

    return


#############################################
#############################################
#       save lines to file - check
#############################################


def _check_to_file(coll=None, keys=None, path=None, name=None, overwrite=None):

    # ---------------
    # check keys
    # ---------------

    if isinstance(keys, str):
        keys = [keys]

    lok = coll.dobj.get(coll._which_lines, {}).keys()
    keys = ds._generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        default=sorted(lok),
        allowed=lok,
    )

    # ---------------
    # check inputs
    # ---------------

    if path is None:
        path = '.'

    if not (isinstance(path, str) and os.path.isdir(path)):
        msg = (
            "Arg path must be a str, a valid path to a directory!\n"
            f"Provided:\n\t{path}\n"
        )
        raise Exception(msg)

    path = os.path.abspath(path)

    # ---------------
    # check inputs
    # ---------------

    name = ds._generic_check._check_var(
        name, 'name',
        types=str,
        default='spectrallines',
    )

    if name.count('.') == 0:
        name = f'{name}.json'

    if not any([name.endswith(ss) for ss in ['.npz', '.json']]):
        msg = (
            "Arg name must a file name ending in '.json' or '.npz'!\n"
            f"Provided:\n\t{name}\n"
        )
        raise Exception(msg)

    pfe = os.path.join(path, name)

    # ---------------
    # check inputs
    # ---------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    return keys, pfe, overwrite
