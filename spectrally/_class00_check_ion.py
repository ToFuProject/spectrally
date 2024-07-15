# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import os


import numpy as np
import pandas as pd
import datastock as ds


_PATH_HERE = os.path.dirname(__file__)


#############################################
#############################################
#       add
#############################################


def add_ion(
    coll=None,
    key=None,
):

    # -------------------------
    # list of possible elements
    # -------------------------

    symbols, names, Z, A = _get_table()

    # ----------------
    # identify ion
    # ----------------

    # check if all ions should be added
    add_all = _check_key(key, symbols=symbols)

    # add all ions for given element
    if add_all is True:
        ions = [f"{key}{qq}+" for qq in range(0, Z[symbols.index(key)+1])]
        for ion in ions:
            coll.add_ion(ion)

    # add only specified ion
    else:

        # check element
        symbol = ''.join([ss for ss in key[:-1] if not ss.isnumeric()])
        symbol = ds._generic_check._check_var(
            symbol.capitalize(), 'symbol',
            types=str,
            allowed=symbols,
        )
        ind = symbols.index(symbol)

        # check charge
        charge = int(''.join([ss for ss in key[1:-1] if ss.isnumeric()]))
        charge = ds._generic_check._check_var(
            charge, 'charge',
            types=int,
            allowed=np.arange(0, Z[ind]+1),
        )

        # --------------
        # get proper key
        # --------------

        name = names[ind]
        z = Z[ind]
        aa = A[ind]

        # rebuild key
        key = f"{symbol}{charge}+"

        # ----------------
        # get iso-electronic sequence
        # ----------------

        isoelect = _get_isoelect(symbols=symbols, Z=Z, z=z, q=charge)

        # -------------
        # add if not in Collection
        # -------------

        if key not in coll.dobj.get(coll._which_ion, {}).keys():

            coll.add_obj(
                which=coll._which_ion,
                key=key,
                element=symbol,
                element_name=name,
                A=aa,
                Z=z,
                q=charge,
                isoelect=isoelect,
            )

    return


#############################################
#############################################
#       remove
#############################################


def remove_ion(coll=None, key=None, propagate=None):

    # -------------------------
    # list of possible elements
    # -------------------------

    symbols, names, Z, A = _get_table()

    # ----------------
    # identify ion
    # ----------------

    # check if all ions should be added
    add_all = _check_key(key, symbols=symbols)

    # add all ions for given element
    if add_all is True:
        ions = [f"{key}{qq}+" for qq in range(0, Z[symbols.index(key)+1])]
        for ion in ions:
            if ion in coll.dobj.get('ion', {}).keys():
                coll.remove_ion(ion, propagate=propagate)

    # add only specified ion
    else:
        if key in coll.dobj.get('ion', {}).keys():
            coll.remove_obj(which='ion', key=key, propagate=propagate)


#############################################
#############################################
#       check key
#############################################


def _call_err(key):
    msg = (
        "Arg key must be a str of the form 'LLXX+', where:\n"
        "\t- LL: is one or 2 letter\n"
        "\t- XX: is one or 2 integers\n"
        "Or it can be just an element 'LL' => all ions are added\n"
        f"Provided:\n\t{key}\n"
    )
    raise Exception(msg)


def _check_key(key, symbols=None):

    c0 = (
        isinstance(key, str)
        and (not key[0].isnumeric())
    )

    if not c0:
        _call_err(key)

    # only element name provided => add all ions
    if len(key) <= 2 and not any([ss.isnumeric() or ss == '+' for ss in key]):
        key = key.capitalize()
        if key not in symbols:
            _call_err(key)
        return True

    # otherwise => add single ion
    else:

        c0 = (
            len(key) >= 3
            and key.endswith('+')
            and len([ss.isnumeric() for ss in key[1:-1]]) > 0
        )

        if not c0:
            _call_err(key)
        return False


#############################################
#############################################
#       Database
#############################################


def _get_table():
    """ get symbols, names, atomic numbers (Z) and masses, sorted by Z """

    try:
        symbols, names, Z, A = _get_from_periodictable()

    except Exception as err:
        symbols, names, Z, A = _get_from_local()

    return symbols, names, Z, A


def _get_from_periodictable(update=False):
    """ Get elements characteristics from library periodictable
    """

    import periodictable as pt

    symbols = [str(el) for el in pt.elements]
    Z = [getattr(pt, el).number for el in symbols]
    A = [getattr(pt, el).mass for el in symbols]
    names = [getattr(pt, el).name for el in symbols]

    # sorting index (by atomic number)
    inds = np.argsort(Z)

    # sort
    symbols = tuple([symbols[ii] for ii in inds])
    names = tuple([names[ii].lower() for ii in inds])
    Z = tuple([Z[ii] for ii in inds])
    A = tuple([A[ii] for ii in inds])

    return symbols, names, Z, A


def _get_from_local():

    pfe = os.path.join(_PATH_HERE, '_class00_check_ions_table.csv')
    df = pd.read_csv(pfe)

    symbols = df.symbol.to_list()
    names = df.name.to_list()
    Z = df.Z.to_list()
    A = df.A.to_list()

    return symbols, names, Z, A


#############################################
#############################################
#       iso-electronic sequence
#############################################


def _get_isoelect(symbols=None, Z=None, z=None, q=None):

    k0 = symbols[Z.index(z-q)]
    return f"{k0}-like"
