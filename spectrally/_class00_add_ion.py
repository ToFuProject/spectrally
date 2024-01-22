# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import datastock as ds



#############################################
#############################################
#       Main
#############################################


def add_ion(
    coll=None,
    key=None,
):

    # ---------------------
    # list of possible ions
    # ---------------------

    table = _get_table()

    # ----------------
    # identify ion
    # ----------------

    lok = None
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # --------------
    # get proper key
    # --------------


    # ----------------
    # get iso-electronic sequence
    # ----------------

    isoelect = _get_isoelect(Z, q)

    # -------------
    # add if not in Collection
    # -------------

    if key not in coll.dobj.get(coll._which_ion, {}).keys():

        coll.add_obj(
            which=coll._which_ion,
            key=key,
            element=element,
            A=A,
            Z=Z,
            q=q,
            isoelect=isoelect,
        )

    return


#############################################
#############################################
#       Database
#############################################


def _get_dtable():

    # -------------------
    # load periodic table

    import periodictable

    # ----------------
    # build table

    dtable = {
        'H': {
            'name': 'hydrogen',
            'A': 1,
            'Z': 1,
        },
    }



    return table


#############################################
#############################################
#       iso-electronic sequence
#############################################


def _get_isoelect(Z, q, table=table):

    k0 = symb[(Z0-q0 == Z)]
    isoelect = f"{k0}-like"

    return isoelect
