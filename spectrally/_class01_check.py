# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import datastock as ds


#############################################
#############################################
#       DEFAULTS
#############################################


_DTYPES = {
    'linear': {'var': ['c0', 'c1']},
    'exp': {'var': ['amp', 'rate']},
    'gauss': {'var': ['amp', 'shift', 'width']},
    'lorentz': {'var': ['x0', 'gamma']},
    'pvoigt': {'var': ['amp', 'shift', 'width', 't', 'x0', 'gamma']},
}


#############################################
#############################################
#       MODEL CHECK
#############################################


def _model(
    coll=None,
    dmodel=None,
):

    # ----------
    # key
    # ----------


    # --------------
    # check dmodel
    # --------------

    dmodel = _check_dmodel(
        coll=coll,
        key=key,
        dmodel=dmodel,
    )

    # --------------
    # Get initial
    # --------------

    initial = _initial_from_model(dmodel)

    # --------------
    # store
    # --------------

    dobj = {
        coll._which_model: {
            key: {
                'fkeys': sorted(dmodel.keys()),
                'model': dmodel,
                'constraints': None,
                'constants': None,
                'initial': initial,
                'scales': None,
                'domain': None,
            },
        },
    }

    coll.update(dobj=dobj)

    return


#############################################
#############################################
#       check dmodel
#############################################


def _dmodel_err(key, dmodel):
    return (
        f"For model '{key}' dmodel must be  dict of the form:\n"
        "\t'bck': 'exp',\n"
        "\t'l0': 'gauss',\n"
        "\t'l1': 'lorentz',\n"
        "\t...: ...,\n"
        "\t'ln': 'pvoigt',\n"
        f"Provided:\n{dmodel}"
    )


def _check_dmodel(
    coll=None,
    key=None,
    dmodel=None,
):

    # -------------
    # model
    # -------------

    if isinstance(dmodel, str):
        dmodel = [dmodel]

    if isinstance(dmodel, (tuple, list)):
        dmodel = {ii: mm for mm in dmodel}

    if not isinstance(dmodel, dict):
        raise Exception(_dmodel_err(key, dmodel))

    # ------------
    # check dict
    # ------------

    dmod2 = {}
    dout = {}
    ibck, il = 0, 0
    for k0, v0 in dmodel.items():

        # ----------
        # check str vs dict

        if isinstance(v0, dict):
            if isinstance(v0.get('type'), str):
                typ = v0['type']
            else:
                dout[k0] = v0
                continue
        elif isinstance(v0, str):
            typ = v0
        else:
            dout[k0] = v0
            continue

        # ----------
        # check type

        if typ not in _DTYPES.keys():
            dout[k0] = v0
            continue

        # ----------
        # check key

        if isinstance(k0, int):
            if typ in ['linear', 'exp']:
                k1 = f'bck{ibck}'
                ibck += 1
            else:
                k1 = f"l{il}"
                il += 1
        else:
            k1 = k0

        dmod2[k1] = {'type': typ}

    # ---------------
    # raise error
    # ---------------

    if len(dout) > 0:
        raise Exception(_dmodel_err(key, dout))

    return dmodel


#############################################
#############################################
#       Model check from lines
#############################################


def _initial_from_from_model(
    coll=None,
    key=None,
    dmodel=None,
):

    # -----------
    # initialize
    # ----------

    wsl = coll._which_lines

    dinit = {}
    for k0, v0 in dmodel.items():

        # -----
        # bck

        if v0['type'] == 'linear':
            dinit[k0] = {'c0': 0, 'c1': 0}

        elif v0['type'] == 'exp':
            dinit[k0] = {'c0': 0, 'c1': 0}

        else:

            # if from spectral lines
            if k0 in coll.dobj.get(wsl, {}).keys():
                lamb0 = coll.dobj[wsl][k0]['lamb0']
            else:
                lamb0 = 0

            if v0['type'] == 'gauss':
                dinit[k0] = {
                    'amp': 1,
                    'shift': lamb0,
                    'width': 1,
                }

            elif v0['type'] == 'lorentz':
                dinit[k0] = np.r_[0, lamb0, 0]

            elif v0['type'] == 'pvoigt':
                dinit[k0] = np.r_[0, lamb0, 0]


    # -------------
    # lines
    # -------------





    coll.add_spectral_model(key=key, dmodel=dmodel)
    coll.set_spectral_model_initial(key=key, initial=initial)

    return
