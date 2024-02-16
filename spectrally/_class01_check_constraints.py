# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:53:54 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import datastock as ds


#############################################
#############################################
#       MODEL CHECK
#############################################


def _dconstraints(
    coll=None,
    key=None,
    dmodel=None,
    dconstraints=None,
):

    # ----------
    # check
    # ------------

    dconstraints, lvar = _check(
        key,
        dconstraints=dconstraints,
    )

    # --------------
    # compute matrix
    # --------------

    dconstraints = _compute_coefs_offset(
        dconstraints=dconstraints,
        lvar=lvar,
    )

    # --------------
    # store
    # --------------

    coll._dobj[coll._which_model][key]['dconstraints'] = dconstraints

    return


#############################################
#############################################
#       check dmodel
#############################################


def _err(key, dconstraints, lvar):
    return (
        f"For model '{key}' dmodel must be  dict of the form:\n"
        "\t'g0': {'ref': var0, vari: [c0, c1, c2], varj: [c0, c1, c2]},\n"
        "\t'g1': {'ref': varn, varh: [c0, c1, c2], vark: [c0, c1, c2]},\n"
        "Where [c0, c1, c2] are coefficnients of a quadratic polynom of ref:\n"
        "\t\t vari = c0 + c1 * ref + c2 * ref**2\n"
        "Where the var are keys to existing model function:\n"
        f"\t- available keys: {sorted(lvar)}\n\n"
        "Each key can only be used once\n"
        f"Provided:\n{dconstraints}"
    )


def _check(
    coll=None,
    key=None,
    dconstraints=None,
):

    # -------------
    # extract variale groups
    # -------------

    lvar = coll.get_spectral_model_variables(key, concatenate=True)

    # -------------
    # model
    # -------------

    c0 = (
        isinstance(dconstraints, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, dict)
            and all([
                k1 in lvar + ['ref']
                and (
                    (
                        k1 == 'ref'
                        and v1 in lvar
                    )
                    or
                    (
                        k1 in lvar
                        and hasattr(v1, '__iter__')
                        and len(v1) == 3
                    )
                )
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dconstraints.items()
        ])
    )

    if c0 is False:
        raise Exception(_err(key, dconstraints, lvar))

    # ------------
    # Format
    # ------------

    for k0, v0 in dconstraints.items():
        for k1, v1 in v0.keys():
            if k1 != 'ref':
                dconstraints[k0][k1] = np.atleast_1d(v1).astype(float).ravel()

    # --------------------------------
    # check each key is used once only
    # --------------------------------

    lin = itt.chain.from_iterable([
        [v0['ref']] + [k1 for ki in v0.keys() if k1 != 'ref']
        for v0 in dconstraints.values()
    ])

    linu = list(set(lin))
    if len(lin) > len(linu):
        raise Exception(_err(key, dconstraints, lvar))

    # safety double-check
    assert all([k0 in lvar for k0 in lin]), (lvar, lin)

    # ------------------
    # add missing keys
    # ------------------

    for k0 in set(lvar).difference(lin):
        dconstraints[k0] = {'ref': k0}

    return dconstraints, lvar


#############################################
#############################################
#       compute matrix
#############################################


def _compute_coefs_offset(
    dconstraints=None,
    lvar=None,
):

    # ----------------
    # get matrix shape
    # ----------------

    lg = sorted(dconstraints.keys())
    ng = len(dconstraints)
    nvar = len(lvar)

    c2 = np.zeros((nvar, ng), dtype=float)
    c1 = np.zeros((nvar, ng), dtype=float)
    c0 = np.zeros((nvar,), dtype=float)

    for ii, gg in enumerate(lg):

        iref = lvar.index(dconstraints[gg]['ref'])
        c1[iref, ii] = 1.

        for k1 in set(dconstraints[gg].keys()).difference(['ref']):
            ind = lvar.index(k1)

            c0[ind] = dconstraints[gg][k1][0]
            c1[ind, ii] = dconstraints[gg][k1][1]
            c2[ind, ii] = dconstraints[gg][k1][2]

    # ----------------
    # store
    # ----------------

    dconstraints = {
        'groups': lg,
        'dconst': dconstraints,
        'c0': c0,
        'c1': c1,
        'c2': c2,
    }

    return dconstraints