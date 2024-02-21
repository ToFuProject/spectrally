# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:33:22 2024

@author: dvezinet
"""


import copy


import itertools as itt
import numpy as np
import datastock as ds


#############################################
#############################################
#       main
#############################################


def main(
    domain=None,
    ddata=None,
):

    # ------------
    # check inputs
    # ------------

    domain = _check(
        domain=domain,
        ddata=ddata,
    )

    # ----------
    # Apply
    # ----------

    for k0, v0 in domain.items():

        # initialize
        indin = np.zeros(ddata[k0].shape, dtype=bool)
        indout = np.zeros(ddata[k0].shape, dtype=bool)

        # loop on domain bits
        for v1 in v0['spec']:
            indi = (ddata[k0] >= v1[0]) & (ddata[k0] <= v1[1])
            if isinstance(v1, tuple):
                indout |= indi
            else:
                indin |= indi

        # apply
        domain[k0]['ind'] = indin & (~indout)

    return domain


#############################################
#############################################
#       check
#############################################


def _check(
    domain=None,
    ddata=None,
):

    # ----------------
    # check ddata
    # ----------------

    c0 = (
        isinstance(ddata, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, np.ndarray)
            and v0.ndim == 1
            for k0, v0 in ddata.items()
        ])
    )
    if not c0:
        msg = (
            "Arg ddata must be a dict of (str, 1d np.ndarray)\n"
            f"Provided:\n{ddata}\n"
        )
        raise Exception(msg)

    # --------------
    # check domain
    # --------------

    # ------------
    # special case

    if len(ddata) == 1 and not isinstance(domain, dict):
        domain = {list(ddata.keys())[0]: domain}

    # --------
    # if None

    if domain is None:
        domain = {
            k0: {
                'spec': [np.inf*np.r_[-1., 1.]],
                'minmax': np.inf*np.r_[-1., 1.],
            }
            for k0 in ddata.keys()
        }

    # ----------
    # general

    c0 = (
        isinstance(domain, dict)
        and all([k0 in ddata.keys() for k0 in domain.keys()])
    )
    if not c0:
        msg = (
            "Arg domain must be a dict with keys:\n"
            + "\n".join([f"\t- {k0}" for k0 in ddata.keys()])
            + "\nProvided:\n{domain}\n"
        )
        raise Exception(msg)

    # ------------
    # loop on keys
    # ------------

    for k0, v0 in ddata.keys():

        # -------
        # trivial

        if domain.get(k0) is None:
            domain[k0] = {
                'spec': [np.inf*np.r_[-1., 1.]],
                'minmax': np.inf*np.r_[-1., 1.],
            }
            continue

        # --------
        # sequence

        if not isinstance(domain[k0], dict):
            domain[k0] = {'spec': domain[k0]}
        else:
            domain = copy.deepcopy(domain)

        # --------
        # subdict

        c0 = (
            isinstance(domain[k0], dict)
            and all([k1 in ['spec', 'minmax'] for k1 in domain[k0].keys()])
        )
        if not c0:
            msg = (
                f"Arg domain['{k0}'] must be a dict with keys:\n"
                + "\n".join([f"\t- {k0}" for k0 in ['spec', 'minmax']])
                + "\nProvided:\n{domain[k0]}\n"
            )
            raise Exception(msg)

        # -------------
        # check spec

        spec = domain[k0]['spec']
        if isinstance(spec, tuple):
            spec = [spec]

        c0 = (
            isinstance(spec, (list, np.ndarray))
            and len(spec) == 2
            and all([np.isscalar(ss) for ss in spec])
        )
        if c0:
            spec = [spec]

        c0 = (
            isinstance(spec, list)
            and all([
                isinstance(s0, (list, np.ndarray, tuple))
                and len(s0) == 2
                and all([np.isscalar(s1) for s1 in s0])
                and not np.any(np.isnan(s0))
                and s0[0] < s0[1]
                for s0 in spec
            ])
        )
        if not c0:
            msg = (
                f"Arg domain['{k0}']['spec'] must be a list of list/tuples\n"
                "\t Each must be a sequence of 2 increasing floats\n"
                f"Provided:\n{domain[k0]['spec']}"
            )
            raise Exception(msg)

        # -------
        # minmax

        domain[k0]['minmax'] = (
            np.nanmin(domain[k0]['spec']),
            np.nanmax(domain[k0]['spec']),
        )

    return domain