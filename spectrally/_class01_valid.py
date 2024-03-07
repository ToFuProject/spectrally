# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:33:22 2024

@author: dvezinet
"""


import os
import copy


import numpy as np
import datastock as ds


_DINDOK = {
    0: 'ok',
    -1: 'mask',
    -2: 'out of domain',
    -3: 'neg, inf or NaN',
    -4: 'S/N valid, excluded',
    -5: 'S/N non-valid, included',
    -6: 'S/N non-valid, excluded',
}


#############################################
#############################################
#       main
#############################################


def mask_domain(
    coll=None,
    key_data=None,
    ddata=None,
    dvalid=None,
    ref=None,
    ref0=None,
    shape0=None,
):

    # ------------
    # check dvalid
    # ------------

    dvalid = _check_dvalid(dvalid)

    # ------------
    # check domain
    # ------------

    domain = _check_domain(
        domain=dvalid.get('domain'),
        ddata=ddata,
    )

    # ----------
    # Apply
    # ----------

    ind = np.zeros(shape0, dtype=bool)
    for k0, v0 in domain.items():

        # initialize
        indin = np.zeros(ddata[k0]['data'].shape, dtype=bool)
        indout = np.zeros(ddata[k0]['data'].shape, dtype=bool)

        # loop on domain bits
        for v1 in v0['spec']:
            indi = (ddata[k0]['data'] >= v1[0]) & (ddata[k0]['data'] <= v1[1])
            if isinstance(v1, tuple):
                indout |= indi
            else:
                indin |= indi

        # store
        indi = (indin & (~indout))
        domain[k0]['ind'] = indi

        # apply to ind
        sli = tuple([
            indi if ii == ref0.index(ddata[k0]['ref']) else slice(None)
            for ii in range(len(ref0))
        ])
        ind[sli] = True

    # ---------------
    # check mask
    # ---------------

    dmask = _check_mask(
        coll=coll,
        mask=dvalid.get('mask'),
        ref0=ref0,
        shape0=shape0,
    )

    # -----------------
    # initialize iok
    # -----------------

    iok = np.zeros(coll.ddata[key_data]['data'].shape, dtype=int)

    # mask
    if dmask['key'] is not None:
        iout = ~coll.ddata[dmask['key']]['data']
        sli = tuple([
            iout
            if rr == ref0[0]
            else slice(None)
            for rr in ref
            if rr not in ref0[1:]
        ])
        iok[sli] = -1

    # domain
    iout = (~ind)
    if dmask['key'] is not None:
        iout &= coll.ddata[dmask['key']]['data']
    sli = tuple([
        iout
        if rr == ref0[0]
        else slice(None)
        for rr in ref
        if rr not in ref0[1:]
    ])
    iok[sli] = -2

    # -----------------
    # store in dvalid
    # -----------------

    dvalid = {
        'domain': domain,
        'mask': dmask,
        'iok': iok,
        'meaning': _DINDOK,
    }

    return dvalid


#############################################
#############################################
#       check dvalid
#############################################


def _check_dvalid(dvalid=None):

    # ------------
    # None
    # ------------

    if dvalid is None:
        dvalid = {}

    # ------------
    # check
    # ------------

    c0 = (
        isinstance(dvalid, dict)
    )
    if not c0:
        msg = (
            "dvalid must be a dict with (optional) keys:\n"
        )
        raise Exception(msg)

    return dvalid


#############################################
#############################################
#       check domain
#############################################


def _check_domain(
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
            and isinstance(v0, dict)
            and isinstance(v0['data'], np.ndarray)
            and v0['data'].ndim == 1
            for k0, v0 in ddata.items()
        ])
    )
    if not c0:
        msg = (
            "Arg ddata must be a dict of shape:\n"
            "{'lamb': {'data': 1d np.ndarray, 'ref': str}}\n"
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

    for k0, v0 in ddata.items():

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


#############################################
#############################################
#       check mask
#############################################


def _mask_err(mask, ref=None, shape=None, lok=None):
    return Exception(
        "Arg mask must be either:\n"
        "\t- a str (path/file.ext) to a valid .npy file\n"
        "\t- a key to a known data array with ref = {ref}\n"
        "\t\tavailable: {lok}\n"
        "\t- a np.narray of bool and of shape = {shape}\n"
        f"Provided:\n{mask}\n"
    )


def _check_mask(
    coll=None,
    mask=None,
    ref0=None,
    shape0=None,
):

    # ----------------
    # prepare
    # ----------------

    pfe = None
    key = None

    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['ref'] == ref0
        and v0['data'].dtype.name == 'bool'
    ]

    err = _mask_err(mask, ref=ref0, shape=shape0, lok=lok)


    # default key
    lout = [
        int(k0[4:]) for k0 in coll.ddata.keys()
        if k0.startswith('mask')
        and all([ss.isnumeric() for ss in k0[4:]])
    ]
    if len(lout) == 0:
        nmax = 0
    else:
        nmax = max(lout) + 1
    key0 = f"mask{nmax:02.0f}"

    # ----------------
    # str
    # ----------------

    if isinstance(mask, str):

        if os.path.isfile(mask) and mask.endswith('.npy'):
            pfe = str(mask)
            mask = np.load(pfe)
            key = key0

        elif mask in lok:
            key = mask

        else:
            raise err

    # --------------
    # numpy array
    # --------------

    if isinstance(mask, np.ndarray):

        c0 = (
            mask.shape == shape0
            and mask.dtype.name == 'bool'
        )
        if c0:
            key = key0

        else:
            raise err

    elif mask is None:
        pass

    else:
        raise err

    # ----------------------
    # store
    # ----------------------

    dmask = {
        'pfe': pfe,
        'key': key,
    }

    if key is not None and key not in coll.ddata.keys():
        coll.add_data(
            key=key,
            data=mask,
            ref=ref0,
            units=None,
            quant='bool',
            dim='mask',
        )

    return dmask


#############################################
#############################################
#       validity
#############################################


def valid(
    coll=None,
    dvalid=None,
):

    # -----------------
    # check inputs
    # -----------------

    # check nsigma, fraction, focus
    dvalid = _check_dvalid_valid(dvalid)

    # -----------------
    # prepare
    # -----------------

    iok = dvalid['iok']
    data = None

    # -----------------
    # nan, neg, inf
    # -----------------

    if dvalid['positive'] is True:
        data[data < 0] = np.nan

    iokb = ((iok == 0) & (~np.isfinite(data)))
    iok[iokb] = -3

    # update iokb
    iokb = (iok == 0)

    # -----------------
    # Recompute domain
    # -----------------

    if dvalid['update_domain'] is True:
        for k0, v0 in dvalid['domain'].items():
            domain[k0]['minmax'] = [
                np.nanmin(lamb[np.any(indok_bool, axis=0)]),
                np.nanmax(lamb[np.any(indok_bool, axis=0)]),
            ]

    # --------------
    # Intermediate safety checks
    # --------------

    if np.any(np.isnan(data[iokb])):
        msg = (
            "Some NaNs in data not caught by iok!"
        )
        raise Exception(msg)

    if np.sum(iokb) == 0:
        msg = "There does not seem to be any usable data (no indok)"
        raise Exception(msg)

    # -----------------
    # validity
    # -----------------

    # Get indices of pts with enough signal
    ind = np.zeros(data.shape, dtype=bool)
    isafe = np.isfinite(data)
    isafe[isafe] = data[isafe] >= 0.
    if indok_bool is not None:
        isafe &= indok_bool

    # Ok with and w/o binning if data provided as counts
    ind[isafe] = np.sqrt(data[isafe]) > valid_nsigma

    # Derive indt and optionally dphi and indknots
    indbs, ldphi = False, False
    if focus is False:
        lambok = np.ones(tuple(np.r_[lamb.shape, 1]), dtype=bool)
        indall = ind[..., None]
    else:
        # TBC
        lambok = np.rollaxis(
            np.array([np.abs(lamb - ff[0]) < ff[1] for ff in focus]),
            0,
            lamb.ndim + 1,
        )
        indall = ind[..., None] & lambok[None, ...]
    nfocus = lambok.shape[-1]

    # ------------------------
    # more backup
    # ------------------------

    # Update indok with non-valid phi
    # non-valid = ok but out of dphi
    for ii in range(dinput['dprepare']['indok'].shape[0]):
        iphino = dinput['dprepare']['indok'][ii, ...] == 0
        for jj in range(len(dinput['valid']['ldphi'][ii])):
            iphino &= (
                (
                    dinput['dprepare']['phi']
                    < dinput['valid']['ldphi'][ii][jj][0]
                )
                | (
                    dinput['dprepare']['phi']
                    >= dinput['valid']['ldphi'][ii][jj][1]
                )
            )

        # valid, but excluded (out of dphi)
        iphi = (
            (dinput['dprepare']['indok'][ii, ...] == 0)
            & (dinput['valid']['ind'][ii, ...])
            & (iphino)
        )
        dinput['dprepare']['indok'][ii, iphi] = -5

        # non-valid, included (in dphi)
        iphi = (
            (dinput['dprepare']['indok'][ii, ...] == 0)
            & (~dinput['valid']['ind'][ii, ...])
            & (~iphino)
        )
        dinput['dprepare']['indok'][ii, iphi] = -6

        # non-valid, excluded (out of dphi)
        iphi = (
            (dinput['dprepare']['indok'][ii, ...] == 0)
            & (~dinput['valid']['ind'][ii, ...])
            & (iphino)
        )
        dinput['dprepare']['indok'][ii, iphi] = -7

    # indok_bool True if indok == 0 or -5 (because ...)
    dinput['dprepare']['indok_bool'] = (
        (dinput['dprepare']['indok'] == 0)
        | (dinput['dprepare']['indok'] == -6)
    )

    # add lambmin for bck
    dinput['lambmin_bck'] = np.min(dinput['dprepare']['lamb'])

    # -----------------
    # store
    # -----------------

    dvalid['iok'] = iok

    return dvalid


#############################################
#############################################
#       validity - check dvalid
#############################################


def _check_dvalid_valid(dvalid=None):

    if not isinstance(dvalid, dict):
        msg = "Arg dvalid must be a dict\nProvided:\n{dvalid}"
        raise Exception(msg)

    # ---------------
    # positive
    # ---------------

    dvalid['positive'] = ds._generic_check._check_var(
        dvalid.get('positive'), "dvalid['positive']",
        types=bool,
        default=True,
    )

    # ---------------
    # update_domain
    # ---------------

    dvalid['update_domain'] = ds._generic_check._check_var(
        dvalid.get('update_domain'), "dvalid['update_domain']",
        types=bool,
        default=True,
    )

    # ---------------
    # nsigma
    # ---------------

    dvalid['nsigma'] = float(ds._generic_check._check_var(
        dvalid.get('nsigma'), "dvalid['nsigma']",
        types=(int, float),
        default=6,
        sign=">=0",
    ))

    # ---------------
    # fraction
    # ---------------

    dvalid['fraction'] = float(ds._generic_check._check_var(
        dvalid.get('fraction'), "dvalid['fraction']",
        types=(int, float),
        default=0.51,
        sign=">=0",
    ))

    return dvalid