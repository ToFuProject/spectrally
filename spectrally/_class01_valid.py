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
    -4: 'S/N valid, excl. (fract.)',
    -5: 'S/N non-valid, incl. (fract.)',
    -6: 'S/N non-valid, excl. (fract.)',
    -7: 'S/N valid, excl. (bs)',
    -8: 'S/N non-valid, incl. (bs)',
}


#############################################
#############################################
#       main
#############################################


def mask_domain(
    # resources
    coll=None,
    key_data=None,
    key_lamb=None,
    key_bs_vect=None,
    # options
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
        key_lamb=key_lamb,
        key_bs_vect=key_bs_vect,
    )

    # ----------
    # Apply
    # ----------

    lk = [k0 for k0 in [key_lamb, key_bs_vect] if k0 is not None]
    ind = np.zeros(shape0, dtype=bool)
    for k0 in lk:

        # initialize
        c0 = (
            len(domain[k0]['spec']) == 1
            or all([isinstance(v1, tuple) for v1 in domain[k0]['spec']])
        )
        if c0 is True:
            indin = np.ones(coll.ddata[k0]['data'].shape, dtype=bool)
        else:
            indin = np.zeros(coll.ddata[k0]['data'].shape, dtype=bool)
        indout = np.zeros(coll.ddata[k0]['data'].shape, dtype=bool)

        # loop on domain bits
        for v1 in domain[k0]['spec']:

            indi = (
                (coll.ddata[k0]['data'] >= v1[0])
                & (coll.ddata[k0]['data'] <= v1[1])
            )

            # tuple => excluded
            if isinstance(v1, tuple):
                indout |= indi

            # list => included
            else:
                indin |= indi

        # store
        indi = (indin & (~indout))
        domain[k0]['ind'] = indi

        # apply to ind
        sli = tuple([
            indi if ii == ref0.index(coll.ddata[k0]['ref'][0]) else slice(None)
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

    if key_bs_vect is None:
        lindok = sorted([
            k0 for k0, v0 in _DINDOK.items()
            if '(bs)' not in v0
        ])[::-1]
    else:
        lindok = sorted(_DINDOK.keys())[::-1]

    dvalid.update({
        'domain': domain,
        'mask': dmask,
        'iok': iok,
        'meaning': {k0: _DINDOK[k0] for k0 in lindok},
    })

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
    key_lamb=None,
    key_bs_vect=None,
):

    # --------------
    # check domain
    # --------------

    # ------------
    # special case

    if key_bs_vect is None and not isinstance(domain, dict):
        domain = {key_lamb: domain}

    # --------
    # if None

    lk = [k0 for k0 in [key_lamb, key_bs_vect] if k0 is not None]
    if domain is None:
        domain = {
            k0: {
                'spec': [np.inf*np.r_[-1., 1.]],
                'minmax': np.inf*np.r_[-1., 1.],
            }
            for k0 in lk
        }

    # ----------
    # general

    c0 = (
        isinstance(domain, dict)
        and all([k0 in lk for k0 in domain.keys()])
    )
    if not c0:
        msg = (
            "Arg domain must be a dict with keys:\n"
            + "\n".join([f"\t- {k0}" for k0 in lk])
            + "\nProvided:\n{domain}\n"
        )
        raise Exception(msg)

    # deepcopy to avoid modifying in place
    domain = copy.deepcopy(domain)

    # ------------
    # loop on keys
    # ------------

    for k0 in lk:

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
                + f"\nProvided:\n{domain[k0]}\n"
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
        f"\t- a key to a known data array with ref = {ref}\n"
        f"\t\tavailable: {lok}\n"
        f"\t- a np.narray of bool and of shape = {shape}\n"
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
    key=None,
    key_data=None,
    key_lamb=None,
    key_bs=None,
    dvalid=None,
    ref=None,
    ref0=None,
):

    # -----------------
    # check inputs
    # -----------------

    # check nsigma, fraction, focus
    dvalid = _check_dvalid_valid(
        dvalid=dvalid,
        key_lamb=key_lamb,
    )

    # -----------------
    # prepare
    # -----------------

    wsf = coll._which_fit
    iok = dvalid['iok']
    data = coll.ddata[key_data]['data']
    lamb = coll.ddata[key_lamb]['data']

    # -----------------
    # nan, neg, inf
    # -----------------

    iokb = ((iok == 0) & (~np.isfinite(data)))
    if dvalid['positive'] is True:
        iokb &= (data >= 0)

    iok[iokb] = -3

    # update iokb
    iokb = (iok == 0)

    # safety check
    if not np.any(iokb):
        msg = (
            "Not a single valid data point for\n"
            f"\t- {wsf}: '{key}'\n"
            f"\t- key_data: '{key_data}'\n"
            f"\t- key_lamb: '{key_lamb}'\n"
            f"\t- mask: {dvalid['mask']['key']}\n"
            f"\t- domain: {dvalid['domain']}\n"
        )
        raise Exception(msg)

    # ----------------------------
    # Recompute domain min, max form valid data
    # ----------------------------

    if dvalid['update_domain'] is True:
        for k0, v0 in dvalid['domain'].items():
            refi = coll.ddata[k0]['ref'][0]
            axis = tuple([ii for ii, rr in enumerate(ref) if rr != refi])
            dvalid['domain'][k0]['minmax'] = [
                np.nanmin(coll.ddata[k0]['data'][np.any(iokb, axis=axis)]),
                np.nanmax(coll.ddata[k0]['data'][np.any(iokb, axis=axis)]),
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
    # validity vs nsigma
    # -----------------

    # Ok with and w/o binning if data provided as counts
    indv = np.zeros(data.shape, dtype=bool)
    indv[iokb] = np.sqrt(data[iokb]) > dvalid['nsigma']

    # update iokb

    # -----------------
    # validity vs faction
    # -----------------

    refi = coll.ddata[key_lamb]['ref'][0]
    axis = tuple([ii for ii, rr in enumerate(ref) if rr != refi])

    nlamb = coll.ddata[key_lamb]['data'].size

    axis = ref.index(refi)
    if dvalid.get('focus') is None:
        nlambi = np.sum(iokb, axis=axis, keepdims=True)
        frac = np.sum(indv, axis=axis, keepdims=True) / nlambi
        iout = frac < dvalid['fraction']

    else:
        shape = list(data.shape)
        shape[axis] = 1
        shape.append(len(dvalid['focus']))
        iout = np.zeros(tuple(shape), dtype=bool)
        frac = np.zeros(tuple(shape), dtype=float)
        nl = np.zeros(tuple(shape), dtype=int)
        sli = [slice(None) for ii in ref]
        for ii, ff in enumerate(dvalid['focus']):
            ilambok = (lamb >= ff[0]) & (lamb <= ff[1])
            nlambi = np.sum(ilambok)
            sli[axis] = ilambok
            frac[..., ii] = (
                np.sum(indv[tuple(sli)], axis=axis, keepdims=True) / nlambi
            )
            nl[..., ii] = nlambi

        if dvalid['focus_logic'] == 'min':
            frac = np.min(frac, axis=-1)
        elif dvalid['focus_logic'] == 'max':
            frac = np.max(frac, axis=-1)
        else:
            frac = np.sum(frac * nl, axis=-1) / np.sum(nl, axis=-1)

        iout = frac < dvalid['fraction']

    # reshape
    iout = np.repeat(iout, nlamb, axis=axis)

    # valid, excluded
    iokb = indv & iout
    iok[iokb] = -4

    # non-valid, included
    iokb = (~indv) & (iok == 0) & (~iout)
    iok[iokb] = -5

    # non-valid, excluded
    iokb = (~indv) & (iok == 0) & (iout)
    iok[iokb] = -6

    # -----------------
    # validity vs bspline mesh
    # -----------------

    if key_bs is not None:

        raise NotImplementedError()


    # ------------------------
    # more backup
    # ------------------------

    """
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

    # -----------
    # backup

    if data2d is True:
        # Code ok with and without binning :-)

        # Get knots intervals that are ok
        fract = np.full((nspect, knots.size-1, nfocus), np.nan)
        for ii in range(knots.size - 1):
            iphi = (phi >= knots[ii]) & (phi < knots[ii + 1])
            fract[:, ii, :] = (
                np.sum(np.sum(indall & iphi[None, ..., None],
                              axis=1), axis=1)
                / np.sum(np.sum(iphi[..., None] & lambok,
                                axis=0), axis=0)
            )
        indknots = np.all(fract > valid_fraction, axis=2)

        # Deduce ldphi
        ldphi = [[] for ii in range(nspect)]
        for ii in range(nspect):
            for jj in range(indknots.shape[1]):
                if indknots[ii, jj]:
                    if jj == 0 or not indknots[ii, jj-1]:
                        ldphi[ii].append([knots[jj]])
                    if jj == indknots.shape[1] - 1:
                        ldphi[ii][-1].append(knots[jj+1])
                else:
                    if jj > 0 and indknots[ii, jj-1]:
                        ldphi[ii][-1].append(knots[jj])

        # Safety check
        assert all([
            all([len(dd) == 2 and dd[0] < dd[1] for dd in ldphi[ii]])
            for ii in range(nspect)
        ])

        # Deduce indbs that are ok
        nintpbs = nknotsperbs - 1
        indbs = np.zeros((nspect, nbs), dtype=bool)
        for ii in range(nbs):
            ibk = np.arange(max(0, ii-(nintpbs-1)), min(knots.size-1, ii+1))
            indbs[:, ii] = np.any(indknots[:, ibk], axis=1)

        assert np.all(
            (np.sum(indbs, axis=1) == 0) | (np.sum(indbs, axis=1) >= deg + 1)
        )

        # Deduce indt
        indt = np.any(indbs, axis=1)


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
    """

    # -----------------
    # store
    # -----------------

    # iok
    kiok = f"{key}_iok"
    dvalid['iok'] = kiok
    coll.add_data(
        key=kiok,
        data=iok,
        ref=ref,
        unit='',
        dim='bool',
    )

    # frac
    if data.ndim == 1:
        dvalid['frac'] = frac

    else:
        kfrac = f"{key}_frac"
        dvalid['frac'] = kfrac
        ref_lamb = coll.ddata[key_lamb]['ref'][0]
        ref_frac = tuple([rr for rr in ref if rr != ref_lamb])
        coll.add_data(
            key=kfrac,
            data=frac.squeeze(),
            ref=ref_frac,
            unit='',
            dim='bool',
        )

    return dvalid


#############################################
#############################################
#       validity - check dvalid
#############################################


def _check_dvalid_valid(
    dvalid=None,
    key_lamb=None,
):

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
        default=10,
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

    # ---------------
    # focus
    # ---------------

    if dvalid.get('focus') is not None:

        dvalid['focus'] = _check_domain(
            domain=dvalid['focus'],
            key_lamb=key_lamb,
            key_bs_vect=None,
        )[key_lamb]['spec']

        # logic
        dvalid['focus_logic'] = ds._generic_check._check_var(
            dvalid.get('focus_logic'), "dvalid['focus_logic']",
            types=str,
            default='min',
            allowed=['min', 'max', 'sum'],
        )

    return dvalid