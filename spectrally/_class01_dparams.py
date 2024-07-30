# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:01:39 2024

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

"""
def main(
    coll=None,
    dparams=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    key_bs=None,
):

    # ------------
    # check inputs
    # ------------

    dparams = _check(dparams)

    # ------------
    # compute
    # ------------

    # ------------------
    # list all variables
    lvar = coll.get_spectral_model_variables(
        key=key_model,
        all_free_tied='all',
        concatenate=True,
    )

    # ----------------------
    # prepare data if needed

    c0 = any([dparams.get(k0) is None for k0 in lvar])
    if c0:

        lamb = coll.ddata[key_lamb]['data']
        data = coll.ddata[key_data]['data']

        # TBF
        Dlamb = np.diff(dinput['dprepare']['domain']['lamb']['minmax'])
        lambm = dinput['dprepare']['domain']['lamb']['minmax'][0]
        if not (np.isfinite(Dlamb)[0] and Dlamb > 0):
            msg = (
                "lamb min, max seems to be non-finite or non-positive!\n"
                + "\t- dinput['dprepare']['domain']['lamb']['minmax'] = {}".format(
                    dinput['dprepare']['domain']['lamb']['minmax']
                )
                + "\n  => Please provide domain['lamb']"
            )
            raise Exception(msg)
        if lambm == 0:
            lambm = Dlamb / 100.

    # ----------------------
    # compute if needed

    # check all variables
    for k0 in lvar:

        if dparams.get(k0) is None:

            key, var = k0.split('_')
            typ = coll.dobj[key_model]['dmodel'][key]['type']

            scale = _get_scale(
                key_model=key_model,
                key=k0,
                typ=typ,
                var=var,
                # data
                lamb=lamb,
                data=data,
            )

            dparams[k0] = scale

    return


#############################################
#############################################
#       check
#############################################


def _check(
    coll=None,
    dparams=None,
    key_model=None,
    key_data=None,
    key_lamb=None,
    key_bs=None,
):

    # --------------------
    # get model variables
    # --------------------

    lvar = coll.get_spectral_model_variables(
        key=key_model,
        all_free_tied='all',
        concatenate=True,
    )

    # --------------------
    # Trivial: False = 1
    # --------------------

    if dparams is False:
        dparams = {k0: 1. for k0 in lvar}
        return dparams

    # --------------------
    # Trivial: scalar => uniform
    # --------------------

    if np.isscalar(dparams):
        dparams = {k0: dparams for k0 in lvar}
        return dparams

    # --------------------
    # Non-trivial
    # --------------------

    # None
    if dparams is None:
        dparams = {}

    c0 = (
        isinstance(dparams, dict)
        and all([
            k0 in lvar
            and ((np.isscalar(v0) and np.isfinite(v0)) or v0 is None)
            for k0, v0 in dparams.items()
        ])
    )
    if not c0:
        msg = (
            "Arg dparams must be a dict with keys = variables of model"
            " and values = floats\n"
            f"\t- model: {key_model}\n"
            f"\t- variables: {lvar}\n"
            f"Provided:\n{dparams}\n"
        )
        raise Exception(msg)

    return dparams


#############################################
#############################################
#       get scale from data
#############################################


def _get_scale(
    key_model=None,
    key=None,
    typ=None,
    var=None,
):

    # ------------
    # bck
    # ------------

    if typ == 'poly':

        if var == 'c0':
            pass

        if var == 'c1':
            pass

        else:
            err = True

    # ------------
    # exp
    # ------------

    elif typ == 'exp':

        indbck = (data > np.nanmean(data, axis=1)[:, None]) | (~indok)
        bcky = np.array(np.ma.masked_where(indbck, data).mean(axis=1))
        bckstd = np.array(np.ma.masked_where(indbck, data).std(axis=1))

        iok = (bcky > 0) & (bckstd > 0)
        if (bck_rate is None or nbck_amp is None) and not np.any(iok):
            bcky = 0.1*np.array(np.ma.masked_where(~indbck, data).mean(axis=1))
            bckstd = 0.1*bcky
        elif not np.all(iok):
            bcky[~iok] = np.mean(bcky[iok])
            bckstd[~iok] = np.mean(bckstd[iok])

        # bck_rate
        if bck_rate is None:
            bck_rate = (
                np.log((bcky + bckstd)/bcky) / (lamb.max()-lamb.min())
            )
        if bck_amp is None:
            # Assuming bck = A*exp(rate*(lamb-lamb.min()))
            bck_amp = bcky

        dparams = _fit12d_filldef_dparamsx0_float(
            din=dparams, din_name='dparams', key='bck_amp',
            vref=bck_amp, nspect=nspect,
        )

        if var == 'amp':
            pass

        elif var == 'rate':
            pass

        else:
            err = True

    # ------------
    # gauss
    # ------------

    elif typ == 'gauss':

        if var == 'amp':
            pass

        elif var == 'w2':
            pass

        elif var == 'x0':
            pass

        else:
            err = True

    # ------------
    # lorentz
    # ------------

    elif typ == 'lorentz':

        if var == 'amp':
            pass

        elif var == 'gamma':
            pass

        elif var == 'x0':
            pass

        else:
            err = True

    # ------------
    # pvoigt
    # ------------

    elif typ == 'pvoigt':

        if var == 'amp':
            pass

        elif var == 'gamma':
            pass

        elif var == 'x0':
            pass

        elif var == 'w2':
            pass

        elif var == 't':
            pass

        else:
            err = True

    # ------------
    # voigt
    # ------------

    # ------------
    # error
    # ------------

    if err is True:
        msg = (
            f"Spectral model '{key_model}', issue with variable '{key}':\n"
            f"unknown variable '{var}' for type '{typ}'"
        )
        raise Exception(msg)

    return val


#############################################
#############################################
#       Backup
#############################################


def _fit12d_filldef_dscalesx0_dict(
    din=None, din_name=None,
    key=None, vref=None,
    nspect=None, dinput=None,
):

    # Check vref
    if vref is not None:
        if type(vref) not in _LTYPES and len(vref) not in [1, nspect]:
            msg = (
                "Non-conform vref for "
                + "{}['{}']\n".format(din_name, key)
                + "\t- expected: float or array (size {})\n".format(nspect)
                + "\t- provided: {}".format(vref)
            )
            raise Exception(msg)
        if type(vref) in _LTYPES:
            vref = np.full((nspect,), vref)
        elif len(vref) == 1:
            vref = np.full((nspect,), vref[0])

    # check din[key]
    if din.get(key) is None:
        assert vref is not None
        din[key] = {k0: vref for k0 in dinput[key]['keys']}

    elif not isinstance(din[key], dict):
        assert type(din[key]) in _LTYPES + [np.ndarray]
        if hasattr(din[key], '__len__') and len(din[key]) == 1:
            din[key] = din[key][0]
        if type(din[key]) in _LTYPES:
            din[key] = {
                k0: np.full((nspect,), din[key])
                for k0 in dinput[key]['keys']
            }
        elif din[key].shape == (nspect,):
            din[key] = {k0: din[key] for k0 in dinput[key]['keys']}
        else:
            msg = (
                "{}['{}'] not conform!".format(dd_name, key)
            )
            raise Exception(msg)

    else:
        for k0 in dinput[key]['keys']:
            if din[key].get(k0) is None:
                din[key][k0] = vref
            elif type(din[key][k0]) in _LTYPES:
                din[key][k0] = np.full((nspect,), din[key][k0])
            elif len(din[key][k0]) == 1:
                din[key][k0] = np.full((nspect,), din[key][k0][0])
            elif din[key][k0].shape != (nspect,):
                msg = (
                    "Non-conform value for "
                    + "{}['{}']['{}']\n".format(din_name, key, k0)
                    + "\t- expected: float or array (size {})\n".format(nspect)
                    + "\t- provided: {}".format(din[key][k0])
                )
                raise Exception(msg)
    return din


def _fit12d_filldef_dscalesx0_float(
    din=None, din_name=None,
    key=None, vref=None,
    nspect=None,
):
    if din.get(key) is None:
        if type(vref) in _LTYPES:
            din[key] = np.full((nspect,), vref)
        elif np.array(vref).shape == (1,):
            din[key] = np.full((nspect,), vref[0])
        elif np.array(vref).shape == (nspect,):
            din[key] = np.array(vref)
        else:
            msg = (
                "Non-conform vref for {}['{}']\n".format(din_name, key)
                + "\t- expected: float or array (size {})\n".format(nspect)
                + "\t- provided: {}".format(vref)
            )
            raise Exception(msg)
    else:
        if type(din[key]) in _LTYPES:
            din[key] = np.full((nspect,), din[key])
        elif din[key].shape == (1,):
            din[key] = np.full((nspect,), din[key][0])
        elif din[key].shape != (nspect,):
            msg = (
                "Non-conform vref for {}['{}']\n".format(din_name, key)
                + "\t- expected: float or array (size {})\n".format(nspect)
                + "\t- provided: {}".format(din[key])
            )
            raise Exception(msg)
    return din


def _check_finit_dict(dd=None, dd_name=None, indtok=None, indbs=None):
    dfail = {}
    for k0, v0 in dd.items():
        if k0 in ['amp', 'width', 'shift']:
            for k1, v1 in v0.items():
                if np.any(~np.isfinite(v1[indtok, ...])):
                    dfail[f"'{k0}'['{k1}']"] = v1
        elif k0 == 'bs':
            if np.any(~np.isfinite(v0[indbs])):
                dfail[f"'{k0}'"] = v0
        else:
            if np.any(~np.isfinite(v0[indtok, ...])):
                dfail[f"'{k0}'"] = v0

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            f"The following {dd_name} values are non-finite:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)



# Double-check 1d vs 2d: TBF / TBC
def fit12d_dscales(dscales=None, dinput=None):

    # --------------
    # Input checks
    dscales = _fit12d_checkformat_dscalesx0(
        din=dscales, dinput=dinput, name='dscales',
    )

    data = dinput['dprepare']['data']
    lamb = dinput['dprepare']['lamb']
    nspect = data.shape[0]

    # --------------
    # 2d spectrum = 1d spectrum + vert. profile
    is2d = data.ndim == 3
    if is2d is True:
        data = dinput['dprepare']['datalamb1d']
        datavert = dinput['dprepare']['dataphi1d']
        lamb = dinput['dprepare']['lamb1d']
        phi = dinput['dprepare']['phi1d']
        indok = np.any(dinput['dprepare']['indok_bool'], axis=1)

        # bsplines modulation of bck and amp, if relevant
        # fit bsplines on datavert (vertical profile)
        # to modulate scales (bck and amp)

        if dinput['symmetry'] is True:
            phitemp = np.abs(phi[None, :] - dinput['symmetry_axis'][:, None])
        else:
            phitemp = np.tile(phi, (nspect, 1))

        # Loop on time and bsplines
        dscales['bs'] = np.full((nspect, dinput['nbs']), np.nan)
        for ii in dinput['valid']['indt'].nonzero()[0]:
            for jj, jbs in enumerate(range(dinput['nbs'])):
                if dinput['valid']['indbs'][ii, jj]:
                    kn0 = dinput['knots_mult'][jj]
                    kn1 = dinput['knots_mult'][jj + dinput['nknotsperbs'] - 1]
                    indj = (
                        (~np.isnan(datavert[ii, :]))
                        & (kn0 <= phitemp[ii, :])
                        & (phitemp[ii, :] <= kn1)
                    )
                    if not np.any(indj):
                        msg = "Unconsistent indbs!"
                        raise Exception(msg)
                    dscales['bs'][ii, jj] = np.mean(datavert[ii, indj])

        # Normalize to avoid double-amplification when amp*bs
        corr = np.nanmax(dscales['bs'][dinput['valid']['indt'], :], axis=1)
        dscales['bs'][dinput['valid']['indt'], :] /= corr[:, None]
    else:
        indok = dinput['dprepare']['indok_bool']

    # --------------
    # Default values for filling missing fields
    Dlamb = np.diff(dinput['dprepare']['domain']['lamb']['minmax'])
    lambm = dinput['dprepare']['domain']['lamb']['minmax'][0]
    if not (np.isfinite(Dlamb)[0] and Dlamb > 0):
        msg = (
            "lamb min, max seems to be non-finite or non-positive!\n"
            + "\t- dinput['dprepare']['domain']['lamb']['minmax'] = {}".format(
                dinput['dprepare']['domain']['lamb']['minmax']
            )
            + "\n  => Please provide domain['lamb']"
        )
        raise Exception(msg)
    if lambm == 0:
        lambm = Dlamb / 100.

    # bck_amp
    bck_amp = dscales.get('bck_amp')
    bck_rate = dscales.get('bck_rate')
    if bck_amp is None or bck_rate is None:
        indbck = (data > np.nanmean(data, axis=1)[:, None]) | (~indok)
        bcky = np.array(np.ma.masked_where(indbck, data).mean(axis=1))
        bckstd = np.array(np.ma.masked_where(indbck, data).std(axis=1))

        iok = (bcky > 0) & (bckstd > 0)
        if (bck_rate is None or nbck_amp is None) and not np.any(iok):
            bcky = 0.1*np.array(np.ma.masked_where(~indbck, data).mean(axis=1))
            bckstd = 0.1*bcky
        elif not np.all(iok):
            bcky[~iok] = np.mean(bcky[iok])
            bckstd[~iok] = np.mean(bckstd[iok])

        # bck_rate
        if bck_rate is None:
            bck_rate = (
                np.log((bcky + bckstd)/bcky) / (lamb.max()-lamb.min())
            )
        if bck_amp is None:
            # Assuming bck = A*exp(rate*(lamb-lamb.min()))
            bck_amp = bcky

    dscales = _fit12d_filldef_dscalesx0_float(
        din=dscales, din_name='dscales', key='bck_amp',
        vref=bck_amp, nspect=nspect,
    )
    dscales = _fit12d_filldef_dscalesx0_float(
        din=dscales, din_name='dscales', key='bck_rate',
        vref=bck_rate, nspect=nspect,
    )

    # amp
    dscales['amp'] = dscales.get('amp', dict.fromkeys(dinput['amp']['keys']))
    for ii, ij in enumerate(dinput['dind']['amp_x0']):
        key = dinput['amp']['keys'][ii]
        if dscales['amp'].get(key) is None:
            # convoluate and estimate geometric mean
            conv = np.exp(
                    -(lamb - dinput['lines'][ij])**2 / (2*(Dlamb / 25.)**2)
                )[None, :]
            dscales['amp'][key] = np.nanmax(data*conv, axis=1)
        else:
            if type(dscales['amp'][key]) in _LTYPES:
                dscales['amp'][key] = np.full((nspect,), dscales['amp'][key])
            else:
                assert dscales['amp'][key].shape == (nspect,)

    # width
    if dinput.get('same_spectrum') is True:
        lambm2 = (
            lambm
            + dinput['same_spectrum_dlamb']
            * np.arange(0, dinput['same_spectrum_nspect'])
        )
        nw0 = iwx.size / dinput['same_spectrum_nspect']
        lambmw = np.repeat(lambm2, nw0)
        widthref = (Dlamb/(25*lambmw))**2
    else:
        widthref = (Dlamb/(25*lambm))**2

    dscales = _fit12d_filldef_dscalesx0_dict(
        din=dscales, din_name='dscales', key='width', vref=widthref,
        nspect=nspect, dinput=dinput,
    )

    # shift
    shiftref = Dlamb/(25*lambm)
    dscales = _fit12d_filldef_dscalesx0_dict(
        din=dscales, din_name='dscales', key='shift', vref=shiftref,
        nspect=nspect, dinput=dinput,
    )

    # Double
    if dinput['double'] is not False:
        dratio = 1.
        dshift = float(Dlamb/(25*lambm))
        if dinput['double'] is True:
            pass
        else:
            if dinput['double'].get('dratio') is not None:
                dratio = dinput['double']['dratio']
            if dinput['double'].get('dshift') is not None:
                dshift = dinput['double']['dshift']
        din = {'dratio': dratio, 'dshift': dshift}
        for k0 in din.keys():
            dscales = _fit12d_filldef_dscalesx0_float(
                din=dscales, din_name='dscales', key=k0,
                vref=din[k0], nspect=nspect,
            )
    elif 'dratio' in dscales.keys():
        del dscales['dratio'], dscales['dshift']

    # check
    _check_finit_dict(
        dd=dscales,
        dd_name='dscales',
        indtok=dinput['valid']['indt'],
        indbs=dinput['valid']['indbs'],
    )
    return dscales
"""