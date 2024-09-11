# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:36:24 2024

@author: dvezinet
"""


# common
import numpy as np


#############################################
#############################################
#       DEFAULTS
#############################################


_LFIT_ORDER = [
    'sol',
    'model', 'data', 'sigma', 'lamb', 'bs',
    'positive', 'nsigma', 'fraction',
    'mask', 'domain', 'focus',
]


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([which] + _LFIT_ORDER)

    # ---------------------------
    # data
    # ---------------------------

    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    lar0 = []
    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # add nb of func of each type
        dfit = coll.dobj[which][k0]
        for k1 in _LFIT_ORDER:

            if k1 in ['model', 'data', 'sigma', 'lamb', 'bs', 'sol']:
                nn = '' if dfit[f"key_{k1}"] is None else str(dfit[f"key_{k1}"])

            elif k1 in ['nsigma', 'fraction', 'positive']:
                nn = str(dfit['dvalid'][k1])

            elif k1 in ['mask']:
                nn = str(dfit['dvalid']['mask']['key'] is not None)

            elif k1 in ['domain']:
                c0 = all([
                    len(v0['spec']) == 1
                    and np.allclose(v0['spec'][0], np.inf*np.r_[-1, 1])
                    for k0, v0 in dfit['dvalid']['domain'].items()
                ])
                if c0:
                    nn = ''
                else:
                    lk = list(dfit['dvalid']['domain'].keys())
                    if len(lk) == 2 and lk[0] != dfit['key_lamb']:
                        lk = [lk[1], lk[0]]

                    nn = ', '.join([
                        str(len(dfit['dvalid']['domain'][k0]['spec']))
                        for k0 in lk
                    ])

            elif k1 in ['focus']:
                if dfit['dvalid'].get('focus') is None:
                    nn = ''
                else:
                    nn = len(dfit['dvalid']['focus'])
                    nn = f"{nn} / {dfit['dvalid']['focus_logic']}"

            arr.append(nn)

        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar