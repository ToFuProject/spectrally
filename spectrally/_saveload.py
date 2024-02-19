# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 08:25:06 2023

@author: dvezinet
"""


import datastock as ds


# ############################################################
# ############################################################
#                 loading
# ############################################################


def load(
    pfe=None,
    cls=None,
    allow_pickle=None,
    sep=None,
    verb=None,
):

    # --------------------
    # use datastock.load()

    if cls is None:
        from ._class01_SpectralFit import SpectralFit as cls

    coll = ds.load(
        pfe=pfe,
        cls=cls,
        allow_pickle=allow_pickle,
        sep=sep,
        verb=verb,
    )

    return coll
