# -*- coding: utf-8 -*-
"""
The nist-compatibility module of tofu

"""


import traceback


_PKG = 'spectrally'


try:
    try:
        from spectrally.nist._requests import *
    except Exception:
        from ._requests import *

except Exception as err:
    msg = (
        str(traceback.format_exc())
        + f"\n\n\t=> optional sub-package {_PKG}.nist not usable\n"
    )
    raise Exception(msg)


del traceback, _PKG
