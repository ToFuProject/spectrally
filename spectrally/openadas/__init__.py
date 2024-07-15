# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
The openadas-compatibility module of spectrally

"""


import traceback


try:
    try:
        from spectrally.openadas._requests import *
        from spectrally.openadas._read_files import *
    except Exception:
        from ._requests import *
        from ._read_files import *

except Exception as err:
    msg = str(traceback.format_exc())
    msg += "\n\n\t=> optional sub-package spectrally.openadas not usable\n"
    raise Exception(msg)

del traceback
