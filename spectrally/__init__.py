# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import warnings


from .version import __version__
from ._class02_SpectralFit import SpectralFit as Collection
from ._saveload import load
from ._class01_show import *
from ._class01_display_models import *
from . import tests
from . import tutorials


_PKG = 'spectrally'


# -------------------------------------
#   Try importing optional subpackages
# -------------------------------------

msg = None
dsub = dict.fromkeys([
    'openadas', 'nist'
])

for sub in dsub.keys():
    try:
        exec(f'import {_PKG}.{sub} as {sub}')
        dsub[sub] = True

    except Exception as err:
        dsub[sub] = str(err)

# -------------------------------------
# If any error, populate warning and store error message
# -------------------------------------

lsubout = [sub for sub in dsub.keys() if dsub[sub] is not True]
if len(lsubout) > 0:
    lsubout = [f'{_PKG}.{ss}' for ss in lsubout]
    msg = (
        "\nThe following subpackages are not available:\n"
        + "\n".join([f"\t- {sub}" for sub in lsubout])
        + f"\n  => see print({_PKG}.dsub[<subpackage>]) for details."
    )
    warnings.warn(msg)

# -------------------------------------
# Add optional subpackages to __all__
# -------------------------------------

__all__ = []
for sub in dsub.keys():
    if dsub[sub] is True:
        __all__.append(sub)


# clean-up the mess
del warnings, lsubout, sub, msg, _PKG