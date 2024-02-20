"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import os
import shutil


__all__ = ['setup_module0', 'teardown_module0']


#######################################################
#
#     DEFAULTS
#
#######################################################


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUT = os.path.join(_PATH_HERE, 'input')
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


_PKG = 'spectrally'
_PATH_SP = os.path.join(os.path.expanduser('~'), f'.{_PKG}')
_PATH_OPAD = os.path.join(_PATH_SP, 'openadas')
_PATH_NIST = os.path.join(_PATH_SP, 'nist')


_CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(_PATH_HERE)))
_CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')


_CLEAN = False
if not os.path.isdir(_PATH_SP):
    os.mkdir(_PATH_SP)
    _CLEAN = True
if not os.path.isdir(_PATH_OPAD):
    os.mkdir(_PATH_OPAD)
    _CLEAN = True
if not os.path.isdir(_PATH_NIST):
    os.mkdir(_PATH_NIST)
    _CLEAN = True


#######################################################
#
#     Setup and Teardown
#
#######################################################


def clean_output(path=_PATH_OUTPUT):
    """ Remove all temporary output files that may have been forgotten """
    lf = [
        ff for ff in os.listdir(path)
        if ff.endswith('.npz')
        or ff.endswith('.json')
    ]
    if len(lf) > 0:
        for ff in lf:
            os.remove(os.path.join(path, ff))


def create_local_path(clean=_CLEAN):
    if clean is True:
        os.system(f'python {_CUSTOM}')


def clean_local_path(clean=_CLEAN):
    if clean is True and os.path.isdir(_PATH_SP):
        shutil.rmtree(_PATH_SP)


def setup_module0(module):
    clean_output()
    create_local_path()


def teardown_module0(module):
    clean_output()
    clean_local_path()
