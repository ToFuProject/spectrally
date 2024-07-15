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
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


_PKG = 'spectrally'
_PATH_SP = os.path.join(os.path.expanduser('~'), f'.{_PKG}')


_CUSTOM = os.path.dirname(_PATH_HERE)
_CUSTOM = os.path.join(_CUSTOM, 'scripts', f'{_PKG}custom.py')


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


def create_local_path():

    # clean
    if os.path.isdir(_PATH_SP):
        shutil.rmtree(_PATH_SP)

    # re-create
    os.system(f'python {_CUSTOM}')


def setup_module0(module):
    create_local_path()


def teardown_module0(module):
    clean_output()
    # clean
    if os.path.isdir(_PATH_SP):
        shutil.rmtree(_PATH_SP)
