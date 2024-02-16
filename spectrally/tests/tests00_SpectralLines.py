# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:53:45 2024

@author: dvezinet
"""
# Built-in
import os
import shutils
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt
import datastock as ds


# spectrally-specific
from .._class00_SpectralLines import SpectralLines as Collection


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUT = os.path.join(_PATH_HERE, 'input')
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


_PATH_SP = os.path.join(os.path.expanduser('~'), '.spectrally')
_PATH_OPAD = os.path.join(_PATH_SP, 'openadas')
_PATH_NIST = os.path.join(_PATH_SP, 'nist')


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


def clean_local_path():
    if _CLEAN is True:
        shutils.rmtree(_PATH_SP)


def setup_module(module):
    clean_output()


def teardown_module(module):
    clean_output()
    clean_local_path()


#######################################################
#
#     Utilities
#
#######################################################


def _add_ref(st=None, nc=None, nx=None, lnt=None):
    pass


#######################################################
#
#     Instanciate
#
#######################################################


class Test00_SpectralLines():

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(cls):
        cls.coll = Collection()
        cls.pfe_json = os.path.join(_PATH_INPUT, 'spectrallines.json')

    # ------------------------
    #   Populating
    # ------------------------

    def test01_add_spectral_lines_from_file(self):
        self.coll.add_spectral_lines_from_file(self.pfe_json)

    def test02_add_spectral_lines_from_openadas(self):
        # if openadas:
        self.coll.add_spectral_lines_from_openadas(
            lambmin=3.94e-10,
            lambmax=4e-10,
            element='Ar',
            online=True,
        )
        self.coll.remove_obj(which=self.coll._which_lines)

    def test03_add_spectral_lines_from_nist(self):
        # if nist:
        self.coll.add_spectral_lines_from_nist(
            lambmin=3.94e-10,
            lambmax=4e-10,
            element='Ar',
        )
        self.coll.remove_obj(which=self.coll._which_lines)

    # ------------------------
    #   Plotting
    # ------------------------

    def test04_plot_spectral_lines(self):
        self.coll.plot_spectral_lines()
        plt.close('all')

    # ------------------------
    #   saving / loading
    # ------------------------

    def test98_save_spectral_lines_to_file(self):
        self.coll.save_spectral_lines_to_file(path=_PATH_OUTPUT)

    def test99_saveload(self):
        pfe = self.coll.save(path=_PATH_OUTPUT, return_pfe=True)
        coll2 = ds.load(pfe)
        assert self.coll == coll2