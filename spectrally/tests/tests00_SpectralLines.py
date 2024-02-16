# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:53:45 2024

@author: dvezinet
"""
# Built-in
import os
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt


# spectrally-specific
from .._class00_SpectralLines import SpectralLines as Collection


_PATH_HERE = os.path.dirname(__file__)
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


#######################################################
#
#     Setup and Teardown
#
#######################################################


def clean(path=_PATH_OUTPUT):
    """ Remove all temporary output files that may have been forgotten """
    lf = [ff for ff in os.listdir(path) if ff.endswith('.npz')]
    if len(lf) > 0:
        for ff in lf:
            os.remove(os.path.join(path, ff))


def setup_module(module):
    clean()


def teardown_module(module):
    clean()


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
        cls.coll = Collection()

    # ------------------------
    #   Populating
    # ------------------------

    def test01_add_spectral_lines_from_module(self):
        coll.add_spectral_lines()

    def test02_add_spectral_lines_from_openadas(self):
        # if openadas:
        coll.add_spectral_lines_from_openadas()

    def test03_add_spectral_lines_from_nist(self):
        # if nist:
        coll.add_spectral_lines_from_nist()

    # ------------------------
    #   Plotting
    # ------------------------

    def test04_plot_spectral_lines(self):
        pass

    # ------------------------
    #   saving / loading
    # ------------------------

    def test99_saveload(self):
        self.coll.save(path=_PATH_OUTPUT)