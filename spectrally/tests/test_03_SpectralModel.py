# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:53:45 2024

@author: dvezinet
"""
# Built-in
import os


# Standard
# import matplotlib.pyplot as plt


# spectrally-specific
from ._setup_teardown import setup_module0, teardown_module0
from .._class01_SpectralModel import SpectralModel as Collection
# from .._saveload import load
from . import _spectralfit_input as _inputs


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUT = os.path.join(_PATH_HERE, 'input')
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module(module):
    setup_module0(module)


def teardown_module(module):
    teardown_module0(module)


#######################################################
#
#     Instanciate and populate
#
#######################################################


class Test00_Populate():

    # ------------------------
    #   setup and teardown
    # ------------------------

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(self):

        # instanciate
        self.coll = Collection()

        # add data
        _inputs.add_data(self.coll)

    # ------------------------
    #   Populating
    # ------------------------

    # -------------
    # add models

    def test00_add_spectral_model(self):
        _inputs.add_models(self.coll)

    def test01_get_spectral_model_func(self):
        _inputs.get_spectral_model_func(self.coll)

    def test02_interpolate_spectral_model(self):
        _inputs.interpolate_spectral_model(self.coll)

    def test03_plot_spectral_model(self):
        _inputs.plot_spectral_model(self.coll)
