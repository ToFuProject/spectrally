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
from .._class01_SpectralFit import SpectralFit as Collection
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

    # ---------------
    # 1d spectral fit

    def test04_add_spectral_fit_1d(self):
        # add spectral fit 1d
        _inputs.add_fit(self.coll, key_data='data1d')

    def test05_plot_spectral_fit_input_validity_1d(self):
        # plot 1d
        _inputs.plot_input_validity(self.coll, key_data='data1d')

    def test06_compute_spectral_fit_1d(self):
        # compute 1d
        _inputs.compute_fit(self.coll, key_data='data1d')

    # ---------------
    # 2d spectral fit

    def test07_add_spectral_fit_2d(self):
        # add spectral fit 2d
        _inputs.add_fit(self.coll, key_data='data2d')

    def test08_plot_spectral_fit_input_validity_2d(self):
        # plot 2d
        _inputs.plot_input_validity(self.coll, key_data='data2d')
