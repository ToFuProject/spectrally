# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:53:45 2024

@author: dvezinet
"""
# Built-in
import os


# Standard
import matplotlib.pyplot as plt


# spectrally-specific
from ._setup_teardown import setup_module0, teardown_module0
from .._class01_SpectralFit import SpectralFit as Collection
from .._saveload import load
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

    def test00_add_spectral_model(self):
        _inputs.add_models(self.coll)

    def test01_add_spectral_fit_1d(self):

        if self.coll.dobj.get('spect_model') is None:
            _inputs.add_models(self.coll)

        # add spectral fit 1d
        _inputs.add_fit1d(self.coll, key_data='data1d')

    def test02_plot_spectral_fit_input_validity_1d(self):

        if self.coll.dobj.get('spect_model') is None:
            _inputs.add_models(self.coll)
            _inputs.add_fit1d(self.coll, key_data='data1d')

        # plot 1d
        _inputs._plot_input_validity_1d(self.coll, key_data='data1d')

    def test03_add_spectral_fit_2d(self):

        if self.coll.dobj.get('spect_model') is None:
            _inputs.add_models(self.coll)

        # add spectral fit 2d
        _inputs.add_fit1d(self.coll, key_data='data2d')

    def test04_plot_spectral_fit_input_validity_2d(self):

        if self.coll.dobj.get('spect_model') is None:
            _inputs.add_models(self.coll)
            _inputs.add_fit2d(self.coll, key_data='data2d')

        # plot 2d
        _inputs._plot_input_validity_1d(self.coll, key_data='data2d')
