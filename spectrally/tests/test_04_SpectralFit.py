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
from .._class02_SpectralFit import SpectralFit as Collection
# from .._saveload import load
from . import _spectralfit_input as _inputs
from . import _hxr_pulses


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

        # add spectral lines
        pfe_json = os.path.join(_PATH_INPUT, 'spectrallines.json')
        self.coll.add_spectral_lines_from_file(pfe_json)

    # ------------------------
    #   Populating
    # ------------------------

    # ---------------
    # 1d spectral fit

    def test01_add_spectral_fit_1d_single(self):
        # add spectral fit 1d
        _inputs.add_fit_single(self.coll)

    def test02_add_spectral_fit_1d_double(self):
        # add spectral fit 1d
        _inputs.add_fit_double(self.coll)

    def test03_add_spectral_fit_1d_multi(self):
        # add spectral fit 1d
        _inputs.add_fit_multi(self.coll)

    def test04_plot_spectral_fit_input_validity_1d(self):
        # plot 1d
        _inputs.plot_input_validity(self.coll, key_data='data1d')

    def test05_compute_spectral_fit_1d_single(self):
        # compute 1d
        _inputs.compute_fit_single(self.coll)

    def test06_compute_spectral_fit_1d_double(self):
        # compute 1d
        _inputs.compute_fit_double(self.coll)

    def test07_compute_spectral_fit_1d_single_binning(self):
        # compute 1d
        _inputs.compute_fit_single(self.coll, binning=10)

    def test08_compute_spectral_fit_1d_multi(self):
        # compute 1d
        _inputs.compute_fit_multi(self.coll, key_data='data1d')

    # ---------------
    # 2d spectral fit

    # def test06_add_spectral_fit_2d(self):
    #     # add spectral fit 2d
    #     _inputs.add_fit(self.coll, key_data='data2d')

    # def test07_plot_spectral_fit_input_validity_2d(self):
    #     # plot 2d
    #     _inputs.plot_input_validity(self.coll, key_data='data2d')

    # ------------------------
    #   Special cases
    # ------------------------

    def test99_compute_HXR_pulses(self):
        _hxr_pulses.main(self.coll)