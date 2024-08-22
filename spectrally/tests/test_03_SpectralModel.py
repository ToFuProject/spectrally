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
from .._class02_SpectralFit import SpectralFit as Collection
# from .._saveload import load
from .._class01_show import get_available_spectral_model_functions
from .._class01_display_models import display_spectral_model_function
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
        _inputs.add_xfree(self.coll)

        # add spectral lines
        pfe_json = os.path.join(_PATH_INPUT, 'spectrallines.json')
        self.coll.add_spectral_lines_from_file(pfe_json)

    # ------------------------
    #   Populating
    # ------------------------

    # -------------
    # add models

    def test00_get_available_spectral_model_funcstions(self):
        get_available_spectral_model_functions()

    def test01_display_spectral_model_function(self):
        lm = [
            'poly', 'exp_lamb',
            'gauss', 'lorentz', 'pvoigt',
            'pulse_exp', 'pulse_gauss', 'lognorm',
        ]
        for kk in lm:
            _ = display_spectral_model_function(kk)
        plt.close('all')

    def test02_add_spectral_model(self):
        _inputs.add_models(self.coll)

    def test03_get_spectral_model_func(self):
        _inputs.get_spectral_model_func(self.coll)

    def test04_interpolate_spectral_model(self):
        _inputs.interpolate_spectral_model(self.coll)

    def test05_get_spectral_model_moments(self):
        _inputs.get_spectral_model_moments(self.coll)

    def test06_plot_spectral_model(self):
        _inputs.plot_spectral_model(self.coll)