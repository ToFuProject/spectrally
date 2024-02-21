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
        self.coll = Collection()
        self.dmodel = {
            'model00': {
                'bck0': 'linear',
                'l00': 'gauss',
                'l01': 'gauss',
                'l02': 'lorentz',
            },
            'model01': {
                'bck0': 'exp',
                'l00': 'gauss',
                'l01': 'lorentz',
                'l02': 'pvoigt',
            },
            'model02': {
                'bck0': 'exp',
                'l00': 'gauss',
                'l01': 'lorentz',
                'l02': 'voigt',
            },
        }

    # ------------------------
    #   Populating
    # ------------------------

    def test00_add_spectral_model(self):

        # no constraints
        self.coll.add_spectral_model(
            key='model00',
            dmodel=self.dmodel['model00'],
            dconstraints=None,
        )

        # with constraints
        self.coll.add_spectral_model(
            key='model01',
            dmodel=self.dmodel['model01'],
            dconstraints={
                'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
                'g01': {'ref': 'l00_width', 'l01_gamma': [0, 1, 0]},
            },
        )

        # with voigt
        self.coll.add_spectral_model(
            key='model02',
            dmodel=self.dmodel['model02'],
            dconstraints={
                'g00': {'ref': 'l00_amp', 'l01_amp': [0, 1, 0]},
                'g01': {'ref': 'l00_width', 'l01_gamma': [0, 1, 0]},
            },
        )