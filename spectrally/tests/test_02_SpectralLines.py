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
from .._class00_SpectralLines import SpectralLines as Collection
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
        self.pfe_json = os.path.join(_PATH_INPUT, 'spectrallines.json')

    # ------------------------
    #   Populating
    # ------------------------

    def test01_add_spectral_lines_from_file(self):
        self.coll.add_spectral_lines_from_file(self.pfe_json)

    def test02_add_spectral_lines_from_openadas(self):
        pass
        # self.coll.add_spectral_lines_from_openadas(
        #     lambmin=3.94e-10,
        #     lambmax=4e-10,
        #     element='Ar',
        #     online=True,
        # )

    def test03_add_spectral_lines_from_nist(self):
        try:
            self.coll.add_spectral_lines_from_nist(
                lambmin=3.94e-10,
                lambmax=4e-10,
                element='Ar',
            )
        except Exception as err:
            lstr = [
                '503 Server Error: Service Unavailable for url:',
                'File could not be downloaded:',
                '=> Maybe check internet connection?',
                # flag that it is running on Github
                (
                    '/runner/.spectrally/nist/',       # MacOS and linux
                    'runneradmin',    # Windows
                ),
            ]
            din = {
                ii: ss in str(err) if isinstance(ss, str)
                else any([s2 in str(err) for s2 in ss])
                for ii, ss in enumerate(lstr)
            }
            if all([vv for vv in din.values()]):
                pass
            else:
                lstr = [f"\t- {k0}: {v0}" for k0, v0 in din.items()]
                msg = "\n\n" + "\n".join(lstr) + "\n"
                raise Exception(msg) from err

    # ----------------
    # removing
    # ----------------

    def test04_remove_spectral_lines(self):
        # populate
        wsl = self.coll._which_lines
        if len(self.coll.dobj.get(wsl, {})) == 0:
            self.coll.add_spectral_lines_from_file(self.pfe_json)

        # remove
        lines = [
            k0 for k0, v0 in self.coll.dobj[wsl].items()
            if v0['ion'] != 'Ar16+'
        ]
        self.coll.remove_spectral_lines(lines)


#######################################################
#
#     Manipulate
#
#######################################################


class Test01_Manipulate():

    # ------------------------
    #   setup and teardown
    # ------------------------

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(self):
        self.coll = Collection()
        self.pfe_json = os.path.join(_PATH_INPUT, 'spectrallines.json')
        self.coll.add_spectral_lines_from_file(self.pfe_json)

    # ------------------------
    #   Plotting
    # ------------------------

    def test00_plot_spectral_lines(self):
        self.coll.plot_spectral_lines()
        plt.close('all')

    # ------------------------
    #   saving / loading
    # ------------------------

    def test98_save_spectral_lines_to_file(self):
        self.coll.save_spectral_lines_to_file(path=_PATH_OUTPUT)

    def test99_saveload(self):
        pfe = self.coll.save(path=_PATH_OUTPUT, return_pfe=True)
        coll2 = load(pfe, cls=Collection)
        assert self.coll == coll2