"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import itertools as itt


# Standard
import numpy as np


# local
from ._setup_teardown import setup_module0, teardown_module0
from .. import openadas


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
#     Creating Ves objects and testing methods
#
#######################################################

# DOES NOT SEEM TO WORK ON GITUB ANYMORE
# class Test_openadas(object):

#     # ------------------------
#     #   setup and teardown
#     # ------------------------

#     @classmethod
#     def setup_class(cls):
#         pass

#     @classmethod
#     def setup_method(self):
#         pass

#     def teardown_method(self):
#         pass

#     @classmethod
#     def teardown_class(cls):
#         pass

#     # ------------------------
#     #   search online
#     # ------------------------

#     def test01_search_online_by_searchstr(self):
#         out = openadas.step01_search_online(
#             'ar+16 ADF15',
#             verb=True,
#             # verb=False,
#             returnas=False,
#         )
#         assert out is None

#         out = openadas.step01_search_online(
#             'ar+16 ADF11',
#             verb=False,
#             returnas=np.ndarray,
#         )
#         assert isinstance(out, np.ndarray)

#     def test02_search_online_by_wavelength(self):
#         # lret = [None, str, np.ndarray]
#         lret = [str]
#         # lverb = [True, False]
#         lverb = [False]
#         lelement = [None, 'ar', ['ar', 'w'], ('w',)]
#         # lcharge = [None, 14, [15, 16], (16,)]
#         lcharge = [[15, 16], (16,)]
#         lres = ['transition', 'file']
#         for comb in itt.product(lret, lverb, lelement, lcharge, lres):
#             out = openadas.step01_search_online_by_wavelengthA(
#                 lambmin=3.94,
#                 lambmax=4.,
#                 returnas=comb[0],
#                 verb=comb[1],
#                 element=comb[2],
#                 charge=comb[3],
#                 resolveby=comb[4],
#             )
#             assert out is comb[0] or isinstance(out, comb[0])

#     # ------------------------
#     #       download
#     # ------------------------

#     def test03_download(self):
#         out = openadas.step02_download(
#             filename='/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
#             update=False,
#             verb=False,
#             returnas=False,
#         )
#         assert out is None

#         out = openadas.step02_download(
#             filename='/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
#             update=True,
#             verb=False,
#             returnas=str,
#         )
#         assert isinstance(out, str)

#         out = openadas.step02_download_all(
#             lambmin=3.94,
#             lambmax=4,
#             element='Mo',
#             update=False,
#             verb=True,
#         )
#         assert out is None

#         lf = [
#             '/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
#             '/adf11/scd74/scd74_ar.dat',
#             '/adf15/pec40][w/pec40][w_ls][w64.dat',
#             '/adf11/plt41/plt41_xe.dat',
#         ]
#         out = openadas.step02_download_all(
#             files=lf,
#             update=False,
#             verb=False,
#         )
#         assert out is None

#         out = openadas.step02_download_all(
#             searchstr='ar+16 ADF15',
#             include_partial=False,
#             update=False,
#             verb=False,
#         )
#         assert out is None

#     # ------------------------
#     #   readfiles
#     # ------------------------

#     def test04_readfiles(self):
#         out = openadas.step03_read(
#             '/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
#         )
#         assert isinstance(out, dict)

#         out = openadas.step03_read(
#             'adf11/plt41/plt41_xe.dat',
#         )
#         assert isinstance(out, dict)

#         out = openadas.step03_read_all(
#             element='ar',
#             charge=16,
#             typ1='adf15',
#             lambmin=3.94,
#             lambmax=4.,
#         )
#         assert isinstance(out, dict)

#         out = openadas.step03_read_all(
#             element='ar',
#             charge=16,
#             typ1='adf11',
#             typ2='scd',
#         )
#         assert isinstance(out, dict)

#         out = openadas.step03_read_all(
#             element='ar',
#             charge=16,
#             typ1='adf11',
#             typ2='plt',
#         )
#         assert isinstance(out, dict)

#     # ------------------------
#     #   clear
#     # ------------------------

#     def test05_clear_downloads(self):
#         openadas.clear_downloads()