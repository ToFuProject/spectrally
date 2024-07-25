"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import itertools as itt


# local
from ._setup_teardown import setup_module0, teardown_module0
from .. import nist


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


class Test_nist(object):

    # ------------------------
    #   setup and teardown
    # ------------------------

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    # ------------------------
    #  search online
    # ------------------------

    def test01_search_online_by_wavelengthA(self):

        llambmin = [None, 3.94]
        llambmax = [None, 4.]
        lion = [None, 'H', ['ar', 'W44+']]

        lcache_from = [False, True]
        ldatacol = [False, True]

        # Search by searchstr
        llcomb = [
            llambmin, llambmax, lion,
            lcache_from,
            ldatacol,
        ]
        ii, itot = -1, 2*2*3*2*2
        for comb in itt.product(*llcomb):
            ii += 1
            if all([vv is None for vv in comb[:2]]):
                continue
            if comb[2] == 'H' and comb[1] is None:
                continue
            if comb[2] == 'H' and all([vv is not None for vv in comb[:2]]):
                continue
            if any([vv is None for vv in comb[:2]]) and comb[2] != 'H':
                continue
            print(f'{ii} / {itot}  -  {comb}')

            try:
                out = nist.step01_search_online_by_wavelengthA(
                    lambmin=comb[0],
                    lambmax=comb[1],
                    ion=comb[2],
                    verb=True,
                    return_dout=True,
                    return_dsources=True,
                    cache_from=comb[3],
                    cache_info=True,
                    format_for_DataStock=comb[4],
                    create_custom=True,
                )
                del out

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

    # ------------------------
    #  clear cache
    # ------------------------

    def test02_clear_cache(self):
        nist.clear_cache()