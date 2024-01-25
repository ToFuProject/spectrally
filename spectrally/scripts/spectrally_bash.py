#!/usr/bin/env python

# Built-in
import sys
import os
import argparse


_HERE = os.path.abspath(os.path.dirname(__file__))


# import parser dict
sys.path.insert(1, _HERE)
from _dparser import _DPARSER
_ = sys.path.pop(1)

_PKG = 'spectrally'
_PKGPATH = os.path.dirname(_HERE)
_ENTRYPOINTS_PATH = os.path.join(_PKGPATH, _PKG, 'entrypoints')


###################################################
###################################################
#       default values
###################################################


_LOPTIONS = ['--version', 'custom', 'plot', 'calc']
_LOPSTRIP = [ss.strip('--') for ss in _LOPTIONS]


###################################################
###################################################
#       function
###################################################


def spectrally_bash(option=None, ddef=None, **kwdargs):
    f""" Print {_PKG} version and / or store in environment variable """

    # --------------
    # Check inputs
    if option not in _LOPSTRIP:
        msg = ("Provided option is not acceptable:\n"
               + "\t- available: {}\n".format(_LOPSTRIP)
               + "\t- provided:  {}".format(option))
        raise Exception(msg)

    # --------------
    # call corresponding bash command
    if option == 'version':
        sys.path.insert(1, _HERE)
        import spectrallyversion
        _ = sys.path.pop(1)
        spectrallyversion.get_version(ddef=ddef, **kwdargs)

    elif option == 'custom':
        sys.path.insert(1, _HERE)
        import spectrallycustom
        _ = sys.path.pop(1)
        spectrallycustom.custom(ddef=ddef, **kwdargs)

    elif option == 'plot':
        sys.path.insert(1, _ENTRYPOINTS_PATH)
        import spectrallyplot
        _ = sys.path.pop(1)
        spectrallyplot.call_tfloadimas(ddef=ddef, **kwdargs)


###################################################
###################################################
#       bash call (main)
###################################################


def main():
    # Parse input arguments
    msg = f""" Get {_PKG} version from bash optionally set an enviroment variable

    If run from a git repo containing {_PKG}, simply returns git describe
    Otherwise reads the {_PKG} version stored in {_PKG}/version.py

    """

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('option',
                        nargs='?',
                        type=str,
                        default='None')
    parser.add_argument('-v', '--version',
                        help=f'get {_PKG} current version',
                        required=False,
                        action='store_true')
    parser.add_argument('kwd', nargs='?', type=str, default='None')

    if sys.argv[1] not in _LOPTIONS:
        msg = ("Provided option is not acceptable:\n"
               + "\t- available: {}\n".format(_LOPTIONS)
               + "\t- provided:  {}".format(sys.argv[1]))
        raise Exception(msg)
    if len(sys.argv) > 2:
        if any([ss in sys.argv[2:] for ss in _LOPTIONS]):
            lopt = [ss for ss in sys.argv[1:] if ss in _LOPTIONS]
            msg = ("Only one option can be provided!\n"
                   + "\t- provided: {}".format(lopt))
            raise Exception(msg)

    option = sys.argv[1].strip('--')
    ddef, parser = _DPARSER[option]()
    if len(sys.argv) > 2:
        kwdargs = dict(parser.parse_args(sys.argv[2:])._get_kwargs())
    else:
        kwdargs = {}

    # Call function
    spectrally_bash(option=option, ddef=ddef, **kwdargs)


if __name__ == '__main__':
    main()
