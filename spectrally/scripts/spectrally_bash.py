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


###################################################
###################################################
#       default values
###################################################


_PKG = 'spectrally'
_PKGPATH = os.path.dirname(_HERE)
_ENTRYPOINTS_PATH = os.path.join(_PKGPATH, _PKG, 'entrypoints')


_LOPTIONS = ['--version', 'custom']
_LOPSTRIP = [ss.strip('--') for ss in _LOPTIONS]


###################################################
###################################################
#       function
###################################################


def spectrally_bash(option=None, ddef=None, **kwdargs):
    f""" Print {_PKG} version and / or store in environment variable """

    # --------------
    # Check inputs
    # --------------

    if option not in _LOPSTRIP:
        msg = (
            "Provided option is not acceptable:\n"
            f"\t- available: {_LOPSTRIP}\n"
            f"\t- provided:  {option}"
        )
        raise Exception(msg)

    # --------------------------------
    # call corresponding bash command
    # --------------------------------

    # version
    if option == 'version':
        sys.path.insert(1, _HERE)
        import spectrallyversion
        _ = sys.path.pop(1)
        spectrallyversion.get_version(ddef=ddef, **kwdargs)

    # custom
    elif option == 'custom':
        sys.path.insert(1, _HERE)
        import spectrallycustom
        _ = sys.path.pop(1)
        spectrallycustom.custom(ddef=ddef, **kwdargs)

    return


###################################################
###################################################
#       bash call (main)
###################################################


def main():

    # ----------------------
    # Parse input arguments
    # ----------------------

    # description msg
    msg = (
        f"Get {_PKG} version from bash\n\n"
        "Optionally set an enviroment variable\n"
        "If run from a git repo containing {_PKG}, returns git describe\n"
        "Otherwise reads the {_PKG} version stored in {_PKG}/version.py\n"
    )

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # ----------------------
    # Define input arguments
    # ----------------------

    # option
    parser.add_argument(
        'option',
        nargs='?',
        type=str,
        default='None',
    )

    # version
    parser.add_argument(
        '-v', '--version',
        help=f'get {_PKG} current version',
        required=False,
        action='store_true',
    )

    # kwd
    parser.add_argument(
        'kwd',
        nargs='?',
        type=str,
        default='None',
    )

    # ----------------------
    # Options
    # ----------------------

    # check provided options
    if sys.argv[1] not in _LOPTIONS:
        msg = (
            "Provided option is not acceptable:\n"
            f"\t- available: {_LOPTIONS}\n"
            f"\t- provided:  {sys.argv[1]}"
        )
        raise Exception(msg)

    # check nb of options
    if len(sys.argv) > 2:
        if any([ss in sys.argv[2:] for ss in _LOPTIONS]):
            lopt = [ss for ss in sys.argv[1:] if ss in _LOPTIONS]
            msg = (
                "Only one option can be provided!\n"
                f"\t- provided: {lopt}"
            )
            raise Exception(msg)

    # get option
    option = sys.argv[1].strip('--')

    # parse to get kwdargs
    ddef, parser = _DPARSER[option]()
    if len(sys.argv) > 2:
        kwdargs = dict(parser.parse_args(sys.argv[2:])._get_kwargs())
    else:
        kwdargs = {}

    # ----------------------
    # call function
    # ----------------------

    spectrally_bash(option=option, ddef=ddef, **kwdargs)

    return


###################################################
###################################################
#       __main__
###################################################


if __name__ == '__main__':
    main()