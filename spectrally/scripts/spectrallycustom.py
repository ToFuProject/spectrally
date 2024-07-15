#!/usr/bin/env python

# Built-in
import sys
import os
from shutil import copyfile


_HERE = os.path.abspath(os.path.dirname(__file__))


# import parser dict
sys.path.insert(1, _HERE)
from _dparser import _DPARSER
_ = sys.path.pop(1)


_PKG_NAME = 'spectrally'


###################################################
###################################################
#        Main
###################################################


def main():
    # Parse input arguments
    # Instanciate parser
    ddef, parser = _DPARSER['custom']()

    # Parse arguments
    args = parser.parse_args()

    # Call function
    custom(ddef=ddef, **dict(args._get_kwargs()))

    return


###################################################
###################################################
#       function
###################################################


def custom(
    target=None,
    source=None,
    files=None,
    directories=None,
    ddef=None,
):

    # --------------
    # Check inputs
    # --------------

    # get input
    kwd = locals()
    for k0 in set(ddef.keys()).intersection(kwd.keys()):
        if kwd[k0] is None:
            kwd[k0] = ddef[k0]

    # extract parameters
    target, source = kwd['target'], kwd['source']
    files, directories = kwd['files'], kwd['directories']

    # Caveat (up to now only relevant for _TARGET)
    if target != ddef['target']:
        msg = ""
        raise Exception(msg)

    # --------------
    # Check files
    # --------------

    if isinstance(files, str):
        files = [files]

    c0 = (
        not isinstance(files, list)
        or any([ff not in ddef['files'] for ff in files])
    )
    if c0 is True:
        msg = "All files should be in {}".format(ddef['files'])
        raise Exception(msg)

    # -------------------------------------------
    # Try creating directory and copying modules
    # -------------------------------------------

    try:
        # Create .spectrally/ if non-existent
        if not os.path.isdir(target):
            os.mkdir(target)

        # Create directories
        for dd in directories:
            if not os.path.isdir(os.path.join(target, dd)):
                os.mkdir(os.path.join(target, dd))

        # Copy files
        for ff in files:
            mod, f0 = ff.split('_')[1:]
            copyfile(
                os.path.join(source, mod, f'_{f0}'),
                os.path.join(target, ff),
            )

        msg = (
            f"A local copy of default {_PKG_NAME} parameters is now in:\n"
            f"\t{target}/\n"
            f"You can edit it to spice up your {_PKG_NAME}"
        )
        print(msg)

    except Exception as err:
        msg = (
            str(err)
            + "\n\nA problem occured\n"
            f"{_PKG_NAME}-custom failed to create a dir .{_PKG_NAME}/ in "
            f"your home {target}\n"
            "But it could not, check the error message above to debug\n"
            "Most frequent cause is a permission issue"
        )
        raise Exception(msg)

    return


###################################################
###################################################
#        __main__
###################################################


if __name__ == '__main__':
    main()
