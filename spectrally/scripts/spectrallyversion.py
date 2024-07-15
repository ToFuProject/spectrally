#!/usr/bin/env python

# Built-in
import sys
import os
import warnings


_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG = 'spectrally'


# import parser dict
sys.path.insert(1, _HERE)
from _dparser import _DPARSER
_ = sys.path.pop(1)


###################################################
###################################################
#       Main
###################################################


def main():
    # Parse input arguments

    # Parse arguments
    ddef, parser = _DPARSER['version']()

    # Parse arguments
    args = parser.parse_args()

    # Call function
    get_version(ddef=ddef, **dict(args._get_kwargs()))

    return


###################################################
###################################################
#       function
###################################################


def get_version(
    verb=None,
    envvar=None,
    path=None,
    warn=None,
    force=None,
    ddef=None,
):
    """ Print tofu version and / or store in environment variable """

    # --------------
    # Check inputs
    # --------------

    # extract input
    kwd = locals()
    for k0 in set(ddef.keys()).intersection(kwd.keys()):
        if kwd[k0] is None:
            kwd[k0] = ddef[k0]
    verb, envvar, path = kwd['verb'], kwd['envvar'], kwd['path']
    warn, force = kwd['warn'], kwd['force']

    # --------------
    # Check
    # --------------

    # verb, warn, force
    dbool = {'verb': verb, 'warn': warn, 'force': force}
    for k0, v0 in dbool.items():
        if v0 is None:
            dbool[k0] = ddef[k0]
        if not isinstance(dbool[k0], bool):
            msg = (
                f"Arg {k0} must be a bool\n"
                "\t- provided: {dbool[k0]}\n"
            )
            raise Exception(msg)

    # --------------
    # Check inputs
    # --------------

    # set envvar
    if isinstance(envvar, bool):
        if envvar is True:
            envvar = ddef['name']

    elif isinstance(envvar, str):
        envvar = envvar.upper()

    else:
        msg = (
            "Arg envvar must be either:\n"
            f"\t- None:  set to default ({ddef['envvar']})\n"
            "\t- False: no setting of environment variable\n"
            f"\t- True:  set default env. var. ({ddef['name']})\n"
            "\t- str:   set provided env. variable\n\n"
            f" you provided: {envvar}"
        )
        raise Exception(msg)

    # warning on name of env variable
    if envvar is not False and envvar in os.environ.keys():
        if dbool['warn'] is True:
            msg = (
                "Chosen environment variable name already exists:\n"
                f"\t- {envvar}: {os.environ[envvar]}\n"
                "  => Are you sure you want to change it?\n"
                f"       use force=True to overwrite {envvar}\n"
                "       use warn=False to disable this warning"
            )
            warnings.warn(msg)

    # --------------
    # Fetch version from git tags, and write to version.py
    # Also, when git is not available (PyPi package), use stored version.py
    # --------------

    pfe = os.path.join(path, 'version.py')
    if not os.path.isfile(pfe):
        msg = (
            f"It seems your current {_PKG} install has no version.py:\n"
            f"\t- looked for: {pfe}"
        )
        raise Exception(msg)

    # --------------
    # Read file
    # --------------

    with open(pfe, 'r') as fh:
        version = fh.read().strip().split("=")[-1].replace("'", '')
    version = version.lower().replace('v', '').replace(' ', '')

    # --------------
    # Outputs
    # --------------

    if dbool['verb'] is True:
        print(version)
    if envvar is not False:
        c0 = ((envvar in os.environ.keys() and dbool['force'] is True)
              or envvar not in os.environ.keys())
        if c0 is True:
            os.environ[envvar] = version

    return


###################################################
###################################################
#       __main__
###################################################


if __name__ == '__main__':
    main()
