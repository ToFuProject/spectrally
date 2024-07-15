import sys
import os
import argparse


# ########################################################
# ########################################################
#               DEFAULTS
# ########################################################


_PKG = 'spectrally'


_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_PKG = os.path.dirname(os.path.dirname(_HERE))
_PATH_LOCALDIR = os.path.join(os.path.expanduser('~'), f'.{_PKG}')


# ########################################################
# ########################################################
#               get mods
# ########################################################


def get_mods():
    """ Test if git repo """

    # ----------------------------------
    # isgit repo? (load local vs global)

    isgit = False
    if '.git' in os.listdir(_PATH_PKG) and _PKG in _PATH_PKG:
        isgit = True

    if isgit:
        # Make sure we load the corresponding pkg
        sys.path.insert(1, _PATH_PKG)
        import spectrally as sp
        _ = sys.path.pop(1)
    else:
        import spectrally as sp

    # ------------------
    # default parameters

    pfe = os.path.join(_PATH_LOCALDIR, '_scripts_def.py')
    if os.path.isfile(pfe):
        # Make sure we load the user-specific file
        # sys.path method
        # sys.path.insert(1, os.path.join(os.path.expanduser('~'), '.tofu'))
        # import _scripts_def as _defscripts
        # _ = sys.path.pop(1)
        # importlib method
        import importlib.util
        spec = importlib.util.spec_from_file_location("_defscripts", pfe)
        _defscripts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_defscripts)
    else:
        try:
            import spectrally.entrypoints._def as _defscripts
        except Exception:
            from . import _def as _defscripts

    return sp, _defscripts


# ########################################################
# ########################################################
#       utility functions
# ########################################################


def _str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['yes', 'true', 'y', 't', '1']:
        return True
    elif v.lower() in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


def _str2boolstr(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ['yes', 'true', 'y', 't', '1']:
            return True
        elif v.lower() in ['no', 'false', 'n', 'f', '0']:
            return False
        elif v.lower() == 'none':
            return None
        else:
            return v
    else:
        raise argparse.ArgumentTypeError('Boolean, None or str expected!')


def _str2tlim(v):
    c0 = (v.isdigit()
          or ('.' in v
              and len(v.split('.')) == 2
              and all([vv.isdigit() for vv in v.split('.')])))
    if c0 is True:
        v = float(v)
    elif v.lower() == 'none':
        v = None
    return v


# ########################################################
# ########################################################
#       Parser for version
# ########################################################


def parser_version():

    # -------------
    # defaults
    # -------------

    ddef = {
        'path': os.path.join(_PATH_PKG, _PKG),
        'envvar': False,
        'verb': True,
        'warn': True,
        'force': False,
        'name': f'{_PKG.upper()}_VERSION',
    }

    # ------------------
    # Instanciate parser
    # ------------------

    # description msg
    msg = (
        "Get pkg version from bash optionally set an enviroment variable\n\n"
        "If run from a git repo containing {_PKG}, returns git describe\n"
        "Otherwise reads the {_PKG} version stored in {_PKG}/version.py\n"
    )

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # ----------------------
    # Define input arguments
    # ----------------------

    # path
    parser.add_argument(
        '-p', '--path',
        type=str,
        help=f'{_PKG} source directory to version.py',
        required=False, default=ddef['path'],
    )

    # verb
    parser.add_argument(
        '-v', '--verb',
        type=_str2bool,
        help='flag indicating whether to print the version',
        required=False,
        default=ddef['verb'],
    )

    # env variable
    parser.add_argument(
        '-ev', '--envvar',
        type=_str2boolstr,
        help='name of the environment variable to set, if any',
        required=False,
        default=ddef['envvar'],
    )

    # warnings
    parser.add_argument(
        '-w', '--warn',
        type=_str2bool,
        help=(
            'flag indicatin whether to print a warning when '
            'desired environment variable (envvar) already exists'
        ),
        required=False,
        default=ddef['warn'],
    )

    # force
    parser.add_argument(
        '-f', '--force',
        type=_str2bool,
        help=(
            'flag indicating whether to force the update of '
            'desired environment variable (envvar) even if it already exists'
        ),
        required=False,
        default=ddef['force'],
    )

    return ddef, parser


# ########################################################
# ########################################################
#       Parser for custom
# ########################################################


def parser_custom():

    # -------------
    # defaults
    # -------------

    ddef = {
        'target': _PATH_LOCALDIR,
        'source': os.path.join(_PATH_PKG, _PKG),
        'files': [
            '_entrypoints_def.py',
        ],
        'directories': [
            'openadas',
            'nist',
            os.path.join('nist', 'ASD'),
        ],
    }

    # ------------------
    # Instanciate parser
    # ------------------

    # description msg
    msg = (
        f" Create a local copy of {_PKG} default parameters\n\n"
        f"A directory '.{_PKG}' is created in your home directory:\n"
        f"\t{_PATH_LOCALDIR}\n"
        "In this directory, modules containing default parameters are copied\n"
        "You can then customize them without impacting other users\n"
    )

    # instanciate
    parser = argparse.ArgumentParser(description=msg)

    # ----------------------
    # Define input arguments
    # ----------------------

    # source
    parser.add_argument(
        '-s', '--source',
        type=str,
        help=f'{_PKG} source directory',
        required=False,
        default=ddef['source'],
    )

    # target
    parser.add_argument(
        '-t', '--target',
        type=str,
        help=(f'directory where .{_PKG}/ should be created'
              + ' (default: {})'.format(ddef['target'])),
        required=False,
        default=ddef['target'],
    )

    # files
    parser.add_argument(
        '-f', '--files',
        type=str,
        help='list of files to be copied',
        required=False,
        nargs='+',
        default=ddef['files'],
        choices=ddef['files'],
    )

    return ddef, parser


# ########################################################
# ########################################################
#       Parser dict
# ########################################################


_DPARSER = {
    'version': parser_version,
    'custom': parser_custom,
}
