import sys
import os
import getpass
import argparse


_PKG = 'spectrally'


# test if in a git repo
_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_PKG = os.path.dirname(_HERE)


def get_mods():
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

    # default parameters
    pfe = os.path.join(os.path.expanduser('~'), '.{_PKG}', '_scripts_def.py')
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
        except Exception as err:
            from . import _def as _defscripts
    return tf, _defscripts


# #############################################################################
#       utility functions
# #############################################################################


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


# #############################################################################
#       Parser for version
# #############################################################################


def parser_version():
    msg = f""" Get pkg version from bash optionally set an enviroment variable

    If run from a git repo containing {_PKG}, simply returns git describe
    Otherwise reads the {_PKG} version stored in {_PKG}/version.py

    """
    ddef = {
        'path': os.path.join(_TOFUPATH, _PKG),
        'envvar': False,
        'verb': True,
        'warn': True,
        'force': False,
        'name': 'TOFU_VERSION',
    }

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('-p', '--path',
                        type=str,
                        help=f'{_PKG} source directory where version.py is found',
                        required=False, default=ddef['path'])
    parser.add_argument('-v', '--verb',
                        type=_str2bool,
                        help='flag indicating whether to print the version',
                        required=False, default=ddef['verb'])
    parser.add_argument('-ev', '--envvar',
                        type=_str2boolstr,
                        help='name of the environment variable to set, if any',
                        required=False, default=ddef['envvar'])
    parser.add_argument('-w', '--warn',
                        type=_str2bool,
                        help=('flag indicatin whether to print a warning when'
                              + 'the desired environment variable (envvar)'
                              + 'already exists'),
                        required=False, default=ddef['warn'])
    parser.add_argument('-f', '--force',
                        type=_str2bool,
                        help=('flag indicating whether to force the update of '
                              + 'the desired environment variable (envvar)'
                              + ' even if it already exists'),
                        required=False, default=ddef['force'])

    return ddef, parser


# #############################################################################
#       Parser for custom
# #############################################################################


def parser_custom():
    msg = f""" Create a local copy of {_PKG} default parameters

    This creates a local copy, in your home, of {_PKG} default parameters
    A directory .{_PKG} is created in your home directory
    In this directory, modules containing default parameters are copied
    You can then customize them without impacting other users

    """
    _USER = getpass.getuser()
    _USER_HOME = os.path.expanduser('~')

    ddef = {
        'target': os.path.join(_USER_HOME, f'.{_PKG}'),
        'source': os.path.join(_TOFUPATH, _PKG),
        'files': [
            '_entrypoints_def.py',
        ],
        'directories': [
            'openadas',
            'nist',
            os.path.join('nist', 'ASD'),
        ],
    }

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('-s', '--source',
                        type=str,
                        help=f'{_PKG} source directory',
                        required=False,
                        default=ddef['source'])
    parser.add_argument('-t', '--target',
                        type=str,
                        help=(f'directory where .{_PKG}/ should be created'
                              + ' (default: {})'.format(ddef['target'])),
                        required=False,
                        default=ddef['target'])
    parser.add_argument('-f', '--files',
                        type=str,
                        help='list of files to be copied',
                        required=False,
                        nargs='+',
                        default=ddef['files'],
                        choices=ddef['files'])
    return ddef, parser


# #############################################################################
#       Parser dict
# #############################################################################


_DPARSER = {
    'version': parser_version,
    'custom': parser_custom,
}
