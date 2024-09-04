# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:04:47 2024

@author: dvezinet
"""


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds


# ##################################################################
# ##################################################################
#                    main
# ##################################################################


def main(
    coll=None,
    keys=None,
    labels=None,
    colors=None,
):

    # ---------------
    # check inputs
    # ---------------

    keys, defcolor, colors = _check(
        coll=coll,
        keys=keys,
        labels=labels,
        colors=colors,
    )

    # ----------------
    # labels
    # ----------------

    if labels is None:

        dlabels = {k0: {'label': k0} for k0 in keys}

    elif isinstance(labels, str):

        wsl = coll._which_lines
        dlabels = {
            k0: {'label': coll.dobj[wsl].get(k0, {}).get(labels, k0)}
            for k0 in keys
        }

    else:

        dlabels = {
            k0: {'label': labels.get(k0, k0)}
            for k0 in keys
        }

    # ----------------
    # colors
    # ----------------

    if colors is None:

        for k0 in dlabels.keys():
            dlabels[k0]['color'] = defcolor

    elif isinstance(colors, str):

        if mcolors.is_color_like(colors):
            for k0 in dlabels.keys():
                dlabels[k0]['color'] = colors

        elif colors in ['sum', 'details']:
            pass

        else:
            wsl = coll._which_lines

            # get unique values
            lunique = set([
                coll.dobj[wsl].get(k0, {}).get(colors, k0)
                for k0 in dlabels.keys()
            ])

            # get color cycle and loop
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for ii, uu in enumerate(lunique):
                for k0 in dlabels.keys():
                    if coll.dobj[wsl].get(k0, {}).get(colors, k0) == uu:
                        dlabels[k0]['color'] = color_cycle[ii]

    else:

        for k0 in dlabels.keys():
            dlabels[k0]['color'] = colors.get(k0, defcolor)

    return dlabels


# ##################################################################
# ##################################################################
#                    check
# ##################################################################


def _check(
    coll=None,
    keys=None,
    labels=None,
    colors=None,
):

    # ----------------
    # keys
    # ----------------

    # which
    wsl = coll._which_lines
    wsm = coll._which_model
    wsf = coll._which_fit

    # lok
    lok_lines = list(coll.dobj.get(wsl, {}).keys())
    lok_model = list(coll.dobj.get(wsm, {}).keys())
    lok_fit = list(coll.dobj.get(wsf, {}).keys())

    # --------
    # deal with by case

    if isinstance(keys, str):

        if keys in lok_fit + lok_model:

            if keys in lok_fit:
                keym = coll.dobj[wsf][keys][wsm]
            else:
                keym = keys

            keys = [
                k0 for k0, v0 in coll.dobj[wsm][keym]['dmodel'].items()
                if v0['type'] not in ['poly', 'exp_lamb']
            ]
            lok = keys

        else:
            keys = [keys]
            lok = lok_lines

    else:
        lok = lok_lines

    # ----------
    # check

    keys = ds._generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

    # ----------------
    # labels
    # ----------------

    if isinstance(labels, str):

        if coll.dobj.get(wsl) is None:
            msg = (
                "Arg 'labels', if str, must be a spectral line parameter!\n"
                "But you apparently have no spectral lines in your Collection!"
                f"\nProvided: '{labels}'\n"
            )
            raise Exception(msg)

        lparam = coll.get_lparam(wsl)

        if labels not in lparam:
            msg = (
                "Arg 'labels', if str, must be a spectral line parameter!\n"
                "\t available: {lparam}\n"
                f"\t Provided: '{labels}'\n"
            )
            raise Exception(msg)

    elif isinstance(labels, dict):

        c0 = all([
            isinstance(k0, str)
            and k0 in keys
            and isinstance(v0, str)
            for k0, v0 in labels.items()
        ])
        if not c0:
            msg = (
                "Arg 'labels', if dict, must be of the form:\n"
                "\t- dict: {'key0': 'label0', 'key1': 'label1', ...}\n"
                f"Provided:\n{labels}\n"
            )
            raise Exception(msg)

    elif labels is not None:
        msg = (
            "Arg 'labels' must be either:\n"
            "\t- str: a valid spectral line parameter"
            "\t\t e.g: 'symbol', 'ion', ...\n"
            "\t- dict: {'key0': 'label0', 'key1': 'label1', ...}\n"
            f"Provided:\n{labels}\n"
        )
        raise Exception(msg)

    # ----------------
    # colors
    # ----------------

    # default
    defcolor = 'k'
    if colors is None:
        if coll.dobj.get(wsl) is None:
            colors = defcolor
        else:
            colors = 'ion'

    # case by case
    if isinstance(colors, str):

        if mcolors.is_color_like(colors):
            pass

        elif colors in ['sum', 'details']:
            pass

        else:

            if coll.dobj.get(wsl) is None:
                msg = (
                    "Arg 'colors', if str and not any of these:\n"
                    "\t- any color-like str\n"
                    "\t- 'sum'\n"
                    "\t- 'details'\n"
                    "Then must be a spectral line parameter!\n"
                    "But you apparently have no spectral lines in your Collection!"
                    f"\nProvided: '{colors}'\n"
                )
                raise Exception(msg)

            lparam = coll.get_lparam(wsl)

            if colors not in lparam:
                msg = (
                    "Arg 'colors', if str, must be a spectral line parameter!\n"
                    "\t available: {lparam}\n"
                    f"\t Provided: '{colors}'\n"
                )
                raise Exception(msg)

    elif isinstance(colors, dict):

        c0 = all([
            isinstance(k0, str)
            and k0 in keys
            and isinstance(v0, str)
            for k0, v0 in colors.items()
        ])
        if not c0:
            msg = (
                "Arg 'colors', if dict, must be of the form:\n"
                "\t- dict: {'key0': 'color0', 'key1': 'color1', ...}\n"
                f"Provided:\n{colors}\n"
            )
            raise Exception(msg)

    elif colors is not None:
        msg = (
            "Arg 'colors' must be either:\n"
            "\t- str: a valid color-like str, 'sum', 'details' "
            "or a valid spectral line parameter"
            "\t\t e.g: 'symbol', 'ion', ...\n"
            "\t- dict: {'key0': 'color0', 'key1': 'color1', ...}\n"
            f"Provided:\n{colors}\n"
        )
        raise Exception(msg)

    return keys, defcolor, colors