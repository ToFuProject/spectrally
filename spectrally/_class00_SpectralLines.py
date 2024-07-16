# #!/usr/bin/python3
# -*- coding: utf-8 -*-


import copy


import numpy as np
from bsplines2d import BSplines2D as Previous


# from . import _class00_check as _check
from . import _class00_check_ion as _check_ion
from . import _class00_check_lines as _check_lines
from . import _class00_compute as _compute
from . import _class00_plot as _plot


__all__ = ['SpectralLines']


#############################################
#############################################
#       DEFAULT VALUES
#############################################


_QUANT_NE = 'ne'
_QUANT_TE = 'Te'
_UNITS_LAMBDA0 = 'm'


#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralLines(Previous):

    _ddef = copy.deepcopy(Previous._ddef)

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _dshow = dict(Previous._dshow)
    _dshow['ion'] = ['element', 'A', 'Z', 'q', 'isoelect']

    _ddef['params']['dobj'] = {
        # 'ion': {
            # 'A': {'cls': float},
            # 'Z': {'cls': int},
            # 'q': {'cls': int},
            # 'isoelect': {'cls': str},
        # },
        'lines': {
            'lambda0': {'cls': float, 'def': 0.},
            'source': {'cls': str, 'def': 'unknown'},
            'transition': {'cls': str, 'def': 'unknown'},
            'element':  {'cls': str, 'def': 'unknown'},
            'charge':  {'cls': int, 'def': 0},
            'ion':  {'cls': str, 'def': 'unknown'},
            'symbol':   {'cls': str, 'def': 'unknown'},
        },
    }

    _which_ion = 'ion'
    _which_lines = 'spect_lines'
    _quant_ne = _QUANT_NE
    _quant_Te = _QUANT_TE

    _units_lamb0 = _UNITS_LAMBDA0

    # -------------------
    # add ion
    # -------------------

    def add_ion(self, key=None):
        """ Add an ion
        """
        _check_ion.add_ion(
            coll=self,
            key=key,
        )

    def remove_ion(self, key=None, propagate=None):
        """ Add an ion
        """
        _check_ion.remove_ion(
            coll=self,
            key=key,
            propagate=propagate,
        )

    # -------------------
    # add spectral lines
    # -------------------

    def add_spectral_line(
        self,
        key=None,
        ion=None,
        lamb0=None,
        transition=None,
        pec=None,
        source=None,
        symbol=None,
        **kwdargs,
    ):
        """ Add a spectral line by key and rest wavelength, optionally with

        """
        _check_lines.add_lines(
            coll=self,
            key=key,
            lamb0=lamb0,
            pec=pec,
            source=source,
            transition=transition,
            ion=ion,
            symbol=symbol,
            **kwdargs,
        )

    def add_spectral_lines_from_file(
        self,
        pfe=None,
    ):
        """ Add a spectral line from json, npz or py file

        """
        dobj = _compute.from_file(
            coll=self,
            pfe=pfe,
        )

        return _check_lines._add_lines_from_dobj(
            coll=self,
            dobj=dobj,
        )

    # ---------------------
    # Add PEC
    # ---------------------

    def add_pec(self, key=None, pec=None, ref=None):
        pass

    # -----------------
    # from openadas
    # ------------------

    def add_spectral_lines_from_openadas(
        self,
        lambmin=None,
        lambmax=None,
        element=None,
        charge=None,
        online=None,
        update=None,
        create_custom=None,
    ):
        """
        Load lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.spectrally/openadas/

        Provide wavelengths in m

        Example:
        --------
                >>> import spectrally as sp
                >>> coll = sp.Collection()
                >>> coll.add_spectral_lines_from_openadas(
                    element='Mo',
                    lambmin=3.94e-10,
                    lambmax=4e-10,
                )

        """

        ddata, dref, dobj = _compute.from_openadas(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            online=online,
            update=update,
            create_custom=create_custom,
            dsource0=self._dobj.get('source'),
            dref0=self._dref if len(self._dref) > 0 else None,
            ddata0=self._ddata if len(self._ddata) > 0 else None,
            dobj0=self._dobj,
            which_lines=self._which_lines,
        )

        return _check_lines._add_lines_from_dobj(
            coll=self,
            dref=dref,
            ddata=ddata,
            dobj=dobj,
        )

    # -----------------
    # from nist
    # ------------------

    def add_spectral_lines_from_nist(
        self,
        lambmin=None,
        lambmax=None,
        element=None,
        charge=None,
        ion=None,
        wav_observed=None,
        wav_calculated=None,
        transitions_allowed=None,
        transitions_forbidden=None,
        cache_from=None,
        cache_info=None,
        verb=None,
        create_custom=None,
    ):
        """
        Load lines from nist, either:
            - cache_from = False:  directly from the website
            - cache_from = True: from pre-downloaded files in ~/.spectrally/nist/

        Provide wavelengths in m

        Example:
        --------
                >>> import spectrally as sp
                >>> coll = sp.Collection()
                >>> sp.add_spectral_lines_from_nist(
                    element='Mo',
                    lambmin=3.94e-10,
                    lambmax=4e-10,
                )

        """

        dobj = _compute.from_nist(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            ion=ion,
            wav_observed=wav_observed,
            wav_calculated=wav_calculated,
            transitions_allowed=transitions_allowed,
            transitions_forbidden=transitions_forbidden,
            cache_from=cache_from,
            cache_info=cache_info,
            verb=verb,
            create_custom=create_custom,
            dsource0=self._dobj.get('source'),
            dlines0=self._dobj.get('lines'),
            group_lines=self._which_lines,
        )

        return _check_lines._add_lines_from_dobj(
            coll=self,
            dobj=dobj,
        )

    # ---------------
    # remove lines
    # ---------------

    def remove_spectral_lines(self, keys=None, propagate=None):
        return _check_lines.remove_lines(
            coll=self,
            keys=keys,
            propagate=propagate,
        )

    # -----------------
    # conversion wavelength - energy - frequency
    # ------------------

    def convert_lines(self, units=None, key=None, ind=None, returnas=None):
        """ Convert wavelength (m) to other units or other quantities

        Avalaible units:
            wavelength: km, m, mm, um, nm, pm, A
            energy:     J, eV, keV, MeV, GeV
            frequency:  Hz, kHz, MHz, GHz, THz

        Return the result as a np.ndarray (returnas = 'data')

        Can also just return the conversion coef if returnas='coef'
        In that case, a bool is also returned indicating whether the result is
        the proportional to the inverse of lambda0::
            - False: data = coef * lambda0
            - True: data = coef / lambda0
        """
        if units is None:
            units = self._units_lambda0
        if returnas is None:
            returnas = dict

        lok = [dict, np.ndarray, 'data', 'coef']
        if returnas not in lok:
            msg = (
                "Arg returnas must be in:\n"
                + "\t- {}\n".format(lok)
                + "\t- provided: {}".format(returnas)
            )
            raise Exception(msg)
        if returnas in [dict, np.ndarray, 'data']:
            returnas2 = 'data'
        else:
            returnas2 = 'coef'

        # get keys of desired lines
        key = self._ind_tofrom_key(
            which=self._which_lines, key=key, ind=ind, returnas=str,
        )

        # get wavelength in m
        lamb_in = self.get_param(
            which=self._which_lines, param='lambda0',
            key=key, returnas=np.ndarray,
        )['lambda0']

        # conversion
        out = _compute.convert_spectral(
            data_in=lamb_in,
            units_in='m',
            units_out=units,
            returnas=returnas2,
        )
        if returnas is dict:
            out = {k0: out[ii] for ii, k0 in enumerate(key)}
        return out

    # -----------------
    # PEC interpolation
    # ------------------

# =============================================================================
#     def calc_pec(
#         self,
#         key=None,
#         ind=None,
#         ne=None,
#         Te=None,
#         deg=None,
#         grid=None,
#         return_params=None,
#     ):
#         """ Compute the pec (<sigma v>) by interpolation for chosen lines
#
#         Assumes Maxwellian electron distribution
#
#         Provide ne and Te and 1d np.ndarrays
#
#         if grid=False:
#             - ne is a (n,) 1d array
#             - Te is a (n,) 1d array
#           => the result is a dict of (n,) 1d array
#
#         if grid=True:
#             - ne is a (n1,) 1d array
#             - Te is a (n2,) 1d array
#           => the result is a dict of (n1, n2) 2d array
#         """
#
#         # Check keys
#         key = self._ind_tofrom_key(
#             which=self._which_lines, key=key, ind=ind, returnas=str,
#         )
#         dlines = self._dobj[self._which_lines]
#
#         key, dnTe, return_params = _check._check_compute_pec(
#             # check keys
#             key=key,
#             dlines=dlines,
#             ddata=self._ddata,
#             _quant_ne=self._quant_ne,
#             _quant_Te=self._quant_Te,
#             # check ne, Te
#             ne=ne,
#             Te=Te,
#             return_params=return_params,
#         )
#         keypec = [f'{k0}-pec' for k0 in key]
#
#         # group lines per common ref
#         lref = set([self._ddata[k0]['ref'] for k0 in keypec])
#         dref = {
#             k0: [k1 for k1 in keypec if self._ddata[k1]['ref'] == k0]
#             for k0 in lref
#         }
#
#         # Interpolate
#         for ii, (k0, v0) in enumerate(dref.items()):
#             douti, dparami = self.interpolate(
#                 # interpolation base
#                 keys=v0,
#                 ref_key=k0,
#                 # interpolation pts
#                 x0=dnTe['ne'],
#                 x1=dnTe['Te'],
#                 # parameters
#                 deg=deg,
#                 deriv=0,
#                 grid=grid,
#                 log_log=True,
#                 return_params=True,
#             )
#
#             # update dict
#             if ii == 0:
#                 dout = douti
#                 dparam = dparami
#             else:
#                 dout.update(**douti)
#                 dparam['keys'] += dparami['keys']
#                 dparam['ref_key'] += dparami['ref_key']
#
#         # -------
#         # return
#
#         if return_params is True:
#             dparam['key'] = dparam['keys']
#             del dparam['keys']
#             dparam['ne'] = dparam['x0']
#             dparam['Te'] = dparam['x1']
#             del dparam['x0'], dparam['x1']
#             return dout, dparam
#         else:
#             return dout
#
#     def calc_intensity(
#         self,
#         key=None,
#         ind=None,
#         ne=None,
#         Te=None,
#         concentration=None,
#         deg=None,
#         grid=None,
#     ):
#         """ Compute the lines intensities by pec interpolation for chosen lines
#
#         Assumes Maxwellian electron distribution
#         Assumes concentration = nz / ne
#
#         Provide ne and Te and 1d np.ndarrays
#
#         Provide concentration as:
#             - a np.ndarray (same concentration assumed for all lines)
#             - a dict of {key: np.ndarray}
#
#         if grid=False:
#             - ne is a (n,) 1d array
#             - Te is a (n,) 1d array
#             - concentration is a (dict of) (n,) 1d array(s)
#           => the result is a dict of (n1, n2) 2d array
#
#         if grid=True:
#             - ne is a (n1,) 1d array
#             - Te is a (n2,) 1d array
#             - concentration is a (dict of) (n1, n2) 2d array(s)
#           => the result is a dict of (n1, n2) 2d array
#
#
#         """
#
#         # Check keys
#         key = self._ind_tofrom_key(
#             which=self._which_lines, key=key, ind=ind, returnas=str,
#         )
#
#         # interpolate pec
#         dout, dparam = self.calc_pec(
#             key=key,
#             ind=ind,
#             ne=ne,
#             Te=Te,
#             grid=grid,
#             deg=deg,
#             return_params=True,
#         )
#
#         # check concentrations
#         concentration = _check._check_compute_intensity(
#             key=[k0[:-4] for k0 in dparam['key']],
#             concentration=concentration,
#             shape=dparam['ne'].shape,
#         )
#
#         # Derive intensity
#         for k0, v0 in dout.items():
#             dout[k0] = v0['data']*dparam['ne']**2*concentration[k0[:-4]]
#
#         return dout
# =============================================================================

    # -----------------
    # plotting
    # ------------------

    def plot_spectral_lines(
        self,
        key=None,
        ind=None,
        ax=None,
        sortby=None,
        param_x=None,
        param_txt=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dsize=None,
        dcolor=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):
        """ plot rest wavelengths as vertical lines """

        # Check inputs
        key = self._ind_tofrom_key(
            which=self._which_lines, key=key, ind=ind, returnas=str,
        )

        return _plot.plot_axvlines(
            din=self._dobj[self._which_lines],
            key=key,
            param_x=param_x,
            param_txt=param_txt,
            sortby=sortby,
            dsize=dsize,
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor, fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )

# =============================================================================
#     def plot_pec_single(
#         self,
#         key=None,
#         ind=None,
#         ne=None,
#         Te=None,
#         concentration=None,
#         deg=None,
#         grid=None,
#         ax=None,
#         sortby=None,
#         param_x=None,
#         param_txt=None,
#         ymin=None,
#         ymax=None,
#         ls=None,
#         lw=None,
#         fontsize=None,
#         side=None,
#         dcolor=None,
#         fraction=None,
#         figsize=None,
#         dmargin=None,
#         wintit=None,
#         tit=None,
#     ):
#         """ Same as plot_spectral_lines() with extra scatter plot with circles
#
#         The circles' diameters depend on the pec value for each line
#
#         Requires:
#             - Te = scalar (eV)
#             - ne = scalar (/m3)
#
#         """
#
#         # ------------
#         # Check ne, Te
#
#         ltypes = [int, float, np.integer, np.floating]
#         dnTe = {'ne': ne, 'Te': Te}
#         single = all([
#             type(v0) in ltypes or len(v0) == 1 for v0 in dnTe.values()
#         ])
#         if not single:
#             msg = ("Arg ne and Te must be floats!")
#             raise Exception(msg)
#
#         # --------
#         # Get dpec
#         dpec = self.calc_pec(
#             key=key,
#             ind=ind,
#             ne=ne,
#             Te=Te,
#             deg=deg,
#             grid=False,
#             return_params=False,
#         )
#         key = [k0[:-4] for k0 in dpec.keys()]
#
#         ne = float(ne)
#         Te = float(Te)
#         tit = (
#             r'$n_e$' + f'= {ne} ' + r'$/m^3$'
#             + r' -  $T_e$ = ' + f'{Te/1000.} keV'
#         )
#
#         pmax = np.max([np.log10(v0['data']) for v0 in dpec.values()])
#         pmin = np.min([np.log10(v0['data']) for v0 in dpec.values()])
#         dsize = {
#             k0[:-4]: (np.log10(v0['data']) - pmin) / (pmax - pmin)*19 + 1
#             for k0, v0 in dpec.items()
#         }
#
#         return _plot.plot_axvlines(
#             din=self._dobj[self._which_lines],
#             key=key,
#             param_x=param_x,
#             param_txt=param_txt,
#             sortby=sortby,
#             dsize=dsize,
#             ax=ax, ymin=ymin, ymax=ymax,
#             ls=ls, lw=lw, fontsize=fontsize,
#             side=side, dcolor=dcolor, fraction=fraction,
#             figsize=figsize, dmargin=dmargin,
#             wintit=wintit, tit=tit,
#         )
#
#     def plot_pec(
#         self,
#         key=None,
#         ind=None,
#         ne=None,
#         Te=None,
#         norder=None,
#         ne_scale=None,
#         Te_scale=None,
#         param_txt=None,
#         param_color=None,
#         deg=None,
#         dax=None,
#         proj=None,
#         ymin=None,
#         ymax=None,
#         ls=None,
#         lw=None,
#         fontsize=None,
#         side=None,
#         dcolor=None,
#         fraction=None,
#         figsize=None,
#         dmargin=None,
#         dtit=None,
#         tit=None,
#         wintit=None,
#     ):
#
#         # -----------
#         # Check input
#
#         # Check ne, Te
#         lc = [np.isscalar(ne) or len(ne) == 1, np.isscalar(Te) or len(Te) == 1]
#         if all(lc):
#             msg = "For a single (ne, Te) space, use plot_pec_singe()"
#             raise Exception(msg)
#
#         # Get dpec
#         dpec = self.calc_pec(
#             key=key,
#             ind=ind,
#             ne=ne,
#             Te=Te,
#             deg=deg,
#             grid=False,
#         )
#         damp = {k0: {'data': v0} for k0, v0 in dpec.items()}
#
#         # Create grid
#         ne_grid = ds._class1_compute._get_grid1d(
#             ne, scale=ne_scale, npts=ne.size*2, nptsmin=3,
#         )
#         Te_grid = ds._class1_compute._get_grid1d(
#             Te, scale=Te_scale, npts=Te.size*2, nptsmin=3,
#         )
#
#         # get dpec for grid
#         dpec_grid = self.calc_pec(
#             key=key,
#             ind=ind,
#             ne=ne_grid,
#             Te=Te_grid,
#             deg=deg,
#             grid=True,
#         )
#
#         raise NotImplementedError()
#
#         return _plot.plot_dominance_map(
#             din=self._dobj['lines'], im=im, extent=extent,
#             xval=ne, yval=Te, damp=damp,
#             x_scale=ne_scale, y_scale=Te_scale, amp_scale='log',
#             param_txt='symbol',
#             dcolor=dcolor,
#             dax=dax, proj=proj,
#             figsize=figsize, dmargin=dmargin,
#             wintit=wintit, tit=tit, dtit=dtit,
#         )
# =============================================================================

    # -------------------
    # units conversion
    # -------------------

# =============================================================================
#     def convert_spectral_lines_units(
#         self,
#         data=None,
#         units=None,
#         units_in=None,
#     ):
#
#         return _compute.convert_spectral_units(
#             coll=self,
#             data=data,
#             units=units,
#             units_in=units_in,
#         )
# =============================================================================

    # -------------------
    # saving
    # -------------------

    def save_spectral_lines_to_file(
        self,
        path=None,
        name=None,
        keys=None,
        overwrite=None,
    ):
        """ Save a set of spectral lines as a json file (or npz) """

        return _check_lines._save_to_file(
            coll=self,
            keys=keys,
            name=name,
            path=path,
            overwrite=overwrite,
        )