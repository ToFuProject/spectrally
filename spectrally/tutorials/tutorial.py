'''

Tutorial script for spectrally

Shows both a SXR (gaussian spectrum) and HXR (pulse shape) example

cjperks
July 16, 2024

'''

# built-in
import os


# common
import numpy as np
import scipy.constants as scpct


__all__ = ['main']


# ######################################################
# ######################################################
#               DEFAULTS
# ######################################################


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUT = os.path.join(os.path.dirname(_PATH_HERE), 'tests', 'input')


# ######################################################
# ######################################################
#               Main
# ######################################################


def main(
    data='ebit',
    compute=True,
    chain=None,
    solver=None,
    binning=None,
):
    """ Tutorial on how to use spectrally

    Two examples 'sxr' and 'hxr' are available
    Both are fitted using several different models and constraints


    Parameters
    ----------
    data : str, optional
        Flag indicating which example to run. The default is 'sxr'.
        - 'sxr': a double-gaussian spectrum with poly background
        - 'hxr': a series of experimental pulse shapes

    Returns
    -------
    coll : Collection instance
        The Collection instance containing all inputs, fit models and data

    """

    import spectrally as sp

    # ------------------------
    # instanciate a Collection
    # ------------------------

    coll = sp.Collection()

    # make sure data is lower-case
    data = data.lower()

    # -----------------
    # create wavelength vector and time-dependent gaussian spectrum
    # -----------------

    _add_data(coll, data=data)

    # -----------------
    # print available spectral model functions
    # -----------------

    sp.get_available_spectral_model_functions()

    # -----------------
    # display a figure showing details of each spectral model function
    # -----------------

    if data in ['sxr', 'ebit']:
        _ = sp.display_spectral_model_function('gauss')

    elif data == 'hxr':
        _ = sp.display_spectral_model_function('pulse_exp')
        _ = sp.display_spectral_model_function('pulse_gauss')
        _ = sp.display_spectral_model_function('lognorm')

    # ------------------
    # add spectral model
    # ------------------

    _add_spectral_model(coll=coll, data=data)

    # ------------------
    # add spectral fit (check input data validity, but not compute yet)
    # ------------------

    _add_spectral_fit(coll=coll, data=data)

    # ---------------------
    # plot validity
    # ---------------------

    if data == 'sxr':
        dax = coll.plot_spectral_fit_input_validity('sf1')
    elif data == 'ebit':
        pass
        # dax = coll.plot_spectral_fit_input_validity('sf0')
    else:
        dax = coll.plot_spectral_fit_input_validity('sf_exp')

    # ---------------------
    # compute spectral fit
    # ---------------------

    if compute is True:

        _compute_spectral_fit(
            coll=coll,
            data=data,
            chain=chain,
            solver=solver,
            binning=binning,
        )

        # ---------------------
        # plot fit
        # ---------------------

        if data == 'sxr':
            dax = coll.plot_spectral_fit('sf1')
        elif data == 'ebit':
            dax = coll.plot_spectral_fit('sf0')
        else:
            dax = coll.plot_spectral_fit('sf_lognorm')

        # ---------------------
        # extract moments
        # ---------------------

        if data == 'sxr':
            dout = coll.get_spectral_model_moments('sf1')
        elif data == 'ebit':
            dout = coll.get_spectral_model_moments('sf0')
        else:
            dout = coll.get_spectral_model_moments('sf_exp')

    return coll


# ######################################################
# ######################################################
#               add data
# ######################################################


def _add_data(coll=None, data=None):

    # ------------
    # SXR
    # ------------

    if data == 'sxr':

        # ------------
        # lamb

        nlamb = 300
        lamb = np.linspace(3.94e-10, 4.e-10, nlamb)

        # ------------
        # time vector

        nt = 50
        t = np.linspace(0, 10, nt)

        # -----------
        # spectra

        lamb0 = 3.96e-10 + 0.001e-10 * np.cos(t[:, None])
        lamb1 = 3.98e-10 + 0.001e-10 * np.sin(t[:, None])


        # poly
        poly = (1 + np.cos(t)[:, None]) * 15 * np.ones((1, nlamb))

        gauss0 = (
            (1 + np.cos(t)[:, None]) * 50
            * np.exp(-(lamb[None, :]-lamb0)**2/(2*0.001e-10**2))
        )

        gauss1 = (
            (1 + np.sin(t)[:, None]) * 80
            * np.exp(-(lamb[None, :]-lamb1)**2/(2*0.0005e-10**2))
        )


        spectra = (
            poly
            + gauss0
            + gauss1
            + np.random.random((nt, nlamb)) * 5
        )

        # --------------------
        # populate collection

        # refs
        coll.add_ref('nlamb', size=nlamb)
        coll.add_ref('nt', size=nt)

        # data - lamb
        coll.add_data(
            'lamb',
            data=lamb,
            ref='nlamb',
            units='m',
        )


        # data - t
        coll.add_data(
            't',
            data=t,
            ref='nt',
            units='s',
        )


        # data - spectrum
        coll.add_data(
            'spectra',
            data=spectra,
            ref=('nt', 'nlamb'),
            units='counts',
        )

    # ------------
    # HXR
    # ------------

    elif data == 'hxr':

        # --------------
        # load file

        # pfe = path/file.ext
        pfe = os.path.join(_PATH_INPUT, 'HXR_pulses.npz')

        # load
        dout = dict(np.load(pfe, allow_pickle=True))

        # -------------
        # ref

        coll.add_ref('npulse', size=dout['pulse'].size)
        coll.add_ref('nsamp', size=dout['data'].shape[0])

        # -------------
        # data

        # pulses
        coll.add_data(
            key='pulses',
            data=dout['pulse'],
            ref='npulse',
            units='index',
        )

        # indices
        coll.add_data(
            key='sample',
            data=np.arange(0, dout['data'].shape[0]),
            ref='nsamp',
            units='index',
        )

        # data
        coll.add_data(
            key='current',
            data=dout['data'],
            ref=('nsamp', 'npulse'),
            units='A',
        )

    # ------------
    # EBIT
    # ------------

    elif data == 'ebit':

        # --------------
        # load file

        # pfe = path/file.ext
        pfe = os.path.join(_PATH_INPUT, 'EBIT_spectrally_test.npz')

        # load
        dout = dict(np.load(pfe, allow_pickle=True))['arr_0'].tolist()

        # -------------
        # ref

        coll.add_ref('nv', size=dout['Vs'].size)
        coll.add_ref('nlamb', size=dout['lambs'].size)

        # -------------
        # data

        # voltages
        coll.add_data(
            key='voltages',
            data=dout['Vs'],
            ref='nv',
            units='kV',
        )

        # wavelength
        coll.add_data(
            key='lamb',
            data=dout['lambs'],
            ref='nlamb',
            units='m',
        )

        # data
        coll.add_data(
            key='spectra',
            data=dout['data'],
            ref=('nv', 'nlamb'),
            units='counts',
        )

    # ------------
    # Error
    # ------------

    else:

        msg = "Unknown data for tutorial"
        raise Exception(msg)

    return


# ######################################################
# ######################################################
#               spectral model
# ######################################################


def _add_spectral_model(coll=None, data=None):

    # ------------
    # SXR
    # ------------

    if data == 'sxr':

        # argon mass
        mz = scpct.m_u * 39.9

        # first mode : bck + 2 gauss
        coll.add_spectral_model(
            key='sm0',
            dmodel={
                'bck': 'poly',
                'l0': {'type': 'gauss', 'lamb0': 3.96e-10, 'mz': mz},
                'l1': {'type': 'gauss', 'lamb0': 3.98e-10, 'mz': mz},
            },
        )

        # first mode : bck with 0 slope + 2 gauss with same width
        coll.add_spectral_model(
            key='sm1',
            dmodel={
                'bck': 'poly',
                'l0': {'type': 'gauss', 'lamb0': 3.96e-10, 'mz': mz},
                'l1': {'type': 'gauss', 'lamb0': 3.98e-10, 'mz': mz},
            },
            dconstraints={
                # 'g0': {
                #     'ref': 'l0_sigma',
                #     'l1_sigma': [0, 1, 0],
                #     },
                'g2': {
                    'ref': 'bck_a0',
                    'bck_a1': [0, 0, 0],
                    'bck_a2': [0, 0, 0],
                },
            },
        )

    # ------------
    # HXR
    # ------------

    elif data == 'hxr':

        # constraints for all models: the poly back has a slope of 0
        dconstraints = {
            'g0': {
                'ref': 'bck_a0',
                'bck_a1': [0, 0, 0],
                'bck_a2': [0, 0, 0],
            },
        }

        # first model : bck with only offset + pulse_exp
        coll.add_spectral_model(
            key='sm_exp',
            dmodel={
                'bck': 'poly',
                'p0': {'type': 'pulse_exp'},
            },
            dconstraints=dconstraints,
        )

        # first model : bck with only offset + pulse_gauss
        coll.add_spectral_model(
            key='sm_gauss',
            dmodel={
                'bck': 'poly',
                'p0': {'type': 'pulse_gauss'},
            },
            dconstraints=dconstraints,
        )

        # first model : bck with only offset + lognorm
        coll.add_spectral_model(
            key='sm_lognorm',
            dmodel={
                'bck': 'poly',
                'p0': {'type': 'lognorm'},
            },
            dconstraints=dconstraints,
        )

    elif data == 'ebit':

        # argon mass
        mz = scpct.m_u * 39.9

        # first mode : bck with 0 slope + 2 gauss with same width
        coll.add_spectral_model(
            key='sm0',
            dmodel={
                'bck': 'poly',
                'l0': {'type': 'gauss', 'lamb0': 2.719e-10, 'mz': mz},
                'l1': {'type': 'gauss', 'lamb0': 2.726e-10, 'mz': mz},
                'l2': {'type': 'gauss', 'lamb0': 2.735e-10, 'mz': mz},
            },
            dconstraints={
                'g0': {
                    'ref': 'bck_a0',
                    'bck_a1': [0, 0, 0],
                    'bck_a2': [0, 0, 0],
                },
                'g1': {
                    'ref': 'l0_sigma',
                    'l1_sigma': [0, 1, 0],
                    'l2_sigma': [0, 1, 0],
                },
            },
        )

    return


# ######################################################
# ######################################################
#               spectral fits
# ######################################################


def _add_spectral_fit(coll=None, data=None):

    # ------------
    # SXR
    # ------------

    if data == 'sxr':

        # first spectral model
        coll.add_spectral_fit(
            key='sf0',
            key_model='sm0',
            key_data='spectra',
            key_lamb='lamb',
            dvalid={
                'nsigma': 3,
                'fraction': 0.31,
            },
        )

        # second spectral model
        coll.add_spectral_fit(
            key='sf1',
            key_model='sm1',
            key_data='spectra',
            key_lamb='lamb',
            dvalid={
                'nsigma': 3,
                'fraction': 0.31,
            },
        )

        # second spectral model again (for later use)
        coll.add_spectral_fit(
            key='sf2',
            key_model='sm1',
            key_data='spectra',
            key_lamb='lamb',
            dvalid={
                'nsigma': 3,
                'fraction': 0.31,
            },
        )

    # ------------
    # HXR
    # ------------

    elif data == 'hxr':

        for k0 in coll.dobj['spect_model'].keys():
            coll.add_spectral_fit(
                key=k0.replace('sm_', 'sf_'),
                key_model=k0,
                key_data='current',
                key_sigma=None,
                key_lamb='sample',
                # params
                dvalid={
                    'nsigma': 2,
                    'fraction': 0.11,
                },
            )

    # ------------
    # EBIT
    # ------------

    elif data == 'ebit':

        for k0 in coll.dobj['spect_model'].keys():
            coll.add_spectral_fit(
                key='sf0',
                key_model=k0,
                key_data='spectra',
                key_sigma=None,
                key_lamb='lamb',
                # params
                dvalid={
                    'nsigma': 2,
                    'fraction': 0.11,
                },
            )

    return


# ######################################################
# ######################################################
#               compute
# ######################################################


def _compute_spectral_fit(
    coll=None,
    data=None,
    chain=None,
    solver=None,
    binning=None,
):

    # ------------
    # check inputs
    # ------------

    if chain is None:
        if data == 'sxr':
            chain = True
        else:
            chain = False

    # ------------
    # SXR
    # ------------

    if data == 'sxr':

        for k0 in ['sf0', 'sf1']:
            coll.compute_spectral_fit(
                k0,
                solver=solver,
                dsolver_options={'nfev': 1000},
                chain=chain,
                verb=2,
                strict=True,
            )
        # ---------------------
        # compute spectral fit with user-provided scales, bounds and x0
        # ---------------------

        # Focus on the second spectral model again
        # this time the bounds and initial of free variablesis set by the user
        # (otherwise they are set automatically

        # we first extract the list of free variables
        xfree = coll.get_spectral_model_variables('sm1', returnas='free')['free']
        print(xfree)
        # ['bck_a0', 'l0_amp', 'l0_vccos', 'l0_sigma', 'l1_amp', 'l1_vccos']

        # Build a dict for lower bounds
        dbounds_low = {
            'bck_a0': 0,
            'l0_amp': 0,
            'l0_vccos': -0.03 / 3.96,
            'l0_sigma': 0.0001e-10,
            'l1_amp': 0,
            'l1_vccos': -0.03 / 3.98,
        }

        # Build a dict for upper bounds
        dbounds_up = {
            'bck_a0': 50,  # 25
            'l0_amp': 200,
            'l0_vccos': 0.03 / 3.96,
            'l0_sigma': 0.005e-10,
            'l1_amp': 200,
            'l1_vccos': 0.03 / 3.98,
        }

        # Build a dict for inital guesses
        dx0 = {
            'bck_a0': 20,
            'l0_amp': 100,
            'l0_vccos': 0. / 3.96, # 0.02 / 3.96
            'l0_sigma': 0.001e-10,
            'l1_amp': 100,
            'l1_vccos': 0. / 3.98, # -0.02 / 3.98
        }

        # second spectral model with user-provided scales, bounds and x0
        coll.compute_spectral_fit(
            'sf2',
            # user-provided
            dbounds_low=dbounds_low,
            dbounds_up=dbounds_up,
            dx0=dx0,
            # options
            solver=solver,
            chain=chain,
            verb=2,
            strict=True,
        )

    # ------------
    # HXR
    # ------------

    else:

        for k0 in coll.dobj['spect_fit'].keys():
            coll.compute_spectral_fit(
                k0,
                solver=solver,
                dsolver_options={'nfev': 1000},
                chain=chain,
                binning=binning,
                verb=2,
                strict=True,
            )

    return


# ######################################################
# ######################################################
#               __main__
# ######################################################


if __name__ == "__main__":
    coll = main()