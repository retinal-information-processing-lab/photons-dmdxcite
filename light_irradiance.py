"""
From LED source activation (%) to photon flux conversion utilities.
Context: irradiance_computation.pdf
Author: Chiara Boscarino, 2026

"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# LED source activation (Xcite 0-100%, x) to power look-up table
#   Key   : LED source activation (%)
#   Value : Power(x [%]) [mW cm⁻²]

# Power physically measured for different Xcite levels of the
# WHITE LED SOURCE (MEA1) without ND filter (Awen, 23.06.2025)
SOURCE_PTG_POWER_mW_cm2 = {
    1:   0.345,
    10:  3.2,
    20:  6.9,
    30:  10.5,
    40:  14.0,
    50:  17.4,
    60:  20.9,
    70:  24.7,
    80:  28.3,
    90:  32.3,
    100: 37.7,
}
P100_mW_cm2 = SOURCE_PTG_POWER_mW_cm2[100]

def _interpolate_power_lookup(
        levels: np.ndarray,
        P_lut: dict
) -> dict:
    """
    Interpolate power lookup table for given levels.
    Args:
        levels: desired activation levels (array-like)
        P_lut: lookup table for available levels

    Returns:
        lookup table for given levels {x: P(x)}
    """
    assert isinstance(levels, np.ndarray), "Levels should be a numpy array for interpolation."
    assert isinstance(P_lut, dict), "P(x) look up table should be a dictionary."

    # Sorted calibration points for interpolation
    cal_levels = sorted(P_lut.keys())
    cal_powers = [P_lut[k] for k in cal_levels]

    # Interpolate lookup table in desired values
    interpolated_power = np.interp(levels, cal_levels, cal_powers, left=np.nan, right=np.nan)
    assert not np.any(np.isnan(interpolated_power)), "DMD levels out of bounds for interpolation."
    lookup_table = dict(zip(levels, interpolated_power))
    return lookup_table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WHITE LIGHT - LED source spectrum
# Normalized spectrum of the white LED source used in the experiments -> I(λ [nm]) [a.u.]
WHITE_SPECTRUM_FP = f"/home/tommaso/Documents/GitHub/hiOsight/white_mea1.pkl"

# So given:
#   P(x) [mW cm⁻²] — measured power at each activation level x
#   I(λ [nm]) [a.u.] — normalized spectrum (peak = 1), assumed shape doesn't change with x
# We can compute:
#   I_tot = ∫ I(λ) dλ - total normalized irradiance (area under the spectrum curve)
#   S₁₀₀ [mW cm⁻² nm⁻¹] = P₁₀₀ / I_tot
#   S(x) [mW cm⁻² nm⁻¹] - spectral irradiance at each activation level x as
#                S(x) = (S₁₀₀ / P₁₀₀) · P(x)
#                S(x) = P(x) / I_tot
# So, then we can check that with the obtained S(x) for each x we retrieve the measured using
#                P(x) = S(x) · I_tot
# And finally get:
#   L(λ, x) [mW cm⁻² nm⁻¹] - the full spectral irradiance at activation x
#                L(λ, x) = S(x) · I(λ) = I(λ) · P(x) / I_tot

def _get_spectrum(fp):
    data = np.load(fp, allow_pickle=True)
    wavelengths, spectrum = data[0], data[1]
    return wavelengths, spectrum

def _plot_spectrum(fp, figsize=None,
                   fontsize=12, color='tab:blue', lw=2,
                   title="White LED source spectrum (normalized)",
                   xlabel="Wavelength [nm]",
                   test_x_levels=None,
                   P_lut=None):
    if test_x_levels is None:
        if figsize is None: figsize = (8, 4)
        nrows, ncols = 1, 1
        plot_x_levels = False
    else:
        if figsize is None: figsize = (12, 4)
        nrows, ncols = 1, 2
        plot_x_levels = True
        assert isinstance(test_x_levels, (list, tuple, np.ndarray)), "test_x_levels should be a list, tuple or numpy array of activation levels to plot."
        assert P_lut is not None and isinstance(P_lut, dict), "P(x) look up table should be not None and a dictionary to test spectral irradiance at specific activation levels."
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()
    wavelengths, spectrum = _get_spectrum(fp)

    ax = axs[0]
    ax.plot(wavelengths, spectrum, color=color, lw=lw)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel("Normalized spectrum [a.u.]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    if plot_x_levels:
        ax = axs[1]
        # alphas = np.linspace(0.1, 1.0, len(test_x_levels))
        for ix, x in enumerate(test_x_levels):
            L_x = compute_spectral_irradiance(x, P_lut, spectrum, wavelengths)
            # ax.plot(wavelengths, L_x, label=f"{x}%", lw=lw, alpha=alphas[ix], color=color)
            ax.plot(wavelengths, L_x, label=f"{x}%", lw=lw)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel("Spectral irradiance\n[mW cm⁻² nm⁻¹]", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_title("Spectral irradiance at\ndifferent activation levels", fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    return fig

def compute_spectral_irradiance(
        x_value: float,
        P_lut: dict,
        I_norm: np.ndarray,
        lambdas: np.ndarray,
) -> np.ndarray:
    """
    Compute spectral irradiance L(λ, x) [mW cm⁻² nm⁻¹] for desired activation level

    P(x) [mW cm⁻²] = S(x) · ∫ I(λ) dλ  →  S(x) = P(x) / ∫ I(λ) dλ
    L(λ, x) [mW cm⁻² nm⁻¹] = S(x) · I(λ)

    Parameters
    ----------
    x_value : activation level
    P_lut    : {x: P(x) [mW cm⁻²]} measured at available activation levels (don't need to include desired x_values, interpolation will be used)
    I_norm   : I(λ) [a.u.], normalized spectrum (peak = 1), shape (N,)
    lambdas  : λ [nm], shape (N,)

    Returns
    -------
    L(λ, x) [mW cm⁻² nm⁻¹]: np.ndarray of shape (N,)
    """
    x_values, _ = _uniform_input_to_array(x_value)
    P_x = _interpolate_power_lookup(x_values, P_lut)

    # ∫ I(λ) dλ [nm]
    I_integral = np.trapz(I_norm, lambdas)

    # S(x) = P(x) / ∫ I(λ) dλ  [mW cm⁻² nm⁻¹]
    # L(λ, x) = S(x) · I(λ)    [mW cm⁻² nm⁻¹]
    L_x = (P_x[x_value] / I_integral) * I_norm
    assert L_x.shape == I_norm.shape, f"Error in spectral irradiance computation: {L_x.shape} vs {I_norm.shape}"
    return L_x

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# POWER P(x) [mW cm⁻²] and PHOTON FLUX P(x) [photons cm⁻² s⁻¹] at each activation level x

def compute_irradiance(
        x_value: float,
        P_lut: dict,
        I_norm: np.ndarray,
        lambdas: np.ndarray,
        unit: str = 'photon_flux_cm2_s' # power_mW_cm2 or photon_flux_cm2_s
) -> float:
    """ Compute power from spectral irradiance for a given light activation level,
    either in P [mW cm⁻²] or in Φ [photons cm⁻² s⁻¹] depending on the specified unit.

    Args:
        x_value: activation level
        P_lut: {x: P(x) [mW cm⁻²]} measured at available activation levels (don't need to include desired x_value, interpolation will be used)
        I_norm   : I(λ) [a.u.], normalized spectrum (peak = 1), shape (N,)
        lambdas  : λ [nm], shape (N,)
        unit: 'power_mW_cm2' to compute power in mW cm⁻², 'photon_flux_cm2_s' to compute photon flux in photons cm⁻² s⁻¹

    Returns:
        P(x) (float): power or photon flux at the given activation level, depending on the specified unit.

    """
    L_x = compute_spectral_irradiance(x_value, P_lut, I_norm, lambdas)
    if unit == 'photon_flux_cm2_s':
        return _compute_photon_flux_cm2_s(L_x, lambdas)
    elif unit == 'power_mW_cm2':
        return _compute_power_mW_cm2(L_x, lambdas)
    else: raise ValueError(f"Invalid unit: {unit}. Supported units: 'power_mW_cm2', 'photon_flux_cm2_s'.")


def _compute_power_mW_cm2(
        L: np.ndarray,
        lambdas: np.ndarray,
) -> float:
    """
    Compute power from spectral irradiance.

    P [mW cm⁻²] = ∫ L(λ) dλ

    Parameters
    ----------
    L       : L(λ) [mW cm⁻² nm⁻¹], shape (N,)
    lambdas : λ [nm], shape (N,)

    Returns
    -------
    P [mW cm⁻²]
    """
    return np.trapz(L, lambdas)


# Pre-computed photon flux factor
# E_photon = h·c / λ
H_PLANCK = 6.626e-34   # J·s  (h: Planck constant)
C_LIGHT   = 2.998e8    # m/s   (c: Speed of light)
C_LIGHT_NM_S = C_LIGHT * 1e9  # nm/s
# h·c constant in [mW nm s] = [J nm s⁻¹ s]
_HC = H_PLANCK * C_LIGHT_NM_S  # [J·nm]
_MW = 1e-3                   # [J/s per mW]

def _compute_photon_flux_cm2_s(
        L: np.ndarray,
        lambdas: np.ndarray
) -> float:
    """
    Compute photon flux from spectral irradiance.

    Φ [photons cm⁻² s⁻¹] = ∫ L(λ) / E_photon(λ) dλ
                          = ∫ L(λ) · λ / (h·c) dλ

    Parameters
    ----------
    L       : L(λ) [mW cm⁻² nm⁻¹], shape (N,)
    lambdas : λ [nm], shape (N,)

    Returns
    -------
    Φ [photons cm⁻² s⁻¹]
    """
    # L [mW cm⁻² nm⁻¹] → [J s⁻¹ cm⁻² nm⁻¹]
    # E_photon = h·c / λ [J]
    # L / E_photon = L·λ / (h·c) [photons s⁻¹ cm⁻² nm⁻¹]
    photon_flux_density = (L * _MW * lambdas) / _HC  # [photons s⁻¹ cm⁻² nm⁻¹]
    return np.trapz(photon_flux_density, lambdas)    # [photons s⁻¹ cm⁻²]

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# UTILS
def _uniform_input_to_array(input_value):
    is_scalar = np.ndim(input_value) == 0
    return np.atleast_1d(np.asarray(input_value)), is_scalar

def save_figure(fig, filename, figuredir, remove_blank_space=False, verbose=False):
    """Save a figure in the specified directory"""
    filepath = os.path.join(figuredir, filename)
    if remove_blank_space:
        fig.savefig(filepath, bbox_inches='tight')
    else:
        fig.savefig(filepath)
    plt.close(fig)
    if verbose:
        print(f"Saved figure: {filepath}")
    return
# ---------------------------------------------------------------------------


# # ---------------------------------------------------------------------------
# # Quick self-test / demo
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("Xcite % power table (mW/cm²):")
#     for lvl, pwr in SOURCE_PTG_POWER_mW_cm2.items():
#         print(f"  Xcite {lvl:3d}%  →  {pwr:.3f} mW/cm²")
#
#     TEST_X_LEVELS = [1, 2, 7, 8, 10, 30, 44, 60]  # Light % levels to test (including some non-calibrated ones)
#     P_x = _interpolate_power_lookup(np.array(TEST_X_LEVELS), P_lut=SOURCE_PTG_POWER_mW_cm2)
#     wavelengths, spectrum = _get_spectrum(WHITE_SPECTRUM_FP)
#     fig = _plot_spectrum(WHITE_SPECTRUM_FP, test_x_levels=TEST_X_LEVELS, P_lut=SOURCE_PTG_POWER_mW_cm2)
#     # plt.show()
#     save_figure(fig, WHITE_SPECTRUM_FP.replace(".pkl", ".jpg"), '', remove_blank_space=True)
#
#     print()
#     power = [compute_irradiance(x, SOURCE_PTG_POWER_mW_cm2, spectrum, wavelengths, unit='power_mW_cm2') for x in TEST_X_LEVELS]
#     flux = [compute_irradiance(x, SOURCE_PTG_POWER_mW_cm2, spectrum, wavelengths, unit='photon_flux_cm2_s') for x in TEST_X_LEVELS]
#     for x, pw, fx in zip(TEST_X_LEVELS, power, flux):
#         assert np.isclose(pw, P_x[x]), f"Error in power computation for Xcite {x}%: {pw:.3f} vs {P_x[x]:.3f} mW/cm²"
#         print(f"Xcite {x:3d}%  →  Power: {pw:.3f} mW/cm², Photon flux: {fx:.3e} photons cm⁻² s⁻¹")
#
#     fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#     fontsize = 16
#     axs = axs.flatten()
#     ax = axs[0]
#     ax.scatter(TEST_X_LEVELS, [P_x[x] for x in TEST_X_LEVELS], label='Power (Measured)', marker='o', color='k')
#     ax.plot(TEST_X_LEVELS, power, label='Power (Computed)', color='r')
#     ax.set_xlabel("Xcite activation level (%)", fontsize=fontsize)
#     ax.set_ylabel("Power\n[mW cm⁻²]", fontsize=fontsize)
#     ax.set_title("Power at different\nlight levels", fontsize=fontsize)
#     for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
#     ax.tick_params(axis='both', which='major', labelsize=fontsize)
#     ax.legend()
#     ax = axs[1]
#     ax.plot(TEST_X_LEVELS, flux, label='Flux (Computed)', marker='o', color='r')
#     ax.set_xlabel("Xcite activation level (%)", fontsize=fontsize)
#     ax.set_ylabel("Photon flux\n[photons cm⁻² s⁻¹]", fontsize=fontsize)
#     ax.set_title("Photon flux at different\nlight levels", fontsize=fontsize)
#     for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
#     ax.tick_params(axis='both', which='major', labelsize=fontsize)
#     ax.legend()
#     # plt.show()
#     save_figure(fig, WHITE_SPECTRUM_FP.replace(".pkl", "_power_flux_computation.jpg"), '', remove_blank_space=True)