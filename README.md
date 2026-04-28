# Light Irradiance Conversion

Utility functions for converting a white LED source activation level (%) into **spectral irradiance** and **photon flux**, based on empirical power measurements and a measured source spectrum.

---

## Background

When driving a broadband white LED (e.g. Xcite) at a given activation level `x` (0–100 %), the physical output is a broadband spectrum whose *shape* is assumed fixed while its *amplitude* scales with `x`. Given:

- `P(x)` — irradiance measured at each calibration level \[mW cm⁻²\]
- `I(λ)` — normalized LED spectrum (peak = 1) \[a.u.\]

the full **spectral irradiance** is:

```
L(λ, x) = P(x) / ∫I(λ)dλ  ·  I(λ)     [mW cm⁻² nm⁻¹]
```

and the **photon flux** is obtained by integrating over the photon energy:

```
Φ(x) = ∫ L(λ,x) · λ / (h·c) dλ         [photons cm⁻² s⁻¹]
```

See `Rationale.pdf` for the full derivation.

---

## Files

| File | Description                                        |
|------|----------------------------------------------------|
| `light_irradiance.py` | Main module — all functions witha small usage demo |
| `white_mea1.pkl` | Measured normalized spectrum of the white LED source (MEA1) |
| `white_mea1.jpg` | Plot of the normalized spectrum                    |
| `white_mea1_power_flux_computation.jpg` | Validation plot: measured vs computed power and photon flux |
| `Rationale.pdf` | Physical derivation and calibration methodology    |

---

## Usage
Use the small demo at the end of `light_irradiance.py` to see how to compute the irradiance and photon flux for a given activation level `x`, using the built-in calibration table and spectrum. 
Then, comment the demo and import the functions into your own code to use them with custom calibration data.

---

## Public API

### `compute_irradiance(x, P_lut, I_norm, lambdas, unit)`
Returns the integrated irradiance at activation level `x`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | LED activation level (%) |
| `P_lut` | `dict` | Calibration table `{x: P(x) [mW cm⁻²]}` |
| `I_norm` | `np.ndarray` | Normalized spectrum `I(λ)`, shape `(N,)` |
| `lambdas` | `np.ndarray` | Wavelengths \[nm\], shape `(N,)` |
| `unit` | `str` | `'power_mW_cm2'` or `'photon_flux_cm2_s'` |

### `compute_spectral_irradiance(x, P_lut, I_norm, lambdas)`
Returns `L(λ, x)` \[mW cm⁻² nm⁻¹\] as an array of shape `(N,)`.

### `SOURCE_PTG_POWER_mW_cm2`
Built-in calibration lookup table for the **white MEA1** source (measured without ND filter, Awen, June 2025). Covers 1–100 % in steps; intermediate values are linearly interpolated.

---

## Requirements

```
numpy
matplotlib
```

---

## Author

Chiara Boscarino, 2026
