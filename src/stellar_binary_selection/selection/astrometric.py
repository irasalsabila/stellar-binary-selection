"""Gaia astrometric selection function (TODO M21).

Implements a RUWE-based unresolved-binary detectability model following
Castro-Ginard et al. (2024): given binary orbital parameters and
distance, forward-model the expected excess astrometric noise and
compare it against the sky-dependent RUWE threshold.

This is a physically-motivated approximation of the GaiaUnlimited
`BinarySystemsSelectionFunction`, implemented locally so the pipeline
does not depend on external package internals. The behaviour to
reproduce is:

  * sensitivity peaks around P ~ 100–1000 d for typical masses/distances;
  * falls off at short P (photocentre wobble too small to resolve) and
    long P (motion absorbed into proper motion over the DR3 baseline);
  * grows with mass ratio q and primary mass (larger photocentre orbit);
  * degrades with distance (angular size shrinks as 1/d).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "photocentre_semimajor_axis_mas",
    "ruwe_excess_model",
    "astro_detection_probability",
]


# Gaia DR3 baseline in years (~34 months of nominal mission data).
DR3_BASELINE_YR = 2.8


def photocentre_semimajor_axis_mas(
    m1_msun: np.ndarray,
    q: np.ndarray,
    period_days: np.ndarray,
    distance_pc: np.ndarray,
) -> np.ndarray:
    """Angular semi-major axis of the photocentre orbit, in milliarcsec.

    a_phot [AU] = (M_tot)^(1/3) * P[yr]^(2/3) * (q/(1+q) - B)
    where B is the light ratio contribution. We approximate the
    photocentre offset using the luminosity-weighted position:

        a_phot = a_orb * (q/(1+q) - l),  l = L2/(L1+L2)

    with a rough main-sequence luminosity relation L ∝ M^3.5, so
    l = q^3.5 / (1 + q^3.5).
    """
    m1 = np.asarray(m1_msun, dtype=float)
    q = np.asarray(q, dtype=float)
    p_yr = np.asarray(period_days, dtype=float) / 365.25
    d_pc = np.asarray(distance_pc, dtype=float)

    m_tot = m1 * (1.0 + q)
    # Kepler's third law: a_orb[AU] = (M_tot)^(1/3) P[yr]^(2/3)
    a_orb_au = np.cbrt(m_tot) * np.power(p_yr, 2.0 / 3.0)

    # luminosity ratio from a rough MS mass-luminosity relation
    lum_ratio = np.power(q, 3.5) / (1.0 + np.power(q, 3.5))
    # fraction of the orbital separation traced by the photocentre
    f_phot = q / (1.0 + q) - lum_ratio

    a_phot_au = np.abs(a_orb_au * f_phot)
    # angular size: 1 AU at 1 pc = 1000 mas
    return a_phot_au * 1000.0 / d_pc


def ruwe_excess_model(
    a_phot_mas: np.ndarray,
    period_days: np.ndarray,
    *,
    parallax_mas: np.ndarray | None = None,
    n_obs: int = 200,
) -> np.ndarray:
    """Model the RUWE excess produced by an unresolved companion.

    The astrometric fit is degraded when the photocentre wobble is not
    well-averaged over the observing baseline. Two regimes:

      * P << baseline: many orbits observed → wobble averages out in the
        along-scan residuals only partially; RUWE grows with a_phot.
      * P >> baseline: the residual motion is nearly linear over the
        baseline and is largely absorbed into proper motion, so the
        excess is suppressed (scales as baseline/P).

    Returns an approximate RUWE value (1.0 = perfect single-star fit).
    """
    a = np.asarray(a_phot_mas, dtype=float)
    p_yr = np.asarray(period_days, dtype=float) / 365.25
    baseline = DR3_BASELINE_YR

    # number of orbits sampled within the DR3 baseline
    n_orbits = baseline / np.maximum(p_yr, 1e-6)

    # Suppression factor: for P >> baseline the signal is absorbed into
    # the proper-motion solution; for P <~ baseline it is resolved.
    # 1 at n_orbits >> 1, falls as ~1/n_orbits for n_orbits << 1.
    suppression = n_orbits / np.sqrt(1.0 + n_orbits ** 2)

    # along-scan residual in mas relative to the single-star astrometric
    # precision. RUWE is a NORMALISED statistic: a coherent residual of
    # size r in every observation gives chi2 excess = n_obs*(r/sigma)^2
    # and RUWE = sqrt(chi2/nu) ~ sqrt(1 + (r/sigma)^2). There is no
    # 1/n_obs dilution because the residual is systematic, not random.
    resid_mas = a * suppression
    # Effective astrometric noise floor. Calibrated so the model
    # reproduces the Castro-Ginard et al. (2024) behaviour: high
    # completeness for q >~ 0.5 at d <~ 100 pc in the P ~ 100-1000 d
    # window, falling off at lower q, larger d and outside that window.
    # This is larger than the formal per-epoch along-scan precision
    # because the RUWE statistic integrates systematic fit degradation
    # across the whole solution, not a single observation.
    sigma_eff = 2.0  # mas
    excess = (resid_mas / sigma_eff) ** 2
    return np.sqrt(1.0 + excess)


def sky_ruwe_threshold(n: int | np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Approximate the sky-varying single-star RUWE threshold.

    Castro-Ginard et al. find the threshold is ~1.0–1.4 depending on
    sky position, colour and magnitude. We draw from an empirically
    motivated distribution centred near 1.25.
    """
    if isinstance(n, (int, np.integer)):
        size = int(n)
    else:
        size = len(n)
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.normal(1.25, 0.08, size=size).clip(1.0, 1.6)


def astro_detection_probability(
    m1_msun: np.ndarray,
    q: np.ndarray,
    period_days: np.ndarray,
    distance_pc: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """P(astrometric detection) via the modelled RUWE threshold.

    A system is 'detected' when its modelled RUWE exceeds the local
    single-star threshold. We return a smoothed probability so the
    resulting selection function is continuous.
    """
    a_phot = photocentre_semimajor_axis_mas(m1_msun, q, period_days, distance_pc)
    ruwe = ruwe_excess_model(a_phot, period_days)
    thr = sky_ruwe_threshold(ruwe, rng=np.random.default_rng(seed))

    # logistic smoothing around the threshold (width ~ 0.05 in RUWE)
    width = 0.05
    prob = 1.0 / (1.0 + np.exp(-(ruwe - thr) / width))
    return np.clip(prob, 0.0, 1.0)