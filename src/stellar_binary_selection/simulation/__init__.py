"""Synthetic binary population + orbital sampling + noise model."""

from .noise import EmpiricalNoiseModel, default_noise_model
from .orbit import sample_angles, sample_eccentricity, sample_inclination, sample_phase
from .population import (
    BANDS,
    generate,
    sample_age_gyr,
    sample_distance_pc,
    sample_log_period_days,
    sample_mass_ratio,
    sample_metallicity,
    sample_primary_mass,
    synthetic_absolute_magnitudes,
)

__all__ = [
    "BANDS",
    "EmpiricalNoiseModel",
    "default_noise_model",
    "generate",
    "sample_age_gyr",
    "sample_distance_pc",
    "sample_eccentricity",
    "sample_angles",
    "sample_inclination",
    "sample_log_period_days",
    "sample_mass_ratio",
    "sample_metallicity",
    "sample_phase",
    "sample_primary_mass",
    "synthetic_absolute_magnitudes",
]