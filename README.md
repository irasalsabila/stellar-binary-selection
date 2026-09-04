# Stellar Binary Selection

Physics-informed tools for studying unresolved stellar binaries using Gaia DR3 and 2MASS photometry, parallax, and a Gaia-inspired astrometric sensitivity model.

## Overview

The project provides a reproducible Python workflow to:

- Build a quality-controlled Gaia DR3 and 2MASS parent catalogue.
- Generate synthetic single-star and unresolved-binary observations.
- Combine component fluxes exactly in Gaia and 2MASS bands.
- Inject magnitude-dependent photometric and parallax noise.
- Train classical machine-learning and neural binary-selection models.
- Measure photometric completeness as a function of mass ratio, primary mass, and distance.
- Compare photometric and astrometric detection regimes across mass ratio and orbital period.

The main scientific objective is to characterize where broadband photometry and astrometric information detect unresolved binaries, where the channels complement one another, and where both remain insensitive.

## Repository Contents

```text
src/       Python package containing data, physics, simulation, model, and selection modules
config/    Data, synthetic-population, model, and experiment configuration
queries/   Gaia ADQL queries
scripts/   Data-ingestion, simulation, training, evaluation, and export commands
```

Large input catalogues, generated model outputs, figures, checkpoints, and run logs are kept outside the source distribution.

## Data Archive

A compact reproducibility archive is provided separately:

```text
stellar-binary-selection-data.zip
```

The archive contains:

- The 100,000-system synthetic population.
- The 47,731-source FGK parent sample.
- The apparently-single Sample B catalogue.
- The Gaia NSS Sample C catalogue.
- The FGK-restricted Sample B and Sample C validation populations, including scored versions.
- The fitted Gaia and 2MASS noise model.

The archive is approximately 58 MB compressed. The full Gaia parent catalogue, raw archive downloads, isochrone files, generated figures, and model checkpoints are not included.

## Installation

The environment specification is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate stellar-binary-selection
pip install -e .
```

## Main Workflows

### Build the observational catalogue

```bash
PYTHONPATH=src python scripts/download_gaia.py
PYTHONPATH=src python scripts/download_nss.py
PYTHONPATH=src python scripts/build_dataset.py --stage derived
PYTHONPATH=src python scripts/build_dataset.py --stage fgk_cut
```

The Gaia and 2MASS catalogue is restricted to high-quality nearby sources. Gaia NSS products are treated as a selected validation population rather than as a complete binary census.

### Generate synthetic observations

```bash
PYTHONPATH=src python scripts/generate_synthetic.py
```

The default configuration generates 100,000 systems, including 50,000 binaries and 50,000 nominal single stars. Configuration is controlled by `config/synthetic.yaml`.

### Train models

```bash
PYTHONPATH=src python scripts/train.py --model baselines
PYTHONPATH=src python scripts/train.py --model plain_mlp
PYTHONPATH=src python scripts/train.py --model pinn
PYTHONPATH=src python scripts/ablation.py --epochs 60 --seeds 42 43 44
```

### Evaluate selection and complementarity

```bash
PYTHONPATH=src python scripts/complementarity.py
PYTHONPATH=src python scripts/validate_nss.py
```

## Data and Model Conventions

- Photometric model inputs exclude binary and NSS diagnostic columns.
- Gaia NSS sources are external validation examples, not confirmed representatives of all binaries.
- Unresolved binary fluxes are added in linear flux before converting back to magnitudes.
- Photometric thresholds are calibrated from synthetic single-star scores.
- Photometric completeness is reported in explicit mass-ratio, primary-mass, distance, and period bins where applicable.
- The astrometric component is a local Gaia-inspired approximation, not a direct implementation of the Gaia scanning law or a published Gaia selection-function package.

## License

See `LICENSE` for the terms governing use and distribution.
