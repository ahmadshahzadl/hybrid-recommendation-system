# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and follows the spirit of [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2025-12-04

- Stabilize end-to-end pipeline: preprocess → CB/CF → hybrid → evaluate
- Wire `run.py` as the CLI entry for full runs
- Pin/refresh `requirements.txt` and align docs (Getting Started, API/CLI, Evaluation)
- Refresh notebooks for reproducibility and parity with `src/`
- Add/track artifacts: `models/cb_similarity.csv`, `data/processed_data.csv`
- Tidy structure and naming across `src` modules

### Breaking Changes

- None

### Upgrade Notes

- Install dependencies: `pip install -r requirements.txt`
- See `docs/GETTING_STARTED.md` for usage and examples
