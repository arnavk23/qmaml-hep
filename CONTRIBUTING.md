# Contributing

Contributions are welcome: new Hamiltonian families, ansätze or initializers, more seeds or larger systems, and
fixes. Results in this repository are only trusted if they are produced by code and checked, so a few rules
apply.

## Setup

```bash
pip install -e ".[dev,plots,docs]"
pre-commit install
make test
```

## Adding a system

1. Add `src/qmaml_hep/systems/<name>.py` with the Hamiltonian, `exact_ground_energy`, `sample_task_space`, and a
   `verify_construction()` that checks something non-trivial: a known limit, a symmetry, or an independent
   construction.
2. Register a track in `src/qmaml_hep/tracks.py`. If the circuit is new, add an exact simulator to
   `simulators.py` and a branch in `make_fast`. The benchmark cross-checks it against the PennyLane circuit on
   every run.
3. Add the construction check to `tests/test_systems.py`, and any conserved quantity to `tests/test_invariants.py`.

## Adding an initializer

Initializers must go through `benchmark.adapt` unchanged, so that every method gets the same budget, optimizer and
learning rate. Charge any one-off cost (pretraining, data collection) in the result file so the compute accounting
stays honest.

## Results

- Never edit numbers by hand. Figures and tables come from `scripts/`, and the summary comes from `qmaml-score`.
- The scoring rule in `analysis.py` is frozen for the published results. Changing it needs its own version and
  changelog entry, with the old summary kept for comparison.
- Add an entry to `CHANGELOG.md` under an `Unreleased` heading.

## Releases

Bump `version` in `pyproject.toml`, `src/qmaml_hep/__init__.py` and `CITATION.cff`. Move the changelog entries under
the new version, then tag `vX.Y.Z` and push the tag. The release workflow builds the package and attaches a
tarball of `results/`.
