# Changelog

All notable changes to `scitex-gen` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.12]

- **BREAKING**: `to_even` and `to_odd` have been moved to the new
  standalone [`scitex-math`](https://pypi.org/project/scitex-math/)
  package (v0.1.0+). The top-level exports `scitex_gen.to_even` and
  `scitex_gen.to_odd` and the corresponding private submodules
  `scitex_gen._numeric._to_even` / `scitex_gen._numeric._to_odd` have
  been removed. Import from `scitex_math` instead:

  ```python
  # Before
  from scitex_gen import to_even, to_odd

  # After
  from scitex_math import to_even, to_odd
  ```

  Downstream consumers (`scitex-nn`, `scitex-dsp`) are migrating in
  parallel. `scitex-math` is pure-stdlib with zero runtime
  dependencies.

## [0.1.10] — 2026-05-27

- fix(ci): remove broken ecosystem-wide quality audit workflow that referenced
  non-existent files (ecosystem.py lives in scitex-dev, not scitex-gen).
  Per-package audit via tests/develop/test_audit.py remains.
- fix(version): reconcile develop with PyPI — bump 0.1.8 → 0.1.10
  (0.1.9 was released from a branch that never merged back).

## [0.1.7]

- Initial CHANGELOG entry — see git log for prior history.
