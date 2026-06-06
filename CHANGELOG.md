# Changelog

All notable changes to `scitex-gen` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.13] — 2026-06-07 — RETIRED

The full public surface of `scitex-gen` has been redistributed across
the SciTeX ecosystem. As of this release, `scitex_gen` exports no
public symbols. The package is kept as an installable shell so that
consumers can `pip install scitex-gen` without errors while their
import paths migrate.

### Migration map

| Old import path                         | New canonical home          |
|-----------------------------------------|-----------------------------|
| `scitex_gen.symlog / to_rank / transpose / connect_nums / float_linspace` | `scitex_math` |
| `scitex_gen.to_z / to_01 / to_nan01 / to_nanz / unbias / clip_perc / to_rank` | `scitex_math` |
| `scitex_gen.DimHandler`                 | `scitex_nn`                 |
| `scitex_gen.mat2dict / mat2npa / mat2npy / dir2npy / save_npa / keys2npa / public_keys` | `scitex_io` |
| `scitex_gen.xml2dict / XmlDictConfig / XmlListConfig` | `scitex_io`   |
| `scitex_gen.print_config / print_config_main` | `scitex_io`           |
| `scitex_gen.list_packages / main / src` | `scitex_introspect`         |
| `scitex_gen.title2path`                 | `scitex_path`               |
| `scitex_gen.var_info`                   | `scitex_types`              |
| `scitex_gen.is_ipython / is_script`     | `scitex_context`            |
| `scitex_gen.embed / less / paste`       | `scitex_context`            |
| `scitex_gen.cache / alternate_kwarg`    | `scitex_decorators`         |

### Dropped — not ported, no replacement port warranted

The destination peer already had a strictly more capable
implementation; consumers should use the peer's version directly:

- `scitex_gen.symlink`   → use `scitex_path.symlink` (richer API)
- `scitex_gen.ArrayLike` → use `scitex_types.ArrayLike` (lazy / optional-dep aware)
- `scitex_gen.wrap`      → use `scitex_decorators.wrap` (preserves `_original_func`)

### Dropped — was already deprecated

- `scitex_gen.start`, `close`, `running2finished`, `TimeStamper`

### Ecosystem re-exports dropped (use the canonical home directly)

`ci`, `check_host`, `is_host`, `verify_host`, `list_api`,
`run_shellcommand`, `run_shellscript`, `title_case`,
`detect_environment`, `is_notebook`, `get_notebook_path`,
`get_notebook_name`, `get_notebook_directory`, `get_output_directory`.

### What remains

- `scitex_gen/misc.py` retains the legacy private helpers
  `find_closest`, `isclose`, `is_nan`, `partial_at`, `describe`,
  `unique`, `uq`, `ThreadWithReturnValue`, etc. None of these were
  ever part of the public surface (no `__all__` entry, no `__init__`
  re-export); they are pending an operator-confirmed disposition. A
  follow-up PR will drop them.
- The package shell, the dispatch tables (now empty), and a single
  installable `__version__` attribute.

### Notes for consumers

- Attribute access on any retired symbol now raises `AttributeError`
  immediately at the call site — no silent `None`. This matches the
  operator's "no silent fallbacks" rule.
- See ADR-0001 for the architecture decision behind this retirement.

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
