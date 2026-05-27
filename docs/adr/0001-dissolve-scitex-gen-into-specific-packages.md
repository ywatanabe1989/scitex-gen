# ADR-0001: Dissolve `scitex-gen` into specific packages

## Status
Accepted (2026-05-27)

## Context
`scitex-gen` ("gen" = general) is a transitional grab-bag. Its own
`__init__.py` docstring already declares the intent: *"This module is being
refactored. Many functions are being moved to more appropriate locations."*
About half its public surface is already just deprecation re-exports from
specific packages (`ci`→stats, `check_host`→os, `run_shellcommand`→sh,
`title_case`→str, `detect_environment`→context, `start/close`→session, …).

The remaining real code is a junk drawer: numeric helpers, filesystem
helpers, IPython helpers, introspection/converters, a few types and
decorators. Every one of these has a natural home in an **existing**
ecosystem package — no new package is required.

A live-consumer scan (2026-05-27, excluding docs/examples/tests/generated
html and the legacy `_scitex` mngs tree) shows the real blast radius is
tiny: **only 6 of ~50 exported symbols** have downstream consumers, across
**4 packages / 11 files**.

| Symbol | Target | Consumers |
| --- | --- | --- |
| `to_even` | scitex-linalg | scitex-dsp, scitex-nn |
| `to_odd` | scitex-linalg | scitex-dsp, scitex-nn |
| `DimHandler` | scitex-linalg | scitex-nn (`_PAC.py`) |
| `float_linspace` | scitex-linalg | scitex-dsp (`_time.py`, dynamic import) |
| `to_z` | scitex-stats | scitex-dsp (`_detect_ripples.py`) |
| `unbias` | scitex-stats | norm family (dsp copy commented out) |

## Decision
Dissolve `scitex-gen` by moving every symbol to a specific existing package,
leaving deprecation re-export shims behind, then collapsing `scitex-gen` to a
shell. No new packages.

**Placement principles**
1. **Classify by what code *is*, not who calls it.** `to_even` is integer
   parity math even though only filter/kernel code calls it → `scitex-linalg`,
   not `scitex-dsp`.
2. **Shared helpers sink *below* their consumers.** `scitex-nn` deliberately
   does not depend on `scitex-dsp` (it vendors dsp utils). Anything both dsp
   and nn use must live in a foundational package (`scitex-linalg` /
   `scitex-stats`), never in a sibling domain package.
3. **Never break callers mid-flight.** Every move leaves a back-compat
   re-export shim that emits `DeprecationWarning`; shims die only in the final
   phase, after consumers are repointed.
4. **No cycles.** All 9 targets verified clean of real `scitex_gen` imports
   (only stray comments reference it): linalg, stats, types, decorators, path,
   io, introspect, notebook, datetime.

### Symbol → target inventory

Already re-exported (delete shims in Phase 3): `ci`→stats,
`check_host`/`is_host`/`verify_host`→os, `list_api`→introspect,
`run_shellcommand`/`run_shellscript`→sh, `detect_environment`/`is_notebook`/
`get_notebook_*`→context, `title_case`→str, `start`/`close`/
`running2finished`→session.

Real code to move:

| Source | Symbol(s) | Target |
| --- | --- | --- |
| `_numeric/_to_even`, `_to_odd` | `to_even`, `to_odd` | scitex-linalg |
| `_numeric/_transpose`, `_symlog` | `transpose`, `symlog` | scitex-linalg |
| `_introspect/_DimHandler` | `DimHandler` | scitex-linalg |
| `_numeric/_norm` | `to_z`, `to_nanz`, `to_01`, `to_nan01`, `unbias`, `clip_perc` | scitex-stats |
| `_numeric/_to_rank` | `to_rank` | scitex-stats |
| `_type`, `_var_info` | `ArrayLike`, `var_info` | scitex-types |
| `_wrap` | `wrap` | scitex-decorators |
| `_introspect/_cache` | `cache` | scitex-decorators |
| `_introspect/_alternate_kwarg` | `alternate_kwarg` | scitex-decorators |
| `_introspect/_list_packages` | `list_packages`, `main` | scitex-introspect |
| `_introspect/_xml2dict` | `xml2dict`, `XmlDictConfig`, `XmlListConfig` | scitex-io |
| `_introspect/_mat2py` | `mat2npy`, `dir2npy`, `mat2dict`, `mat2npa`, `keys2npa`, `public_keys`, `save_npa` | scitex-io |
| `_fs/_symlink`, `_title2path` | `symlink`, `title2path` | scitex-path |
| `_fs/_src` | `src` | scitex-introspect |
| `_fs/_print_config` | `print_config`, `print_config_main` | scitex-config |
| `_ipython/*` | `embed`, `paste`, `less`, `is_ipython`, `is_script` | scitex-notebook |
| `_legacy/_TimeStamper` | `TimeStamper` | scitex-datetime |

`misc.py` split: `connect_nums`→str; `float_linspace`/`find_closest`/
`isclose`/`is_nan`→linalg; `describe`→stats; `copy_files`/`copy_the_file`/
`_copy_a_file`→path; `is_defined_global`/`is_defined_local`/
`is_later_or_equal`→introspect; `partial_at`→decorators;
`ThreadWithReturnValue`/`wait_key`→parallel.

### Per-move recipe
1. `git mv` source into the target package.
2. Export from target `__init__.py` (import + `__all__`).
3. Replace the scitex-gen module body with a deprecation re-export shim.
4. Add the dependency edge in any consumer's `pyproject.toml` if missing.
5. Run tests in both target and scitex-gen.

Import-path rewrites use the cross-reference-aware bulk tool, dry-run first:

```bash
scitex-dev rename-symbols \
  'from scitex_gen\._to_even import to_even' \
  'from scitex_linalg import to_even' \
  --regex --root /home/ywatanabe/proj --dry-run
```

### Ordered phases
- **Phase 0 — quick wins (no move):** repoint `get_notebook_path` consumers
  (`scitex-notebook/_magic.py`, `scitex-session/_lifecycle/_start.py`)
  straight at `scitex-context`. The session import path
  (`scitex_gen._detect_notebook_path`) is currently *wrong* and always falls
  through to its `except` — this also fixes that latent failure.
- **Phase 1 — live numeric symbols:** the 6 externally-consumed symbols (plus
  their `_norm`/`_numeric` siblings) → linalg / stats; repoint the 11 dsp & nn
  files. After this, nothing imports scitex-gen for real work.
- **Phase 2 — zero-consumer bulk:** everything else (xml2dict, mat2py,
  symlink, wrap, cache, TimeStamper, the `misc.py` split), each with a shim.
- **Phase 3 — collapse:** reduce scitex-gen to deprecation-only (or archive
  it) and delete the redundant re-export shims.

## Consequences
- Symbols live where they conceptually belong; discoverability improves.
- `scitex-nn` gains no dependency on `scitex-dsp` (the placement rule that
  drove the linalg/stats choice for `to_even`/`to_odd`/`DimHandler`/`unbias`).
- A grace period of deprecation shims keeps every existing caller working;
  the ecosystem can migrate import paths at its own pace.
- Downstream churn is small and front-loaded into Phase 1 (11 files, 4
  packages); Phases 2–3 carry near-zero risk.
- `scitex-gen` ultimately disappears from the ecosystem list, removing a
  catch-all that invited future junk-drawer accretion.

## Notes
Surfaced 2026-05-27 when the operator asked whether "gen" (general) could be
refactored into more specific packages. The dsp-vs-linalg/stats placement of
`to_even`/`to_odd`/`DimHandler`/`unbias` was debated explicitly and resolved
by principles 1–2 above. This ADR supersedes the draft
`docs/migration-plan-scitex-gen-dissolution.md`.
