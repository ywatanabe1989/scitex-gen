---
name: scitex-gen
description: |
  [WHAT] General-purpose utilities for SciTeX scripts — small helpers for collections, iteration, and ad-hoc tasks not big enough to deserve their own package.
  [WHEN] Writing research scripts and you reach for one-off Python utilities.
  [HOW] `from scitex_gen import ...` or `scitex-gen --help`.
primary_interface: python
interfaces:
  python: 1
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
canonical-location: scitex-gen/src/scitex_gen/_skills/scitex-gen/SKILL.md
tags: [scitex-gen]
---

# scitex-gen

**Deprecated namespace** — kept for backwards compatibility. Every
import emits `DeprecationWarning` pointing at the new location.

## Migration map

| Old import (deprecated)        | New canonical location                       |
| ------------------------------ | -------------------------------------------- |
| `scitex_gen.ci`                | `scitex_stats.descriptive.ci`                |
| `scitex_gen.check_host`        | `scitex_os.check_host`                       |
| `scitex_gen.is_host`           | `scitex_os.is_host`                          |
| `scitex_gen.verify_host`       | `scitex_os.verify_host`                      |
| `scitex_gen.detect_environment`| `scitex_context.detect_environment`          |
| `scitex_gen.is_notebook`       | `scitex_context.is_notebook`                 |
| `scitex_gen.is_script`         | `scitex_context.is_script`                   |
| `scitex_gen.list_api`          | `scitex_introspect.list_api`                 |
| `scitex_gen.run_shellcommand`  | `scitex_sh.run_shellcommand`                 |
| `scitex_gen.run_shellscript`   | `scitex_sh.run_shellscript`                  |
| `scitex_gen.xml2dict`          | `scitex_io.xml2dict`                         |
| `scitex_gen.title_case`        | `scitex_str.title_case`                      |
| `scitex_gen.symlink`           | `scitex_path.symlink`                        |

## When to use

- ✅ Reading old code — confirm what `scitex.gen.X` resolves to and
  rewrite imports to the new module
- ❌ New code — always import from the canonical location

## Removal plan

`scitex-gen` will be removed when downstream usage drops to zero
(tracked via `audit_doc_examples.py`). Until then, the shim is
maintained but no new functions will be added.

## Sub-skills

### Core (01–09)
- [01_installation.md](01_installation.md) — install + import sanity check
- [02_quick-start.md](02_quick-start.md) — 30-second tour
- [03_python-api.md](03_python-api.md) — Python API surface
