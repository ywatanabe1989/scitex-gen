---
name: scitex-gen
description: |
  [WHAT] General-purpose Python utilities for scientific workflows — caching, normalization, env detection, XML→dict, mat→npy, TimeStamper, numeric helpers, and filesystem utilities.
  [WHEN] Writing research scripts that need small utility functions or backward-compatible re-exports from peer SciTeX packages.
  [HOW] `from scitex_gen import ...`
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

Active collection of general-purpose Python utilities for scientific
workflows. Provides both native functions (`to_even`, `to_odd`, `to_z`,
`to_01`, `symlog`, `TimeStamper`, `cache`, `list_packages`, `xml2dict`,
`transpose`, ...) and backward-compatible re-exports from peer SciTeX
packages.

## Re-exported names (thin wrappers over peer packages)

Some names in `scitex_gen` are re-exported from canonical peer packages
for backward compatibility. New code should import from the canonical
location:

| scitex_gen name             | Canonical location                   |
| --------------------------- | ------------------------------------ |
| `ci`                        | `scitex_stats.descriptive.ci`       |
| `check_host`                | `scitex_os.check_host`              |
| `is_host`                   | `scitex_os.is_host`                 |
| `verify_host`               | `scitex_os.verify_host`             |
| `detect_environment`        | `scitex_context.detect_environment` |
| `is_notebook`               | `scitex_context.is_notebook`        |
| `is_script`                 | `scitex_context.is_script`          |
| `list_api`                  | `scitex_introspect.list_api`        |
| `run_shellcommand`          | `scitex_sh.run_shellcommand`        |
| `run_shellscript`           | `scitex_sh.run_shellscript`         |
| `title_case`                | `scitex_str.title_case`             |
| `symlink`                   | `scitex_path.symlink`               |

## Native functions (defined in scitex-gen)

| Function / Class     | Module                          |
| -------------------- | ------------------------------- |
| `cache`              | `scitex_gen._introspect._cache` |
| `DimHandler`         | `scitex_gen._introspect`        |
| `list_packages`      | `scitex_gen._introspect`        |
| `mat2npy / mat2dict` | `scitex_gen._introspect`        |
| `xml2dict`           | `scitex_gen._introspect`        |
| `alternate_kwarg`    | `scitex_gen._introspect`        |
| `embed`              | `scitex_gen._ipython`           |
| `less`               | `scitex_gen._ipython`           |
| `paste`              | `scitex_gen._ipython`           |
| `TimeStamper`        | `scitex_gen._legacy`            |
| `to_z / to_01`       | `scitex_gen._numeric._norm`     |
| `symlog / symn`      | `scitex_gen._numeric._symlog`   |
| `to_even / to_odd`   | `scitex_gen._numeric`           |
| `to_rank`            | `scitex_gen._numeric`           |
| `transpose`          | `scitex_gen._numeric`           |
| `symlink`            | `scitex_gen._fs`               |
| `src`                | `scitex_gen._fs`               |
| `title2path`         | `scitex_gen._fs`               |
| `print_config`       | `scitex_gen._fs`               |
| `wrap`               | `scitex_gen._wrap`             |

## When to use

- ✅ Writing research scripts that need normalization, caching, env detection,
  XML parsing, or similar small utilities
- ✅ Reading old code that imports from `scitex.gen` — these names resolve
  through the re-export chain
- ✅ Choosing a single import target instead of reaching for `scitex-stats`,
  `scitex-context`, etc. for every small task (especially in exploratory code)

## Sub-skills

### Core (01–09)
- [01_installation.md](01_installation.md) — install + import sanity check
- [02_quick-start.md](02_quick-start.md) — 30-second tour
- [03_python-api.md](03_python-api.md) — Python API surface
