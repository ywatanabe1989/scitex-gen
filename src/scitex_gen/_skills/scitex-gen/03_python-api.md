---
description: |
  [TOPIC] Python API
  [DETAILS] Public Python API of scitex-gen — exported functions, signatures,
  return types, and minimal usage examples per function.
tags: [scitex-gen-python-api]
---

# Python API

```python
import scitex_gen
```

## Normalization

- `to_z(x, axis=-1, dim=None, device="cuda")` → z-score normalization (NaN-aware)
- `to_01(x, axis=-1, dim=None, device="cuda")` → min-max scaling to [0, 1]
- `to_nanz(x, ...)` → NaN-safe z-score
- `to_nan01(x, ...)` → NaN-safe min-max
- `clip_perc(x, ...)` → percentile clipping
- `unbias(x, ...)` → unbias estimator

## Numeric helpers

- `to_rank(x, ...)` — rank transformation

> ``to_even`` and ``to_odd`` moved to
> [`scitex-math`](https://pypi.org/project/scitex-math/);
> import them as ``from scitex_math import to_even, to_odd``.

- `symlog(x, linthresh=1.0)` — symmetric log transform
- `symn(x, linthresh=1.0)` — symmetric n transform
- `transpose(x, src, tgt)` — named-axis transpose for tensors

## Introspection / utilities

- `cache(func)` — `lru_cache`-based function cache
- `list_packages()` — list installed packages
- `alternate_kwarg(func)` — alternate keyword argument dispatch
- `DimHandler` — dimension manipulation for tensors
- `dir2npy(...)`, `mat2npy(...)`, `mat2dict(...)` — MATLAB file conversion
- `xml2dict(xml_string)` → nested dict from XML
- `var_info(array)` — variable information display

## IPython / environment

- `detect_environment()` → `"notebook"|"ipython"|"script"|"test"`
- `is_notebook()` → bool
- `is_script()` → bool
- `is_ipython()` → bool
- `embed()` — launch embedded IPython with clipboard support
- `paste()` — paste clipboard content
- `less(text)` — display text via `less` pager
- `get_notebook_path()` → str or None

## Filesystem

- `symlink(tgt, src, force=False)` — create symbolic link
- `src(obj)` — display source code of object via `less`
- `title2path(title)` → path-friendly string
- `print_config(key)` — print configuration value

## Timing

- `TimeStamper()` — elapsed-time logger
  - `ts(label)` — log elapsed since creation
  - Properties: `elapsed`, `total`

## Miscellaneous

- `wrap(text, width=70)` — word-wrap text
- `connect_nums(a, b)` — connect numbers with operator
- `float_linspace(start, stop, num)` — float-safe linear spacing
- `symlink(...)` — symlink creation (re-exported from scitex-path)
- `check_host(...)`, `is_host(...)`, `verify_host(...)` — host checks (re-exported from scitex-os)
- `run_shellcommand(...)`, `run_shellscript(...)` — shell execution (re-exported from scitex-sh)
- `title_case(text)` — title case conversion (re-exported from scitex-str)
