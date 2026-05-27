<!-- ---
!-- Timestamp: 2025-01-15 10:53:31
!-- Author: ywatanabe
!-- File: ./src/scitex/gen/README.md
!-- --- -->

# `scitex-gen` Quick Start Guide

The `scitex-gen` module is a collection of general-purpose utility functions
and classes designed to simplify common programming tasks in data science
and machine learning workflows.

## Quick Start

```python
import scitex_gen

# Normalization
scitex_gen.to_z(tensor)
scitex_gen.to_01(tensor)

# Caching
@scitex_gen.cache
def expensive_function(x):
    ...

# Environment detection
scitex_gen.detect_environment()
scitex_gen.is_notebook()
scitex_gen.is_script()

# Numeric helpers
scitex_gen.to_even(7)     # 6
scitex_gen.to_odd(8)      # 7
scitex_gen.symn(x)        # symmetric n transform

# XML parsing
scitex_gen.xml2dict(xml_string)

# Filesystem
scitex_gen.symlink(tgt, src)

# Timing
ts = scitex_gen.TimeStamper()
ts("step1")  # logs elapsed time
```

## Subpackages

| Subpackage        | Contents                                                    |
| ----------------- | ----------------------------------------------------------- |
| `scitex_gen._fs`  | Filesystem helpers (symlink, src, title2path, print_config) |
| `scitex_gen._introspect` | Introspection, caching, mat2py, xml2dict, DimHandler        |
| `scitex_gen._ipython` | IPython/Jupyter helpers (embed, paste, less, notebook detection) |
| `scitex_gen._numeric` | Normalization, symlog, to_even, to_odd, to_rank, transpose  |
| `scitex_gen._legacy` | Backward-compat wrappers (TimeStamper, deprecated start/close) |

## Contact
Yusuke Watanabe (ywatanabe@scitex.ai)

For more information and updates, please visit the
[scitex-gen GitHub repository](https://github.com/ywatanabe1989/scitex-gen).
