# scitex-gen

General-purpose utilities (caching, environment detection, normalization, mat→npy, xml→dict, TimeStamper, etc.) extracted from the [SciTeX](https://github.com/ywatanabe1989/scitex-python) ecosystem as a standalone package.

## Install

```bash
pip install scitex-gen
```

## API

```python
import scitex_gen as gen

gen.cache(...)
gen.TimeStamper()
gen.xml2dict(...)
gen.to_z(tensor)          # requires torch
gen.to_even(n)
gen.to_odd(n)
gen.transpose(...)
```

## Status

Standalone fork of `scitex.gen`. The umbrella package's `scitex.gen` import path
is preserved via a `sys.modules`-alias bridge.

Decoupling notes:
- `scitex.{decorators,str,os,introspect,session,context,sh,dict}` →
  `scitex_*` direct imports (peer packages).
- `scitex.torch.nanstd` → optional via `try/except` with a torch-only
  fallback (only matters for `_norm.to_z / to_nanz`).
- `import scitex` removed from `_less.py` (was unused in module body).
- self-references in `_norm_cache.py` rewritten to `scitex_gen.*`.

## License

AGPL-3.0-only (see [LICENSE](./LICENSE)).
