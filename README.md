# SciTeX Gen (<code>scitex-gen</code>) — Retired

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/assets/images/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>This package is retired. Its surface has moved to canonical peers across the SciTeX ecosystem.</b></p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-gen/"><img src="https://img.shields.io/pypi/v/scitex-gen?label=pypi" alt="pypi"></a>
  <a href="https://pypi.org/project/scitex-gen/"><img src="https://img.shields.io/pypi/pyversions/scitex-gen?label=python" alt="python"></a>
</p>
<!-- scitex-badges:end -->

---

## Status

As of **v0.1.13** (2026-06-07), `scitex-gen` is retired. The package
publishes only `__version__`; accessing any former public symbol raises
`AttributeError` immediately at the call site (no silent fallbacks).

The package shell is kept on PyPI so existing pins (`scitex-gen>=0.x`) in
downstream consumer pyprojects continue to install cleanly while
maintainers migrate their import paths.

## Where did everything go?

| Old import path | New canonical home |
|---|---|
| `scitex_gen.symlog / to_rank / transpose / connect_nums / float_linspace` | `scitex_math` |
| `scitex_gen.to_z / to_01 / to_nan01 / to_nanz / unbias / clip_perc` | `scitex_math` |
| `scitex_gen.to_even / to_odd` (already moved in v0.1.12) | `scitex_math` |
| `scitex_gen.DimHandler` | `scitex_nn` |
| `scitex_gen.mat2dict / mat2npa / mat2npy / dir2npy / save_npa / keys2npa / public_keys` | `scitex_io` |
| `scitex_gen.xml2dict / XmlDictConfig / XmlListConfig` | `scitex_io` |
| `scitex_gen.print_config / print_config_main` | `scitex_io` |
| `scitex_gen.list_packages / main / src` | `scitex_introspect` |
| `scitex_gen.title2path` | `scitex_path` |
| `scitex_gen.var_info` | `scitex_types` |
| `scitex_gen.is_ipython / is_script` | `scitex_context` |
| `scitex_gen.embed / less / paste` | `scitex_context` |
| `scitex_gen.cache / alternate_kwarg` | `scitex_decorators` |

### Dropped — use the peer's superior version directly

- `scitex_gen.symlink`   → `scitex_path.symlink`
- `scitex_gen.ArrayLike` → `scitex_types.ArrayLike`
- `scitex_gen.wrap`      → `scitex_decorators.wrap`

### Dropped — were already deprecated

- `scitex_gen.start`, `close`, `running2finished`, `TimeStamper`

### Re-export shims dropped — use the canonical home

`ci`, `check_host`, `is_host`, `verify_host`, `list_api`,
`run_shellcommand`, `run_shellscript`, `title_case`,
`detect_environment`, `is_notebook`, `get_notebook_path`,
`get_notebook_name`, `get_notebook_directory`, `get_output_directory`.

## ADR

See [ADR-0001](docs/adr/ADR-0001-dissolve-scitex-gen.md) for the
architecture decision behind this retirement.

## License

AGPL-3.0-only.
