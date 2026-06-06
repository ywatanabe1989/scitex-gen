# Changelog

All notable changes to `scitex-gen` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **`torch` is no longer a core dependency.** Bare `pip install
  scitex-gen` no longer pulls torch + nvidia-cuda + cublas +
  cuda-toolkit + triton + sympy + networkx (~4GB) as transitive deps.
  The `[torch]` extra (already declared in `pyproject.toml`) gates
  the install: users who need the torch-specific surface install
  with `pip install 'scitex-gen[torch]'`. Motivation surfaced by
  proj-paper-ripple-wm's 4.4GB SIF (2026-06-07): a slim research
  container ate a heavy CUDA chain just for a small slice of
  scitex_gen that used `torch.Tensor` in `isinstance` checks.

  Three modules are now torch-optional:
  - `scitex_gen._type` — `var_info()` still works on np/pd/xr inputs
    without torch. `ArrayLike` and the internal isinstance tuple
    include `torch.Tensor` only when the `[torch]` extra is
    installed. `_TORCH_AVAILABLE` flag exposed for downstream checks.
  - `scitex_gen._var_info` — same shape as `_type`.
  - `scitex_gen._numeric._norm` — every function (`to_z`, `to_nanz`,
    `to_01`, `to_nan01`, `unbias`, `clip_perc`) is torch-native by
    construction; without torch they raise a clear `ImportError`
    naming the extra: "scitex_gen._numeric requires torch for this
    function. Install with: {sys.executable} -m pip install
    'scitex-gen[torch]'". The `_require_torch()` helper centralises
    that gate so each function body stays unchanged otherwise.

  No public API change for users who install with the `[torch]`
  extra. Users on bare install who never call torch-specific
  functions are unaffected.

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
