---
description: |
  [TOPIC] Quick Start
  [DETAILS] Smallest useful example demonstrating the primary use case in
  under 30 seconds.
tags: [scitex-gen-quick-start]
---

# Quick Start

```python
import scitex_gen

# Normalize a tensor / array
z = scitex_gen.to_z(tensor)       # z-score (NaN-aware)
scaled = scitex_gen.to_01(tensor) # min-max → [0, 1]

# Cached function
@scitex_gen.cache
def load_dataset(path: str):
    ...

# Environment detection
env = scitex_gen.detect_environment()
# -> "notebook" | "ipython" | "script" | "test"

# Numeric helpers (parity helpers moved to scitex-math)
from scitex_math import to_even, to_odd
to_even(7)              # 6
to_odd(8)               # 7
scitex_gen.symn(x)      # symmetric n transform

# XML → dict
import scitex_gen
data = scitex_gen.xml2dict("<root><item>value</item></root>")

# Timing
ts = scitex_gen.TimeStamper()
ts("step1")   # logs elapsed since creation
```

Install: `pip install scitex-gen`
