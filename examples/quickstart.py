"""scitex-gen quickstart: small numerical helpers + symmetric-log transform.

scitex-gen is a grab-bag of general-purpose utilities. This script picks a
handful of pure helpers that don't require torch / matplotlib / I/O.
"""

import numpy as np

import scitex_gen


def main():
    # 1. to_rank: ordinal rank of each element.
    arr = np.array([10.0, 30.0, 20.0, 30.0])
    ranks = scitex_gen.to_rank(arr)
    print("array:", arr)
    print("ranks:", ranks)
    assert ranks.shape == arr.shape

    # 2. to_even / to_odd: round to the nearest even / odd integer.
    print("\nto_even(5)  ->", scitex_gen.to_even(5))
    print("to_odd(4)   ->", scitex_gen.to_odd(4))
    assert scitex_gen.to_even(5) % 2 == 0
    assert scitex_gen.to_odd(4) % 2 == 1

    # 3. symlog: signed log transform (linear near zero, logarithmic at scale).
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = scitex_gen.symlog(x, linthresh=1.0)
    print("\nx     :", x)
    print("symlog:", y)
    assert y.shape == x.shape
    # symlog is monotone-increasing
    assert np.all(np.diff(y) > 0)

    # 4. title2path: convert a human title into a filesystem-safe slug.
    slug = scitex_gen.title2path("Hello, World! 2026")
    print("\ntitle2path('Hello, World! 2026') ->", slug)
    assert " " not in slug


if __name__ == "__main__":
    main()
