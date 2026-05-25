#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 18:53:04 (ywatanabe)"
# /home/yusukew/proj/scitex_repo/src/scitex/gen/_print_config.py

"""
1. Functionality:
   - Prints configuration values from YAML files
2. Input:
   - Configuration key (dot-separated for nested structures)
3. Output:
   - Corresponding configuration value
4. Prerequisites:
   - scitex package with load_configs function

Example:
    python _print_config.py PATH.TITAN.MAT
"""

import argparse
import os
import sys
from pprint import pprint


def print_config(key, *, config_loader=None):
    """Print a configuration value addressed by a dot-separated key.

    Parameters
    ----------
    key : str or None
        Dot-separated path into the config (e.g. ``"PATH.TITAN.MAT"``). When
        ``None`` the whole config is pretty-printed.
    config_loader : callable, optional
        Zero-argument callable returning the configuration mapping. Defaults to
        ``scitex.io.load_configs``; injectable so callers (and tests) can
        supply a real config dict without depending on the umbrella package.
        ``DotDict`` configs subclass ``dict`` and are navigated via the ``dict``
        branch.
    """
    if config_loader is None:
        import scitex

        config_loader = scitex.io.load_configs
    CONFIG = config_loader()

    if key is None:
        print("Available configurations:")
        pprint(CONFIG)
        return

    try:
        keys = key.split(".")
        value = CONFIG
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)

            elif isinstance(value, list):
                try:
                    value = value[int(k)]
                except (ValueError, IndexError):
                    value = None

            elif isinstance(value, str):
                break

            else:
                value = None

            if value is None:
                break

        print(value)

    except Exception as e:
        print(f"Error: {e}")
        print("Available configurations:")
        pprint(value)


def print_config_main(args=None, *, config_loader=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Print configuration values")
    parser.add_argument(
        "key",
        nargs="?",
        default=None,
        help="Configuration key (dot-separated for nested structures)",
    )
    parsed_args = parser.parse_args(args)
    print_config(parsed_args.key, config_loader=config_loader)


if __name__ == "__main__":
    print_config_main()
