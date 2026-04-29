#!/usr/bin/env python3
# Timestamp: 2026-04-29
# File: src/scitex_gen/_shell.py

"""Shell command utilities — local module so tests can mock run_shellcommand."""

import os
import subprocess


def run_shellcommand(command, *args):
    """Run a shell command and return stdout, stderr, and exit code."""
    full_command = [command] + list(args)
    result = subprocess.run(
        full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout = result.stdout
    stderr = result.stderr
    exit_code = result.returncode
    if exit_code == 0:
        print("Command executed successfully")
        print("Output:", stdout)
    else:
        print("Command failed with error code:", exit_code)
        print("Error:", stderr)
    return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}


def run_shellscript(lpath_sh, *args):
    """Run a shell script, making it executable first if needed."""
    if not os.access(lpath_sh, os.X_OK):
        subprocess.run(["chmod", "+x", lpath_sh])
    command = [lpath_sh] + list(args)
    return run_shellcommand(*command)


# EOF
