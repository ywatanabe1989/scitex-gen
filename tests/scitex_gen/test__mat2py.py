#!/usr/bin/env python3
# Timestamp: "2025-05-31 19:55:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/scitex_repo/tests/scitex/gen/test__mat2py.py

"""Tests for MATLAB-to-Python conversion helpers.

All tests exercise the real functions against real ``.mat`` / ``.npy`` files
written into temp files — no mocks. (The ``mat2npy`` path previously required a
mock only because ``mat2npa``/``keys2npa`` shipped a stray ``pdb.set_trace()``
and a Python-2 ``dict.keys()[0]`` indexing bug; both are now fixed in source.)
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
from scipy.io import savemat

pytest.importorskip("torch")

from scitex_gen import mat2dict, mat2npy, public_keys, save_npa


@pytest.fixture
def scipy_mat_file():
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        test_data = {
            "matrix1": np.array([[1, 2, 3], [4, 5, 6]]),
            "matrix2": np.array([7, 8, 9]),
        }
        savemat(tmp.name, test_data)
        try:
            yield tmp.name
        finally:
            os.unlink(tmp.name)


@pytest.fixture
def hdf5_mat_file():
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            f.create_dataset("data1", data=np.array([[1, 2], [3, 4]]))
            f.create_dataset("data2", data=np.array([5, 6, 7]))
        try:
            yield tmp.name
        finally:
            os.unlink(tmp.name)


@pytest.fixture
def single_matrix_mat_file():
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        savemat(tmp.name, {"data": np.array([1, 2, 3])})
        try:
            yield tmp.name
        finally:
            for path in (tmp.name, tmp.name.replace(".mat", ".npy")):
                if os.path.exists(path):
                    os.unlink(path)


# --- mat2dict: scipy ------------------------------------------------------


def test_mat2dict_loads_matrix1_key(scipy_mat_file):
    # Arrange
    # Act
    result = mat2dict(scipy_mat_file)
    # Assert
    assert "matrix1" in result


def test_mat2dict_loads_matrix2_key(scipy_mat_file):
    # Arrange
    # Act
    result = mat2dict(scipy_mat_file)
    # Assert
    assert "matrix2" in result


def test_mat2dict_marks_scipy_file_as_non_hdf(scipy_mat_file):
    # Arrange
    # Act
    result = mat2dict(scipy_mat_file)
    # Assert
    assert result["__hdf__"] is False


def test_mat2dict_preserves_matrix1_values(scipy_mat_file):
    # Arrange
    # Act
    result = mat2dict(scipy_mat_file)
    # Assert
    assert np.array_equal(result["matrix1"], np.array([[1, 2, 3], [4, 5, 6]]))


def test_mat2dict_preserves_matrix2_values(scipy_mat_file):
    # Arrange
    # Act
    result = mat2dict(scipy_mat_file)
    # Assert
    assert np.array_equal(result["matrix2"].flatten(), np.array([7, 8, 9]))


# --- mat2dict: hdf5 -------------------------------------------------------


def test_mat2dict_loads_hdf5_data1_key(hdf5_mat_file):
    # Arrange
    # Act
    result = mat2dict(hdf5_mat_file)
    # Assert
    assert "data1" in result


def test_mat2dict_loads_hdf5_data2_key(hdf5_mat_file):
    # Arrange
    # Act
    result = mat2dict(hdf5_mat_file)
    # Assert
    assert "data2" in result


def test_mat2dict_marks_hdf5_file_as_hdf(hdf5_mat_file):
    # Arrange
    # Act
    result = mat2dict(hdf5_mat_file)
    # Assert
    assert result["__hdf__"] is True


def test_mat2dict_raises_for_non_mat_file():
    # Arrange
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"Not a mat file")
        tmp.flush()
        path = tmp.name
    ctx = pytest.raises(Exception)
    # Act
    # Assert
    try:
        with ctx:
            mat2dict(path)
    finally:
        os.unlink(path)


# --- public_keys ----------------------------------------------------------


@pytest.mark.parametrize("public_key", ["public1", "public2", "another_public"])
def test_public_keys_includes_public_keys(public_key):
    # Arrange
    test_dict = {"public1": 1, "public2": 2, "_private1": 3, "another_public": 6}
    # Act
    public = public_keys(test_dict)
    # Assert
    assert public_key in public


@pytest.mark.parametrize("private_key", ["_private1", "__private2__", "_"])
def test_public_keys_excludes_underscore_keys(private_key):
    # Arrange
    test_dict = {"public1": 1, "_private1": 3, "__private2__": 4, "_": 5}
    # Act
    public = public_keys(test_dict)
    # Assert
    assert private_key not in public


def test_public_keys_returns_empty_for_empty_dict():
    # Arrange
    # Act
    public = public_keys({})
    # Assert
    assert public == []


def test_public_keys_returns_empty_when_all_keys_private():
    # Arrange
    test_dict = {"_private1": 1, "_private2": 2, "__dunder__": 3}
    # Act
    public = public_keys(test_dict)
    # Assert
    assert public == []


# --- save_npa -------------------------------------------------------------


def test_save_npa_writes_loadable_array():
    # Arrange
    test_array = np.array([1, 2, 3, 4, 5])
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        path = tmp.name
    # Act
    save_npa(path, test_array)
    loaded = np.load(path)
    os.unlink(path)
    # Assert
    assert np.array_equal(loaded, test_array)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_save_npa_preserves_values_for_dtype(dtype):
    # Arrange
    test_array = np.array([1, 2, 3, 4, 5], dtype=dtype)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        path = tmp.name
    # Act
    save_npa(path, test_array)
    loaded = np.load(path)
    os.unlink(path)
    # Assert
    assert np.array_equal(loaded, test_array)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_save_npa_preserves_dtype(dtype):
    # Arrange
    test_array = np.array([1, 2, 3, 4, 5], dtype=dtype)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        path = tmp.name
    # Act
    save_npa(path, test_array)
    loaded = np.load(path)
    os.unlink(path)
    # Assert
    assert loaded.dtype == dtype


# --- mat2npy (end-to-end, no mocks) ---------------------------------------


def test_mat2npy_creates_npy_file(single_matrix_mat_file):
    # Arrange
    npy_path = single_matrix_mat_file.replace(".mat", ".npy")
    # Act
    mat2npy(single_matrix_mat_file, np.float32)
    # Assert
    assert os.path.exists(npy_path)


def test_mat2npy_npy_contains_matrix_values(single_matrix_mat_file):
    # Arrange
    npy_path = single_matrix_mat_file.replace(".mat", ".npy")
    # Act
    mat2npy(single_matrix_mat_file, np.float32)
    loaded = np.load(npy_path)
    # Assert
    assert np.array_equal(loaded.flatten(), np.array([1, 2, 3], dtype=np.float32))


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_mat2py.py
# --------------------------------------------------------------------------------
# def mat2dict(fname):
#     """Function returns a dictionary with .mat variables"""
#     try:
#         D = h5py.File(fname)
#         d = {}
#         for key, value in D.items():
#             d[key] = value
#         d["__hdf__"] = True
#     except:
#         d = loadmat(fname)
#         d["__hdf__"] = False
#     return d
#
#
# def keys2npa(d, typ):
#     d2 = {}
#     for key in public_keys(d):
#         x = np.array(d[key], dtype=typ)
#         if d["__hdf__"]:
#             x = np.squeeze(np.swapaxes(x, 0, -1))
#         assert type(x.flatten()[0]) == typ
#         d2[key] = x.copy()
#     return d2
#
#
# def public_keys(d):
#     return [k for k in d.keys() if not k.startswith("_")]
#
#
# def mat2npa(fname, typ):
#     """Function returns np array from 1st entry in .mat file"""
#     d = keys2npa(mat2dict(fname), typ)
#     return d[list(d.keys())[0]]
#
#
# def save_npa(fname, x):
#     np.save(fname, x)
#
#
# def mat2npy(fname, typ):
#     """Function save np array from 1st entry in .mat file to .npy file"""
#     x = mat2npa(fname, typ)
#     save_npa(fname=fname.replace(".mat", ""), x=x)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_mat2py.py
# --------------------------------------------------------------------------------
