import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scitex_gen import symlink


class TestSymlinkBasic:
    """Test basic symlink functionality."""

    def test_create_simple_symlink_link_is_symlink(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Hello World")
        # Create symlink
        link = tmp_path / "link.txt"
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Act
        try:
            symlink(str(target), str(link))
        finally:
            sys.stdout = old_stdout
        # Act
        # Assert
        # Assert
        assert link.is_symlink()

    def test_create_simple_symlink_link_resolve_target_resolve(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Hello World")
        # Create symlink
        link = tmp_path / "link.txt"
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Act
        try:
            symlink(str(target), str(link))
        finally:
            sys.stdout = old_stdout
        # Act
        # Assert
        # Assert
        assert link.resolve() == target.resolve()

    def test_create_simple_symlink_link_read_text_hello_world(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Hello World")
        # Create symlink
        link = tmp_path / "link.txt"
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Act
        try:
            symlink(str(target), str(link))
        finally:
            sys.stdout = old_stdout
        # Act
        # Assert
        # Assert
        assert link.read_text() == "Hello World"

    def test_create_simple_symlink_symlink_was_created_in_output(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Hello World")
        # Create symlink
        link = tmp_path / "link.txt"
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Act
        try:
            symlink(str(target), str(link))
        finally:
            sys.stdout = old_stdout
        # Check symlink exists and points to target
        # Assert
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
        assert link.read_text() == "Hello World"
        # Check output message
        output = buffer.getvalue()
        # Act
        # Assert
        assert "Symlink was created" in output

    def test_create_simple_symlink_str_link_in_output(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Hello World")
        # Create symlink
        link = tmp_path / "link.txt"
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Act
        try:
            symlink(str(target), str(link))
        finally:
            sys.stdout = old_stdout
        # Check symlink exists and points to target
        # Assert
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
        assert link.read_text() == "Hello World"
        # Check output message
        output = buffer.getvalue()
        # Act
        # Assert
        assert str(link) in output


    def test_symlink_to_directory_link_dir_is_symlink(self, tmp_path):
        # Arrange
        # Arrange
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        (target_dir / "file1.txt").write_text("File 1")
        (target_dir / "file2.txt").write_text("File 2")
        # Create symlink
        link_dir = tmp_path / "link_dir"
        # Act
        symlink(str(target_dir), str(link_dir))
        # Act
        # Assert
        # Assert
        assert link_dir.is_symlink()

    def test_symlink_to_directory_link_dir_is_dir(self, tmp_path):
        # Arrange
        # Arrange
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        (target_dir / "file1.txt").write_text("File 1")
        (target_dir / "file2.txt").write_text("File 2")
        # Create symlink
        link_dir = tmp_path / "link_dir"
        # Act
        symlink(str(target_dir), str(link_dir))
        # Act
        # Assert
        # Assert
        assert link_dir.is_dir()

    def test_symlink_to_directory_link_dir_file1_txt_read_text_file_1(self, tmp_path):
        # Arrange
        # Arrange
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        (target_dir / "file1.txt").write_text("File 1")
        (target_dir / "file2.txt").write_text("File 2")
        # Create symlink
        link_dir = tmp_path / "link_dir"
        # Act
        symlink(str(target_dir), str(link_dir))
        # Act
        # Assert
        # Assert
        assert (link_dir / "file1.txt").read_text() == "File 1"

    def test_symlink_to_directory_link_dir_file2_txt_read_text_file_2(self, tmp_path):
        # Arrange
        # Arrange
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        (target_dir / "file1.txt").write_text("File 1")
        (target_dir / "file2.txt").write_text("File 2")
        # Create symlink
        link_dir = tmp_path / "link_dir"
        # Act
        symlink(str(target_dir), str(link_dir))
        # Act
        # Assert
        # Assert
        assert (link_dir / "file2.txt").read_text() == "File 2"


    def test_relative_path_calculation_not_os_path_isabs_link_target(self, tmp_path):
        # Arrange
        # Arrange
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "x" / "y").mkdir(parents=True)
        # Create target
        target = tmp_path / "a" / "b" / "target.txt"
        target.write_text("Target content")
        # Create symlink in different directory
        link = tmp_path / "x" / "y" / "link.txt"
        symlink(str(target), str(link))
        # Check that relative path is used
        # Act
        link_target = os.readlink(str(link))
        # Act
        # Assert
        # Assert
        assert not os.path.isabs(link_target)

    def test_relative_path_calculation_link_target_equals_a_b_target_txt(self, tmp_path):
        # Arrange
        # Arrange
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "x" / "y").mkdir(parents=True)
        # Create target
        target = tmp_path / "a" / "b" / "target.txt"
        target.write_text("Target content")
        # Create symlink in different directory
        link = tmp_path / "x" / "y" / "link.txt"
        symlink(str(target), str(link))
        # Check that relative path is used
        # Act
        link_target = os.readlink(str(link))
        # Act
        # Assert
        # Assert
        assert link_target == "../../a/b/target.txt"


    def test_symlink_in_same_directory(self, tmp_path):
        """Test creating symlink in the same directory as target."""
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Same dir")

        link = tmp_path / "link.txt"
        symlink(str(target), str(link))

        # When in same directory, relative path should be simple
        # Act
        link_target = os.readlink(str(link))
        # Assert
        assert link_target == "target.txt"


class TestSymlinkForceOption:
    """Test symlink force option functionality."""

    def test_force_removes_existing_file_raises_fileexistserror(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Target content")
        # Create existing file at link location
        link = tmp_path / "link.txt"
        # Act
        link.write_text("Existing content")
        # Act
        # Assert
        # Assert
        with pytest.raises(FileExistsError):
            symlink(str(target), str(link), force=False)

    def test_force_removes_existing_file_link_is_symlink(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Target content")
        # Create existing file at link location
        link = tmp_path / "link.txt"
        # Act
        link.write_text("Existing content")
        # Without force, should raise error
        # Assert
        with pytest.raises(FileExistsError):
            symlink(str(target), str(link), force=False)
        # With force, should succeed
        symlink(str(target), str(link), force=True)
        # Act
        # Assert
        assert link.is_symlink()

    def test_force_removes_existing_file_link_read_text_target_content(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Target content")
        # Create existing file at link location
        link = tmp_path / "link.txt"
        # Act
        link.write_text("Existing content")
        # Without force, should raise error
        # Assert
        with pytest.raises(FileExistsError):
            symlink(str(target), str(link), force=False)
        # With force, should succeed
        symlink(str(target), str(link), force=True)
        # Act
        # Assert
        assert link.read_text() == "Target content"


    def test_force_removes_existing_symlink_link_read_text_target_1(self, tmp_path):
        # Arrange
        # Arrange
        target1 = tmp_path / "target1.txt"
        target1.write_text("Target 1")
        target2 = tmp_path / "target2.txt"
        target2.write_text("Target 2")
        link = tmp_path / "link.txt"
        # Create initial symlink
        # Act
        symlink(str(target1), str(link))
        # Act
        # Assert
        # Assert
        assert link.read_text() == "Target 1"

    def test_force_removes_existing_symlink_link_read_text_target_2(self, tmp_path):
        # Arrange
        # Arrange
        target1 = tmp_path / "target1.txt"
        target1.write_text("Target 1")
        target2 = tmp_path / "target2.txt"
        target2.write_text("Target 2")
        link = tmp_path / "link.txt"
        # Create initial symlink
        # Act
        symlink(str(target1), str(link))
        # Assert
        assert link.read_text() == "Target 1"
        # Force create new symlink
        symlink(str(target2), str(link), force=True)
        # Act
        # Assert
        assert link.read_text() == "Target 2"


    def test_force_with_nonexistent_link(self, tmp_path):
        """Test that force=True works when link doesn't exist."""
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")

        link = tmp_path / "link.txt"

        # Should work fine even if file doesn't exist
        # Act
        symlink(str(target), str(link), force=True)
        # Assert
        assert link.is_symlink()

    def test_force_does_not_remove_directory(self, tmp_path):
        """Test that force=True fails on existing directory."""
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")

        # Create directory at link location
        link_dir = tmp_path / "link_dir"
        # Act
        link_dir.mkdir()

        # Force should fail on directory (os.remove doesn't work on dirs)
        # Assert
        with pytest.raises(OSError):
            symlink(str(target), str(link_dir), force=True)


class TestSymlinkErrorCases:
    """Test error handling in symlink function."""

    def test_nonexistent_target_link_is_symlink(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "nonexistent.txt"
        link = tmp_path / "link.txt"
        # Should create broken symlink
        # Act
        symlink(str(target), str(link))
        # Act
        # Assert
        # Assert
        assert link.is_symlink()

    def test_nonexistent_target_not_link_exists(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "nonexistent.txt"
        link = tmp_path / "link.txt"
        # Should create broken symlink
        # Act
        symlink(str(target), str(link))
        # Act
        # Assert
        # Assert
        assert not link.exists()  # Broken symlink


    def test_existing_link_without_force(self, tmp_path):
        """Test error when link exists and force=False."""
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")

        link = tmp_path / "link.txt"
        # Act
        link.write_text("Existing")

        # Assert
        with pytest.raises(FileExistsError):
            symlink(str(target), str(link), force=False)

    def test_invalid_link_path(self, tmp_path):
        """Test error with invalid link path."""
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")

        # Try to create symlink in non-existent directory
        # Act
        link = tmp_path / "nonexistent_dir" / "link.txt"

        # Assert
        with pytest.raises(FileNotFoundError):
            symlink(str(target), str(link))

    def test_permission_error_calls_write_text(self, tmp_path):
        """Test handling of permission errors."""
        # Arrange
        # Act
        # Assert
        if os.name == "nt":
            pytest.skip("Permission test not applicable on Windows")

        target = tmp_path / "target.txt"
        target.write_text("Content")

        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, 0o444)

        try:
            link = readonly_dir / "link.txt"
            with pytest.raises(PermissionError):
                symlink(str(target), str(link))
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


class TestSymlinkSpecialCases:
    """Test special cases and edge scenarios."""

    def test_symlink_to_symlink_link2_read_text_original(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Original")
        link1 = tmp_path / "link1.txt"
        symlink(str(target), str(link1))
        link2 = tmp_path / "link2.txt"
        # Act
        symlink(str(link1), str(link2))
        # Act
        # Assert
        # Assert
        assert link2.read_text() == "Original"

    def test_symlink_to_symlink_link2_resolve_target_resolve(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Original")
        link1 = tmp_path / "link1.txt"
        symlink(str(target), str(link1))
        link2 = tmp_path / "link2.txt"
        # Act
        symlink(str(link1), str(link2))
        # Act
        # Assert
        # Assert
        assert link2.resolve() == target.resolve()


    def test_deep_nested_paths_link_read_text_deep_content(self, tmp_path):
        # Arrange
        # Arrange
        deep_path = tmp_path
        for i in range(5):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)
        target = deep_path / "target.txt"
        target.write_text("Deep content")
        link = tmp_path / "shallow_link.txt"
        # Act
        symlink(str(target), str(link))
        # Act
        # Assert
        # Assert
        assert link.read_text() == "Deep content"

    def test_deep_nested_paths_link_target_count_5(self, tmp_path):
        # Arrange
        # Arrange
        deep_path = tmp_path
        for i in range(5):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)
        target = deep_path / "target.txt"
        target.write_text("Deep content")
        link = tmp_path / "shallow_link.txt"
        # Act
        symlink(str(target), str(link))
        # Assert
        assert link.read_text() == "Deep content"
        # Check relative path goes down many levels
        link_target = os.readlink(str(link))
        # Act
        # Assert
        assert link_target.count("/") >= 5


    def test_unicode_filenames_link_is_symlink(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target_文件.txt"
        target.write_text("Unicode content")
        link = tmp_path / "link_链接.txt"
        # Act
        symlink(str(target), str(link))
        # Act
        # Assert
        # Assert
        assert link.is_symlink()

    def test_unicode_filenames_link_read_text_unicode_content(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target_文件.txt"
        target.write_text("Unicode content")
        link = tmp_path / "link_链接.txt"
        # Act
        symlink(str(target), str(link))
        # Act
        # Assert
        # Assert
        assert link.read_text() == "Unicode content"


    def test_spaces_in_paths(self, tmp_path):
        """Test symlink with spaces in paths."""
        # Arrange
        target_dir = tmp_path / "dir with spaces"
        target_dir.mkdir()
        target = target_dir / "file with spaces.txt"
        target.write_text("Spaced content")

        link = tmp_path / "link with spaces.txt"
        # Act
        symlink(str(target), str(link))

        # Assert
        assert link.read_text() == "Spaced content"


class TestSymlinkOutput:
    """Test output messages from symlink function."""

    def test_output_format_symlink_was_created_in_output(self, tmp_path, capsys):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")
        link = tmp_path / "link.txt"
        # Clear any previous output
        capsys.readouterr()
        symlink(str(target), str(link))
        captured = capsys.readouterr()
        # Act
        output = captured.out
        # Act
        # Assert
        # Assert
        assert "Symlink was created:" in output

    def test_output_format_str_link_in_output(self, tmp_path, capsys):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")
        link = tmp_path / "link.txt"
        # Clear any previous output
        capsys.readouterr()
        symlink(str(target), str(link))
        captured = capsys.readouterr()
        # Act
        output = captured.out
        # Act
        # Assert
        # Assert
        assert str(link) in output

    def test_output_format_in_output(self, tmp_path, capsys):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")
        link = tmp_path / "link.txt"
        # Clear any previous output
        capsys.readouterr()
        symlink(str(target), str(link))
        captured = capsys.readouterr()
        # Act
        output = captured.out
        # Act
        # Assert
        # Assert
        assert "->" in output

    def test_output_format_target_txt_in_output(self, tmp_path, capsys):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Content")
        link = tmp_path / "link.txt"
        # Clear any previous output
        capsys.readouterr()
        symlink(str(target), str(link))
        captured = capsys.readouterr()
        # Act
        output = captured.out
        # Act
        # Assert
        # Assert
        assert "target.txt" in output


    def test_output_shows_relative_path(self, tmp_path, capsys):
        """Test that output shows the relative path used."""
        # Arrange
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        target = tmp_path / "a" / "target.txt"
        target.write_text("Content")

        link = tmp_path / "b" / "link.txt"

        capsys.readouterr()  # Clear
        symlink(str(target), str(link))

        # Act
        output = capsys.readouterr().out
        # Should show relative path
        # Assert
        assert "../a/target.txt" in output


class TestSymlinkCrossPlatform:
    """Test cross-platform compatibility."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix-specific test")
    def test_unix_symlink_properties_os_path_islink_str_link(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Unix test")
        link = tmp_path / "link.txt"
        # Act
        symlink(str(target), str(link))
        # Act
        # Assert
        # Assert
        assert os.path.islink(str(link))

    @pytest.mark.skipif(os.name == "nt", reason="Unix-specific test")
    def test_unix_symlink_properties_target_read_text_modified(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Unix test")
        link = tmp_path / "link.txt"
        # Act
        symlink(str(target), str(link))
        # Check it's actually a symlink (not a copy)
        # Assert
        assert os.path.islink(str(link))
        # Modifying through symlink should modify target
        link.write_text("Modified")
        # Act
        # Assert
        assert target.read_text() == "Modified"


    def test_path_separator_handling_link_is_symlink(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Sep test")
        # Use forward slashes even on Windows
        link_path = str(tmp_path) + "/link.txt"
        symlink(str(target), link_path)
        # Act
        link = Path(link_path)
        # Act
        # Assert
        # Assert
        assert link.is_symlink()

    def test_path_separator_handling_link_read_text_sep_test(self, tmp_path):
        # Arrange
        # Arrange
        target = tmp_path / "target.txt"
        target.write_text("Sep test")
        # Use forward slashes even on Windows
        link_path = str(tmp_path) + "/link.txt"
        symlink(str(target), link_path)
        # Act
        link = Path(link_path)
        # Act
        # Assert
        # Assert
        assert link.read_text() == "Sep test"



# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_symlink.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:29:31 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_symlink.py
#
# import os
# from scitex.str._color_text import color_text
#
#
# def symlink(tgt, src, force=False):
#     """Create a symbolic link.
#
#     This function creates a symbolic link from the target to the source.
#     If the force parameter is True, it will remove any existing file at
#     the source path before creating the symlink.
#
#     Parameters
#     ----------
#     tgt : str
#         The target path (the file or directory to be linked to).
#     src : str
#         The source path (where the symbolic link will be created).
#     force : bool, optional
#         If True, remove the existing file at the src path before creating
#         the symlink (default is False).
#
#     Returns
#     -------
#     None
#
#     Raises
#     ------
#     OSError
#         If the symlink creation fails.
#
#     Example
#     -------
#     >>> symlink('/path/to/target', '/path/to/link')
#     >>> symlink('/path/to/target', '/path/to/existing_file', force=True)
#     """
#     if force:
#         try:
#             os.remove(src)
#         except FileNotFoundError:
#             pass
#
#     # Calculate the relative path from src to tgt
#     src_dir = os.path.dirname(src)
#     relative_tgt = os.path.relpath(tgt, src_dir)
#
#     os.symlink(relative_tgt, src)
#     print(color_text(f"\nSymlink was created: {src} -> {relative_tgt}\n", c="yellow"))
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_symlink.py
# --------------------------------------------------------------------------------
