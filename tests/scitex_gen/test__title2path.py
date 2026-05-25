import pytest

pytest.importorskip("torch")

from scitex_gen import title2path


class TestTitle2Path:
    """Test the title2path function on string inputs."""

    @pytest.mark.parametrize(
        "title, expected",
        [
            ("hello world", "hello_world"),
            ("test:file;name=value[0]", "testfilenamevalue0"),
            ("test file name", "test_file_name"),
            ("test___file", "test_file"),
            ("test____file", "test_file"),
            ("test_-_file", "test-file"),
            ("TEST FILE", "test_file"),
            ("TestFile", "testfile"),
            ("Test:File[1];Name=Value _-_ End", "testfile1namevalue_-_end"),
            ("", ""),
            ("::;;==[[]]", ""),
            ("Model: ResNet50; Epochs=100 [Batch: 32]", "model_resnet50_epochs100_batch_32"),
            ("Deep Learning: A Review [2023]", "deep_learning_a_review_2023"),
            (
                "config: lr=0.001; batch_size=32; optimizer=adam",
                "config_lr0.001_batch_size32_optimizeradam",
            ),
            ("data/train/images [processed]", "data/train/images_processed"),
            ("folder/subfolder/file", "folder/subfolder/file"),
            ("test    file", "test_file"),
            ("test file", "test_file"),
            ("café: résumé", "café_résumé"),
            ("test123file456", "test123file456"),
            ("v2.0: release[final]", "v2.0_releasefinal"),
            ("a_-_b_-_c", "a-b-c"),
            ("_-_start", "-start"),
            ("end_-_", "end-"),
            ("a________b", "a_b"),
            ("ResNet50: ImageNet [Epoch: 100]", "resnet50_imagenet_epoch_100"),
            ("BERT Fine-tuning; Task=NER", "bert_fine-tuning_taskner"),
            (
                "GAN Training [Discriminator Loss = 0.5]",
                "gan_training_discriminator_loss_0.5",
            ),
        ],
    )
    def test_title2path_converts_string_to_expected_path(self, title, expected):
        # Arrange
        # Act
        result = title2path(title)
        # Assert
        assert result == expected

    def test_title2path_converts_dict_via_real_to_str(self):
        # title2path delegates dict→str conversion to the real scitex_dict.to_str
        # collaborator; we exercise it end-to-end with no mock.
        # Arrange
        test_dict = {"model": "resnet50", "epochs": 100}
        # Act
        result = title2path(test_dict)
        # Assert
        assert result == "model-resnet50_epochs-100"


class TestTitle2PathFilenameSafety:
    """Verify the output is suitable for filenames."""

    @pytest.mark.parametrize("forbidden", [":", ";", "=", "[", "]"])
    def test_title2path_removes_forbidden_characters(self, forbidden):
        # Arrange
        problematic = "file:name*with?invalid<chars>;a=b[c]"
        # Act
        result = title2path(problematic)
        # Assert
        assert forbidden not in result

    def test_title2path_result_is_lowercase(self):
        # Arrange
        problematic = "File:Name=Value[0]"
        # Act
        result = title2path(problematic)
        # Assert
        assert result == result.lower()

    def test_title2path_collapses_consecutive_underscores(self):
        # Arrange
        problematic = "file   name    here"
        # Act
        result = title2path(problematic)
        # Assert
        assert "__" not in result


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_title2path.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: 2024-05-12 21:02:21 (7)
# # /sshx:ywatanabe@444:/home/ywatanabe/proj/scitex/src/scitex/gen/_title2spath.py
#
#
# def title2path(title):
#     """
#     Convert a title (string or dictionary) to a path-friendly string.
#
#     Parameters
#     ----------
#     title : str or dict
#         The input title to be converted.
#
#     Returns
#     -------
#     str
#         A path-friendly string derived from the input title.
#     """
#     if isinstance(title, dict):
#         from scitex_dict import to_str
#
#         title = to_str(title)
#
#     path = title
#
#     patterns = [":", ";", "=", "[", "]"]
#     for pattern in patterns:
#         path = path.replace(pattern, "")
#
#     path = path.replace("_-_", "-")
#     path = path.replace(" ", "_")
#
#     while "__" in path:
#         path = path.replace("__", "_")
#
#     return path.lower()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_title2path.py
# --------------------------------------------------------------------------------
