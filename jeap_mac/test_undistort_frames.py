"""Unit tests for the geometric rectification interface; no VRS/data downloads needed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
from types import ModuleType

import numpy as np
from PIL import Image

from undistort_frames import list_frames, rectify_one, source_calibration


class FakeCamera:
    def get_image_size(self):
        return (100, 100)
    def rescale(self, size, factor):
        assert size == [50, 50]
        assert abs(factor - .5) < 1e-6
        return self


class FakeDevice:
    def get_camera_calib(self, label):
        return FakeCamera() if label == "camera-rgb" else None


class FakeProvider:
    def get_device_calibration(self):
        return FakeDevice()


class FakeDataProvider:
    @staticmethod
    def create_vrs_data_provider(_):
        return FakeProvider()


class FakeCalibration:
    @staticmethod
    def distort_by_calibration(arr, target, source, interpolation):
        return arr.copy()


class RectificationTests(unittest.TestCase):
    def test_list_frames_preserves_numeric_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in (10, 2, 0):
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(d / f"frame_{i:07d}.jpg")
            self.assertEqual([p.stem for p in list_frames(d)],
                             ["frame_0000000", "frame_0000002", "frame_0000010"])

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "frame_0000001.jpg").touch()
            (d / "frame_0000001.png").touch()
            with self.assertRaisesRegex(ValueError, "duplicati"):
                list_frames(d)

    def test_rectify_one_accepts_valid_shape(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        actual = rectify_one(rgb, object(), object(), FakeCalibration(), object())
        np.testing.assert_array_equal(rgb, actual)
        self.assertEqual(actual.dtype, np.uint8)

    def test_camera_aspect_ratio_guard(self):
        fake_root = ModuleType("projectaria_tools")
        fake_core = ModuleType("projectaria_tools.core")
        fake_core.data_provider = FakeDataProvider()
        fake_core.calibration = FakeCalibration()
        with tempfile.TemporaryDirectory() as d, patch.dict(
            sys.modules, {
                "projectaria_tools": fake_root,
                "projectaria_tools.core": fake_core,
            }
        ):
            vrs = Path(d) / "calibration.vrs"
            vrs.touch()
            cam, _ = source_calibration(vrs, (50, 50), "camera-rgb")
            self.assertIsInstance(cam, FakeCamera)
            with self.assertRaisesRegex(ValueError, "aspect ratio"):
                source_calibration(vrs, (50, 40), "camera-rgb")

    def test_missing_vrs_fails(self):
        fake_root = ModuleType("projectaria_tools")
        fake_core = ModuleType("projectaria_tools.core")
        fake_core.data_provider = FakeDataProvider()
        fake_core.calibration = FakeCalibration()
        with patch.dict(sys.modules, {"projectaria_tools": fake_root, "projectaria_tools.core": fake_core}):
            with self.assertRaises(FileNotFoundError):
                source_calibration(Path("/not/a/real/file.vrs"), (50, 50), "camera-rgb")


if __name__ == "__main__":
    unittest.main()
