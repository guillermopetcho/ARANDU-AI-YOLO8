import unittest
import sys
import os
import shutil
import tempfile
import PIL.Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.augmentation_engine import (
    AugmentationEngine,
    AugmentationPreset,
)


class TestAugmentationEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.pil_img = PIL.Image.new("RGB", (100, 100), color="blue")
        self.shapes = [
            {
                "label": "cat",
                "shape_type": "polygon",
                "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
            },
            {
                "label": "dog",
                "shape_type": "rectangle",
                "points": [[60.0, 60.0], [90.0, 90.0]],
            },
        ]

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_presets_exist(self):
        cfg_light = AugmentationEngine.get_default_config(AugmentationPreset.LIGHT)
        cfg_heavy = AugmentationEngine.get_default_config(AugmentationPreset.HEAVY)
        cfg_std = AugmentationEngine.get_default_config(AugmentationPreset.ROBOFLOW_STANDARD)

        self.assertEqual(cfg_light["preset"], AugmentationPreset.LIGHT)
        self.assertEqual(cfg_heavy["preset"], AugmentationPreset.HEAVY)
        self.assertEqual(cfg_std["preset"], AugmentationPreset.ROBOFLOW_STANDARD)

    def test_apply_transformations(self):
        engine = AugmentationEngine(AugmentationEngine.get_default_config(AugmentationPreset.ROBOFLOW_STANDARD))
        aug_img, aug_shapes = engine.apply_transformations(self.pil_img, self.shapes)

        self.assertIsNotNone(aug_img)
        self.assertIsInstance(aug_shapes, list)
        self.assertGreaterEqual(len(aug_shapes), 1)

        # Check transformed polygon points are within [0, 100]
        for shape in aug_shapes:
            for p in shape.get("points", []):
                self.assertGreaterEqual(p[0], 0.0)
                self.assertLessEqual(p[0], 100.0)
                self.assertGreaterEqual(p[1], 0.0)
                self.assertLessEqual(p[1], 100.0)


if __name__ == "__main__":
    unittest.main()
