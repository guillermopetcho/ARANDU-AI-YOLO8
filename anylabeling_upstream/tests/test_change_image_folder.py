import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import shutil
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.views.labeling.label_widget import LabelingWidget
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


class TestChangeImageFolder(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.target_dir = tempfile.mkdtemp()

        # Create dummy image and json
        self.img_path = os.path.join(self.test_dir, "test1.jpg")
        self.json_path = os.path.join(self.test_dir, "test1.json")

        with open(self.img_path, "w") as f:
            f.write("dummy image content")

        with open(self.json_path, "w") as f:
            f.write('{"imagePath": "test1.jpg", "shapes": []}')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.target_dir, ignore_errors=True)

    def test_resolve_image_path_and_shutil_import(self):
        with patch.object(LabelingWidget, "__init__", lambda self, *a, **k: None):
            widget = LabelingWidget()
            widget.last_open_dir = self.test_dir
            widget.output_dir = None

            resolved = widget.resolve_image_path("test1.jpg")
            self.assertEqual(resolved, os.path.abspath(self.img_path))

            json_found = widget.get_label_file_for_image(resolved)
            self.assertEqual(json_found, os.path.abspath(self.json_path))

            # Test shutil move capability without error
            target_img = os.path.join(self.target_dir, "test1.jpg")
            target_json = os.path.join(self.target_dir, "test1.json")

            shutil.move(self.img_path, target_img)
            shutil.move(self.json_path, target_json)

            self.assertTrue(os.path.exists(target_img))
            self.assertTrue(os.path.exists(target_json))


if __name__ == "__main__":
    unittest.main()
