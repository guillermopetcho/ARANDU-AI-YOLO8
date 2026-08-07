import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import shutil
import tempfile
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.widgets.dataset_gallery_dialog import (
    DatasetGalleryDialog,
    GalleryThumbnailItem,
)


class TestDatasetGalleryAndAudit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Image 1: 0 shapes
        self.img1 = os.path.join(self.test_dir, "img1.jpg")
        with open(self.img1, "w") as f:
            f.write("dummy")

        # Image 2: 2 shapes
        self.img2 = os.path.join(self.test_dir, "img2.jpg")
        with open(self.img2, "w") as f:
            f.write("dummy")

        self.json2 = os.path.join(self.test_dir, "img2.json")
        with open(self.json2, "w") as f:
            json.dump({
                "shapes": [
                    {"label": "cat", "shape_type": "polygon", "points": [[10, 10], [20, 10], [20, 20]]},
                    {"label": "dog", "shape_type": "rectangle", "points": [[30, 30], [50, 50]]}
                ]
            }, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_filter_images_by_shape_count(self):
        with patch.object(LabelingWidget, "__init__", lambda self, *a, **k: None):
            widget = LabelingWidget()
            widget.last_open_dir = self.test_dir
            widget.output_dir = None
            widget.file_list_widget = MagicMock()
            widget.statusBar = MagicMock()
            widget.resolve_image_path = lambda p: os.path.abspath(os.path.join(self.test_dir, p))
            widget.get_label_file_for_image = lambda p: os.path.splitext(p)[0] + ".json"
            widget.scan_all_images = lambda d: [self.img1, self.img2]

            # Filter 0 labels -> should match img1
            widget.filter_images_by_shape_count(0)
            widget.file_list_widget.addItem.assert_called()

    def test_gallery_thumbnail_item(self):
        item = GalleryThumbnailItem(self.img2, shape_count=2, label_names=["cat", "dog"])
        self.assertEqual(item.image_path, self.img2)
        self.assertEqual(item.shape_count, 2)
        self.assertIn("cat", item.label_names)


if __name__ == "__main__":
    unittest.main()
