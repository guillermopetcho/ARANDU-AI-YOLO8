import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication, QDialog
app = QApplication.instance() or QApplication(sys.argv)

from anylabeling.views.labeling.widgets.label_inspector_dialog import LabelInspectorDialog


class TestDatasetInspector(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = self.temp_dir.name

        # Create two sample JSON files
        self.json1 = os.path.join(self.folder, "img_01.json")
        self.json2 = os.path.join(self.folder, "img_02.json")

        data1 = {
            "shapes": [
                {"label": "hoja_sana", "points": [[10, 10], [50, 50]], "shape_type": "polygon"},
                {"label": "plaga", "points": [[100, 100], [150, 150]], "shape_type": "rectangle"}
            ]
        }
        data2 = {
            "shapes": [
                {"label": "hoja_sana", "points": [[20, 20], [60, 60]], "shape_type": "polygon"},
                {"label": "hoja_sana", "points": [[70, 70], [90, 90]], "shape_type": "polygon"}
            ]
        }

        with open(self.json1, "w", encoding="utf-8") as f:
            json.dump(data1, f)
        with open(self.json2, "w", encoding="utf-8") as f:
            json.dump(data2, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_scope_scanning_and_relabeling(self):
        parent_mock = MagicMock()
        parent_mock.last_open_dir = self.folder
        parent_mock.filename = self.json1
        parent_mock.canvas.shapes = []

        dialog = LabelInspectorDialog(parent_mock, target_label="hoja_sana")
        
        # Switch to dataset scope
        dialog.radio_dataset_folder.setChecked(True)
        dialog.on_scope_changed()

        # Should find 3 "hoja_sana" objects across the 2 JSON files
        self.assertEqual(dialog.object_list.count(), 3)

        # Select all items
        dialog.select_all_items()

        # Mock SelectRegisteredLabelDialog & QMessageBox to run non-interactively
        with patch("anylabeling.views.labeling.widgets.label_inspector_dialog.SelectRegisteredLabelDialog") as mock_dlg, \
             patch("anylabeling.views.labeling.widgets.label_inspector_dialog.QMessageBox"):
            mock_inst = MagicMock()
            mock_inst.exec.return_value = QDialog.DialogCode.Accepted
            mock_inst.get_selected_label.return_value = "hoja_trada"
            mock_dlg.return_value = mock_inst

            dialog.relabel_selected_items()

        # Check that JSON files were updated
        with open(self.json1, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        with open(self.json2, "r", encoding="utf-8") as f:
            d2 = json.load(f)

        self.assertEqual(d1["shapes"][0]["label"], "hoja_trada")
        self.assertEqual(d2["shapes"][0]["label"], "hoja_trada")
        self.assertEqual(d2["shapes"][1]["label"], "hoja_trada")


if __name__ == '__main__':
    unittest.main()
