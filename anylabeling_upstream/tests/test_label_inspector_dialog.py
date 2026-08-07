import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.views.labeling.shape import Shape
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor


class TestLabelInspectorDialog(unittest.TestCase):

    def test_inspector_dialog_populates_shapes(self):
        from anylabeling.views.labeling.widgets.label_inspector_dialog import LabelInspectorDialog
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        # Mock parent widget
        parent_widget = MagicMock()
        shape1 = Shape(label="cat", shape_type="polygon")
        shape1.add_point(QPointF(10, 10))
        shape1.add_point(QPointF(20, 20))

        shape2 = Shape(label="dog", shape_type="rectangle")
        shape2.add_point(QPointF(30, 30))
        shape2.add_point(QPointF(40, 40))

        parent_widget.canvas.shapes = [shape1, shape2]

        dialog = LabelInspectorDialog(parent_widget, target_label="cat")
        self.assertEqual(dialog.object_list.count(), 1)
        item = dialog.object_list.item(0)
        self.assertIn("cat", item.text())


if __name__ == '__main__':
    unittest.main()
