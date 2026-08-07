import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.views.labeling.shape import Shape
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor


class TestBatchLabelEditAndTools(unittest.TestCase):

    def test_merge_polygons(self):
        from anylabeling.views.labeling.widgets.canvas import Canvas

        # Mock Qt canvas widget
        with patch.object(Canvas, '__init__', lambda self, *a, **k: None):
            canvas = Canvas()
            canvas.shapes = []
            canvas.selected_shapes = []
            canvas.store_shapes = MagicMock()
            canvas.new_shape = MagicMock()
            canvas.update = MagicMock()

            # Create 2 overlapping square shapes
            shape1 = Shape(label="dog", shape_type="polygon")
            shape1.add_point(QPointF(10, 10))
            shape1.add_point(QPointF(50, 10))
            shape1.add_point(QPointF(50, 50))
            shape1.add_point(QPointF(10, 50))
            shape1.fill_color = QColor("#000000")
            shape1.line_color = QColor("#000000")

            shape2 = Shape(label="dog", shape_type="polygon")
            shape2.add_point(QPointF(40, 10))
            shape2.add_point(QPointF(80, 10))
            shape2.add_point(QPointF(80, 50))
            shape2.add_point(QPointF(40, 50))
            shape2.fill_color = QColor("#000000")
            shape2.line_color = QColor("#000000")

            canvas.shapes = [shape1, shape2]
            canvas.selected_shapes = [shape1, shape2]

            merged = canvas.merge_selected_polygons()
            self.assertIsNotNone(merged)
            self.assertEqual(len(canvas.shapes), 1)
            self.assertEqual(canvas.shapes[0].label, "dog")


if __name__ == '__main__':
    unittest.main()
