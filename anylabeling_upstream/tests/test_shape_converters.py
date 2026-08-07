import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from anylabeling.views.labeling.shape import Shape
from PyQt6.QtCore import QPointF


class TestShapeConverters(unittest.TestCase):

    def test_polygon_to_rectangle_conversion(self):
        from anylabeling.views.labeling.widgets.canvas import Canvas

        with patch.object(Canvas, '__init__', lambda self, *a, **k: None):
            canvas = Canvas()
            canvas.selected_shapes = []
            canvas.store_shapes = MagicMock()
            canvas.update = MagicMock()

            # Create a 4-point polygon
            poly = Shape(label="leaf", shape_type="polygon")
            poly.add_point(QPointF(10.0, 20.0))
            poly.add_point(QPointF(100.0, 20.0))
            poly.add_point(QPointF(100.0, 80.0))
            poly.add_point(QPointF(10.0, 80.0))

            canvas.shapes = [poly]

            count = canvas.convert_to_rectangles()
            self.assertEqual(count, 1)
            self.assertEqual(canvas.shapes[0].shape_type, "rectangle")
            self.assertEqual(len(canvas.shapes[0].points), 2)

    def test_rectangle_to_polygon_conversion(self):
        from anylabeling.views.labeling.widgets.canvas import Canvas

        with patch.object(Canvas, '__init__', lambda self, *a, **k: None):
            canvas = Canvas()
            canvas.selected_shapes = []
            canvas.store_shapes = MagicMock()
            canvas.update = MagicMock()

            # Create a 2-point rectangle
            rect = Shape(label="plant", shape_type="rectangle")
            rect.add_point(QPointF(5.0, 5.0))
            rect.add_point(QPointF(50.0, 50.0))

            canvas.shapes = [rect]

            count = canvas.convert_to_polygons()
            self.assertEqual(count, 1)
            self.assertEqual(canvas.shapes[0].shape_type, "polygon")
            self.assertEqual(len(canvas.shapes[0].points), 4)


if __name__ == '__main__':
    unittest.main()
