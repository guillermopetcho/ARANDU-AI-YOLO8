import os
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from anylabeling.services.auto_labeling.model_manager import ModelManager


class TestAutoLabelingRobustness(unittest.TestCase):

    def test_concurrent_prediction_reentry_blocked(self):
        mm = ModelManager()
        mm.loaded_model_config = {"model": MagicMock()}

        # Set atomic predicting flag
        mm._is_predicting = True

        # Call predict_shapes_threading while predicting
        with patch.object(mm, "prediction_started") as mock_started:
            mm.predict_shapes_threading(image=MagicMock(), filename="test.jpg")
            # Should NOT emit prediction_started because re-entry was blocked safely
            mock_started.emit.assert_not_called()

        # Reset flag
        mm._is_predicting = False

    def test_canvas_synchronous_lock_on_marks(self):
        from anylabeling.views.labeling.widgets.canvas import Canvas

        with patch.object(Canvas, '__init__', lambda self, *a, **k: None):
            canvas = Canvas()
            canvas.is_loading = False
            canvas.shapes = []
            canvas.epsilon = 10.0
            canvas.scale = 1.0
            canvas.auto_labeling_marks_updated = MagicMock()
            canvas.set_loading = MagicMock()
            canvas.tr = lambda s: s

            # Mock a mark shape
            mark_shape = MagicMock()
            mark_shape.label = "AUTOLABEL_ADD"
            mark_shape.shape_type = "point"
            mark_shape.points = [MagicMock(x=lambda: 10, y=lambda: 20)]
            canvas.shapes = [mark_shape]

            canvas.update_auto_labeling_marks()

            # set_loading should have been called synchronously with True
            canvas.set_loading.assert_called_with(True, "Inferenciando modelo de IA...")

    def test_escape_key_deselects_shape_and_accepts_event(self):
        from anylabeling.views.labeling.widgets.canvas import Canvas
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import QEvent, Qt

        with patch.object(Canvas, '__init__', lambda self, *a, **k: None):
            canvas = Canvas()
            canvas.is_loading = False
            canvas.selected_shapes = [MagicMock()]
            canvas.deselect_shape = MagicMock()
            canvas.drawing = MagicMock(return_value=False)

            event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
            canvas.keyPressEvent(event)

            canvas.deselect_shape.assert_called_once()
            self.assertTrue(event.isAccepted())

    def test_stop_inference_resets_state(self):
        mm = ModelManager()
        mock_model = MagicMock()
        mm.loaded_model_config = {"model": mock_model}
        mm._is_predicting = True

        mm.stop_inference()

        self.assertFalse(mm._is_predicting)
        self.assertTrue(mock_model.stop_inference)
        mock_model.set_auto_labeling_marks.assert_called_with([])

    def test_early_return_emits_prediction_finished(self):
        mm = ModelManager()
        mm.loaded_model_config = {"model": MagicMock()}
        mm._is_predicting = True

        with patch.object(mm, "prediction_finished") as mock_finished:
            mm.predict_shapes_threading(image=MagicMock(), filename="test.jpg")
            mock_finished.emit.assert_called_once()

        mm._is_predicting = False


if __name__ == '__main__':
    unittest.main()


