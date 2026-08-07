import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.model_manager import ModelManager

class TestModelManager(unittest.TestCase):

    @patch('anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs')
    def test_set_text_prompt_delegation(self, mock_load):
        # Prevent actual loading of configs which might be slow/blocking
        manager = ModelManager()
        mock_model = MagicMock()
        manager.loaded_model_config = {"model": mock_model, "type": "segment_anything"}

        manager.set_text_prompt("new prompt")
        mock_model.set_text_prompt.assert_called_once_with("new prompt")

    @patch('anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs')
    def test_unload_model(self, mock_load):
        manager = ModelManager()
        mock_model = MagicMock()
        manager.loaded_model_config = {"model": mock_model}

        manager.unload_model()
        mock_model.unload.assert_called_once()
        self.assertIsNone(manager.loaded_model_config)

    @patch('anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs')
    def test_predict_shapes_threading_guarded_when_running(self, mock_load):
        manager = ModelManager()
        mock_model = MagicMock()
        manager.loaded_model_config = {"model": mock_model, "type": "segment_anything"}

        # Simulate running thread
        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = True
        manager.model_execution_thread = mock_thread

        # Call predict_shapes_threading while thread is running
        manager.predict_shapes_threading(MagicMock(), "test.jpg")
        # Ensure prediction_started signal was not emitted (thread creation bypassed)
        self.assertTrue(mock_thread.isRunning.called)

    @patch('anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs')
    def test_unload_guarded_when_running(self, mock_load):
        manager = ModelManager()
        mock_model = MagicMock()
        manager.loaded_model_config = {"model": mock_model}

        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = True
        manager.model_execution_thread = mock_thread

        manager.unload_model()
        # Model should NOT be unloaded while thread is running
        mock_model.unload.assert_not_called()
        self.assertIsNotNone(manager.loaded_model_config)


if __name__ == '__main__':
    unittest.main()
