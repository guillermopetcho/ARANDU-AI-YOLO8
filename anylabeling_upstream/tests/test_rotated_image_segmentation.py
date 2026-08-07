import os
import unittest
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QTransform
from PyQt6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img


class TestRotatedImageSegmentation(unittest.TestCase):

    def test_qt_img_to_rgb_cv_img_preserves_rotation(self):
        # Create a test image: 100 wide x 50 high
        qimg = QImage(100, 50, QImage.Format.Format_RGB888)
        qimg.fill(Qt.GlobalColor.blue)

        self.assertEqual(qimg.width(), 100)
        self.assertEqual(qimg.height(), 50)

        # Rotate image 90 degrees clockwise -> new dimensions: 50 wide x 100 high
        transform = QTransform().rotate(90)
        rotated_qimg = qimg.transformed(transform)

        self.assertEqual(rotated_qimg.width(), 50)
        self.assertEqual(rotated_qimg.height(), 100)

        # Pass rotated image to qt_img_to_rgb_cv_img with a dummy file path
        cv_img = qt_img_to_rgb_cv_img(rotated_qimg, img_path="non_existent.jpg")

        # Verify OpenCV matrix shape matches the ROTATED dimensions (H=100, W=50, C=3)
        self.assertEqual(cv_img.shape[0], 100)
        self.assertEqual(cv_img.shape[1], 50)
        self.assertEqual(cv_img.shape[2], 3)


if __name__ == '__main__':
    unittest.main()
