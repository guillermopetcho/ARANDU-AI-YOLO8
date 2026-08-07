import os.path

import cv2
import numpy as np
import qimage2ndarray
from PyQt6 import QtGui
from PyQt6.QtGui import QImage


def qt_img_to_rgb_cv_img(qt_img, img_path=None):
    """
    Convert 8bit/16bit RGB image or 8bit/16bit Gray image to 8bit RGB image.
    Prioritizes in-memory qt_img so that any rotations, flips, or edits are respected.
    Safely handles missing, invalid, or corrupted images without raising exceptions.
    """
    cv_image = None
    if qt_img is not None and hasattr(qt_img, "isNull") and not qt_img.isNull():
        try:
            if qt_img.format() not in (
                QImage.Format.Format_RGB32,
                QImage.Format.Format_ARGB32,
                QImage.Format.Format_ARGB32_Premultiplied,
            ):
                qt_img = qt_img.convertToFormat(QImage.Format.Format_RGB32)
            cv_image = qimage2ndarray.rgb_view(qt_img).copy()
        except Exception:
            cv_image = None
    elif img_path is not None and os.path.exists(img_path):
        try:
            cv_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if cv_image is not None and len(cv_image.shape) >= 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                cv_image = None
        except Exception:
            cv_image = None

    if cv_image is None or getattr(cv_image, "size", 0) == 0:
        return None

    try:
        # To uint8
        if cv_image.dtype != np.uint8:
            cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
            cv_image = np.array(cv_image, dtype=np.uint8)
        # To RGB
        if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
            cv_image = cv2.merge([cv_image, cv_image, cv_image])
        return cv_image
    except Exception:
        return None


def qt_img_to_cv_img(in_image):
    return qimage2ndarray.rgb_view(in_image)


def cv_img_to_qt_img(in_mat):
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat))
