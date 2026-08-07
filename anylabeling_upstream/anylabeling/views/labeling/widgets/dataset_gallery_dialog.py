"""
Roboflow-Style Visual Dataset Gallery & Audit Dialog for AnyLabeling
Renders image thumbnails with bounding boxes and polygon overlays, allowing
interactive dataset overview, filtering by label count (0 or < N), and click-to-edit navigation.
"""

import os
import json
import logging
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from .. import utils
from ..logger import logger


class GalleryThumbnailItem(QtWidgets.QListWidgetItem):
    """Custom QListWidgetItem holding image path and shape count metadata."""

    def __init__(self, image_path, shape_count=0, label_names=None):
        super().__init__()
        self.image_path = image_path
        self.shape_count = shape_count
        self.label_names = label_names or []
        self.setText(f"{os.path.basename(image_path)}\n({shape_count} etiquetas)")


class DatasetGalleryDialog(QtWidgets.QDialog):
    """Roboflow-style Visual Grid Gallery and Dataset Audit Dialog."""

    def __init__(self, main_widget, parent=None):
        super().__init__(parent or main_widget)
        self.main_widget = main_widget
        self.setWindowTitle(self.tr("🖼️ Galería Visual del Dataset (Estilo Roboflow)"))
        self.resize(980, 720)

        self.thumbnail_size = 180
        self.all_items_data = []  # List of dicts: {path, name, shape_count, labels, json_path}

        self.init_ui()
        self.scan_dataset()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # Header Info
        header_label = QtWidgets.QLabel(
            "<b>🖼️ Galería Visual del Dataset & Auditoría (Estilo Roboflow / CVAT)</b><br>"
            "<small>Visualiza las imágenes cargadas con sus polígonos y rectángulos superpuestos en miniatura. "
            "Haz clic sobre cualquier miniatura para editarla directamente en el lienzo.</small>"
        )
        header_label.setWordWrap(True)
        main_layout.addWidget(header_label)

        # Filter Bar Box
        filter_box = QtWidgets.QGroupBox("Filtros de Auditoría y Búsqueda")
        filter_layout = QtWidgets.QHBoxLayout(filter_box)
        filter_layout.setContentsMargins(10, 8, 10, 8)
        filter_layout.setSpacing(8)

        # Search by filename
        self.edit_search = QtWidgets.QLineEdit()
        self.edit_search.setPlaceholderText("🔍 Buscar por nombre de archivo...")
        self.edit_search.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.edit_search)

        # Filter by Label
        self.combo_label_filter = QtWidgets.QComboBox()
        self.combo_label_filter.addItem("Todas las Etiquetas", "")
        self.combo_label_filter.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.combo_label_filter)

        # Filter by Shape Count (All, 0 Unannotated, < N)
        self.combo_count_filter = QtWidgets.QComboBox()
        self.combo_count_filter.addItem("Cualquier cantidad de etiquetas", "all")
        self.combo_count_filter.addItem("⚠️ Solo sin etiquetas (0)", "zero")
        self.combo_count_filter.addItem("🔢 Menos de 3 etiquetas (< 3)", "less_3")
        self.combo_count_filter.addItem("🔢 Menos de 5 etiquetas (< 5)", "less_5")
        self.combo_count_filter.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.combo_count_filter)

        # Slider for Thumbnail Size
        lbl_size = QtWidgets.QLabel("Tamaño Miniaturas:")
        self.slider_size = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setRange(120, 280)
        self.slider_size.setValue(self.thumbnail_size)
        self.slider_size.valueChanged.connect(self.on_thumbnail_size_changed)

        filter_layout.addWidget(lbl_size)
        filter_layout.addWidget(self.slider_size)

        main_layout.addWidget(filter_box)

        # Status Summary Label
        self.lbl_status = QtWidgets.QLabel("Escaneando dataset...")
        self.lbl_status.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        main_layout.addWidget(self.lbl_status)

        # Thumbnail Grid Widget (QListWidget in IconMode)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(self.thumbnail_size, self.thumbnail_size))
        self.list_widget.setGridSize(QtCore.QSize(self.thumbnail_size + 24, self.thumbnail_size + 44))
        self.list_widget.setSpacing(10)
        self.list_widget.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QtWidgets.QListView.Movement.Static)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QListWidget::item {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item:hover {
                background-color: #313244;
                border: 1px solid #89b4fa;
            }
            QListWidget::item:selected {
                background-color: #45475a;
                border: 2px solid #a6e3a1;
                color: #ffffff;
            }
        """)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.list_widget.itemClicked.connect(self.on_item_clicked)

        main_layout.addWidget(self.list_widget)

        # Bottom Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_open_selected = QtWidgets.QPushButton("👁️ Abrir Imagen Seleccionada en Lienzo")
        btn_open_selected.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 6px 14px;")
        btn_open_selected.clicked.connect(self.open_current_selected_image)

        btn_close = QtWidgets.QPushButton("Cerrar")
        btn_close.clicked.connect(self.reject)

        btn_layout.addWidget(btn_open_selected)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)

        main_layout.addLayout(btn_layout)

    def scan_dataset(self):
        """Scan open dataset images and build thumbnail database."""
        image_list = []
        if hasattr(self.main_widget, "file_list_widget") and self.main_widget.file_list_widget.count() > 0:
            for i in range(self.main_widget.file_list_widget.count()):
                item = self.main_widget.file_list_widget.item(i)
                if item:
                    resolved = self.main_widget.resolve_image_path(item.text())
                    if resolved and resolved not in image_list:
                        image_list.append(resolved)

        if not image_list and hasattr(self.main_widget, "image_list") and self.main_widget.image_list:
            image_list = list(self.main_widget.image_list)

        self.all_items_data = []
        unique_labels = set()

        for img_path in image_list:
            json_path = self.main_widget.get_label_file_for_image(img_path)
            shape_count = 0
            labels = []

            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    shapes = data.get("shapes", [])
                    shape_count = len(shapes)
                    for s in shapes:
                        lbl = s.get("label")
                        if lbl:
                            labels.append(lbl)
                            unique_labels.add(lbl)
                except Exception:
                    pass

            self.all_items_data.append({
                "path": img_path,
                "name": os.path.basename(img_path),
                "shape_count": shape_count,
                "labels": labels,
                "json_path": json_path,
            })

        # Update label combo
        self.combo_label_filter.blockSignals(True)
        self.combo_label_filter.clear()
        self.combo_label_filter.addItem("Todas las Etiquetas", "")
        for lbl in sorted(unique_labels):
            self.combo_label_filter.addItem(f"🏷️ {lbl}", lbl)
        self.combo_label_filter.blockSignals(False)

        self.apply_filters()

    def apply_filters(self):
        """Filter thumbnails by search text, label name, and shape count threshold."""
        search_text = self.edit_search.text().strip().lower()
        label_filter = self.combo_label_filter.currentData() or ""
        count_filter = self.combo_count_filter.currentData() or "all"

        self.list_widget.clear()

        matched_data = []
        for data in self.all_items_data:
            # Search text filter
            if search_text and search_text not in data["name"].lower():
                continue

            # Label name filter
            if label_filter and label_filter not in data["labels"]:
                continue

            # Shape count filter
            cnt = data["shape_count"]
            if count_filter == "zero" and cnt != 0:
                continue
            elif count_filter == "less_3" and cnt >= 3:
                continue
            elif count_filter == "less_5" and cnt >= 5:
                continue

            matched_data.append(data)

        # Populate thumbnails
        total_all = len(self.all_items_data)
        total_matched = len(matched_data)

        zero_count = sum(1 for d in self.all_items_data if d["shape_count"] == 0)
        self.lbl_status.setText(
            f"📊 Mostrando {total_matched} de {total_all} imágenes en el dataset "
            f"(⚠️ {zero_count} imágenes sin etiquetar)."
        )

        for data in matched_data:
            thumb_pixmap = self._generate_thumbnail_with_shapes(data["path"], data["json_path"])
            item = GalleryThumbnailItem(data["path"], data["shape_count"], data["labels"])
            item.setIcon(QtGui.QIcon(thumb_pixmap))
            self.list_widget.addItem(item)

    def _generate_thumbnail_with_shapes(self, img_path, json_path):
        """Render image thumbnail with bounding boxes and polygon overlays drawn on top."""
        if not os.path.exists(img_path):
            pix = QtGui.QPixmap(self.thumbnail_size, self.thumbnail_size)
            pix.fill(QtGui.QColor("#2b2b3b"))
            return pix

        qimg = QtGui.QImage(img_path)
        if qimg.isNull():
            pix = QtGui.QPixmap(self.thumbnail_size, self.thumbnail_size)
            pix.fill(QtGui.QColor("#2b2b3b"))
            return pix

        # Load shapes from JSON if available
        shapes = []
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data.get("shapes", [])
            except Exception:
                pass

        # Draw shape overlays onto QImage before scaling
        painter = QtGui.QPainter(qimg)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pen = QtGui.QPen(QtGui.QColor("#10b981"), max(2, int(qimg.width() / 250)))
        brush = QtGui.QBrush(QtGui.QColor(16, 185, 129, 70))
        painter.setPen(pen)
        painter.setBrush(brush)

        for s in shapes:
            pts = s.get("points", [])
            stype = s.get("shape_type", "polygon")
            if not pts:
                continue

            qpoints = [QtCore.QPointF(p[0], p[1]) for p in pts]
            if stype == "rectangle" and len(qpoints) >= 2:
                rect = QtCore.QRectF(qpoints[0], qpoints[1])
                painter.drawRect(rect)
            elif stype == "polygon" and len(qpoints) >= 3:
                polygon = QtGui.QPolygonF(qpoints)
                painter.drawPolygon(polygon)
            elif stype == "point" and len(qpoints) >= 1:
                r = max(4, int(qimg.width() / 100))
                painter.drawEllipse(qpoints[0], r, r)

        painter.end()

        # Scale to thumbnail size
        scaled_pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.thumbnail_size,
            self.thumbnail_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return scaled_pix

    def on_thumbnail_size_changed(self, value):
        self.thumbnail_size = value
        self.list_widget.setIconSize(QtCore.QSize(self.thumbnail_size, self.thumbnail_size))
        self.list_widget.setGridSize(QtCore.QSize(self.thumbnail_size + 24, self.thumbnail_size + 44))

    def on_item_clicked(self, item):
        if isinstance(item, GalleryThumbnailItem):
            self.main_widget.statusBar().showMessage(
                f"🖼️ Seleccionada: {os.path.basename(item.image_path)} ({item.shape_count} etiquetas)", 3000
            )

    def on_item_double_clicked(self, item):
        if isinstance(item, GalleryThumbnailItem):
            self.open_image_path(item.image_path)

    def open_current_selected_image(self):
        item = self.list_widget.currentItem()
        if isinstance(item, GalleryThumbnailItem):
            self.open_image_path(item.image_path)

    def open_image_path(self, img_path):
        """Load selected image into main canvas and close gallery dialog."""
        if hasattr(self.main_widget, "file_list_widget"):
            items = self.main_widget.file_list_widget.findItems(img_path, Qt.MatchFlag.MatchExactly)
            if items:
                row = self.main_widget.file_list_widget.row(items[0])
                self.main_widget.file_list_widget.setCurrentRow(row)

        if hasattr(self.main_widget, "load_file"):
            self.main_widget.load_file(img_path)

        self.accept()
