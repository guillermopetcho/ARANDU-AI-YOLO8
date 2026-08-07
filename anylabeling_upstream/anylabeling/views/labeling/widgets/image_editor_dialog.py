"""
Interactive Dataset Augmentation Studio & Image Editor Dialog for AnyLabeling
Inspired by Roboflow, Albumentations, CVAT, and Supervisely.
"""

import os
import json
import logging
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import PIL.ImageFilter
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.services.auto_labeling.augmentation_engine import (
    AugmentationEngine,
    AugmentationPreset,
)
from .. import utils
from ..logger import logger


class ImageEditorDialog(QtWidgets.QDialog):
    """Dataset Augmentation Studio and Image Editor Dialog."""

    def __init__(self, main_widget, parent=None):
        super().__init__(parent or main_widget)
        self.main_widget = main_widget
        self.setWindowTitle(self.tr("🎨 Estudio de Aumentación de Dataset & Editor de Imagen"))
        self.resize(920, 680)

        self.engine = AugmentationEngine()
        self.original_pil_img = self._get_current_pil_img()
        self.current_shapes = self._get_current_shapes()

        self.preview_aug_img = None
        self.preview_aug_shapes = []

        self.init_ui()
        self.update_preset_config(AugmentationPreset.ROBOFLOW_STANDARD)
        self.generate_random_preview()

    def _get_current_pil_img(self):
        """Convert current QImage from label_widget into PIL Image."""
        if not self.main_widget.image or self.main_widget.image.isNull():
            return None
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)
        self.main_widget.image.save(buffer, "PNG")
        pil_img = PIL.Image.open(buffer)
        return pil_img.convert("RGB")

    def _get_current_shapes(self):
        """Extract shape dictionaries from main canvas."""
        shapes_data = []
        if hasattr(self.main_widget, "canvas") and self.main_widget.canvas.shapes:
            for s in self.main_widget.canvas.shapes:
                shapes_data.append(s.to_dict())
        return shapes_data

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # Header Info
        header_label = QtWidgets.QLabel(
            "<b>🎨 Estudio de Aumentación de Dataset (Estilo Roboflow / Albumentations)</b><br>"
            "<small>Genera automáticamente nuevas muestras aumentadas con sincronización exacta de polígonos y etiquetas para entrenar YOLO o SAM.</small>"
        )
        header_label.setWordWrap(True)
        main_layout.addWidget(header_label)

        # Splitter: Controls Left, Dual Preview Right
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL: Tabs for Presets, Custom Controls, and Batch Multiplier
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Preset Selector Box
        preset_box = QtWidgets.QGroupBox("1. Seleccionar Preset de Aumentación")
        preset_layout = QtWidgets.QVBoxLayout(preset_box)
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItem("🌿 Light (Volteo H + Brillo/Contraste leve)", AugmentationPreset.LIGHT)
        self.combo_preset.addItem("🚀 Roboflow Standard (Flips, Rotación ±25°, Ruido, Recorte, Brillo)", AugmentationPreset.ROBOFLOW_STANDARD)
        self.combo_preset.addItem("🔥 Heavy (Aumentación Intensa Multimodal)", AugmentationPreset.HEAVY)
        self.combo_preset.addItem("⚙️ Custom (Personalizado)", AugmentationPreset.CUSTOM)
        self.combo_preset.setCurrentIndex(1)
        self.combo_preset.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.combo_preset)
        left_layout.addWidget(preset_box)

        # Tabs Widget for Detailed Configuration & Batch Multiplier
        self.tab_widget = QtWidgets.QTabWidget()

        # TAB 1: Transformations Config & Preview
        tab_config = QtWidgets.QWidget()
        cfg_layout = QtWidgets.QVBoxLayout(tab_config)

        # Scroll area for sliders & checkboxes
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(scroll_content)
        form_layout.setSpacing(6)

        self.chk_flip_h = QtWidgets.QCheckBox("↔️ Volteo Horizontal (Prob 50%)")
        self.chk_flip_v = QtWidgets.QCheckBox("↕️ Volteo Vertical (Prob 30%)")
        self.chk_rotation = QtWidgets.QCheckBox("🔄 Rotación Aleatoria (±25°)")
        self.chk_crop = QtWidgets.QCheckBox("✂️ Recorte & Escala Aleatoria (0.8x-0.98x)")
        self.chk_brightness = QtWidgets.QCheckBox("☀️ Ajuste de Brillo (±20%)")
        self.chk_contrast = QtWidgets.QCheckBox("🌓 Ajuste de Contraste (±20%)")
        self.chk_blur = QtWidgets.QCheckBox("💧 Desenfoque Gaussiano")
        self.chk_noise = QtWidgets.QCheckBox("📻 Ruido Gaussiano / Granulado")
        self.chk_grayscale = QtWidgets.QCheckBox("⬛⬜ Convertir a Escala de Grises")

        form_layout.addRow(self.chk_flip_h)
        form_layout.addRow(self.chk_flip_v)
        form_layout.addRow(self.chk_rotation)
        form_layout.addRow(self.chk_crop)
        form_layout.addRow(self.chk_brightness)
        form_layout.addRow(self.chk_contrast)
        form_layout.addRow(self.chk_blur)
        form_layout.addRow(self.chk_noise)
        form_layout.addRow(self.chk_grayscale)

        for chk in [
            self.chk_flip_h, self.chk_flip_v, self.chk_rotation, self.chk_crop,
            self.chk_brightness, self.chk_contrast, self.chk_blur, self.chk_noise, self.chk_grayscale
        ]:
            chk.stateChanged.connect(self.on_custom_option_changed)

        scroll.setWidget(scroll_content)
        cfg_layout.addWidget(scroll)

        btn_rand_preview = QtWidgets.QPushButton("🎲 Probar Aumentación Aleatoria")
        btn_rand_preview.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 6px;")
        btn_rand_preview.clicked.connect(self.generate_random_preview)
        cfg_layout.addWidget(btn_rand_preview)

        self.tab_widget.addTab(tab_config, "🎛️ Parámetros")

        # TAB 2: Batch Generation / Multiplier
        tab_batch = QtWidgets.QWidget()
        batch_layout = QtWidgets.QVBoxLayout(tab_batch)

        batch_group = QtWidgets.QGroupBox("Generación Masiva de Dataset (Dataset Multiplier)")
        batch_form = QtWidgets.QFormLayout(batch_group)

        self.combo_scope = QtWidgets.QComboBox()
        self.combo_scope.addItem("Imagen Actual Únicamente", "single")
        self.combo_scope.addItem("Toda la Carpeta Abierta", "folder")
        batch_form.addRow("Alcance:", self.combo_scope)

        self.spin_multiplier = QtWidgets.QSpinBox()
        self.spin_multiplier.setRange(1, 50)
        self.spin_multiplier.setValue(3)
        self.spin_multiplier.setSuffix(" x por imagen")
        batch_form.addRow("Multiplicador (Copias):", self.spin_multiplier)

        self.edit_output_dir = QtWidgets.QLineEdit()
        default_aug_dir = os.path.join(self.main_widget.last_open_dir or ".", "augmented_dataset")
        self.edit_output_dir.setText(default_aug_dir)
        btn_browse_dir = QtWidgets.QPushButton("📁 Buscar...")
        btn_browse_dir.clicked.connect(self.browse_output_dir)

        dir_layout = QtWidgets.QHBoxLayout()
        dir_layout.addWidget(self.edit_output_dir)
        dir_layout.addWidget(btn_browse_dir)
        batch_form.addRow("Carpeta Destino:", dir_layout)

        batch_layout.addWidget(batch_group)

        btn_run_batch = QtWidgets.QPushButton("🚀 Generar Dataset Aumentado")
        btn_run_batch.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; font-size: 13px; padding: 8px;")
        btn_run_batch.clicked.connect(self.run_batch_augmentation)
        batch_layout.addWidget(btn_run_batch)

        self.tab_widget.addTab(tab_batch, "⚡ Multiplicador de Dataset")
        left_layout.addWidget(self.tab_widget)
        splitter.addWidget(left_widget)

        # RIGHT PANEL: Dual Live Preview (Original vs. Augmented)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_label_head = QtWidgets.QLabel("<b>Visualización Comparativa (Original vs. Aumentada con Polígonos)</b>")
        right_layout.addWidget(preview_label_head)

        preview_box_layout = QtWidgets.QHBoxLayout()

        # Original Image View
        box_orig = QtWidgets.QGroupBox("Imagen Original")
        layout_orig = QtWidgets.QVBoxLayout(box_orig)
        self.lbl_view_orig = QtWidgets.QLabel()
        self.lbl_view_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_view_orig.setStyleSheet("background-color: #181825; border: 1px solid #313244;")
        layout_orig.addWidget(self.lbl_view_orig)
        preview_box_layout.addWidget(box_orig)

        # Augmented Image View
        box_aug = QtWidgets.QGroupBox("Resultado Aumentado")
        layout_aug = QtWidgets.QVBoxLayout(box_aug)
        self.lbl_view_aug = QtWidgets.QLabel()
        self.lbl_view_aug.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_view_aug.setStyleSheet("background-color: #181825; border: 1px solid #313244;")
        layout_aug.addWidget(self.lbl_view_aug)
        preview_box_layout.addWidget(box_aug)

        right_layout.addLayout(preview_box_layout)
        splitter.addWidget(right_widget)

        # Set splitter balance
        splitter.setSizes([380, 540])
        main_layout.addWidget(splitter)

        # Bottom Dialog Buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

    def on_preset_changed(self, index):
        preset_key = self.combo_preset.currentData()
        if preset_key != AugmentationPreset.CUSTOM:
            self.update_preset_config(preset_key)
            self.generate_random_preview()

    def update_preset_config(self, preset_key):
        self.engine.config = AugmentationEngine.get_default_config(preset_key)
        cfg = self.engine.config

        self.chk_flip_h.setChecked(cfg.get("flip_h", {}).get("enabled", False))
        self.chk_flip_v.setChecked(cfg.get("flip_v", {}).get("enabled", False))
        self.chk_rotation.setChecked(cfg.get("rotation", {}).get("enabled", False))
        self.chk_crop.setChecked(cfg.get("crop", {}).get("enabled", False))
        self.chk_brightness.setChecked(cfg.get("brightness", {}).get("enabled", False))
        self.chk_contrast.setChecked(cfg.get("contrast", {}).get("enabled", False))
        self.chk_blur.setChecked(cfg.get("blur", {}).get("enabled", False))
        self.chk_noise.setChecked(cfg.get("noise", {}).get("enabled", False))
        self.chk_grayscale.setChecked(cfg.get("grayscale", {}).get("enabled", False))

    def on_custom_option_changed(self):
        # Update custom engine config from checkboxes
        cfg = self.engine.config
        cfg["preset"] = AugmentationPreset.CUSTOM
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(3)  # Custom
        self.combo_preset.blockSignals(False)

        cfg["flip_h"]["enabled"] = self.chk_flip_h.isChecked()
        cfg["flip_v"]["enabled"] = self.chk_flip_v.isChecked()
        cfg["rotation"]["enabled"] = self.chk_rotation.isChecked()
        cfg["crop"]["enabled"] = self.chk_crop.isChecked()
        cfg["brightness"]["enabled"] = self.chk_brightness.isChecked()
        cfg["contrast"]["enabled"] = self.chk_contrast.isChecked()
        cfg["blur"]["enabled"] = self.chk_blur.isChecked()
        cfg["noise"]["enabled"] = self.chk_noise.isChecked()
        cfg["grayscale"]["enabled"] = self.chk_grayscale.isChecked()

    def generate_random_preview(self):
        if not self.original_pil_img:
            return

        # Render original with shapes overlay
        orig_qimg = self._render_pil_with_shapes(self.original_pil_img, self.current_shapes)
        self.lbl_view_orig.setPixmap(
            QtGui.QPixmap.fromImage(orig_qimg).scaled(
                260, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

        # Apply augmentation engine pass
        aug_pil, aug_shapes = self.engine.apply_transformations(self.original_pil_img, self.current_shapes)
        self.preview_aug_img = aug_pil
        self.preview_aug_shapes = aug_shapes

        # Render augmented result with transformed shapes overlay
        aug_qimg = self._render_pil_with_shapes(aug_pil, aug_shapes)
        self.lbl_view_aug.setPixmap(
            QtGui.QPixmap.fromImage(aug_qimg).scaled(
                260, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

    def _render_pil_with_shapes(self, pil_img, shapes):
        """Render PIL image with polygon and rectangle shape overlays into QImage."""
        if not pil_img:
            return QtGui.QImage()

        # Convert PIL to QImage
        w, h = pil_img.size
        img_bytes = pil_img.tobytes("raw", "RGB")
        qimg = QtGui.QImage(img_bytes, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()

        painter = QtGui.QPainter(qimg)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pen = QtGui.QPen(QtGui.QColor("#10b981"), 2)
        brush = QtGui.QBrush(QtGui.QColor(16, 185, 129, 60))
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
                painter.drawEllipse(qpoints[0], 4, 4)

        painter.end()
        return qimg

    def browse_output_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("📁 Seleccionar carpeta de destino para el Dataset Aumentado"),
            self.edit_output_dir.text(),
        )
        if dir_path:
            self.edit_output_dir.setText(dir_path)

    def run_batch_augmentation(self):
        scope = self.combo_scope.currentData()
        multiplier = self.spin_multiplier.value()
        target_dir = self.edit_output_dir.text().strip()

        if not target_dir:
            QtWidgets.QMessageBox.warning(self, "Dataset Aumentado", "Especifica una carpeta destino válida.")
            return

        os.makedirs(target_dir, exist_ok=True)

        image_list = []
        if scope == "single":
            if self.main_widget.filename:
                image_list = [self.main_widget.filename]
        else:
            if hasattr(self.main_widget, "image_list") and self.main_widget.image_list:
                image_list = list(self.main_widget.image_list)

        if not image_list:
            QtWidgets.QMessageBox.warning(self, "Dataset Aumentado", "No hay imágenes disponibles para procesar.")
            return

        total_tasks = len(image_list) * multiplier
        progress = QtWidgets.QProgressDialog(
            self.tr("🚀 Generando Dataset Aumentado..."),
            self.tr("Cancelar"),
            0,
            total_tasks,
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        generated_count = 0
        task_idx = 0

        for img_path in image_list:
            if progress.wasCanceled():
                break

            img_name = os.path.basename(img_path)
            base_name, ext = os.path.splitext(img_name)

            # Load PIL image
            try:
                pil_img = PIL.Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error("Error loading image for augmentation %s: %s", img_path, e)
                continue

            # Load corresponding shapes
            json_path = self.main_widget.get_label_file_for_image(img_path)
            shapes = []
            json_meta = {}

            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_meta = json.load(f)
                    shapes = json_meta.get("shapes", [])
                except Exception as e:
                    logger.error("Error loading JSON for augmentation %s: %s", json_path, e)

            for copy_idx in range(1, multiplier + 1):
                if progress.wasCanceled():
                    break

                task_idx += 1
                progress.setValue(task_idx)
                progress.setLabelText(f"Procesando [{task_idx}/{total_tasks}]: {base_name}_aug_{copy_idx}{ext}")
                QtWidgets.QApplication.processEvents()

                # Apply augmentation pass
                aug_pil, aug_shapes = self.engine.apply_transformations(pil_img, shapes)
                if not aug_pil:
                    continue

                aug_img_name = f"{base_name}_aug_{copy_idx}{ext}"
                aug_img_path = os.path.join(target_dir, aug_img_name)
                aug_json_path = os.path.join(target_dir, f"{base_name}_aug_{copy_idx}.json")

                # Save augmented image file
                try:
                    aug_pil.save(aug_img_path)
                except Exception as e:
                    logger.error("Error saving augmented image %s: %s", aug_img_path, e)
                    continue

                # Save augmented JSON label file
                w, h = aug_pil.size
                out_json_data = {
                    "version": json_meta.get("version", "0.3.3"),
                    "flags": json_meta.get("flags", {}),
                    "shapes": aug_shapes,
                    "imagePath": aug_img_name,
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w,
                }

                try:
                    with open(aug_json_path, "w", encoding="utf-8") as f:
                        json.dump(out_json_data, f, ensure_ascii=False, indent=2)
                    generated_count += 1
                except Exception as e:
                    logger.error("Error saving augmented JSON %s: %s", aug_json_path, e)

        progress.setValue(total_tasks)
        progress.close()

        msg = (
            f"✅ Se generaron exitosamente {generated_count} imágenes y archivos `.json` aumentados "
            f"en la carpeta:\n'{target_dir}'"
        )
        self.main_widget.statusBar().showMessage(msg, 6000)
        QtWidgets.QMessageBox.information(self, "🚀 Multiplicador de Dataset Completado", msg)
