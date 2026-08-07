import os

from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt6.QtWidgets import (
    QFileDialog, QWidget, QPushButton, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QMessageBox, QDoubleSpinBox, QCheckBox,
    QComboBox
)

from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.styles.theme import AppTheme


class AutomateWorkerThread(QThread):
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, image_dir, model_path, conf, overwrite, mode="yolo_seg", sam_path="sam_b.pt"):
        super().__init__()
        self.image_dir = image_dir
        self.model_path = model_path
        self.conf = conf
        self.overwrite = overwrite
        self.mode = mode
        self.sam_path = sam_path

    def run(self):
        try:
            if self.mode == "yolo_sam":
                from auto_label_yolo_sam import auto_label_yolo_sam
                auto_label_yolo_sam(
                    image_dir=self.image_dir,
                    detector_path=self.model_path,
                    sam_path=self.sam_path,
                    conf_threshold=self.conf,
                    overwrite=self.overwrite
                )
            else:
                from auto_label_folder import auto_label_folder
                auto_label_folder(
                    image_dir=self.image_dir,
                    model_path=self.model_path,
                    conf_threshold=self.conf,
                    overwrite=self.overwrite
                )
            self.finished_signal.emit(True, "Segmentación masiva completada con éxito.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class AutomateProcessDialog(QDialog):
    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.setWindowTitle("⚡ Automatizar Proceso de Segmentación")
        self.setMinimumWidth(580)

        layout = QVBoxLayout(self)

        header = QLabel("<b>Auto-Segmentación Masiva con Modelo Entrenado</b>")
        header.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(header)

        layout.addWidget(QLabel("<b>1. Modo de Segmentación:</b>"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("🎯 YOLO Segmentación Directa (YOLO-Seg)", userData="yolo_seg")
        self.combo_mode.addItem("⚡ YOLO Detección + SAM Segmentador (Guiado)", userData="yolo_sam")
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        layout.addWidget(self.combo_mode)

        self.label_model1 = QLabel("<b>2. Modelo YOLO-Seg (.pt / .onnx):</b>")
        layout.addWidget(self.label_model1)
        model_layout = QHBoxLayout()
        self.edit_model = QLineEdit()
        self.edit_model.setPlaceholderText("Seleccionar modelo YOLO (.pt / .onnx)...")
        btn_browse_model = QPushButton("Buscar Modelo...")
        btn_browse_model.clicked.connect(self.browse_model)
        model_layout.addWidget(self.edit_model)
        model_layout.addWidget(btn_browse_model)
        layout.addLayout(model_layout)

        # Configuración modelo SAM (para modo guiado YOLO + SAM)
        self.widget_sam = QWidget()
        sam_vlayout = QVBoxLayout(self.widget_sam)
        sam_vlayout.setContentsMargins(0, 0, 0, 0)
        sam_vlayout.addWidget(QLabel("<b>3. Modelo SAM / Segment Anything (.pt):</b>"))
        sam_hlayout = QHBoxLayout()
        self.edit_sam = QLineEdit("sam_b.pt")
        self.edit_sam.setPlaceholderText("Ruta o nombre del modelo SAM (ej: sam_b.pt, mobile_sam.pt)")
        btn_browse_sam = QPushButton("Buscar SAM...")
        btn_browse_sam.clicked.connect(self.browse_sam)
        sam_hlayout.addWidget(self.edit_sam)
        sam_hlayout.addWidget(btn_browse_sam)
        sam_vlayout.addLayout(sam_hlayout)
        layout.addWidget(self.widget_sam)
        self.widget_sam.hide()

        self.label_folder = QLabel("<b>3. Carpeta de Imágenes a Segmentar:</b>")
        layout.addWidget(self.label_folder)
        folder_layout = QHBoxLayout()
        self.edit_folder = QLineEdit()
        current_dir = ""
        if hasattr(parent_widget.parent, "dirname") and parent_widget.parent.dirname:
            current_dir = parent_widget.parent.dirname
        self.edit_folder.setText(current_dir)
        self.edit_folder.setPlaceholderText("Seleccionar carpeta de imágenes...")
        btn_browse_folder = QPushButton("Buscar Carpeta...")
        btn_browse_folder.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.edit_folder)
        folder_layout.addWidget(btn_browse_folder)
        layout.addLayout(folder_layout)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Umbral Confianza:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.05, 0.95)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        param_layout.addWidget(self.spin_conf)

        self.chk_overwrite = QCheckBox("Sobrescribir JSONs existentes")
        self.chk_overwrite.setChecked(True)
        param_layout.addWidget(self.chk_overwrite)
        layout.addLayout(param_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.btn_run = QPushButton("▶ Iniciar Auto-Segmentación Masiva")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.btn_run.clicked.connect(self.start_automation)
        layout.addWidget(self.btn_run)

    def on_mode_changed(self, index):
        mode = self.combo_mode.currentData()
        if mode == "yolo_sam":
            self.widget_sam.show()
            self.label_model1.setText("<b>2. Modelo YOLO Detector (.pt / .onnx):</b>")
            self.label_folder.setText("<b>4. Carpeta de Imágenes a Segmentar:</b>")
        else:
            self.widget_sam.hide()
            self.label_model1.setText("<b>2. Modelo YOLO-Seg (.pt / .onnx):</b>")
            self.label_folder.setText("<b>3. Carpeta de Imágenes a Segmentar:</b>")

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Modelo YOLO", "", "Modelos (*.pt *.onnx *.yaml)"
        )
        if file_path:
            self.edit_model.setText(file_path)

    def browse_sam(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Modelo SAM", "", "Modelos PyTorch (*.pt *.onnx)"
        )
        if file_path:
            self.edit_sam.setText(file_path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Imágenes")
        if folder:
            self.edit_folder.setText(folder)

    def start_automation(self):
        mode = self.combo_mode.currentData()
        model_path = self.edit_model.text().strip()
        sam_path = self.edit_sam.text().strip()
        folder_path = self.edit_folder.text().strip()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Por favor selecciona un archivo de modelo YOLO válido (.pt o .onnx).")
            return
        if mode == "yolo_sam" and not sam_path:
            QMessageBox.warning(self, "Error", "Por favor especifica un modelo SAM válido (ej: sam_b.pt o la ruta a un archivo .pt).")
            return
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Error", "Por favor selecciona una carpeta de imágenes válida.")
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText("Procesando segmentación en lote por IA...")

        self.worker = AutomateWorkerThread(
            image_dir=folder_path,
            model_path=model_path,
            conf=self.spin_conf.value(),
            overwrite=self.chk_overwrite.isChecked(),
            mode=mode,
            sam_path=sam_path
        )
        self.worker.finished_signal.connect(self.on_automation_finished)
        self.worker.start()

    def on_automation_finished(self, success, msg):
        self.progress_bar.hide()
        self.btn_run.setEnabled(True)

        if success:
            QMessageBox.information(self, "Éxito", msg)
            if hasattr(self.parent_widget.parent, "import_dir_images"):
                self.parent_widget.parent.import_dir_images(self.edit_folder.text().strip())
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Fallo en la automatización: {msg}")
            self.status_label.setText("Error en la ejecución.")


class AutoLabelingWidget(QWidget):
    new_model_selected = pyqtSignal(str)
    new_custom_model_selected = pyqtSignal(str)
    auto_segmentation_requested = pyqtSignal()
    auto_segmentation_disabled = pyqtSignal()
    auto_labeling_mode_changed = pyqtSignal(AutoLabelingMode)
    clear_auto_labeling_action_requested = pyqtSignal()
    finish_auto_labeling_object_action_requested = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        current_dir = os.path.dirname(__file__)
        uic.loadUi(os.path.join(current_dir, "auto_labeling.ui"), self)

        # Botón "⚡ Automatizar Proceso"
        self.button_automate_process = QPushButton(self.tr("⚡ Automatizar Proceso"), self)
        self.button_automate_process.setToolTip(
            self.tr("Cargar un modelo entrenado (.pt / .onnx) y auto-segmentar toda la carpeta automáticamente")
        )
        self.button_automate_process.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.button_automate_process.clicked.connect(self.open_automate_dialog)

        if hasattr(self, "model_selection") and self.model_selection is not None:
            self.model_selection.insertWidget(2, self.button_automate_process)

        self.model_manager = ModelManager()
        self.model_manager.model_configs_changed.connect(
            lambda model_list: self.update_model_configs(model_list)
        )
        self.model_manager.new_model_status.connect(self.on_new_model_status)
        self.new_model_selected.connect(self.model_manager.load_model)
        self.new_custom_model_selected.connect(self.model_manager.load_custom_model)
        self.model_manager.model_loaded.connect(self.update_visible_widgets)
        self.model_manager.model_loaded.connect(self.on_new_model_loaded)
        self.model_manager.new_auto_labeling_result.connect(
            lambda auto_labeling_result: self.parent.new_shapes_from_auto_labeling(
                auto_labeling_result
            )
        )
        self.model_manager.auto_segmentation_model_selected.connect(
            self.auto_segmentation_requested
        )
        self.model_manager.auto_segmentation_model_unselected.connect(
            self.auto_segmentation_disabled
        )
        self.model_manager.output_modes_changed.connect(self.on_output_modes_changed)
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

        self.update_model_configs(self.model_manager.get_model_configs())

        # Contenedor para barra de progreso y botón "❌ Cancelar"
        self.progress_container = QWidget(self)
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(4, 2, 4, 2)
        progress_layout.setSpacing(6)

        self.progress_bar = QProgressBar(self.progress_container)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #383838;
                border-radius: 4px;
                text-align: center;
                font-size: 10px;
                font-weight: bold;
                color: #ffffff;
                background-color: #1e1e2e;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #3b82f6);
                border-radius: 3px;
            }
        """)

        self.button_cancel_process = QPushButton("❌ Cancelar", self.progress_container)
        self.button_cancel_process.setToolTip(self.tr("Cancelar el proceso de carga o inferencia actual"))
        self.button_cancel_process.setFixedHeight(22)
        self.button_cancel_process.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_cancel_process.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: #ffffff;
                font-weight: bold;
                font-size: 10px;
                border-radius: 4px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        self.button_cancel_process.clicked.connect(self.cancel_current_process)

        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.button_cancel_process, 0)

        if hasattr(self, "verticalLayout") and self.verticalLayout:
            self.verticalLayout.addWidget(self.progress_container)
        elif self.layout():
            self.layout().addWidget(self.progress_container)

        self.progress_container.hide()

        # Disable tools when loading or inference is running
        def set_enable_tools(enable):
            self.model_select_combobox.setEnabled(enable)
            self.output_select_combobox.setEnabled(enable)
            self.edit_prompt.setEnabled(enable)
            if hasattr(self, "combobox_prompt_mode"):
                self.combobox_prompt_mode.setEnabled(enable)
            if hasattr(self, "double_spin_box_confidence"):
                self.double_spin_box_confidence.setEnabled(enable)
            if hasattr(self, "button_run"):
                self.button_run.setEnabled(enable)
            if hasattr(self, "button_automate_process"):
                self.button_automate_process.setEnabled(enable)
            self.button_add_point.setEnabled(enable)
            self.button_remove_point.setEnabled(enable)
            self.button_add_rect.setEnabled(enable)
            self.button_clear.setEnabled(enable)
            self.button_finish_object.setEnabled(enable)

        def on_loading_started():
            set_enable_tools(False)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("⚡ Cargando modelo...")
            self.progress_container.show()

        def on_prediction_started():
            set_enable_tools(False)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("🧠 Inferenciando IA...")
            self.progress_container.show()

        def on_process_finished():
            set_enable_tools(True)
            self.progress_container.hide()

        self.model_manager.model_loading_started.connect(on_loading_started)
        self.model_manager.model_loading_finished.connect(on_process_finished)
        self.model_manager.prediction_started.connect(on_prediction_started)
        self.model_manager.prediction_finished.connect(on_process_finished)

        # Prompt input
        self.edit_prompt.textChanged.connect(self.on_prompt_changed)
        self.edit_prompt.returnPressed.connect(self.run_prediction)

        # Prompt mode
        self.combobox_prompt_mode.currentIndexChanged.connect(
            self.on_prompt_mode_changed
        )

        # Confidence
        self.double_spin_box_confidence.valueChanged.connect(self.on_confidence_changed)

        # Auto labeling buttons
        self.button_run.setText("▶ Iniciar (I)")
        self.button_run.setShortcut("I")
        self.button_run.clicked.connect(self.run_prediction)

        self.button_add_point.setText("➕ Punto (+)")
        self.button_add_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.POINT
            )
        )

        # Botón Q(++) - Fast Auto-Labeling with Selected Label (Atajo único: Q)
        self.button_q_plus_plus = QPushButton("⚡ Q(++) [Q]", self)
        self.button_q_plus_plus.setCheckable(True)
        self.button_q_plus_plus.setToolTip(
            self.tr("Modo Q(++): Etiquetado automático rápido [Atajo: Q]. Asigna directamente la etiqueta seleccionada en la lista al hacer clic o segmentar sin ventana emergente.")
        )
        self.button_q_plus_plus.toggled.connect(self.on_q_plus_plus_toggled)
        self.button_q_plus_plus.setShortcut("Q")
        self.update_q_plus_plus_button_style(False)

        if hasattr(self, "button_add_point") and self.button_add_point.parentWidget():
            parent_layout = self.button_add_point.parentWidget().layout()
            if parent_layout:
                idx = parent_layout.indexOf(self.button_add_point)
                parent_layout.insertWidget(idx + 1, self.button_q_plus_plus)

        self.button_remove_point.setText("➖ Quitar (-) [E]")
        self.button_remove_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.REMOVE, AutoLabelingMode.POINT
            )
        )
        self.button_remove_point.setShortcut("E")

        self.button_add_rect.setText("🔲 Caja BBox")
        self.button_add_rect.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.RECTANGLE
            )
        )

        self.button_clear.setText("🧹 Limpiar")
        self.button_clear.clicked.connect(self.clear_auto_labeling_action_requested)

        self.button_finish_object.setText("✅ Finalizar (F)")
        self.button_finish_object.clicked.connect(
            self.finish_auto_labeling_object_action_requested
        )
        self.button_finish_object.setShortcut("F")

        # Hide labeling widgets by default
        self.hide_labeling_widgets()

        # Handle close button
        self.button_close.clicked.connect(self.unload_and_hide)

        # Handle model select combobox
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )

        self.auto_labeling_mode_changed.connect(self.update_button_colors)
        self.auto_labeling_mode = AutoLabelingMode.NONE
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    def update_model_configs(self, model_list):
        """Update model list"""
        # Add models to combobox
        self.model_select_combobox.clear()
        self.model_select_combobox.addItem(self.tr("No Model"), userData=None)
        self.model_select_combobox.addItem(
            self.tr("...Load Custom Model"), userData="load_custom_model"
        )
        for model_config in model_list:
            self.model_select_combobox.addItem(
                (
                    self.tr("(User) ")
                    if model_config.get("is_custom_model", False)
                    else ""
                )
                + model_config["display_name"],
                userData=model_config["config_file"],
            )

    @pyqtSlot()
    def update_button_colors(self):
        """Update button colors based on current theme and mode"""
        style_sheet = """
            text-align: center;
            margin-right: 2px;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
            border: 1px solid {border_color};
        """

        border_color = AppTheme.get_color("border")
        normal_bg_color = AppTheme.get_color("button")
        normal_text_color = AppTheme.get_color("button_text")
        active_bg_color = AppTheme.get_color("success")
        remove_bg_color = AppTheme.get_color("error")
        highlighted_text_color = AppTheme.get_color("highlighted_text")

        normal_style = (
            style_sheet.format(border_color=border_color)
            + f"background-color: {normal_bg_color}; color: {normal_text_color};"
        )

        for button in [
            self.button_add_point,
            self.button_remove_point,
            self.button_add_rect,
            self.button_clear,
            self.button_finish_object,
        ]:
            button.setStyleSheet(normal_style)

        if self.auto_labeling_mode == AutoLabelingMode.NONE:
            return

        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_add_point.setStyleSheet(
                    style_sheet.format(border_color=border_color)
                    + f"background-color: {active_bg_color}; color: {highlighted_text_color};"
                )
            elif self.auto_labeling_mode.shape_type == AutoLabelingMode.RECTANGLE:
                self.button_add_rect.setStyleSheet(
                    style_sheet.format(border_color=border_color)
                    + f"background-color: {active_bg_color}; color: {highlighted_text_color};"
                )
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_remove_point.setStyleSheet(
                    style_sheet.format(border_color=border_color)
                    + f"background-color: {remove_bg_color}; color: {highlighted_text_color};"
                )

    def set_auto_labeling_mode(self, edit_mode, shape_type=None):
        """Set auto labeling mode"""
        if edit_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
            if hasattr(self, "button_q_plus_plus") and self.button_q_plus_plus.isChecked():
                self.button_q_plus_plus.blockSignals(True)
                self.button_q_plus_plus.setChecked(False)
                self.update_q_plus_plus_button_style(False)
                self.button_q_plus_plus.blockSignals(False)
                if hasattr(self, "parent") and hasattr(self.parent, "q_plus_plus_mode"):
                    self.parent.q_plus_plus_mode = False
        else:
            self.auto_labeling_mode = AutoLabelingMode(edit_mode, shape_type)
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    @pyqtSlot(bool)
    def on_q_plus_plus_toggled(self, checked):
        """Toggle Q(++) auto labeling mode on parent LabelWidget"""
        self.update_q_plus_plus_button_style(checked)
        if checked:
            if self.auto_labeling_mode == AutoLabelingMode.NONE:
                self.set_auto_labeling_mode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)
        if hasattr(self.parent, "set_q_plus_plus_mode"):
            self.parent.set_q_plus_plus_mode(checked)

    def update_q_plus_plus_button_style(self, checked):
        if not hasattr(self, "button_q_plus_plus"):
            return
        if checked:
            self.button_q_plus_plus.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 11px;
                    border: 1.5px solid #059669;
                    border-radius: 4px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            is_dark = AppTheme.is_dark_mode()
            bg_col = "#2c2c2c" if is_dark else "#ffffff"
            text_col = "#ffffff" if is_dark else "#1b1b1b"
            border_col = "#383838" if is_dark else "#cbd5e1"
            self.button_q_plus_plus.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_col};
                    color: {text_col};
                    font-weight: bold;
                    font-size: 11px;
                    border: 1px solid {border_col};
                    border-radius: 4px;
                    padding: 3px 8px;
                }}
                QPushButton:hover {{
                    border: 1.5px solid #10b981;
                    color: #10b981;
                }}
            """)

    def run_prediction(self):
        """Run prediction"""
        if self.parent.filename is not None:
            self.model_manager.predict_shapes_threading(
                self.parent.image, self.parent.filename
            )

    def open_automate_dialog(self):
        """Abre el diálogo para cargar un modelo de segmentación entrenado y automatizar la carpeta"""
        dialog = AutomateProcessDialog(self)
        dialog.exec()

    def unload_and_hide(self):
        """Unload model and hide widget"""
        self.model_select_combobox.setCurrentIndex(0)
        self.hide()

    def cancel_current_process(self):
        """Cancel current model download, loading, or prediction process."""
        if hasattr(self, "model_manager") and self.model_manager:
            self.model_manager.stop_inference()
        if hasattr(self, "parent") and hasattr(self.parent, "canvas") and self.parent.canvas:
            self.parent.canvas.set_loading(False)
            if hasattr(self.parent, "clear_auto_labeling_marks"):
                self.parent.clear_auto_labeling_marks()
        self.progress_container.hide()
        self.model_status_label.setText(self.tr("⛔ Proceso cancelado por el usuario."))

    def on_new_model_status(self, status):
        self.model_status_label.setText(status)
        if "%" in status:
            try:
                import re
                match = re.search(r"(\d+)%", status)
                if match:
                    val = int(match.group(1))
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(val)
                    self.progress_bar.setFormat(f"Descargando {val}%")
                    self.progress_container.show()
            except Exception:
                pass

    def on_new_model_loaded(self, model_config):
        """Enable model select combobox"""
        self.model_select_combobox.currentIndexChanged.disconnect()
        if "config_file" not in model_config:
            self.model_select_combobox.setCurrentIndex(0)
        else:
            config_file = model_config["config_file"]
            self.model_select_combobox.setCurrentIndex(
                self.model_select_combobox.findData(config_file)
            )
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )
        self.model_select_combobox.setEnabled(True)

    def on_output_modes_changed(self, output_modes, default_output_mode):
        """Handle output modes changed"""
        # Disconnect onIndexChanged signal to prevent triggering
        # on model select combobox change
        self.output_select_combobox.currentIndexChanged.disconnect()

        self.output_select_combobox.clear()
        for output_mode, display_name in output_modes.items():
            self.output_select_combobox.addItem(display_name, userData=output_mode)
        self.output_select_combobox.setCurrentIndex(
            self.output_select_combobox.findData(default_output_mode)
        )

        # Reconnect onIndexChanged signal
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

    def on_model_select_combobox_changed(self, index):
        """Handle model select combobox change"""
        self.clear_auto_labeling_action_requested.emit()
        config_path = self.model_select_combobox.itemData(index)

        # Load custom model?
        if config_path == "load_custom_model":
            # Unload current model
            self.model_manager.unload_model()
            # Open file dialog to select "config.yaml" file for model
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("Config file (*.yaml)")
            if file_dialog.exec():
                config_file = file_dialog.selectedFiles()[0]
                # Disable combobox while loading model
                if config_path:
                    self.model_select_combobox.setEnabled(False)
                self.hide_labeling_widgets()
                self.model_manager.load_custom_model(config_file)
            else:
                self.model_select_combobox.setCurrentIndex(0)
            return

        # Disable combobox while loading model
        if config_path:
            self.model_select_combobox.setEnabled(False)
        self.hide_labeling_widgets()
        self.new_model_selected.emit(config_path)

    def update_visible_widgets(self, model_config):
        """Update widget status"""
        if not model_config or "model" not in model_config:
            return
        model = model_config["model"]
        widgets = model.get_required_widgets()

        is_sam3 = getattr(model, "_is_sam3", False)

        # Always check if prompt mode selection should be shown
        if "label_prompt" in widgets or "edit_prompt" in widgets:
            self.label_prompt_mode.show()
            self.combobox_prompt_mode.show()
            self.label_confidence.show()
            self.double_spin_box_confidence.show()

            if not is_sam3:
                # Force Visual mode for non-SAM3 models
                visual_index = self.combobox_prompt_mode.findText(self.tr("Visual"))
                if visual_index >= 0:
                    self.combobox_prompt_mode.setCurrentIndex(visual_index)
                self.combobox_prompt_mode.setEnabled(False)
            else:
                self.combobox_prompt_mode.setEnabled(True)

        prompt_mode = self.combobox_prompt_mode.currentText().lower()

        for widget in widgets:
            widget_obj = getattr(self, widget)

            # Filter based on prompt mode
            if prompt_mode == "visual":
                if widget in ["label_prompt", "edit_prompt"]:
                    widget_obj.hide()
                    continue
            elif prompt_mode == "text":
                # In text mode hide the geometric-prompt buttons.
                # Inference is triggered by pressing Enter, changing
                # the prompt text, or clicking the Run button.
                if widget in [
                    "button_add_point",
                    "button_remove_point",
                    "button_add_rect",
                    "button_clear",
                    "button_finish_object",
                ]:
                    widget_obj.hide()
                    continue

            widget_obj.show()

        # Set initial values for widgets
        if hasattr(model_config["model"], "text_prompt"):
            self.edit_prompt.setText(model_config["model"].text_prompt)
        if hasattr(model_config["model"], "confidence_threshold"):
            self.double_spin_box_confidence.setValue(
                model_config["model"].confidence_threshold
            )

    def hide_labeling_widgets(self):
        """Hide labeling widgets by default"""
        widgets = [
            "output_label",
            "output_select_combobox",
            "label_prompt_mode",
            "combobox_prompt_mode",
            "label_confidence",
            "double_spin_box_confidence",
            "label_prompt",
            "edit_prompt",
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        for widget in widgets:
            getattr(self, widget).hide()

    def on_prompt_changed(self, text):
        """Handle prompt changed"""
        self.model_manager.set_text_prompt(text)

    def on_confidence_changed(self, value):
        """Handle confidence changed"""
        self.model_manager.set_confidence_threshold(value)

    def on_prompt_mode_changed(self, index):
        """Handle prompt mode changed"""
        mode = self.combobox_prompt_mode.currentText().lower()
        self.model_manager.set_prompt_mode(mode)

        if mode == "visual":
            # Clear and reset the text prompt when switching to visual mode so
            # old text does not linger in the model's language encoder.
            self.edit_prompt.blockSignals(True)
            self.edit_prompt.clear()
            self.edit_prompt.blockSignals(False)
            self.model_manager.set_text_prompt("")

        # Refresh widget visibility
        if self.model_manager.loaded_model_config:
            self.update_visible_widgets(self.model_manager.loaded_model_config)

    def on_new_marks(self, marks):
        """Handle new marks"""
        self.model_manager.set_auto_labeling_marks(marks)
        self.run_prediction()

    def on_open(self):
        pass

    def on_close(self):
        return True
