import os
import json
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QRadioButton, QButtonGroup, QMessageBox, QProgressBar
)

class GlobalClassReplaceDialog(QDialog):
    """Diálogo para renombrar/reemplazar clases globalmente en la imagen actual o en la carpeta."""

    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.setWindowTitle("🔁 Reemplazar Clase Global")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        header = QLabel("<b>Búsqueda y Reemplazo Global de Clases</b>")
        header.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(header)

        # Source Class Selection
        layout.addWidget(QLabel("<b>1. Clase Origen a Reemplazar:</b>"))
        self.combo_source = QComboBox()
        self.populate_source_classes()
        layout.addWidget(self.combo_source)

        # Target Class Input
        layout.addWidget(QLabel("<b>2. Nueva Clase (Destino):</b>"))
        self.edit_target = QLineEdit()
        self.edit_target.setPlaceholderText("Escribe o selecciona la nueva clase...")
        layout.addWidget(self.edit_target)

        # Quick select target from existing classes
        if self.combo_source.count() > 0:
            layout.addWidget(QLabel("<font color='#64748b'><i>O elegir de clases existentes:</i></font>"))
            self.combo_target_existing = QComboBox()
            self.combo_target_existing.addItem("-- Seleccionar de lista --")
            for i in range(self.combo_source.count()):
                self.combo_target_existing.addItem(self.combo_source.itemText(i))
            self.combo_target_existing.currentIndexChanged.connect(self.on_existing_target_selected)
            layout.addWidget(self.combo_target_existing)

        # Scope Selection
        layout.addWidget(QLabel("<b>3. Alcance del Reemplazo:</b>"))
        self.radio_group = QButtonGroup(self)
        self.radio_current = QRadioButton("Imagen Actual")
        self.radio_current.setChecked(True)
        self.radio_folder = QRadioButton("Todas las Imágenes de la Carpeta (Archivos JSON)")
        self.radio_group.addButton(self.radio_current)
        self.radio_group.addButton(self.radio_folder)
        layout.addWidget(self.radio_current)
        layout.addWidget(self.radio_folder)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_replace = QPushButton("▶ Aplicar Reemplazo")
        self.btn_replace.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_replace.clicked.connect(self.execute_replace)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_replace)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def populate_source_classes(self):
        """Poblar clases únicas presentes en la aplicación"""
        self.combo_source.clear()
        labels = set()

        # Obtenemos etiquetas de la lista única
        if hasattr(self.parent_widget, "unique_label_list"):
            for i in range(self.parent_widget.unique_label_list.count()):
                item = self.parent_widget.unique_label_list.item(i)
                label = item.data(Qt.ItemDataRole.UserRole)
                if label:
                    labels.add(label)

        # Obtenemos etiquetas de las figuras del canvas actual
        if hasattr(self.parent_widget, "canvas") and hasattr(self.parent_widget.canvas, "shapes"):
            for shape in self.parent_widget.canvas.shapes:
                if shape.label:
                    labels.add(shape.label)

        for label in sorted(labels):
            self.combo_source.addItem(label)

    def on_existing_target_selected(self, index):
        if index > 0:
            text = self.combo_target_existing.currentText()
            self.edit_target.setText(text)

    def execute_replace(self):
        source_label = self.combo_source.currentText().strip()
        target_label = self.edit_target.text().strip()

        if not source_label:
            QMessageBox.warning(self, "Error", "Debes seleccionar la clase origen a reemplazar.")
            return

        if not target_label:
            QMessageBox.warning(self, "Error", "Debes especificar el nuevo nombre de clase.")
            return

        if source_label == target_label:
            QMessageBox.warning(self, "Aviso", "La clase origen y la clase destino son idénticas.")
            return

        if self.radio_current.isChecked():
            # Reemplazar en imagen actual
            count = self.replace_in_current_image(source_label, target_label)
            QMessageBox.information(
                self, "Éxito", f"Se cambiaron {count} etiquetas de '{source_label}' a '{target_label}' en la imagen actual."
            )
            self.accept()
        else:
            # Reemplazar en la carpeta de etiquetas JSON
            self.replace_in_folder(source_label, target_label)

    def replace_in_current_image(self, source_label, target_label):
        count = 0
        if not hasattr(self.parent_widget, "canvas"):
            return 0

        # Cambiar figuras en canvas
        for shape in self.parent_widget.canvas.shapes:
            if shape.label == source_label:
                shape.label = target_label
                count += 1

        # Actualizar lista de etiquetas
        if hasattr(self.parent_widget, "label_list"):
            for item in self.parent_widget.label_list:
                shape = item.shape()
                if shape and shape.label == target_label:
                    color = shape.fill_color.getRgb()[:3]
                    item.setText(f"{target_label} <font color=\"#{color[0]:02x}{color[1]:02x}{color[2]:02x}\">●</font>")

        # Actualizar lista única de etiquetas
        if hasattr(self.parent_widget, "unique_label_list"):
            # Si no existe la nueva etiqueta, crearla
            if not self.parent_widget.unique_label_list.find_items_by_label(target_label):
                item = self.parent_widget.unique_label_list.create_item_from_label(target_label)
                self.parent_widget.unique_label_list.addItem(item)
                rgb = self.parent_widget._get_rgb_by_label(target_label)
                self.parent_widget.unique_label_list.set_item_label(item, target_label, rgb)

        if count > 0:
            self.parent_widget.set_dirty()
            self.parent_widget.canvas.update()

        return count

    def replace_in_folder(self, source_label, target_label):
        if not hasattr(self.parent_widget, "last_open_dir") or not self.parent_widget.last_open_dir:
            if hasattr(self.parent_widget, "dirname") and self.parent_widget.dirname:
                folder = self.parent_widget.dirname
            else:
                QMessageBox.warning(self, "Error", "No se encontró una carpeta abierta.")
                return
        else:
            folder = self.parent_widget.last_open_dir

        json_files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")
        ]

        if not json_files:
            QMessageBox.warning(self, "Aviso", "No se encontraron archivos JSON de etiquetas en la carpeta.")
            return

        reply = QMessageBox.question(
            self,
            "Confirmar Reemplazo Global",
            f"Se buscarán y reemplazarán todas las ocurrencias de '{source_label}' por '{target_label}' en {len(json_files)} archivos JSON.\n¿Deseas continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.progress_bar.show()
        self.progress_bar.setMaximum(len(json_files))

        modified_files = 0
        total_shapes_changed = 0

        for idx, json_path in enumerate(json_files):
            self.progress_bar.setValue(idx + 1)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                changed_in_file = 0
                if "shapes" in data and isinstance(data["shapes"], list):
                    for shape in data["shapes"]:
                        if shape.get("label") == source_label:
                            shape["label"] = target_label
                            changed_in_file += 1

                if changed_in_file > 0:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    modified_files += 1
                    total_shapes_changed += changed_in_file
            except Exception as e:
                print(f"Error procesando {json_path}: {e}")

        # También aplicar a la imagen actual abierta
        self.replace_in_current_image(source_label, target_label)

        QMessageBox.information(
            self,
            "Reemplazo Completado",
            f"Proceso finalizado con éxito.\n- Archivos modificados: {modified_files}\n- Total de etiquetas reemplazadas: {total_shapes_changed}"
        )
        self.accept()
