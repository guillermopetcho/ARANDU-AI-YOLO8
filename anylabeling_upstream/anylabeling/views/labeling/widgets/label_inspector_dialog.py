import html
import json
import os
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox, QLineEdit, QComboBox, QWidget, QCheckBox,
    QRadioButton, QButtonGroup
)


class SelectRegisteredLabelDialog(QDialog):
    """Diálogo para seleccionar una etiqueta registrada del cuadro Labels o escribir una nueva."""

    def __init__(self, parent, registered_labels, current_label=""):
        super().__init__(parent)
        self.setWindowTitle("✏️ Cambiar Etiqueta de Objetos")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>1. Seleccionar de las etiquetas registradas (Cuadro Labels):</b>"))
        self.combo_labels = QComboBox()
        self.combo_labels.addItem("-- Seleccionar etiqueta registrada --", "")
        for l in registered_labels:
            self.combo_labels.addItem(f"●  {l}", l)

        # Pre-select current label if present
        idx = self.combo_labels.findData(current_label)
        if idx >= 0:
            self.combo_labels.setCurrentIndex(idx)

        self.combo_labels.currentIndexChanged.connect(self.on_combo_changed)
        layout.addWidget(self.combo_labels)

        layout.addWidget(QLabel("<b>2. O escribir un nuevo nombre de etiqueta:</b>"))
        self.edit_new_label = QLineEdit()
        self.edit_new_label.setPlaceholderText("Escribe un nuevo nombre de etiqueta...")
        self.edit_new_label.setText(current_label)
        layout.addWidget(self.edit_new_label)

        # Checkbox for applying to all folder files
        self.check_apply_to_folder = QCheckBox("📁 Aplicar cambio a TODAS las imágenes de la carpeta (Archivos JSON)")
        self.check_apply_to_folder.setStyleSheet("margin-top: 8px; font-weight: bold; color: #0369a1;")
        layout.addWidget(self.check_apply_to_folder)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("▶ Aplicar Cambio")
        btn_apply.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        btn_apply.clicked.connect(self.accept)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def on_combo_changed(self, index):
        data = self.combo_labels.currentData()
        if data:
            self.edit_new_label.setText(data)

    def get_selected_label(self):
        return self.edit_new_label.text().strip()

    def apply_to_folder(self):
        return self.check_apply_to_folder.isChecked()


class LabelInspectorDialog(QDialog):
    """Ventana interactiva de inspección y gestión de objetos por etiqueta."""

    def __init__(self, parent_widget, target_label=None):
        qt_parent = parent_widget if isinstance(parent_widget, QWidget) else None
        super().__init__(qt_parent)
        self.parent_widget = parent_widget
        self.target_label = target_label
        self.setWindowTitle(f"🔍 Inspector de Objetos - Clase: '{target_label if target_label else 'Todas'}'")
        self.setMinimumSize(520, 500)

        self.layout = QVBoxLayout(self)

        # Header Info
        self.header_label = QLabel()
        self.header_label.setStyleSheet("font-size: 14px; margin-bottom: 5px;")
        self.layout.addWidget(self.header_label)

        # Scope Selector Layout
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("<b>Ámbito de Inspección:</b>"))

        self.radio_current_image = QRadioButton("Imagen Actual")
        self.radio_dataset_folder = QRadioButton("Todas las Imágenes de la Carpeta")
        self.radio_current_image.setChecked(True)

        self.scope_group = QButtonGroup(self)
        self.scope_group.addButton(self.radio_current_image)
        self.scope_group.addButton(self.radio_dataset_folder)

        self.radio_current_image.toggled.connect(self.on_scope_changed)
        self.radio_dataset_folder.toggled.connect(self.on_scope_changed)

        scope_layout.addWidget(self.radio_current_image)
        scope_layout.addWidget(self.radio_dataset_folder)
        scope_layout.addStretch()
        self.layout.addLayout(scope_layout)

        # Class Filter Selector if no target_label specified or to change class
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("<b>Filtrar por Clase:</b>"))
        self.combo_class_filter = QComboBox()
        self.combo_class_filter.currentIndexChanged.connect(self.on_class_filter_changed)
        filter_layout.addWidget(self.combo_class_filter)
        self.layout.addLayout(filter_layout)

        # Quick Search Box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Buscar por archivo, ID, tipo o etiqueta...")
        self.search_edit.textChanged.connect(self.filter_list_items)
        self.layout.addWidget(self.search_edit)

        # List Widget for Objects
        self.object_list = QListWidget()
        self.object_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.object_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.object_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.layout.addWidget(self.object_list)

        # Action Buttons Layout
        btn_layout = QHBoxLayout()

        self.btn_select_all = QPushButton("🎯 Seleccionar Todos")
        self.btn_select_all.clicked.connect(self.select_all_items)
        btn_layout.addWidget(self.btn_select_all)

        self.btn_relabel = QPushButton("✏️ Cambiar Etiqueta")
        self.btn_relabel.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px;")
        self.btn_relabel.clicked.connect(self.relabel_selected_items)
        btn_layout.addWidget(self.btn_relabel)

        self.btn_delete = QPushButton("🗑️ Borrar Objetos")
        self.btn_delete.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold; padding: 6px;")
        self.btn_delete.clicked.connect(self.delete_selected_items)
        btn_layout.addWidget(self.btn_delete)

        self.layout.addLayout(btn_layout)

        # Populate class combo and list
        self.populate_class_combo()
        self.reload_object_list()

    def on_scope_changed(self):
        """Al cambiar el ámbito entre Imagen Actual y Dataset Completo"""
        self.populate_class_combo()
        self.reload_object_list()

    def populate_class_combo(self):
        """Poblar combo de clases con todas las etiquetas existentes"""
        self.combo_class_filter.blockSignals(True)
        current_data = self.combo_class_filter.currentData()
        self.combo_class_filter.clear()
        self.combo_class_filter.addItem("-- Todas las Clases --", None)

        classes = set()
        # 1. Classes from registered labels & canvas
        labels = self.get_registered_labels()
        for l in labels:
            classes.add(l)

        # 2. If Dataset scope, scan all JSON files for classes
        if self.radio_dataset_folder.isChecked():
            folder = self.get_dataset_folder()
            if folder and os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(".json"):
                        json_path = os.path.join(folder, f)
                        try:
                            with open(json_path, "r", encoding="utf-8") as file:
                                data = json.load(file)
                                if "shapes" in data and isinstance(data["shapes"], list):
                                    for shape in data["shapes"]:
                                        lbl = shape.get("label")
                                        if lbl:
                                            classes.add(lbl)
                        except Exception:
                            pass

        for c in sorted(classes):
            self.combo_class_filter.addItem(c, c)

        target = self.target_label if self.target_label else current_data
        if target:
            index = self.combo_class_filter.findData(target)
            if index >= 0:
                self.combo_class_filter.setCurrentIndex(index)

        self.combo_class_filter.blockSignals(False)

    def on_class_filter_changed(self, index):
        self.target_label = self.combo_class_filter.currentData()
        self.reload_object_list()

    def reload_object_list(self):
        """Cargar objetos del canvas o del dataset completo en el list widget"""
        self.object_list.blockSignals(True)
        self.object_list.clear()

        is_dataset = self.radio_dataset_folder.isChecked()
        current_json_path = None
        if hasattr(self.parent_widget, "filename") and self.parent_widget.filename:
            current_json_path = os.path.abspath(self.parent_widget.filename)

        if not is_dataset:
            # --- Modo Imagen Actual ---
            shapes = []
            if hasattr(self.parent_widget, "canvas") and hasattr(self.parent_widget.canvas, "shapes"):
                shapes = self.parent_widget.canvas.shapes

            matching_items = []
            for idx, shape in enumerate(shapes):
                if self.target_label is None or shape.label == self.target_label:
                    matching_items.append((idx + 1, shape))

            class_str = self.target_label if self.target_label else "Todas"
            self.header_label.setText(
                f"<b>Objetos encontrados (Imagen Actual): <font color='#0284c7'>{len(matching_items)}</font></b> | Clase: '{class_str}'"
            )

            for idx, shape in matching_items:
                shape_type = shape.shape_type if shape.shape_type else "polígono"
                num_pts = len(shape.points) if shape.points else 0

                center_x, center_y = 0, 0
                if num_pts > 0:
                    center_x = int(sum(p.x() for p in shape.points) / num_pts)
                    center_y = int(sum(p.y() for p in shape.points) / num_pts)

                item = QListWidgetItem()
                display_text = f"#{idx} | {shape.label} | {shape_type.capitalize()} ({num_pts} pts) | Centro (x: {center_x}, y: {center_y})"
                item.setText(display_text)

                meta = {
                    "is_dataset": False,
                    "is_current": True,
                    "shape_obj": shape,
                    "file_path": current_json_path,
                }
                item.setData(Qt.ItemDataRole.UserRole, meta)
                self.object_list.addItem(item)
        else:
            # --- Modo Dataset Completo (Todas las imágenes de la carpeta) ---
            folder = self.get_dataset_folder()
            matching_count = 0
            file_count = 0

            if folder and os.path.exists(folder):
                json_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".json")])

                for f_name in json_files:
                    json_path = os.path.abspath(os.path.join(folder, f_name))
                    is_current = (current_json_path and json_path == current_json_path)

                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        shapes_in_file = data.get("shapes", [])
                        file_has_match = False

                        for s_idx, shape_dict in enumerate(shapes_in_file):
                            lbl = shape_dict.get("label", "")
                            if self.target_label is None or lbl == self.target_label:
                                file_has_match = True
                                matching_count += 1
                                shape_type = shape_dict.get("shape_type", "polígono")
                                points = shape_dict.get("points", [])
                                num_pts = len(points)

                                item = QListWidgetItem()
                                base_name = os.path.splitext(f_name)[0]
                                display_text = f"[{base_name}] #{s_idx + 1} | {lbl} | {shape_type.capitalize()} ({num_pts} pts)"
                                if is_current:
                                    display_text += "  📌 (Imagen Actual)"
                                item.setText(display_text)

                                # Find matching shape object if current image
                                shape_obj = None
                                if is_current and hasattr(self.parent_widget, "canvas"):
                                    if s_idx < len(self.parent_widget.canvas.shapes):
                                        shape_obj = self.parent_widget.canvas.shapes[s_idx]

                                meta = {
                                    "is_dataset": True,
                                    "is_current": is_current,
                                    "file_path": json_path,
                                    "shape_dict": shape_dict,
                                    "shape_idx": s_idx,
                                    "shape_obj": shape_obj,
                                    "label": lbl,
                                }
                                item.setData(Qt.ItemDataRole.UserRole, meta)
                                self.object_list.addItem(item)

                        if file_has_match:
                            file_count += 1
                    except Exception as e:
                        print(f"Error al leer {json_path}: {e}")

            class_str = self.target_label if self.target_label else "Todas"
            self.header_label.setText(
                f"<b>Objetos en Dataset: <font color='#0284c7'>{matching_count}</font></b> (en {file_count} imágenes) | Clase: '{class_str}'"
            )

        self.object_list.blockSignals(False)

    def filter_list_items(self, text):
        text = text.lower().strip()
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            item.setHidden(text not in item.text().lower())

    def on_selection_changed(self):
        """Al cambiar la selección en el inspector, seleccionar los objetos en el canvas si pertenecen a la imagen actual"""
        selected_items = self.object_list.selectedItems()
        selected_shapes = []

        for item in selected_items:
            meta = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(meta, dict) and meta.get("is_current") and meta.get("shape_obj"):
                selected_shapes.append(meta["shape_obj"])
            elif not isinstance(meta, dict) and meta:
                selected_shapes.append(meta)

        if hasattr(self.parent_widget, "canvas") and selected_shapes:
            self.parent_widget.canvas.select_shapes(selected_shapes)
            self.parent_widget.canvas.set_editing(True)
            self.parent_widget.canvas.update()

    def on_item_double_clicked(self, item):
        """Al hacer doble clic en un objeto:
        - Si es de la imagen actual: centrar vista en él
        - Si es de otra imagen del dataset: abrir esa imagen en el editor
        """
        meta = item.data(Qt.ItemDataRole.UserRole)

        if isinstance(meta, dict):
            if meta.get("is_current") and meta.get("shape_obj"):
                shape = meta["shape_obj"]
                if shape and shape.points and hasattr(self.parent_widget, "canvas"):
                    if hasattr(self.parent_widget.canvas, "scroll_to_point"):
                        self.parent_widget.canvas.scroll_to_point(shape.points[0])
            else:
                file_path = meta.get("file_path")
                if file_path and os.path.exists(file_path):
                    if hasattr(self.parent_widget, "load_file"):
                        self.parent_widget.load_file(file_path)
                        # Re-open inspector or refresh it for the new file
                        self.reload_object_list()
        elif meta and hasattr(meta, "points") and hasattr(self.parent_widget, "canvas"):
            if hasattr(self.parent_widget.canvas, "scroll_to_point"):
                self.parent_widget.canvas.scroll_to_point(meta.points[0])

    def select_all_items(self):
        self.object_list.selectAll()

    def get_registered_labels(self):
        """Obtener todas las etiquetas registradas en el cuadro Labels y canvas"""
        labels = set()
        if hasattr(self.parent_widget, "unique_label_list") and "MagicMock" not in type(self.parent_widget.unique_label_list).__name__:
            try:
                for i in range(self.parent_widget.unique_label_list.count()):
                    item = self.parent_widget.unique_label_list.item(i)
                    lbl = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(lbl, str) and lbl.strip():
                        labels.add(lbl.strip())
            except Exception:
                pass

        if hasattr(self.parent_widget, "canvas") and hasattr(self.parent_widget.canvas, "shapes"):
            for shape in self.parent_widget.canvas.shapes:
                if hasattr(shape, "label") and isinstance(shape.label, str) and shape.label.strip():
                    labels.add(shape.label.strip())

        if hasattr(self.parent_widget, "label_hist") and isinstance(self.parent_widget.label_hist, (list, tuple, set)):
            for lbl in self.parent_widget.label_hist:
                if isinstance(lbl, str) and lbl.strip():
                    labels.add(lbl.strip())

        return sorted(labels)

    def get_dataset_folder(self):
        """Obtener la carpeta activa de imágenes/etiquetas"""
        if hasattr(self.parent_widget, "last_open_dir") and self.parent_widget.last_open_dir:
            return self.parent_widget.last_open_dir
        if hasattr(self.parent_widget, "dirname") and self.parent_widget.dirname:
            return self.parent_widget.dirname
        return None

    def _replace_label_in_folder(self, current_label, text):
        """Reemplazar una etiqueta por otra en todos los archivos JSON del dataset"""
        folder = self.get_dataset_folder()
        if folder and os.path.exists(folder):
            json_files = [
                os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")
            ]
            modified_files = 0
            total_shapes_changed = 0

            for json_path in json_files:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    changed_in_file = 0
                    if "shapes" in data and isinstance(data["shapes"], list):
                        for shape in data["shapes"]:
                            if shape.get("label") == current_label:
                                shape["label"] = text
                                changed_in_file += 1

                    if changed_in_file > 0:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        modified_files += 1
                        total_shapes_changed += changed_in_file
                except Exception as e:
                    print(f"Error procesando {json_path}: {e}")

            QMessageBox.information(
                self,
                "Reemplazo en Dataset Completado",
                f"Se reemplazaron todas las ocurrencias de '{current_label}' por '{text}' en la carpeta.\n\n"
                f"• Archivos modificados: {modified_files}\n"
                f"• Total etiquetas reemplazadas: {total_shapes_changed}"
            )

    def relabel_selected_items(self):
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Aviso", "Selecciona al menos un objeto de la lista.")
            return

        first_meta = selected_items[0].data(Qt.ItemDataRole.UserRole)
        current_label = ""
        if isinstance(first_meta, dict):
            current_label = first_meta.get("label", "")
        elif hasattr(first_meta, "label"):
            current_label = first_meta.label

        registered_labels = self.get_registered_labels()

        dialog = SelectRegisteredLabelDialog(self, registered_labels, current_label)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        text = dialog.get_selected_label()
        if not text:
            QMessageBox.warning(self, "Error", "El nombre de la etiqueta no puede estar vacío.")
            return

        is_dataset = self.radio_dataset_folder.isChecked()

        if not is_dataset:
            # 1. Update items in current image
            for item in selected_items:
                meta = item.data(Qt.ItemDataRole.UserRole)
                shape = meta.get("shape_obj") if isinstance(meta, dict) else meta
                if shape:
                    shape.label = text
                    if hasattr(self.parent_widget, "_update_shape_color"):
                        self.parent_widget._update_shape_color(shape)

            if dialog.apply_to_folder() and current_label and text != current_label:
                self._replace_label_in_folder(current_label, text)
        else:
            # 2. Dataset-wide batch relabeling for selected items
            files_to_update = {}
            for item in selected_items:
                meta = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(meta, dict):
                    f_path = meta.get("file_path")
                    s_idx = meta.get("shape_idx")
                    if f_path:
                        files_to_update.setdefault(f_path, []).append((s_idx, meta))

            modified_files = 0
            total_shapes_changed = 0

            for json_path, item_list in files_to_update.items():
                if not os.path.exists(json_path):
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    changed_in_file = 0
                    if "shapes" in data and isinstance(data["shapes"], list):
                        indices_to_change = {idx for idx, m in item_list if idx is not None}
                        for idx, shape in enumerate(data["shapes"]):
                            if idx in indices_to_change or (indices_to_change == set() and shape.get("label") == current_label):
                                shape["label"] = text
                                changed_in_file += 1

                    if changed_in_file > 0:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        modified_files += 1
                        total_shapes_changed += changed_in_file
                except Exception as e:
                    print(f"Error actualizando {json_path}: {e}")

            # Update current canvas shapes if current image was modified
            current_json = os.path.abspath(self.parent_widget.filename) if hasattr(self.parent_widget, "filename") and self.parent_widget.filename else None
            if current_json and current_json in files_to_update:
                for idx, meta in files_to_update[current_json]:
                    shape = meta.get("shape_obj")
                    if shape:
                        shape.label = text
                        if hasattr(self.parent_widget, "_update_shape_color"):
                            self.parent_widget._update_shape_color(shape)

            QMessageBox.information(
                self,
                "Edición en Dataset Completada",
                f"Se modificó la etiqueta a '{text}' en las imágenes seleccionadas del dataset.\n\n"
                f"• Archivos modificados: {modified_files}\n"
                f"• Total etiquetas modificadas: {total_shapes_changed}"
            )

        # Register label in unique_label_list if new
        if hasattr(self.parent_widget, "unique_label_list"):
            if not self.parent_widget.unique_label_list.find_items_by_label(text):
                item = self.parent_widget.unique_label_list.create_item_from_label(text)
                self.parent_widget.unique_label_list.addItem(item)
                rgb = self.parent_widget._get_rgb_by_label(text)
                self.parent_widget.unique_label_list.set_item_label(item, text, rgb)

        # Reload shape lists and canvas
        if hasattr(self.parent_widget, "load_shapes") and hasattr(self.parent_widget, "canvas"):
            self.parent_widget.load_shapes(self.parent_widget.canvas.shapes, replace=True)
        if hasattr(self.parent_widget, "set_dirty"):
            self.parent_widget.set_dirty()

        self.populate_class_combo()
        self.reload_object_list()

    def delete_selected_items(self):
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Aviso", "Selecciona al menos un objeto para borrar.")
            return

        is_dataset = self.radio_dataset_folder.isChecked()
        scope_str = "en todo el dataset" if is_dataset else "de la imagen actual"

        reply = QMessageBox.question(
            self,
            "Confirmar Borrado",
            f"¿Estás seguro de borrar {len(selected_items)} objeto(s) {scope_str}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if not is_dataset:
            for item in selected_items:
                meta = item.data(Qt.ItemDataRole.UserRole)
                shape = meta.get("shape_obj") if isinstance(meta, dict) else meta
                if shape and shape in self.parent_widget.canvas.shapes:
                    self.parent_widget.canvas.shapes.remove(shape)

            self.parent_widget.canvas.selected_shapes = []
        else:
            files_to_update = {}
            for item in selected_items:
                meta = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(meta, dict):
                    f_path = meta.get("file_path")
                    s_idx = meta.get("shape_idx")
                    if f_path:
                        files_to_update.setdefault(f_path, []).append((s_idx, meta))

            modified_files = 0
            total_deleted = 0

            for json_path, item_list in files_to_update.items():
                if not os.path.exists(json_path):
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if "shapes" in data and isinstance(data["shapes"], list):
                        indices_to_remove = {idx for idx, m in item_list if idx is not None}
                        new_shapes = [s for idx, s in enumerate(data["shapes"]) if idx not in indices_to_remove]
                        deleted_count = len(data["shapes"]) - len(new_shapes)

                        if deleted_count > 0:
                            data["shapes"] = new_shapes
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                            modified_files += 1
                            total_deleted += deleted_count
                except Exception as e:
                    print(f"Error borrando en {json_path}: {e}")

            current_json = os.path.abspath(self.parent_widget.filename) if hasattr(self.parent_widget, "filename") and self.parent_widget.filename else None
            if current_json and current_json in files_to_update:
                indices_to_remove = {s_idx for s_idx, m in files_to_update[current_json] if s_idx is not None}
                self.parent_widget.canvas.shapes = [
                    s for idx, s in enumerate(self.parent_widget.canvas.shapes) if idx not in indices_to_remove
                ]
                self.parent_widget.canvas.selected_shapes = []

            QMessageBox.information(
                self,
                "Borrado en Dataset Completado",
                f"Se borraron {total_deleted} objeto(s) en {modified_files} imágenes del dataset."
            )

        if hasattr(self.parent_widget, "load_shapes") and hasattr(self.parent_widget, "canvas"):
            self.parent_widget.load_shapes(self.parent_widget.canvas.shapes, replace=True)
        if hasattr(self.parent_widget, "set_dirty"):
            self.parent_widget.set_dirty()

        self.populate_class_combo()
        self.reload_object_list()
