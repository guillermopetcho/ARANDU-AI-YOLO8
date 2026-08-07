import cv2
import functools
import glob
import html
import json
import math
import numpy as np
import os
import os.path as osp
import re
import shutil
import webbrowser

import imgviz
import natsort
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWhatsThis,
)

from anylabeling.app_info import __appname__
from anylabeling.config import get_config, save_config
from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.styles import AppTheme
from anylabeling.views.labeling import utils
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets import (
    AutoLabelingWidget,
    BrightnessContrastDialog,
    ImageEditorDialog,
    Canvas,
    FileDialogPreview,
    LabelDialog,
    LabelListWidget,
    LabelListWidgetItem,
    ToolBar,
    UniqueLabelQListWidget,
    ZoomWidget,
)

from .widgets.export_dialog import ExportDialog

LABEL_COLORMAP = imgviz.label_colormap().copy()

# Green for the first label
LABEL_COLORMAP[2] = LABEL_COLORMAP[1]
LABEL_COLORMAP[1] = [0, 180, 33]


class DirectoryBookmarksDialog(QtWidgets.QDialog):
    """Dialog to manage and pick from saved/bookmark directories."""

    def __init__(self, parent, saved_dirs):
        super().__init__(parent)
        self.parent = parent
        self.saved_dirs = [d for d in saved_dirs if d] if saved_dirs else []
        self.selected_dir = None

        self.setWindowTitle("📁 Gestor de Directorios Guardados")
        self.setMinimumSize(620, 400)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("<b>📁 Selecciona una Carpeta Guardada o Agrega una Nueva:</b>")
        header.setStyleSheet("font-size: 13px; margin-bottom: 4px;")
        layout.addWidget(header)

        # List widget
        self.dir_list_widget = QtWidgets.QListWidget()
        self.dir_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 4px;
                background-color: #ffffff;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QListWidget::item:selected {
                background-color: #0067c0;
                color: #ffffff;
                font-weight: bold;
            }
        """)
        self.dir_list_widget.itemDoubleClicked.connect(self.accept_selected)
        layout.addWidget(self.dir_list_widget)

        self.populate_list()

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_add = QtWidgets.QPushButton("➕ Agregar Carpeta...")
        self.btn_add.setToolTip("Buscar en el disco y guardar una nueva carpeta en la lista")
        self.btn_add.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.btn_add.clicked.connect(self.browse_and_add)

        self.btn_remove = QtWidgets.QPushButton("🗑️ Eliminar de la Lista")
        self.btn_remove.setToolTip("Quitar la carpeta seleccionada de la lista de guardados")
        self.btn_remove.clicked.connect(self.remove_selected)

        self.btn_open = QtWidgets.QPushButton("📂 Abrir Seleccionada")
        self.btn_open.setToolTip("Cargar imágenes de la carpeta seleccionada")
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #0067c0;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        self.btn_open.clicked.connect(self.accept_selected)

        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_open)

        layout.addLayout(btn_layout)

    def populate_list(self):
        self.dir_list_widget.clear()
        for d in self.saved_dirs:
            if osp.exists(d):
                name = osp.basename(d) or d
                item = QtWidgets.QListWidgetItem(f"📁 {name}   ({d})")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, d)
                self.dir_list_widget.addItem(item)
            else:
                item = QtWidgets.QListWidgetItem(f"⚠️ {d} (No existe en disco)")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, d)
                item.setForeground(QtGui.QColor("#ef4444"))
                self.dir_list_widget.addItem(item)
        if self.dir_list_widget.count() > 0:
            self.dir_list_widget.setCurrentRow(0)

    def browse_and_add(self):
        default_path = self.saved_dirs[0] if self.saved_dirs else "."
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Seleccionar Nueva Carpeta para Guardar",
            default_path,
            QtWidgets.QFileDialog.Option.ShowDirsOnly
        )
        if new_dir:
            new_dir = str(new_dir)
            if new_dir not in self.saved_dirs:
                self.saved_dirs.append(new_dir)
                self.populate_list()
            for i in range(self.dir_list_widget.count()):
                item = self.dir_list_widget.item(i)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) == new_dir:
                    self.dir_list_widget.setCurrentItem(item)
                    break
            self.accept_selected()

    def remove_selected(self):
        current_item = self.dir_list_widget.currentItem()
        if current_item:
            d = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if d in self.saved_dirs:
                self.saved_dirs.remove(d)
                self.populate_list()

    def accept_selected(self):
        current_item = self.dir_list_widget.currentItem()
        if current_item:
            d = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if osp.exists(d):
                self.selected_dir = d
                self.accept()
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Carpeta no encontrada", f"La carpeta '{d}' no existe en el disco."
                )


class LabelingWidget(LabelDialog):
    """The main widget for labeling images"""

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    next_files_changed = QtCore.pyqtSignal(list)

    def __init__(
        self,
        parent=None,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        self.parent = parent
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        self.filename = None
        self.image_path = None
        self.image_data = None
        self.label_file = None
        self.other_data = {}

        # see configs/anylabeling_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(LabelDialog, self).__init__()

        # Whether we need to save or not.
        self.dirty = False

        self._no_selection_slot = False

        self._copied_shapes = None

        # Initialize the QSettings object early
        self.settings = QtCore.QSettings("anylabeling", "anylabeling")

        # Initialize a QMainWindow for dock widget functionality
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AnimatedDocks
        )
        # Set central widget for the main window
        self.main_window.setCentralWidget(QtWidgets.QWidget())
        self.main_window.centralWidget().setLayout(QtWidgets.QVBoxLayout())
        self.main_window.centralWidget().layout().setContentsMargins(0, 0, 0, 0)

        # Main widgets and related state.
        self.label_dialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.label_list = LabelListWidget()
        self.last_open_dir = None
        self.q_plus_plus_mode = False
        self._last_selected_unique_label = None
        self.saved_directories = self.settings.value("saved_directories", []) or []

        features = (
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        # Apply dock title styling
        dock_title_style = (
            "QDockWidget::title {"
            "text-align: center;"
            "border-radius: 4px;"
            "margin-bottom: 2px;"
            f"background-color: {AppTheme.get_color('dock_title_bg')};"
            f"color: {AppTheme.get_color('dock_title_text')};"
            "}"
        )

        # Create right sidebar with shape text editor
        shape_text_widget = QtWidgets.QWidget()
        shape_text_layout = QVBoxLayout()
        shape_text_layout.setContentsMargins(0, 0, 0, 0)
        self.shape_text_label = QLabel("Object Text")
        self.shape_text_label.setStyleSheet(
            "QLabel {"
            "text-align: center;"
            "padding: 0px;"
            "font-size: 11px;"
            "margin-bottom: 5px;"
            "}"
        )
        self.shape_text_edit = QPlainTextEdit()
        shape_text_layout.addWidget(
            self.shape_text_label, 0, Qt.AlignmentFlag.AlignCenter
        )
        shape_text_layout.addWidget(self.shape_text_edit)
        shape_text_widget.setLayout(shape_text_layout)

        # Add shape text widget to dock
        self.shape_text_dock = QtWidgets.QDockWidget(
            self.tr("Text Editor"), self.main_window
        )
        self.shape_text_dock.setObjectName("TextEditor")
        self.shape_text_dock.setFeatures(features)
        self.shape_text_dock.setWidget(shape_text_widget)
        self.shape_text_dock.setStyleSheet(dock_title_style)
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.shape_text_dock
        )

        # Text Editor Actions - created after dock is initialized
        # Set shortcut for the text editor toggle view action
        self.shape_text_dock.toggleViewAction().setShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.Key.Key_T
            )
        )

        # Create dock widgets with movable feature enabled
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self.main_window)
        self.flag_dock.setObjectName("Flags")
        self.flag_dock.setFeatures(features)
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.load_flags(dict.fromkeys(config["flags"], False))
        else:
            self.flag_dock.hide()
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.set_dirty)
        self.flag_dock.setStyleSheet(dock_title_style)
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.flag_dock
        )

        self.label_list.item_selection_changed.connect(self.label_selection_changed)
        self.label_list.item_double_clicked.connect(self.edit_label)
        self.label_list.item_changed.connect(self.label_item_changed)
        self.label_list.item_dropped.connect(self.label_order_changed)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Objects"), self.main_window)
        self.shape_dock.setObjectName("Objects")
        self.shape_dock.setFeatures(features)
        self.shape_dock.setWidget(self.label_list)
        self.shape_dock.setStyleSheet(dock_title_style)
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.shape_dock
        )

        self.unique_label_list = UniqueLabelQListWidget()
        self.unique_label_list.setToolTip(
            self.tr("Select label to start annotating for it. Press 'Esc' to deselect.")
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(item, label, rgb)
        self.unique_label_list.itemClicked.connect(self.on_unique_label_item_clicked)
        self.unique_label_list.itemDoubleClicked.connect(
            lambda item: self.open_label_inspector(item.data(Qt.ItemDataRole.UserRole))
        )
        self.unique_label_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.unique_label_list.customContextMenuRequested.connect(self.pop_unique_label_list_menu)
        self.label_dock = QtWidgets.QDockWidget(self.tr("Labels"), self.main_window)
        self.label_dock.setObjectName("Labels")
        self.label_dock.setFeatures(features)
        self.label_dock.setWidget(self.unique_label_list)
        self.label_dock.setStyleSheet(dock_title_style)
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.label_dock
        )

        self.btn_saved_dirs = QtWidgets.QPushButton("📁 Carpetas Guardadas...")
        self.btn_saved_dirs.setToolTip("Abrir gestor de directorios guardados y favoritos")
        self.btn_saved_dirs.clicked.connect(self.open_saved_directories_dialog)
        self.btn_saved_dirs.setStyleSheet("""
            QPushButton {
                background-color: #0067c0;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 10px;
                border-radius: 4px;
                margin: 2px 2px 4px 2px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)

        self.btn_filter_status = QtWidgets.QPushButton("🔍 Filtro Activo [❌ Quitar]")
        self.btn_filter_status.setToolTip("Haz clic para quitar el filtro de etiqueta y ver todas las imágenes")
        self.btn_filter_status.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                font-weight: bold;
                padding: 5px 8px;
                border-radius: 4px;
                margin: 2px 2px 4px 2px;
            }
            QPushButton:hover {
                background-color: #ef4444;
            }
        """)
        self.btn_filter_status.clicked.connect(self.clear_label_filter)
        self.btn_filter_status.setVisible(False)

        self.file_search = QtWidgets.QLineEdit()
        self.file_search.setPlaceholderText(self.tr("Search Filename"))
        self.file_search.textChanged.connect(self.file_search_changed)
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list_widget.viewport().installEventFilter(self)
        self.file_list_widget.itemSelectionChanged.connect(self.file_selection_changed)
        self.file_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list_widget.customContextMenuRequested.connect(self.pop_file_list_menu)
        file_list_layout = QtWidgets.QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(0)
        file_list_layout.addWidget(self.btn_saved_dirs)
        file_list_layout.addWidget(self.btn_filter_status)
        file_list_layout.addWidget(self.file_search)
        file_list_layout.addWidget(self.file_list_widget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("Files"), self.main_window)
        self.file_dock.setObjectName("Files")
        self.file_dock.setFeatures(features)
        file_list_widget = QtWidgets.QWidget()
        file_list_widget.setLayout(file_list_layout)
        self.file_dock.setWidget(file_list_widget)
        self.file_dock.setStyleSheet(dock_title_style)
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.file_dock
        )

        # Clean Startup View: Only Files and Labels remain open by default
        self.shape_text_dock.hide()
        self.flag_dock.hide()
        self.shape_dock.hide()

        self.zoom_widget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.label_list.canvas = Canvas(
            parent=self,
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
        )
        self.canvas.zoom_request.connect(self.zoom_request)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Orientation.Vertical: scroll_area.verticalScrollBar(),
            Qt.Orientation.Horizontal: scroll_area.horizontalScrollBar(),
        }
        self.canvas.scroll_request.connect(self.scroll_request)

        self.canvas.new_shape.connect(self.new_shape)
        self.canvas.shape_moved.connect(self.set_dirty)
        self.canvas.selection_changed.connect(self.shape_selection_changed)
        self.canvas.drawing_polygon.connect(self.toggle_drawing_sensitive)

        self._central_widget = scroll_area

        # Actions
        create_action = functools.partial(utils.new_action, self)
        shortcuts = self._config["shortcuts"]
        open_ = create_action(
            self.tr("📂 &Open"),
            self.open_file,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = create_action(
            self.tr("📁 &Open Dir"),
            self.open_folder_dialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        open_saved_dirs = create_action(
            self.tr("📁 &Carpetas Guardadas..."),
            self.open_saved_directories_dialog,
            "Ctrl+Shift+D",
            "open",
            self.tr("Gestor de directorios guardados y favoritos"),
        )
        open_next_image = create_action(
            self.tr("⏩ &Next Image"),
            self.open_next_image,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctrl+Shift to copy labels)"),
            enabled=False,
        )
        open_prev_image = create_action(
            self.tr("⏪ &Prev Image"),
            self.open_prev_image,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctrl+Shift to copy labels)"),
            enabled=False,
        )
        save = create_action(
            self.tr("💾 &Save"),
            self.save_file,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        save_as = create_action(
            self.tr("📥 &Save As"),
            self.save_file_as,
            shortcuts["save_as"],
            "save",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        delete_file = create_action(
            self.tr("🗑️ &Delete File"),
            self.delete_file,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        change_output_dir = create_action(
            self.tr("&Change Output Dir"),
            slot=self.change_output_dir_dialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        save_auto = create_action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.save_auto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        save_auto.setChecked(self._config["auto_save"])

        save_with_image_data = create_action(
            text=self.tr("Save With Image Data"),
            slot=self.enable_save_image_with_data,
            icon="save",
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = create_action(
            self.tr("&Close"),
            self.close_file,
            shortcuts["close"],
            "cancel",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = create_action(
            self.tr("Keep Previous Annotation"),
            self.toggle_keep_prev_mode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "Keep Previous Annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        toggle_auto_use_last_label_mode = create_action(
            self.tr("Auto Use Last Label"),
            self.toggle_auto_use_last_label,
            shortcuts["toggle_auto_use_last_label"],
            None,
            self.tr('Toggle "Auto Use Last Label" mode'),
            checkable=True,
        )
        toggle_auto_use_last_label_mode.setChecked(self._config["auto_use_last_label"])

        create_mode = create_action(
            self.tr("🔺 Polygons (W)"),
            lambda: self.toggle_draw_mode(False, create_mode="polygon"),
            shortcuts["create_polygon"],
            "polygon",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        create_rectangle_mode = create_action(
            self.tr("🔲 Rectangle (R)"),
            lambda: self.toggle_draw_mode(False, create_mode="rectangle"),
            shortcuts["create_rectangle"],
            "rectangle",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        create_cirle_mode = create_action(
            self.tr("⭕ Circle (C)"),
            lambda: self.toggle_draw_mode(False, create_mode="circle"),
            shortcuts["create_circle"],
            "circle",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        create_line_mode = create_action(
            self.tr("📏 Line (L)"),
            lambda: self.toggle_draw_mode(False, create_mode="line"),
            shortcuts["create_line"],
            "line",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        create_point_mode = create_action(
            self.tr("📍 Point (P)"),
            lambda: self.toggle_draw_mode(False, create_mode="point"),
            shortcuts["create_point"],
            "point",
            self.tr("Start drawing points"),
            enabled=False,
        )
        create_line_strip_mode = create_action(
            self.tr("📐 LineStrip"),
            lambda: self.toggle_draw_mode(False, create_mode="linestrip"),
            shortcuts["create_linestrip"],
            "line-strip",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        edit_mode = create_action(
            self.tr("✏️ Edit Object"),
            self.set_edit_mode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )
        group_selected_shapes = create_action(
            self.tr("🔗 Group Shapes"),
            self.canvas.group_selected_shapes,
            shortcuts["group_selected_shapes"],
            "group",
            self.tr("Group shapes by assigning a same group_id"),
            enabled=True,
        )
        ungroup_selected_shapes = create_action(
            self.tr("🔓 Ungroup Shapes"),
            self.canvas.ungroup_selected_shapes,
            shortcuts["ungroup_selected_shapes"],
            "group",
            self.tr("Ungroup shapes"),
            enabled=True,
        )

        delete = create_action(
            self.tr("❌ Delete Polygon"),
            self.delete_selected_shape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = create_action(
            self.tr("Duplicate Polygons"),
            self.duplicate_selected_shape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = create_action(
            self.tr("Copy Object"),
            self.copy_selected_shape,
            shortcuts["copy_polygon"],
            "copy",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = create_action(
            self.tr("Paste Object"),
            self.paste_selected_shape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undo_last_point = create_action(
            self.tr("Undo last point"),
            self.canvas.undo_last_point,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        remove_point = create_action(
            text=self.tr("Remove Selected Point"),
            slot=self.remove_selected_point,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = create_action(
            self.tr("Undo"),
            self.undo_shape_edit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hide_all = create_action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.toggle_polygons, False),
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        show_all = create_action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.toggle_polygons, True),
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )

        documentation = create_action(
            self.tr("&Documentation"),
            self.documentation,
            icon="help",
            tip=self.tr("Show documentation"),
        )

        contact = create_action(
            self.tr("&Contact me"),
            self.contact,
            icon="help",
            tip=self.tr("Show contact page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoom_widget)
        self.zoom_widget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmt_shortcut(f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"),
                utils.fmt_shortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoom_widget.setEnabled(False)

        zoom_in = create_action(
            self.tr("Zoom &In"),
            functools.partial(self.add_zoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoom_out = create_action(
            self.tr("&Zoom Out"),
            functools.partial(self.add_zoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoom_org = create_action(
            self.tr("&Original size"),
            functools.partial(self.set_zoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keep_prev_scale = create_action(
            self.tr("&Keep Previous Scale"),
            self.enable_keep_prev_scale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fit_window = create_action(
            self.tr("&Fit Window"),
            self.set_fit_window,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fit_width = create_action(
            self.tr("Fit &Width"),
            self.set_fit_width,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightness_contrast = create_action(
            self.tr("&Brightness Contrast"),
            self.brightness_contrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        show_cross_line = create_action(
            self.tr("&Show Cross Line"),
            self.enable_show_cross_line,
            tip=self.tr("Show cross line for mouse position"),
            icon="cartesian",
            checkable=True,
            checked=self._config["show_cross_line"],
            enabled=True,
        )
        show_groups = create_action(
            self.tr("&Show Groups"),
            self.enable_show_groups,
            tip=self.tr("Show shape groups"),
            icon=None,
            checkable=True,
            checked=self._config["show_groups"],
            enabled=True,
        )
        show_texts = create_action(
            self.tr("&Show Texts"),
            self.enable_show_texts,
            tip=self.tr("Show text above shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_texts"],
            enabled=True,
        )

        reset_views = create_action(
            self.tr("&Reset Views"),
            self.reset_dock_layout,
            shortcuts.get("reset_views", "Ctrl+Shift+V"),
            "refresh",
            self.tr("Reset dock widgets layout to default"),
            enabled=True,
        )

        # Languages
        select_lang_en = create_action(
            "English",
            functools.partial(self.set_language, "en_US"),
            icon="us",
            checkable=True,
            checked=self._config["language"] == "en_US",
            enabled=True,  # Always enable all language options
        )
        select_lang_vi = create_action(
            "Tiếng Việt",
            functools.partial(self.set_language, "vi_VN"),
            icon="vn",
            checkable=True,
            checked=self._config["language"] == "vi_VN",
            enabled=True,  # Always enable all language options
        )
        select_lang_zh = create_action(
            "中文",
            functools.partial(self.set_language, "zh_CN"),
            icon="cn",
            checkable=True,
            checked=self._config["language"] == "zh_CN",
            enabled=True,  # Always enable all language options
        )

        # Create action group for language actions to make them mutually exclusive
        lang_action_group = QtGui.QActionGroup(self)
        lang_action_group.setExclusive(True)
        lang_action_group.addAction(select_lang_en)
        lang_action_group.addAction(select_lang_vi)
        lang_action_group.addAction(select_lang_zh)

        # Store language actions for later use
        lang_actions = (select_lang_en, select_lang_vi, select_lang_zh)

        # Theme selector
        current_theme = self._config.get("theme", "system")
        select_theme_system = create_action(
            self.tr("System"),
            functools.partial(self.set_theme, "system"),
            icon="computer",
            checkable=True,
            checked=current_theme == "system",
            enabled=True,
        )
        select_theme_light = create_action(
            self.tr("Light"),
            functools.partial(self.set_theme, "light"),
            icon="sun",
            checkable=True,
            checked=current_theme == "light",
            enabled=True,
        )
        select_theme_dark = create_action(
            self.tr("Dark"),
            functools.partial(self.set_theme, "dark"),
            icon="moon",
            checkable=True,
            checked=current_theme == "dark",
            enabled=True,
        )

        # Create action group for theme actions to make them mutually exclusive
        theme_action_group = QtGui.QActionGroup(self)
        theme_action_group.setExclusive(True)
        theme_action_group.addAction(select_theme_system)
        theme_action_group.addAction(select_theme_light)
        theme_action_group.addAction(select_theme_dark)

        # Store theme actions for later use
        theme_actions = (select_theme_system, select_theme_light, select_theme_dark)

        # Group zoom controls into a list for easier toggling.
        zoom_actions = (
            self.zoom_widget,
            zoom_in,
            zoom_out,
            zoom_org,
            fit_window,
            fit_width,
        )
        self.zoom_mode = self.FIT_WINDOW
        fit_window.setChecked(True)
        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            self.FIT_WIDTH: self.scale_fit_width,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = create_action(
            self.tr("&Edit Label"),
            self.edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = create_action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.set_fill_drawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        fill_drawing.trigger()

        # AI Actions
        toggle_auto_labeling_widget = create_action(
            self.tr("&Auto Labeling"),
            self.toggle_auto_labeling_widget,
            shortcuts["auto_label"],
            "brain",
            self.tr("Auto Labeling"),
        )

        # Label list context menu.
        label_menu = QtWidgets.QMenu()
        utils.add_actions(label_menu, (edit, delete))
        self.label_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(self.pop_label_list_menu)

        clone_image_action = create_action(
            self.tr("📋 Clonar Imagen Actual"),
            self.clone_current_image,
            "Ctrl+Shift+C",
            "copy",
            self.tr("Duplicar la imagen actual y sus etiquetas en el dataset"),
        )
        rotate_90_cw_action = create_action(
            self.tr("🔄 Girar 90° (Derecha)"),
            lambda: self.rotate_image(90),
            "Ctrl+R",
            "undo",
            self.tr("Girar imagen 90 grados a la derecha"),
        )
        rotate_90_ccw_action = create_action(
            self.tr("🔄 Girar 90° (Izquierda)"),
            lambda: self.rotate_image(270),
            "Ctrl+Shift+R",
            "undo",
            self.tr("Girar imagen 90 grados a la izquierda"),
        )
        rotate_180_action = create_action(
            self.tr("🔄 Girar 180°"),
            lambda: self.rotate_image(180),
            None,
            "undo",
            self.tr("Girar imagen 180 grados"),
        )
        open_image_editor_action = create_action(
            self.tr("🎨 Mini Editor & Aumentación..."),
            self.open_image_editor_dialog,
            "Ctrl+E",
            "edit",
            self.tr("Ajustar brillo, contraste, saturación, nitidez y aumentación de dataset"),
        )
        move_image_folder_action = create_action(
            self.tr("📁 Cambiar de Carpeta..."),
            self.change_image_folder,
            "Ctrl+M",
            "open",
            self.tr("Mover las imágenes seleccionadas y sus archivos de segmentación (.json) a otra carpeta"),
        )
        open_dataset_gallery_action = create_action(
            self.tr("🖼️ Galería Dataset (Estilo Roboflow)..."),
            self.open_dataset_gallery_dialog,
            "Ctrl+G",
            "open",
            self.tr("Abrir galería visual con miniaturas y polígonos renderizados estilo Roboflow"),
        )
        delete_image_action = create_action(
            self.tr("🗑️ Eliminar Imagen de Disco"),
            self.delete_image_and_json,
            "Shift+Delete",
            "delete",
            self.tr("Eliminar permanentemente la imagen y sus etiquetas de tu disco"),
        )

        # Store actions for further handling.
        self.actions = utils.Struct(
            save_auto=save_auto,
            save_with_image_data=save_with_image_data,
            change_output_dir=change_output_dir,
            save=save,
            save_as=save_as,
            open=open_,
            close=close,
            delete_file=delete_file,
            toggle_keep_prev_mode=toggle_keep_prev_mode,
            toggle_auto_use_last_label_mode=toggle_auto_use_last_label_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undo_last_point=undo_last_point,
            undo=undo,
            remove_point=remove_point,
            create_mode=create_mode,
            edit_mode=edit_mode,
            create_rectangle_mode=create_rectangle_mode,
            create_cirle_mode=create_cirle_mode,
            create_line_mode=create_line_mode,
            create_point_mode=create_point_mode,
            create_line_strip_mode=create_line_strip_mode,
            zoom=zoom,
            zoom_in=zoom_in,
            zoom_out=zoom_out,
            zoom_org=zoom_org,
            keep_prev_scale=keep_prev_scale,
            fit_window=fit_window,
            fit_width=fit_width,
            brightness_contrast=brightness_contrast,
            show_cross_line=show_cross_line,
            show_groups=show_groups,
            show_texts=show_texts,
            zoom_actions=zoom_actions,
            open_next_image=open_next_image,
            open_prev_image=open_prev_image,
            file_menu_actions=(open_, opendir, save, save_as, close),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                delete,
                None,
                undo,
                undo_last_point,
                None,
                remove_point,
                None,
                toggle_keep_prev_mode,
                toggle_auto_use_last_label_mode,
            ),
            # menu shown at right click
            menu=(
                create_mode,
                create_rectangle_mode,
                create_cirle_mode,
                create_line_mode,
                create_point_mode,
                create_line_strip_mode,
                edit_mode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undo_last_point,
                remove_point,
            ),
            on_load_active=(
                close,
                create_mode,
                create_rectangle_mode,
                create_cirle_mode,
                create_line_mode,
                create_point_mode,
                create_line_strip_mode,
                edit_mode,
                brightness_contrast,
                move_image_folder_action,
                clone_image_action,
                open_image_editor_action,
                open_dataset_gallery_action,
                rotate_90_cw_action,
                rotate_90_ccw_action,
                rotate_180_action,
                delete_image_action,
            ),
            on_shapes_present=(save_as, hide_all, show_all),
            group_selected_shapes=group_selected_shapes,
            ungroup_selected_shapes=ungroup_selected_shapes,
            move_image_folder_action=move_image_folder_action,
            clone_image_action=clone_image_action,
            open_image_editor_action=open_image_editor_action,
            open_dataset_gallery_action=open_dataset_gallery_action,
            delete_image_action=delete_image_action,
        )

        self.canvas.vertex_selected.connect(self.actions.remove_point.setEnabled)

        # Tools
        create_action(
            self.tr("Tools"),
            self.toggle_tools,
            "tools",
            "tools",
            self.tr("Tools"),
            enabled=False,
        )

        export_annotations = create_action(
            self.tr("Export Annotations"),
            self.export_annotations,
            None,
            "box",
            self.tr("Export annotations to other formats"),
        )

        refine_polygons_action = create_action(
            self.tr("⚡ Perfeccionar Polígonos Actuales"),
            self.refine_current_polygons,
            "Shift+W",
            "polygon",
            self.tr("Suavizar contornos y eliminar micro-ruido poligonal"),
        )

        audit_health_action = create_action(
            self.tr("📊 Auditoría de Salud del Dataset"),
            self.audit_dataset_health,
            "Ctrl+Shift+A",
            "open",
            self.tr("Reporte de calidad y distribución del dataset"),
        )

        convert_to_rectangles_action = create_action(
            self.tr("📦 Convertir Polígonos a Bounding Boxes"),
            self.convert_polygons_to_rectangles_slot,
            "Ctrl+Shift+B",
            "box",
            self.tr("Convertir polígonos a cajas delimitadoras (Bounding Boxes)"),
        )

        convert_to_polygons_action = create_action(
            self.tr("🔷 Convertir Bounding Boxes a Polígonos"),
            self.convert_rectangles_to_polygons_slot,
            "Ctrl+Shift+P",
            "polygon",
            self.tr("Convertir cajas delimitadoras a polígonos de 4 puntos"),
        )

        toggle_auto_next_mode_action = create_action(
            self.tr("⏩ Avance Automático tras Etiquetar"),
            self.toggle_auto_next_image_mode,
            "Ctrl+Shift+N",
            "next",
            self.tr("Guardar y avanzar a la siguiente imagen automáticamente al etiquetar un objeto"),
            checkable=True,
            checked=False,
        )

        # Store theme actions for later use
        theme_actions = (select_theme_system, select_theme_light, select_theme_dark)

        self.menus = utils.Struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            language=self.menu(self.tr("&Language")),
            theme=self.menu(self.tr("&Theme")),
            tools=self.menu(self.tr("&Tools")),
            help=self.menu(self.tr("&Help")),
            recent_files=QtWidgets.QMenu(self.tr("Open &Recent")),
            label_list=label_menu,
        )

        utils.add_actions(
            self.menus.edit,
            (
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undo_last_point,
                remove_point,
                None,
                clone_image_action,
                open_image_editor_action,
                None,
                rotate_90_cw_action,
                rotate_90_ccw_action,
                rotate_180_action,
                None,
                delete_image_action,
                None,
                toggle_keep_prev_mode,
                toggle_auto_use_last_label_mode,
            ),
        )

        utils.add_actions(
            self.menus.file,
            (
                open_,
                open_next_image,
                open_prev_image,
                opendir,
                open_saved_dirs,
                self.menus.recent_files,
                save,
                save_as,
                save_auto,
                change_output_dir,
                save_with_image_data,
                close,
                delete_file,
                delete_image_action,
                None,
            ),
        )
        utils.add_actions(
            self.menus.help,
            (
                documentation,
                contact,
            ),
        )
        utils.add_actions(
            self.menus.tools,
            (
                clone_image_action,
                open_image_editor_action,
                rotate_90_cw_action,
                rotate_90_ccw_action,
                rotate_180_action,
                delete_image_action,
                None,
                refine_polygons_action,
                convert_to_rectangles_action,
                convert_to_polygons_action,
                toggle_auto_next_mode_action,
                audit_health_action,
                export_annotations,
            ),
        )
        utils.add_actions(
            self.menus.language,
            lang_actions,
        )
        utils.add_actions(
            self.menus.theme,
            (
                select_theme_system,
                select_theme_light,
                select_theme_dark,
            ),
        )

        utils.add_actions(
            self.menus.view,
            (
                self.shape_text_dock.toggleViewAction(),
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                reset_views,
                None,
                fill_drawing,
                None,
                hide_all,
                show_all,
                None,
                zoom_in,
                zoom_out,
                zoom_org,
                keep_prev_scale,
                None,
                fit_window,
                fit_width,
                None,
                brightness_contrast,
                show_cross_line,
                show_texts,
                show_groups,
                group_selected_shapes,
                ungroup_selected_shapes,
            ),
        )

        self.menus.file.aboutToShow.connect(self.update_file_menu)

        # Custom context menu for the canvas widget:
        utils.add_actions(self.canvas.menus[0], self.actions.menu)
        utils.add_actions(
            self.canvas.menus[1],
            (
                utils.new_action(self, "&Copy here", self.copy_shape),
                utils.new_action(self, "&Move here", self.move_shape),
            ),
        )

        # Tool actions definition (shape tools are accessible via right-click canvas menu)
        self.actions.tool = (
            open_,
            opendir,
            open_next_image,
            open_prev_image,
            save,
            delete_file,
            None,
            clone_image_action,
            rotate_90_cw_action,
            rotate_90_ccw_action,
            None,
            zoom,
            fit_width,
            toggle_auto_labeling_widget,
        )

        # Create a movable dock widget for tools
        self.tools_dock = QtWidgets.QDockWidget(
            self.tr("..."), self.main_window
        )  # Empty title
        self.tools_dock.setObjectName("ToolsDock")
        # Allow moving and detaching, but disable closing
        self.tools_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # We need visible handle, so don't hide the title bar completely
        # self.tools_dock.setTitleBarWidget(QtWidgets.QWidget())

        # Create toolbar widget to place inside dock
        tools_widget = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout()
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(0)

        # Create toolbar for tools
        self.tools = ToolBar("Tools")
        self.tools.setObjectName("ToolsToolBar")
        self.tools.setOrientation(Qt.Orientation.Vertical)
        self.tools.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.tools.setIconSize(QtCore.QSize(24, 24))

        # Set initial size constraints for vertical layout
        self.tools_dock.setMinimumWidth(40)
        self.tools_dock.setMaximumWidth(40)

        # Add actions to toolbar
        utils.add_actions(self.tools, self.actions.tool)

        # Add toolbar to layout and set as dock widget
        tools_layout.addWidget(self.tools)
        tools_widget.setLayout(tools_layout)
        self.tools_dock.setWidget(tools_widget)

        # Apply styling for tools dock with visible handle
        tools_dock_style = (
            "QDockWidget {"
            f"background-color: {AppTheme.get_color('dock_title_bg')};"
            "border: none;"
            "}"
            "QDockWidget::title {"
            "text-align: center;"
            "background-color: " + AppTheme.get_color("dock_title_bg") + ";"
            "color: " + AppTheme.get_color("dock_title_text") + ";"
            "border-radius: 4px;"
            "margin-bottom: 2px;"
            "}"
        )
        self.tools_dock.setStyleSheet(tools_dock_style)

        # Add dock to main window
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.tools_dock
        )

        # Connect signal for location changes to update toolbar orientation
        self.tools_dock.dockLocationChanged.connect(self.on_tools_dock_location_changed)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.main_window)

        # Setup central area
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.label_instruction = QLabel(self.get_labeling_instruction())
        self.label_instruction.setContentsMargins(0, 0, 0, 0)
        self.auto_labeling_widget = AutoLabelingWidget(self)
        self.auto_labeling_widget.auto_segmentation_requested.connect(
            self.on_auto_segmentation_requested
        )
        self.auto_labeling_widget.auto_segmentation_disabled.connect(
            self.on_auto_segmentation_disabled
        )
        self.canvas.auto_labeling_marks_updated.connect(
            self.auto_labeling_widget.on_new_marks
        )
        self.auto_labeling_widget.auto_labeling_mode_changed.connect(
            self.canvas.set_auto_labeling_mode
        )
        self.auto_labeling_widget.clear_auto_labeling_action_requested.connect(
            self.clear_auto_labeling_marks
        )
        self.auto_labeling_widget.finish_auto_labeling_object_action_requested.connect(
            self.finish_auto_labeling_object
        )
        self.auto_labeling_widget.model_manager.model_loading_started.connect(
            lambda: self.on_auto_labeling_started(self.tr("Loading model..."))
        )
        self.auto_labeling_widget.model_manager.model_loading_finished.connect(
            self.on_auto_labeling_finished
        )
        self.auto_labeling_widget.model_manager.prediction_started.connect(
            lambda: self.on_auto_labeling_started(self.tr("Please wait..."))
        )
        self.auto_labeling_widget.model_manager.prediction_finished.connect(
            self.on_auto_labeling_finished
        )
        self.next_files_changed.connect(
            self.auto_labeling_widget.model_manager.on_next_files_changed
        )
        self.auto_labeling_widget.model_manager.request_next_files_requested.connect(
            lambda: self.inform_next_files(self.filename)
        )
        self.auto_labeling_widget.hide()  # Hide by default

        central_layout.addWidget(self.label_instruction)
        central_layout.addWidget(self.auto_labeling_widget)
        central_layout.addWidget(scroll_area)

        # Set the central widget content
        center_widget = QtWidgets.QWidget()
        center_widget.setLayout(central_layout)
        self.main_window.centralWidget().layout().addWidget(center_widget)

        # Stretch central area (image view)
        layout.setStretch(0, 1)

        # Arrange dock widgets separately rather than tabbing them
        # All docks are initially added to RightDockWidgetArea but can be moved by the user
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.shape_text_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.flag_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.label_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.shape_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.file_dock
        )

        self.shape_text_edit.textChanged.connect(self.shape_text_changed)

        self.setLayout(layout)

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.image_path = None
        self.recent_files = []
        self.max_recent = 7
        self.other_data = {}
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightness_contrast_values = {}
        self.scroll_values = {
            Qt.Orientation.Horizontal: {},
            Qt.Orientation.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.import_image_folder(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.file_search.setText(config["file_search"])
            self.file_search_changed()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.recent_files = self.settings.value("recent_files", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        # state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']

        # Populate the File menu dynamically.
        self.update_file_menu()

        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queue_event(functools.partial(self.load_file, self.filename))

        # Callbacks:
        self.zoom_widget.valueChanged.connect(self.paint_canvas)

        self.populate_mode_actions()

        self.first_start = False
        if self.first_start:
            QWhatsThis.enterWhatsThisMode()

        self.set_text_editing(False)

        # We'll load dock state with a longer delay to ensure UI is fully ready
        QtCore.QTimer.singleShot(100, self.load_dock_state)

        # Setup periodic dock state saving
        self._dock_save_timer = QtCore.QTimer(self)
        self._dock_save_timer.setInterval(60000)  # Save state every minute
        self._dock_save_timer.timeout.connect(lambda: self.save_dock_state(force=True))
        self._dock_save_timer.start()

    def set_language(self, language):
        if self._config["language"] == language:
            return
        self._config["language"] = language
        save_config(self._config)

        # Show dialog to restart application
        msg_box = QMessageBox()
        msg_box.setText(self.tr("Please restart the application to apply changes."))
        msg_box.exec()
        self.parent.parent.close()

    def get_labeling_instruction(self):
        text_mode = self.tr("Mode:")
        text_shortcuts = self.tr("Shortcuts:")
        text_previous = self.tr("Previous:")
        text_next = self.tr("Next:")
        text_rectangle = self.tr("Rectangle:")
        text_polygon = self.tr("Polygon:")
        return (
            f"<b>{text_mode}</b> {self.canvas.get_mode()} - <b>{text_shortcuts}</b>"
            f" {text_previous} <b>A</b>, {text_next} <b>D</b>,"
            f" {text_rectangle} <b>R</b>,"
            f" {text_polygon} <b>P</b>"
        )

    @pyqtSlot()
    def on_auto_segmentation_requested(self):
        self.canvas.set_auto_labeling(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    @pyqtSlot()
    def on_auto_segmentation_disabled(self):
        self.canvas.set_auto_labeling(False)
        self.label_instruction.setText(self.get_labeling_instruction())

    def menu(self, title, actions=None):
        menu = self.parent.parent.menuBar().addMenu(title)
        if actions:
            utils.add_actions(menu, actions)
        return menu

    def central_widget(self):
        """Return the central widget for the application."""
        return self.main_window.centralWidget()

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(f"{title}ToolBar")
        toolbar.setOrientation(Qt.Orientation.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.setMaximumWidth(40)
        if actions:
            utils.add_actions(toolbar, actions)
        return toolbar

    def statusBar(self):
        return self.parent.parent.statusBar()

    def no_shape(self):
        return len(self.label_list) == 0

    def populate_mode_actions(self):
        tool = self.actions.tool
        menu = self.actions.menu
        self.tools.clear()
        utils.add_actions(self.tools, tool)

        self.canvas.menus[0].clear()
        utils.add_actions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.create_mode,
            self.actions.create_rectangle_mode,
            self.actions.create_cirle_mode,
            self.actions.create_line_mode,
            self.actions.create_point_mode,
            self.actions.create_line_strip_mode,
            self.actions.edit_mode,
        )
        utils.add_actions(self.menus.edit, actions + self.actions.editMenu)

    def set_dirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)

        if self._config["auto_save"] or self.actions.save_auto.isChecked():
            label_file = osp.splitext(self.image_path)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.save_labels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = f"{title} - {self.filename}*"
        self.setWindowTitle(title)

    def set_clean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create_mode.setEnabled(True)
        self.actions.create_rectangle_mode.setEnabled(True)
        self.actions.create_cirle_mode.setEnabled(True)
        self.actions.create_line_mode.setEnabled(True)
        self.actions.create_point_mode.setEnabled(True)
        self.actions.create_line_strip_mode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = f"{title} - {self.filename}"
        self.setWindowTitle(title)

        if self.has_label_file():
            self.actions.delete_file.setEnabled(True)
        else:
            self.actions.delete_file.setEnabled(False)

    def toggle_actions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for act in self.actions.zoom_actions:
            act.setEnabled(value)
        for act in self.actions.on_load_active:
            act.setEnabled(value)

    def queue_event(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def reset_state(self):
        self.label_list.blockSignals(True)
        try:
            self.label_list.clear()
        finally:
            self.label_list.blockSignals(False)
        self.filename = None
        self.image_path = None
        self.image_data = None
        self.label_file = None
        self.other_data = {}
        self.canvas.reset_state()

    def current_item(self):
        items = self.label_list.selected_items()
        if items:
            return items[0]
        return None

    def add_recent_file(self, filename):
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        elif len(self.recent_files) >= self.max_recent:
            self.recent_files.pop()
        self.recent_files.insert(0, filename)

    # Callbacks

    def undo_shape_edit(self):
        self.canvas.restore_shape()
        self.label_list.clear()
        self.load_shapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)

    def documentation(self):
        url = "https://anylabeling.nrl.ai/"  # NOQA
        webbrowser.open(url)

    def contact(self):
        url = "https://www.nrl.ai/contact"  # NOQA
        webbrowser.open(url)

    def toggle_drawing_sensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.edit_mode.setEnabled(not drawing)
        self.actions.undo_last_point.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggle_draw_mode(
        self, edit=True, create_mode="rectangle", disable_auto_labeling=True
    ):
        # Disable auto labeling if needed
        if (
            disable_auto_labeling
            and hasattr(self, "auto_labeling_widget")
            and self.auto_labeling_widget.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            self.clear_auto_labeling_marks()
            self.auto_labeling_widget.set_auto_labeling_mode(None)

        if disable_auto_labeling and getattr(self, "q_plus_plus_mode", False):
            self.q_plus_plus_mode = False
            if hasattr(self, "auto_labeling_widget") and hasattr(self.auto_labeling_widget, "button_q_plus_plus"):
                self.auto_labeling_widget.button_q_plus_plus.blockSignals(True)
                self.auto_labeling_widget.button_q_plus_plus.setChecked(False)
                self.auto_labeling_widget.update_q_plus_plus_button_style(False)
                self.auto_labeling_widget.button_q_plus_plus.blockSignals(False)

        self.set_text_editing(False)

        self.canvas.set_editing(edit)
        self.canvas.create_mode = create_mode
        if edit:
            self.actions.create_mode.setEnabled(True)
            self.actions.create_rectangle_mode.setEnabled(True)
            self.actions.create_cirle_mode.setEnabled(True)
            self.actions.create_line_mode.setEnabled(True)
            self.actions.create_point_mode.setEnabled(True)
            self.actions.create_line_strip_mode.setEnabled(True)
        else:
            if create_mode == "polygon":
                self.actions.create_mode.setEnabled(False)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_cirle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "rectangle":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(False)
                self.actions.create_cirle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "line":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_cirle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(False)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "point":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_cirle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(False)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "circle":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_cirle_mode.setEnabled(False)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "linestrip":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_cirle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(False)
            else:
                raise ValueError(f"Unsupported create_mode: {create_mode}")
        self.actions.edit_mode.setEnabled(not edit)
        self.label_instruction.setText(self.get_labeling_instruction())

    def set_edit_mode(self):
        # Disable auto labeling
        self.clear_auto_labeling_marks()
        self.auto_labeling_widget.set_auto_labeling_mode(None)

        self.toggle_draw_mode(True)
        self.set_text_editing(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    def update_file_menu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recent_files
        menu.clear()
        files = [f for f in self.recent_files if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.new_icon("labels")
            menu_action = QtGui.QAction(
                icon, f"&{i + 1} {QtCore.QFileInfo(f).fileName()}", self
            )
            menu_action.triggered.connect(functools.partial(self.load_recent, f))
            menu.addAction(menu_action)

    def pop_unique_label_list_menu(self, point):
        """Context menu for right clicking on unique label list items."""
        item = self.unique_label_list.itemAt(point)
        menu = QtWidgets.QMenu(self)

        label = item.data(Qt.ItemDataRole.UserRole) if item else None

        if label:
            action_filter = QtGui.QAction(
                utils.new_icon("labels"),
                f"🔍 Filtrar imágenes por etiqueta '{label}'",
                self,
            )
            action_filter.triggered.connect(
                functools.partial(self.filter_images_by_label, label)
            )
            menu.addAction(action_filter)

            action_del_all = QtGui.QAction(
                utils.new_icon("delete"),
                f"🗑️ Eliminar etiqueta '{label}' de TODAS las imágenes de la lista",
                self,
            )
            action_del_all.triggered.connect(
                functools.partial(self.delete_label_across_dataset, label)
            )
            menu.addAction(action_del_all)
            menu.addSeparator()

        active_filter = getattr(self, "_active_label_filter", None)
        if active_filter:
            action_clear = QtGui.QAction(
                f"❌ Quitar filtro (Mostrar todas, actual: '{active_filter}')",
                self,
            )
            action_clear.triggered.connect(self.clear_label_filter)
            menu.addAction(action_clear)

        if menu.actions():
            menu.exec(self.unique_label_list.mapToGlobal(point))

    def pop_label_list_menu(self, point):
        """Context menu for right clicking on active shape list items."""
        item = self.label_list.itemAt(point)
        menu = QtWidgets.QMenu(self)

        menu.addAction(self.actions.edit)
        menu.addAction(self.actions.delete)

        if item:
            try:
                shape = item.shape() if hasattr(item, "shape") else item.data(Qt.ItemDataRole.UserRole)
                if shape and hasattr(shape, "label") and shape.label:
                    menu.addSeparator()
                    action_filter = QtGui.QAction(
                        utils.new_icon("labels"),
                        f"🔍 Filtrar imágenes por etiqueta '{shape.label}'",
                        self,
                    )
                    action_filter.triggered.connect(
                        functools.partial(self.filter_images_by_label, shape.label)
                    )
                    menu.addAction(action_filter)

                    action_del_all = QtGui.QAction(
                        utils.new_icon("delete"),
                        f"🗑️ Eliminar etiqueta '{shape.label}' de TODAS las imágenes de la lista",
                        self,
                    )
                    action_del_all.triggered.connect(
                        functools.partial(self.delete_label_across_dataset, shape.label)
                    )
                    menu.addAction(action_del_all)
            except Exception:
                pass

        active_filter = getattr(self, "_active_label_filter", None)
        if active_filter:
            menu.addSeparator()
            action_clear = QtGui.QAction(
                f"❌ Quitar filtro (Mostrar todas, actual: '{active_filter}')",
                self,
            )
            action_clear.triggered.connect(self.clear_label_filter)
            menu.addAction(action_clear)

        menu.exec(self.label_list.mapToGlobal(point))

    def pop_file_list_menu(self, point):
        """Show context menu for right clicking items in file list widget."""
        item = self.file_list_widget.itemAt(point)
        if not item:
            return

        # If right clicked item wasn't selected, select only it
        if not item.isSelected():
            self.file_list_widget.clearSelection()
            item.setSelected(True)

        menu = QtWidgets.QMenu(self)
        if hasattr(self.actions, "clone_image_action"):
            menu.addAction(self.actions.clone_image_action)
        if hasattr(self.actions, "move_image_folder_action"):
            menu.addAction(self.actions.move_image_folder_action)

        rotate_menu = menu.addMenu(utils.new_icon("undo"), self.tr("🔄 Girar Imagen"))
        if hasattr(self.actions, "rotate_90_cw_action"):
            rotate_menu.addAction(self.actions.rotate_90_cw_action)
        if hasattr(self.actions, "rotate_90_ccw_action"):
            rotate_menu.addAction(self.actions.rotate_90_ccw_action)
        if hasattr(self.actions, "rotate_180_action"):
            rotate_menu.addAction(self.actions.rotate_180_action)

        if hasattr(self.actions, "open_dataset_gallery_action"):
            menu.addAction(self.actions.open_dataset_gallery_action)

        filter_count_menu = menu.addMenu(utils.new_icon("labels"), self.tr("🔍 Auditoría por Cantidad de Etiquetas"))
        filter_count_menu.addAction("⚠️ Filtrar Imágenes Sin Etiquetas (0)", lambda: self.filter_images_by_shape_count(0))
        filter_count_menu.addAction("🔢 Filtrar Imágenes con < 3 Etiquetas", lambda: self.filter_images_by_shape_count(3))
        filter_count_menu.addAction("🔢 Filtrar Imágenes con < 5 Etiquetas", lambda: self.filter_images_by_shape_count(5))

        if hasattr(self.actions, "open_image_editor_action"):
            menu.addAction(self.actions.open_image_editor_action)
        menu.addSeparator()
        if hasattr(self.actions, "delete_image_action"):
            menu.addAction(self.actions.delete_image_action)

        global_pos = self.file_list_widget.viewport().mapToGlobal(point)
        menu.exec(global_pos)

    def filter_images_by_label(self, label):
        """Filter the file list widget to show only images containing annotations with the given label."""
        if not label or not self.last_open_dir or not osp.exists(self.last_open_dir):
            self.statusBar().showMessage("⚠️ Abre una carpeta con imágenes primero para filtrar.")
            return

        self._active_label_filter = label

        # Scan folder for images containing this label in their JSON files
        all_images = self.scan_all_images(self.last_open_dir)
        matched_images = []

        for img_path in all_images:
            json_path = osp.splitext(img_path)[0] + ".json"
            if self.output_dir:
                json_path = osp.join(self.output_dir, osp.basename(json_path))

            if osp.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for shape in data.get("shapes", []):
                            if shape.get("label") == label:
                                matched_images.append(img_path)
                                break
                except Exception:
                    pass

        # Populate file_list_widget with matched_images
        self.filename = None
        self.file_list_widget.clear()

        for file in matched_images:
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.file_list_widget.addItem(item)

        # Update filter status button on UI
        if hasattr(self, "btn_filter_status"):
            self.btn_filter_status.setText(f"🔍 Filtro: '{label}' ({len(matched_images)}/{len(all_images)}) [❌ Quitar]")
            self.btn_filter_status.setVisible(True)

        msg = f"🔍 Filtro por Etiqueta '{label}': {len(matched_images)} de {len(all_images)} imágenes encontradas."
        self.statusBar().showMessage(msg, 5000)

        if matched_images:
            self.actions.open_next_image.setEnabled(True)
            self.actions.open_prev_image.setEnabled(True)
            self.open_next_image()
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Filtro por Etiqueta",
                f"No se encontraron imágenes guardadas con la etiqueta '{label}' en esta carpeta."
            )

    def clear_label_filter(self):
        """Clear label filter and reload all images in the current folder."""
        self._active_label_filter = None
        if hasattr(self, "btn_filter_status"):
            self.btn_filter_status.setVisible(False)
        if self.last_open_dir and osp.exists(self.last_open_dir):
            self.import_image_folder(self.last_open_dir)
            self.statusBar().showMessage("✅ Filtro desactivado. Mostrando todas las imágenes.", 3000)

    def validate_label(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        # These labels are produced by the auto-labeling feature and shouldn't emit errors
        if label in ("AUTOLABEL_ADD", "AUTOLABEL_REMOVE"):
            return True

        for i in range(self.unique_label_list.count()):
            label_i = self.unique_label_list.item(i).data(Qt.ItemDataRole.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def edit_label(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return

        # Determine all target shapes and items to edit (multi-selection support)
        selected_items = self.label_list.selected_items()
        selected_shapes = list(self.canvas.selected_shapes)

        target_items = set()
        if item:
            target_items.add(item)
        for it in selected_items:
            target_items.add(it)

        # Also get items corresponding to canvas.selected_shapes
        for shape in selected_shapes:
            it = self.label_list.find_item_by_shape(shape)
            if it:
                target_items.add(it)

        if not target_items:
            curr = self.current_item()
            if curr:
                target_items.add(curr)

        if not target_items:
            return

        target_items = list(target_items)
        first_item = target_items[0]
        first_shape = first_item.shape()
        if first_shape is None:
            return

        text, flags, group_id = self.label_dialog.pop_up(
            text=first_shape.label,
            flags=first_shape.flags,
            group_id=first_shape.group_id,
        )
        if text is None:
            return
        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        # Apply edits to all target items and their shapes
        for it in target_items:
            s = it.shape()
            if s is None:
                continue
            s.label = text
            s.flags = flags
            s.group_id = group_id

            # Add to label history
            self.label_dialog.add_label_history(s.label)

            # Update unique label list
            if not self.unique_label_list.find_items_by_label(s.label):
                unique_label_item = self.unique_label_list.create_item_from_label(
                    s.label
                )
                self.unique_label_list.addItem(unique_label_item)
                rgb = self._get_rgb_by_label(s.label)
                self.unique_label_list.set_item_label(unique_label_item, s.label, rgb)

            self._update_shape_color(s)
            if s.group_id is None:
                color = s.fill_color.getRgb()[:3]
                it.setText(
                    '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                        html.escape(s.label), *color
                    )
                )
            else:
                it.setText(f"{s.label} ({s.group_id})")

        self.set_dirty()
        self.canvas.update()

    def open_global_replace_dialog(self):
        """Open Global Class Find & Replace dialog"""
        from .widgets.global_class_replace_dialog import GlobalClassReplaceDialog
        dialog = GlobalClassReplaceDialog(self)
        dialog.exec()

    def open_label_inspector(self, label=None):
        """Open Label Inspector Window for managing shapes of a class"""
        from .widgets.label_inspector_dialog import LabelInspectorDialog
        if not label and self.unique_label_list.currentItem():
            label = self.unique_label_list.currentItem().data(Qt.ItemDataRole.UserRole)
        dialog = LabelInspectorDialog(self, target_label=label)
        dialog.exec()

    def select_all_by_class(self, label=None):
        """Select all shapes matching the specified class label"""
        if not label and self.unique_label_list.currentItem():
            label = self.unique_label_list.currentItem().data(Qt.ItemDataRole.UserRole)
        if not label:
            return

        matching_shapes = [s for s in self.canvas.shapes if s.label == label]
        if matching_shapes:
            self.canvas.select_shapes(matching_shapes)
            self.canvas.set_editing(True)

    def merge_selected_polygons_action(self):
        """Merge selected polygons into a single shape"""
        if hasattr(self.canvas, "merge_selected_polygons"):
            merged = self.canvas.merge_selected_polygons()
            if merged:
                # Reload shapes in label list
                self.load_shapes(self.canvas.shapes, replace=True)
                self.set_dirty()

    def convert_polygons_to_rectangles_slot(self):
        """Convert selected or all polygons to rectangle bounding boxes"""
        if hasattr(self.canvas, "convert_to_rectangles"):
            count = self.canvas.convert_to_rectangles()
            if count > 0:
                self.load_shapes(self.canvas.shapes, replace=True)
                self.set_dirty()
                self.statusBar().showMessage(f"📦 Se convirtieron {count} polígono(s) a Bounding Boxes.", 4000)

    def convert_rectangles_to_polygons_slot(self):
        """Convert selected or all rectangles to polygons"""
        if hasattr(self.canvas, "convert_to_polygons"):
            count = self.canvas.convert_to_polygons()
            if count > 0:
                self.load_shapes(self.canvas.shapes, replace=True)
                self.set_dirty()
                self.statusBar().showMessage(f"🔷 Se convirtieron {count} Bounding Box(es) a Polígonos.", 4000)

    def toggle_auto_next_image_mode(self, value=None):
        """Toggle Auto-Next image mode after object finalization"""
        if value is None:
            self.auto_next_image_mode = not getattr(self, "auto_next_image_mode", False)
        else:
            self.auto_next_image_mode = value

        status = "Activado ⏩" if self.auto_next_image_mode else "Desactivado ⏸️"
        self.statusBar().showMessage(f"Modo Avance Automático tras etiquetar: {status}", 4000)

    def file_search_changed(self):
        self.import_image_folder(
            self.last_open_dir,
            pattern=self.file_search.text(),
            load=False,
        )

    def file_selection_changed(self):
        items = self.file_list_widget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.may_continue():
            return

        current_index = self.image_list.index(str(item.text()))
        if current_index < len(self.image_list):
            filename = self.image_list[current_index]
            if filename:
                self.load_file(filename)

    # React to canvas signals.
    def shape_selection_changed(self, selected_shapes):
        self._no_selection_slot = True
        for shape in self.canvas.selected_shapes:
            shape.selected = False
        self.label_list.clearSelection()
        self.canvas.selected_shapes = selected_shapes
        for shape in self.canvas.selected_shapes:
            shape.selected = True
            item = self.label_list.find_item_by_shape(shape)
            self.label_list.select_item(item)
            self.label_list.scroll_to_item(item)
        self._no_selection_slot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        self.set_text_editing(True)

    def add_label(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = f"{shape.label} ({shape.group_id})"
        label_list_item = LabelListWidgetItem(text, shape)
        self.label_list.add_iem(label_list_item)
        # Don't add special autolabeling labels to the unique_label_list
        if shape.label not in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ] and not self.unique_label_list.find_items_by_label(shape.label):
            item = self.unique_label_list.create_item_from_label(shape.label)
            self.unique_label_list.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.unique_label_list.set_item_label(item, shape.label, rgb)

        # Add label to history if it is not a special label
        if shape.label not in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            self.label_dialog.add_label_history(shape.label)

        for action in self.actions.on_shapes_present:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def shape_text_changed(self):
        text = self.shape_text_edit.toPlainText()
        if self.canvas.current is not None:
            self.canvas.current.text = text
        elif self.canvas.editing() and len(self.canvas.selected_shapes) == 1:
            self.canvas.selected_shapes[0].text = text
        else:
            self.other_data["image_text"] = text
        self.set_dirty()

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            # For special autolabeling labels, use fixed colors
            if label in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                if label == AutoLabelingMode.OBJECT:
                    return (0, 255, 255)  # Cyan color for object
                elif label == AutoLabelingMode.ADD:
                    return (0, 255, 0)  # Green color for add
                elif label == AutoLabelingMode.REMOVE:
                    return (255, 0, 0)  # Red color for remove

            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
            item = self.unique_label_list.find_items_by_label(label)[0]
            label_id = self.unique_label_list.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        if (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        if self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remove_labels(self, shapes):
        for shape in shapes:
            item = self.label_list.find_item_by_shape(shape)
            self.label_list.remove_item(item)

    def load_shapes(self, shapes, replace=True):
        self._no_selection_slot = True
        for shape in shapes:
            self.add_label(shape)
        self.label_list.clearSelection()
        self._no_selection_slot = False
        self.canvas.load_shapes(shapes, replace=replace)

    def load_labels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            text = shape.get("text", "")
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                text=text,
                shape_type=shape_type,
                group_id=group_id,
            )
            for x, y in points:
                shape.add_point(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            if flags:
                shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.load_shapes(s)

    def load_flags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if flag else Qt.CheckState.Unchecked
            )
            self.flag_widget.addItem(item)

    def save_labels(self, filename):
        label_file = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                {
                    "label": s.label,
                    "text": s.text,
                    "points": [(p.x(), p.y()) for p in s.points],
                    "group_id": s.group_id,
                    "shape_type": s.shape_type,
                    "flags": s.flags,
                }
            )
            return data

        # Get current shapes
        # Excluding auto labeling special shapes
        shapes = [
            format_shape(item.shape())
            for item in self.label_list
            if item.shape().label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.CheckState.Checked
            flags[key] = flag
        try:
            image_path = osp.relpath(self.image_path, osp.dirname(filename))
            image_data = self.image_data if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            label_file.save(
                filename=filename,
                shapes=shapes,
                image_path=image_path,
                image_data=image_data,
                image_height=self.image.height(),
                image_width=self.image.width(),
                other_data=self.other_data,
                flags=flags,
            )
            self.label_file = label_file
            items = self.file_list_widget.findItems(
                self.image_path, Qt.MatchFlag.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.CheckState.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.error_message(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicate_selected_shape(self):
        added_shapes = self.canvas.duplicate_selected_shapes()
        self.label_list.clearSelection()
        for shape in added_shapes:
            self.add_label(shape)
        self.set_dirty()

    def paste_selected_shape(self):
        self.load_shapes(self._copied_shapes, replace=False)
        self.set_dirty()

    def copy_selected_shape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selected_shapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def label_selection_changed(self):
        if self._no_selection_slot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.label_list.selected_items():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.select_shapes(selected_shapes)
            else:
                self.canvas.deselect_shape()

    def label_item_changed(self, item):
        shape = item.shape()
        self.canvas.set_shape_visible(shape, item.checkState() == Qt.CheckState.Checked)

    def label_order_changed(self):
        self.set_dirty()
        self.canvas.load_shapes([item.shape() for item in self.label_list])

    # Callback functions:

    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.unique_label_list.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.ItemDataRole.UserRole)
        flags = {}
        group_id = None

        if self.canvas.shapes[-1].label in [
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            text = self.canvas.shapes[-1].label
        elif (
            self._config["display_label_popup"]
            or not text
            or self.canvas.shapes[-1].label == AutoLabelingMode.OBJECT
        ):
            last_label = self.find_last_label()
            if self._config["auto_use_last_label"] and last_label:
                text = last_label
            else:
                previous_text = self.label_dialog.edit.text()
                text, flags, group_id = self.label_dialog.pop_up(text)
                if not text:
                    self.label_dialog.edit.setText(previous_text)

        if text and not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
            return

        if text:
            self.label_list.clearSelection()
            shape = self.canvas.set_last_label(text, flags)
            shape.group_id = group_id
            shape.label = text
            self.add_label(shape)
            self.actions.edit_mode.setEnabled(True)
            self.actions.undo_last_point.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.set_dirty()
        else:
            self.canvas.undo_last_line()
            self.canvas.shapes_backups.pop()

    def scroll_request(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        scroll_bar = self.scroll_bars[orientation]
        value = scroll_bar.value() + scroll_bar.singleStep() * units
        self.set_scroll(orientation, value)

    def set_scroll(self, orientation, value):
        self.scroll_bars[orientation].setValue(round(value))
        self.scroll_values[orientation][self.filename] = value

    def set_zoom(self, value):
        self.actions.fit_width.setChecked(False)
        self.actions.fit_window.setChecked(False)
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)

    def add_zoom(self, increment=1.1):
        zoom_value = self.zoom_widget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.set_zoom(zoom_value)

    def zoom_request(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.add_zoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor - pos.x())
            y_shift = round(pos.y() * canvas_scale_factor - pos.y())

            self.set_scroll(
                Qt.Orientation.Horizontal,
                self.scroll_bars[Qt.Orientation.Horizontal].value() + x_shift,
            )
            self.set_scroll(
                Qt.Orientation.Vertical,
                self.scroll_bars[Qt.Orientation.Vertical].value() + y_shift,
            )

    def set_fit_window(self, value=True):
        if value:
            self.actions.fit_width.setChecked(False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def set_fit_width(self, value=True):
        if value:
            self.actions.fit_window.setChecked(False)
        self.zoom_mode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def enable_keep_prev_scale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keep_prev_scale.setChecked(enabled)
        save_config(self._config)

    def enable_show_cross_line(self, enabled):
        self._config["show_cross_line"] = enabled
        self.actions.show_cross_line.setChecked(enabled)
        self.canvas.set_show_cross_line(enabled)
        save_config(self._config)

    def enable_show_groups(self, enabled):
        self._config["show_groups"] = enabled
        self.actions.show_groups.setChecked(enabled)
        self.canvas.set_show_groups(enabled)
        save_config(self._config)

    def enable_show_texts(self, enabled):
        self._config["show_texts"] = enabled
        self.actions.show_texts.setChecked(enabled)
        self.canvas.set_show_texts(enabled)
        save_config(self._config)

    def on_new_brightness_contrast(self, qimage):
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightness_contrast(self, _):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.image_data),
            self.on_new_brightness_contrast,
            parent=self,
        )
        brightness, contrast = self.brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightness_contrast_values[self.filename] = (brightness, contrast)

    def toggle_polygons(self, value):
        for item in self.label_list:
            item.setCheckState(
                Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
            )

    def get_next_files(self, filename, num_files):
        """Get the next files in the list."""
        if not self.image_list:
            return []
        filenames = []
        current_index = 0
        if filename is not None:
            try:
                current_index = self.image_list.index(filename)
            except ValueError:
                return []
            filenames.append(filename)
        for _ in range(num_files):
            if current_index + 1 < len(self.image_list):
                filenames.append(self.image_list[current_index + 1])
                current_index += 1
            else:
                filenames.append(self.image_list[-1])
                break
        return filenames

    def inform_next_files(self, filename):
        """Inform the next files to be annotated.
        This list can be used by the user to preload the next files
        or running a background process to process them
        """
        next_files = self.get_next_files(filename, 5)
        if next_files:
            self.next_files_changed.emit(next_files)

    def load_file(self, filename=None):  # noqa: C901
        """Load the specified file, or the last opened file if None."""

        # Guard contra re-entrancia para evitar llamadas recursivas al cambiar de imagen
        if getattr(self, "_is_loading_file", False):
            return False
        self._is_loading_file = True

        try:
            # Pausar y desactivar Q(++) temporalmente durante la carga de imagen para evitar colisiones
            self._was_q_pp_active_before_load = getattr(self, "q_plus_plus_mode", False)
            if hasattr(self, "auto_labeling_widget") and self.auto_labeling_widget:
                if hasattr(self.auto_labeling_widget, "button_q_plus_plus") and self.auto_labeling_widget.button_q_plus_plus.isChecked():
                    self.auto_labeling_widget.button_q_plus_plus.blockSignals(True)
                    self.auto_labeling_widget.button_q_plus_plus.setChecked(False)
                    self.auto_labeling_widget.update_q_plus_plus_button_style(False)
                    self.auto_labeling_widget.button_q_plus_plus.blockSignals(False)
                if hasattr(self.auto_labeling_widget, "model_manager") and self.auto_labeling_widget.model_manager:
                    self.auto_labeling_widget.model_manager.stop_inference()
            self.q_plus_plus_mode = False

            # For auto labeling, clear the previous marks
            # and stop background preloading to prevent GUI freezes on image change
            self.clear_auto_labeling_marks()
            if hasattr(self, "canvas") and hasattr(self.canvas, "shapes_backups"):
                self.canvas.shapes_backups.clear()
            self.inform_next_files(filename)

            # Sincronizar fila en file_list_widget de forma limpia sin emitir señales reentrantes
            if filename in self.image_list:
                target_row = self.image_list.index(filename)
                if self.file_list_widget.currentRow() != target_row:
                    self.file_list_widget.blockSignals(True)
                    self.file_list_widget.setCurrentRow(target_row)
                    self.file_list_widget.blockSignals(False)

            self.reset_state()
            self.canvas.setEnabled(False)
            if filename is None:
                filename = self.settings.value("filename", "")
            filename = str(filename)
            if not QtCore.QFile.exists(filename):
                self.error_message(
                    self.tr("Error opening file"),
                    self.tr("No such file: <b>%s</b>") % filename,
                )
                return False

            # assumes same name, but json extension
            self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                try:
                    self.label_file = LabelFile(label_file)
                except LabelFileError as e:
                    self.error_message(
                        self.tr("Error opening file"),
                        self.tr(
                            "<p><b>%s</b></p><p>Make sure <i>%s</i> is a valid label file."
                        )
                        % (e, label_file),
                    )
                    self.status(self.tr("Error reading %s") % label_file)
                    return False
                self.image_data = self.label_file.image_data
                self.image_path = osp.join(
                    osp.dirname(label_file),
                    self.label_file.image_path,
                )
                self.other_data = self.label_file.other_data
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(self.other_data.get("image_text", ""))
                self.shape_text_edit.textChanged.connect(self.shape_text_changed)
            else:
                self.image_data = LabelFile.load_image_file(filename)
                if self.image_data:
                    self.image_path = filename
                self.label_file = None
            image = QtGui.QImage.fromData(self.image_data)

            if image.isNull():
                formats = [
                    f"*.{fmt.data().decode()}"
                    for fmt in QtGui.QImageReader.supportedImageFormats()
                ]
                self.error_message(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                        "Supported image formats: {1}</p>"
                    ).format(filename, ",".join(formats)),
                )
                self.status(self.tr("Error reading %s") % filename)
                self.canvas.load_pixmap(QtGui.QPixmap())
                self.canvas.setEnabled(False)
                return False
            self.image = image
            self.filename = filename
            if self._config["keep_prev"]:
                prev_shapes = self.canvas.shapes
            self.canvas.load_pixmap(QtGui.QPixmap.fromImage(image))
            flags = dict.fromkeys(self._config["flags"] or [], False)
            if self.label_file:
                self.load_labels(self.label_file.shapes)
                if self.label_file.flags is not None:
                    flags.update(self.label_file.flags)
            self.load_flags(flags)
            if self._config["keep_prev"] and self.no_shape():
                self.load_shapes(prev_shapes, replace=False)
                self.set_dirty()
            else:
                self.set_clean()
            self.canvas.setEnabled(True)
            # set zoom values
            is_initial_load = not self.zoom_values
            if self.filename in self.zoom_values:
                self.zoom_mode = self.zoom_values[self.filename][0]
                self.set_zoom(self.zoom_values[self.filename][1])
            elif is_initial_load or not self._config["keep_prev_scale"]:
                self.adjust_scale(initial=True)
            # set scroll values
            for orientation in self.scroll_values:
                if self.filename in self.scroll_values[orientation]:
                    self.set_scroll(
                        orientation, self.scroll_values[orientation][self.filename]
                    )
            # set brightness contrast values
            dialog = BrightnessContrastDialog(
                utils.img_data_to_pil(self.image_data),
                self.on_new_brightness_contrast,
                parent=self,
            )
            brightness, contrast = self.brightness_contrast_values.get(
                self.filename, (None, None)
            )
            if self._config["keep_prev_brightness"] and self.recent_files:
                brightness, _ = self.brightness_contrast_values.get(
                    self.recent_files[0], (None, None)
                )
            if self._config["keep_prev_contrast"] and self.recent_files:
                _, contrast = self.brightness_contrast_values.get(
                    self.recent_files[0], (None, None)
                )
            if brightness is not None:
                dialog.slider_brightness.setValue(brightness)
            if contrast is not None:
                dialog.slider_contrast.setValue(contrast)
            self.brightness_contrast_values[self.filename] = (brightness, contrast)
            if brightness is not None or contrast is not None:
                dialog.on_new_value(None)
            self.paint_canvas()
            self.add_recent_file(self.filename)
            self.toggle_actions(True)
            self.canvas.setFocus()
            self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))

            # Save dock state after loading file (to capture any UI adjustments)
            QtCore.QTimer.singleShot(100, self.save_dock_state)
            self.update_unique_label_counts()

            # Si el modo Q(++) estaba activo antes de cambiar de imagen, reactivarlo de forma segura
            if getattr(self, "_was_q_pp_active_before_load", False):
                self._was_q_pp_active_before_load = False
                QtCore.QTimer.singleShot(0, self._reactivate_q_plus_plus)

            return True
        finally:
            self._is_loading_file = False

    # QT Overload
    def resizeEvent(self, _):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoom_mode != self.MANUAL_ZOOM
        ):
            self.adjust_scale()

        # Save dock state after resize (after a short delay to let layout settle)
        if hasattr(self, "_resize_timer"):
            self._resize_timer.stop()
        else:
            self._resize_timer = QtCore.QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self.save_dock_state)

        self._resize_timer.start(100)

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        value = int(100 * value)
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)

    def scale_fit_window(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.central_widget().width() - e
        h1 = self.central_widget().height() - e
        wh_ratio1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        wh_ratio2 = w2 / h2
        return w1 / w2 if wh_ratio2 >= wh_ratio1 else h1 / h2

    def scale_fit_width(self):
        # The epsilon does not seem to work too well here.
        w = self.central_widget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enable_save_image_with_data(self, enabled):
        self._config["store_data"] = enabled
        self.actions.save_with_image_data.setChecked(enabled)

    # QT Overload
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if hasattr(self, "canvas") and self.canvas:
                self.canvas.deselect_shape()
            if hasattr(self, "label_list") and self.label_list:
                self.label_list.clearSelection()
            if hasattr(self, "unique_label_list") and self.unique_label_list:
                self.unique_label_list.clearSelection()
            event.accept()
            return
        super().keyPressEvent(event)

    # QT Overload
    def closeEvent(self, event):
        if not self.may_continue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.parent.parent.saveState())

        # Save dock layout to config (final save on exit)
        self.save_dock_state(force=True)

        self.settings.setValue("recent_files", self.recent_files)
        self.settings.setValue("saved_directories", self.saved_directories)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    # QT Overload
    def dragEnterEvent(self, event):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any(i.lower().endswith(tuple(extensions)) for i in items):
                event.accept()
        else:
            event.ignore()

    # QT Overload
    def dropEvent(self, event):
        if not self.may_continue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.import_dropped_image_files(items)

    def load_recent(self, filename):
        if self.may_continue():
            self.load_file(filename)

    def open_prev_image(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ):
            self._config["keep_prev"] = True
            save_config(self._config)

        if not self.may_continue():
            return

        if len(self.image_list) <= 0:
            return

        if self.filename is None:
            return

        # Save dock state before changing images
        self.save_dock_state()

        current_index = self.image_list.index(self.filename)
        if current_index - 1 >= 0:
            filename = self.image_list[current_index - 1]
            if filename:
                self.load_file(filename)

        self._config["keep_prev"] = keep_prev
        save_config(self._config)

    def open_next_image(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ):
            self._config["keep_prev"] = True
            save_config(self._config)

        if not self.may_continue():
            return

        if len(self.image_list) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.image_list[0]
        else:
            current_index = self.image_list.index(self.filename)
            if current_index + 1 < len(self.image_list):
                filename = self.image_list[current_index + 1]
            else:
                filename = self.image_list[-1]
        self.filename = filename

        # Save dock state before changing images
        self.save_dock_state()

        if self.filename and load:
            self.load_file(self.filename)

        self._config["keep_prev"] = keep_prev
        save_config(self._config)

    def open_file(self, _value=False):
        if not self.may_continue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        file_dialog = FileDialogPreview(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter(filters)
        file_dialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        file_dialog.setWindowFilePath(path)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        if file_dialog.exec():
            filename = file_dialog.selectedFiles()[0]
            if filename:
                self.load_file(filename)

    def change_output_dir_dialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.current_path()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.import_image_folder(self.last_open_dir, load=False)

        if current_filename in self.image_list:
            # retain currently selected file
            self.file_list_widget.setCurrentRow(self.image_list.index(current_filename))
            self.file_list_widget.repaint()

    def save_file(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.label_file:
            # DL20180323 - overwrite when in directory
            self._save_file(self.label_file.filename)
        elif self.output_file:
            self._save_file(self.output_file)
            self.close()
        else:
            self._save_file(self.save_file_dialog())

    def save_file_as(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._save_file(self.save_file_dialog())

    def clone_current_image(self):
        """Duplicate / clone current image and its annotations JSON file in the dataset directory."""
        if not self.filename or not osp.exists(self.filename):
            QtWidgets.QMessageBox.warning(self, "Clonar Imagen", "Abre una imagen primero para clonarla.")
            return

        # Ensure current changes on canvas are saved before cloning
        if self.dirty:
            try:
                self.save_file()
            except Exception as e:
                logger.warning("Error saving file before cloning: %s", e)

        import shutil
        dir_name = osp.dirname(self.filename)
        base_name = osp.basename(self.filename)
        name, ext = osp.splitext(base_name)

        # Generate unique clone filename
        clone_base = f"{name}_clone"
        clone_image_path = osp.join(dir_name, f"{clone_base}{ext}")
        counter = 1
        while osp.exists(clone_image_path):
            clone_base = f"{name}_clone{counter}"
            clone_image_path = osp.join(dir_name, f"{clone_base}{ext}")
            counter += 1

        clone_image_filename = osp.basename(clone_image_path)

        try:
            # Copy image file
            shutil.copyfile(self.filename, clone_image_path)

            # Copy JSON label file if it exists or save from canvas
            src_json = self.get_label_file_for_image(self.filename)
            json_dir = self.output_dir if (self.output_dir and osp.exists(self.output_dir)) else dir_name
            dst_json = osp.join(json_dir, f"{clone_base}.json")

            if src_json and osp.exists(src_json):
                try:
                    with open(src_json, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    data["imagePath"] = clone_image_filename
                    if "imageData" in data and data["imageData"]:
                        with open(clone_image_path, "rb") as img_f:
                            data["imageData"] = utils.img_arr_to_b64(utils.img_data_to_arr(img_f.read()))

                    with open(dst_json, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as ex:
                    logger.warning("Error copying JSON label file: %s", ex)
                    shutil.copyfile(src_json, dst_json)
            elif self.canvas.shapes:
                try:
                    shapes_data = [s.to_dict() for s in self.canvas.shapes]
                    data = {
                        "version": "0.3.3",
                        "flags": {},
                        "shapes": shapes_data,
                        "imagePath": clone_image_filename,
                        "imageData": None,
                        "imageHeight": self.image.height() if self.image else 0,
                        "imageWidth": self.image.width() if self.image else 0,
                    }
                    with open(dst_json, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as ex:
                    logger.warning("Error creating cloned JSON from canvas: %s", ex)

            # Insert cloned file into list widget and open it
            if self.last_open_dir and osp.exists(self.last_open_dir):
                self.import_image_folder(self.last_open_dir, load=False)

            if clone_image_path in self.image_list:
                idx = self.image_list.index(clone_image_path)
                self.file_list_widget.setCurrentRow(idx)
                self.load_file(clone_image_path)

            msg = f"📋 Imagen y etiquetas clonadas exitosamente: '{clone_image_filename}'"
            self.statusBar().showMessage(msg, 5000)
            QtWidgets.QMessageBox.information(self, "📋 Clonar Imagen", msg)

        except Exception as e:
            self.error_message("Error al clonar imagen", f"No se pudo clonar la imagen: {str(e)}")

    def rotate_image(self, angle):
        """Rotate current image by angle (90, 180, 270 degrees) and transform annotation shapes."""
        if not self.image or self.image.isNull() or not self.filename:
            return

        old_w = self.image.width()
        old_h = self.image.height()

        # Transform QImage
        transform = QtGui.QTransform().rotate(angle)
        rotated_qimage = self.image.transformed(transform)
        self.image = rotated_qimage

        # Save rotated QImage to disk
        try:
            self.image.save(self.filename)
        except Exception as e:
            logger.error("Error saving rotated image to disk: %s", e)

        # Transform shape points on canvas to matching rotated coordinates
        for shape in list(self.canvas.shapes):
            new_points = []
            for p in shape.points:
                x, y = p.x(), p.y()
                if angle == 90:
                    # 90 degrees Clockwise
                    nx = old_h - y
                    ny = x
                elif angle in (270, -90):
                    # 90 degrees Counter-Clockwise (270 CW)
                    nx = y
                    ny = old_w - x
                elif angle == 180:
                    # 180 degrees
                    nx = old_w - x
                    ny = old_h - y
                else:
                    nx, ny = x, y
                new_points.append(QtCore.QPointF(float(nx), float(ny)))

            if shape.shape_type == "rectangle" and len(new_points) == 2:
                p1, p2 = new_points[0], new_points[1]
                shape.points = [
                    QtCore.QPointF(min(p1.x(), p2.x()), min(p1.y(), p2.y())),
                    QtCore.QPointF(max(p1.x(), p2.x()), max(p1.y(), p2.y()))
                ]
            else:
                shape.points = new_points

        # Save updated transformed shapes to JSON label file on disk
        json_file = self.get_label_file_for_image(self.filename)
        if json_file:
            try:
                self._save_file(json_file)
            except Exception as e:
                logger.error("Error saving rotated annotation JSON: %s", e)

        # Update canvas pixmap & redraw shapes
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(self.image))
        self.canvas.update()

        # Clear auto labeling marks and SAM image embedding cache
        self.clear_auto_labeling_marks()
        if (
            hasattr(self, "auto_labeling_widget")
            and self.auto_labeling_widget.model_manager
            and self.auto_labeling_widget.model_manager.loaded_model_config
        ):
            model = self.auto_labeling_widget.model_manager.loaded_model_config.get("model")
            if model and hasattr(model, "image_embedding_cache"):
                model.image_embedding_cache.clear()

        # Update image data if stored
        if self.image_data and self._config["store_data"]:
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
            self.image.save(buffer, "PNG")
            self.image_data = buffer.data().data()

        msg = f"🔄 Imagen y {len(self.canvas.shapes)} etiqueta(s) giradas {angle}° exitosamente."
        self.statusBar().showMessage(msg, 4000)

    def resolve_image_path(self, raw_path):
        """Resolve a raw path from file list widget to an existing absolute path."""
        if not raw_path:
            return None
        if osp.isabs(raw_path) and osp.exists(raw_path):
            return osp.abspath(raw_path)
        if hasattr(self, "last_open_dir") and self.last_open_dir:
            combined = osp.join(self.last_open_dir, raw_path)
            if osp.exists(combined):
                return osp.abspath(combined)
        if osp.exists(raw_path):
            return osp.abspath(raw_path)
        return None

    def get_label_file_for_image(self, img_path):
        """Get path to the corresponding JSON label file for a given image path."""
        if not img_path:
            return None
        base_json = osp.splitext(img_path)[0] + ".json"
        if self.output_dir and osp.exists(self.output_dir):
            custom_json = osp.join(self.output_dir, osp.basename(base_json))
            if osp.exists(custom_json):
                return custom_json
        return base_json

    def change_image_folder(self):
        """Move selected image(s) and their corresponding segmentation/annotation file(s) to another directory."""
        selected_items = self.file_list_widget.selectedItems()
        image_paths = []
        if selected_items:
            for item in selected_items:
                raw_text = item.text()
                resolved = self.resolve_image_path(raw_text)
                if resolved and resolved not in image_paths:
                    image_paths.append(resolved)

        if not image_paths and self.filename:
            resolved = self.resolve_image_path(self.filename)
            if resolved:
                image_paths = [resolved]

        if not image_paths:
            QtWidgets.QMessageBox.warning(
                self,
                "📁 Cambiar de Carpeta",
                "No se pudo encontrar ninguna de las imágenes seleccionadas en el disco para mover."
            )
            return

        current_dir = self.last_open_dir or (osp.dirname(image_paths[0]) if image_paths else "")
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("📁 Seleccionar carpeta de destino para mover imagen(es) y segmentaciones"),
            current_dir
        )

        if not target_dir:
            return

        target_dir = osp.abspath(target_dir)

        # Configurar diálogo de progreso modal estilizado
        total_files = len(image_paths)
        progress = QtWidgets.QProgressDialog(
            self.tr("📁 Moviendo archivos e imágenes..."),
            self.tr("Cancelar"),
            0,
            total_files,
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setStyleSheet("""
            QProgressDialog {
                background-color: #1e1e2e;
                color: #ffffff;
                font-weight: bold;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }
            QProgressBar {
                border: 1px solid #383838;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                background-color: #2b2b3b;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #ef4444;
                color: #ffffff;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)

        moved_images_count = 0
        moved_json_count = 0
        moved_paths = []
        error_messages = []

        if hasattr(self, "auto_labeling_widget") and self.auto_labeling_widget:
            if hasattr(self.auto_labeling_widget, "model_manager") and self.auto_labeling_widget.model_manager:
                self.auto_labeling_widget.model_manager.stop_inference()

        current_file_was_moved = False

        for i, img_path in enumerate(image_paths):
            if progress.wasCanceled():
                break

            img_path = osp.abspath(img_path)
            src_dir = osp.dirname(img_path)
            img_name = osp.basename(img_path)

            progress.setValue(i)
            progress.setLabelText(
                self.tr(f"📁 Moviendo [{i + 1}/{total_files}]: {img_name}...")
            )
            QtWidgets.QApplication.processEvents()

            if src_dir == target_dir:
                continue

            target_img_path = osp.join(target_dir, img_name)
            json_src = self.get_label_file_for_image(img_path)
            json_name = osp.basename(json_src) if json_src else f"{osp.splitext(img_name)[0]}.json"
            target_json_path = osp.join(target_dir, json_name)

            try:
                if osp.exists(target_img_path):
                    os.remove(target_img_path)
                shutil.move(img_path, target_img_path)
                moved_images_count += 1

                if json_src and osp.exists(json_src):
                    if osp.exists(target_json_path):
                        os.remove(target_json_path)

                    try:
                        with open(json_src, "r", encoding="utf-8") as f:
                            json_data = json.load(f)
                        json_data["imagePath"] = img_name
                        with open(json_src, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                    shutil.move(json_src, target_json_path)
                    moved_json_count += 1

                moved_paths.append(img_path)

                if self.filename and osp.abspath(self.filename) == img_path:
                    current_file_was_moved = True

            except Exception as e:
                logger.error("Error al mover archivo %s: %s", img_path, e)
                error_messages.append(f"{img_name}: {str(e)}")

        progress.setValue(total_files)
        progress.close()

        if error_messages:
            err_summary = "\n".join(error_messages[:5])
            if len(error_messages) > 5:
                err_summary += f"\n...y {len(error_messages) - 5} errores más."
            QtWidgets.QMessageBox.warning(
                self,
                "⚠️ Error al Mover Archivos",
                f"Ocurrieron errores al mover los siguientes archivos:\n\n{err_summary}"
            )

        if moved_images_count == 0:
            self.statusBar().showMessage("⚠️ No se movió ninguna imagen (las imágenes ya estaban en la carpeta destino).", 4000)
            return

        curr_row = self.file_list_widget.currentRow()

        # Remover elementos de file_list_widget y de image_list
        for m_path in moved_paths:
            m_base = osp.basename(m_path)
            idx = 0
            while idx < self.file_list_widget.count():
                item = self.file_list_widget.item(idx)
                if item:
                    item_text = item.text()
                    resolved = self.resolve_image_path(item_text)
                    if (
                        resolved == m_path
                        or item_text == m_path
                        or item_text == m_base
                    ):
                        self.file_list_widget.takeItem(idx)
                        continue
                idx += 1

            if m_path in self.image_list:
                self.image_list.remove(m_path)
            else:
                for img_in_list in list(self.image_list):
                    if osp.abspath(img_in_list) == m_path or self.resolve_image_path(img_in_list) == m_path:
                        self.image_list.remove(img_in_list)

        self.clear_auto_labeling_marks()

        # Refrescar filtro activo si estaba activo
        active_filter = getattr(self, "_active_label_filter", None)
        if active_filter:
            self.filter_images_by_label(active_filter)
        elif current_file_was_moved:
            if self.file_list_widget.count() > 0:
                next_row = min(max(0, curr_row), self.file_list_widget.count() - 1)
                self.file_list_widget.setCurrentRow(next_row)
                item = self.file_list_widget.item(next_row)
                if item:
                    target_file = self.resolve_image_path(item.text()) or item.text()
                    self.load_file(target_file)
            else:
                self.reset_state()
                self.canvas.load_pixels(None)

        msg = f"📁 Movid@(s) {moved_images_count} imagen(es) y {moved_json_count} archivo(s) de segmentación a:\n'{target_dir}'"
        self.statusBar().showMessage(msg, 5000)
        QtWidgets.QMessageBox.information(self, "📁 Cambiar de Carpeta", msg)

    def open_dataset_gallery_dialog(self):
        """Open Roboflow-style Visual Dataset Gallery and Audit Dialog."""
        if not self.last_open_dir or not osp.exists(self.last_open_dir):
            QtWidgets.QMessageBox.warning(self, "Galería del Dataset", "Abre una carpeta con imágenes primero para ver la galería.")
            return

        from anylabeling.views.labeling.widgets.dataset_gallery_dialog import DatasetGalleryDialog
        dialog = DatasetGalleryDialog(self, parent=self)
        dialog.exec()

    def filter_images_by_shape_count(self, max_count=0):
        """Filter file list widget to show only images with 0 or < max_count shapes/annotations."""
        if not self.last_open_dir or not osp.exists(self.last_open_dir):
            self.statusBar().showMessage("⚠️ Abre una carpeta con imágenes primero para filtrar.")
            return

        filter_desc = "Sin Etiquetas (0)" if max_count == 0 else f"< {max_count} Etiquetas"
        self._active_label_filter = f"Count:{filter_desc}"

        all_images = self.scan_all_images(self.last_open_dir)
        matched_images = []

        for img_path in all_images:
            json_path = self.get_label_file_for_image(img_path)
            shape_count = 0
            if json_path and osp.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    shape_count = len(data.get("shapes", []))
                except Exception:
                    pass

            if max_count == 0:
                if shape_count == 0:
                    matched_images.append(img_path)
            else:
                if shape_count < max_count:
                    matched_images.append(img_path)

        self.filename = None
        self.file_list_widget.clear()

        for file in matched_images:
            label_file = self.get_label_file_for_image(file)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if label_file and QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.file_list_widget.addItem(item)

        try:
            btn_filter = getattr(self, "btn_filter_status", None)
            if btn_filter:
                btn_filter.setText(f"🔍 Filtro: '{filter_desc}' ({len(matched_images)}/{len(all_images)}) [❌ Quitar]")
                btn_filter.setVisible(True)
        except (AttributeError, RuntimeError):
            pass

        msg = f"🔍 Filtro '{filter_desc}': Encontradas {len(matched_images)} de {len(all_images)} imágenes."
        self.statusBar().showMessage(msg, 5000)

    def open_image_editor_dialog(self):
        """Open mini image editor and dataset augmentation dialog."""
        if not self.image or self.image.isNull():
            QtWidgets.QMessageBox.warning(self, "Editor de Imagen", "Abre una imagen primero para editarla.")
            return
        dialog = ImageEditorDialog(self, parent=self)
        dialog.exec()

    def delete_image_and_json(self):
        """Permanently delete current image file and its JSON annotations from disk."""
        if not self.filename or not osp.exists(self.filename):
            QtWidgets.QMessageBox.warning(self, "Eliminar Imagen", "No hay ninguna imagen cargada.")
            return

        image_filename = osp.basename(self.filename)
        mb = QtWidgets.QMessageBox
        msg = f"¿Estás seguro de que deseas eliminar permanentemente la imagen '{image_filename}' y sus etiquetas de tu disco?"
        answer = mb.warning(
            self,
            "🗑️ Confirmar Eliminación",
            msg,
            mb.StandardButton.Yes | mb.StandardButton.No,
        )
        if answer != mb.StandardButton.Yes:
            return

        try:
            # Delete label JSON file if it exists
            label_file = self.get_label_file()
            if osp.exists(label_file):
                os.remove(label_file)

            # Delete image file from disk
            if osp.exists(self.filename):
                os.remove(self.filename)

            # Clear auto labeling marks
            self.clear_auto_labeling_marks()

            # Remove from file list widget
            curr_row = self.file_list_widget.currentRow()
            if self.filename in self.image_list:
                idx = self.image_list.index(self.filename)
                self.image_list.pop(idx)
                self.file_list_widget.takeItem(idx)

            msg = f"🗑️ Imagen '{image_filename}' eliminada de tu disco."
            self.statusBar().showMessage(msg, 4000)

            # Load next image if available
            if self.image_list:
                next_row = min(curr_row, len(self.image_list) - 1)
                self.file_list_widget.setCurrentRow(next_row)
                self.load_file(self.image_list[next_row])
            else:
                self.reset_state()

        except Exception as e:
            self.error_message("Error al eliminar imagen", f"No se pudo eliminar la imagen: {str(e)}")

    def save_file_dialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            file_dialog = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            file_dialog = QtWidgets.QFileDialog(
                self, caption, self.current_path(), filters
            )
        file_dialog.setDefaultSuffix(LabelFile.suffix[1:])
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setOption(QtWidgets.QFileDialog.Option.DontConfirmOverwrite, False)
        file_dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.current_path(), basename + LabelFile.suffix
            )
        filename = file_dialog.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _save_file(self, filename):
        if filename and self.save_labels(filename):
            self.add_recent_file(filename)
            self.set_clean()

    def close_file(self, _value=False):
        if not self.may_continue():
            return
        self.reset_state()
        self.set_clean()
        self.toggle_actions(False)
        self.canvas.setEnabled(False)
        self.actions.save_as.setEnabled(False)

    def get_label_file(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def delete_file(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, proceed anyway?"
        )
        answer = mb.warning(
            self,
            self.tr("Attention"),
            msg,
            mb.StandardButton.Yes | mb.StandardButton.No,
        )
        if answer != mb.StandardButton.Yes:
            return

        label_file = self.get_label_file()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: %s", label_file)

            item = self.file_list_widget.currentItem()
            item.setCheckState(Qt.CheckState.Unchecked)

            self.reset_state()

    # Message Dialogs. #
    def has_labels(self):
        if self.no_shape():
            self.error_message(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def has_label_file(self):
        if self.filename is None:
            return False

        label_file = self.get_label_file()
        return osp.exists(label_file)

    def may_continue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr(f'Save annotations to "{self.filename!r}" before closing?')
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.StandardButton.Save
            | mb.StandardButton.Discard
            | mb.StandardButton.Cancel,
            mb.StandardButton.Save,
        )
        if answer == mb.StandardButton.Discard:
            return True
        if answer == mb.StandardButton.Save:
            self.save_file()
            return True
        # answer == mb.StandardButton.Cancel
        return False

    def error_message(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def current_path(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggle_keep_prev_mode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]
        save_config(self._config)

    def toggle_auto_use_last_label(self):
        self._config["auto_use_last_label"] = not self._config["auto_use_last_label"]
        save_config(self._config)

    def remove_selected_point(self):
        self.canvas.remove_selected_point()
        self.canvas.update()
        if self.canvas.h_hape is not None and not self.canvas.h_hape.points:
            self.canvas.delete_shape(self.canvas.h_hape)
            self.remove_labels([self.canvas.h_hape])
            self.set_dirty()
            if self.no_shape():
                for act in self.actions.on_shapes_present:
                    act.setEnabled(False)

    def delete_selected_shape(self):
        selected_shapes = list(self.canvas.selected_shapes)
        if not selected_shapes:
            selected_items = self.label_list.selected_items()
            if selected_items:
                selected_shapes = [item.shape() for item in selected_items if hasattr(item, "shape")]
                self.canvas.select_shapes(selected_shapes)

        if not selected_shapes:
            return

        yes, no = (
            QtWidgets.QMessageBox.StandardButton.Yes,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        msg = self.tr(
            "You are about to permanently delete {} polygons, proceed anyway?"
        ).format(len(selected_shapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            deleted = self.canvas.delete_selected()
            if not deleted and selected_shapes:
                for s in selected_shapes:
                    if s in self.canvas.shapes:
                        self.canvas.shapes.remove(s)
                deleted = selected_shapes

            self.remove_labels(deleted)
            self.set_dirty()

            # Save immediately to disk so JSON deletion persists across filter switching
            if self.filename:
                try:
                    self.save_file()
                except Exception as e:
                    logger.error("Error saving file after deleting shapes: %s", e)

            if self.no_shape():
                for act in self.actions.on_shapes_present:
                    act.setEnabled(False)

            # Refresh active label filter if present
            active_filter = getattr(self, "_active_label_filter", None)
            if active_filter:
                self.filter_images_by_label(active_filter)

    def delete_label_across_dataset(self, label):
        """Delete all instances of a given label across all open/filtered dataset images."""
        if not label:
            return

        image_paths = []
        if self.file_list_widget.count() > 0:
            for i in range(self.file_list_widget.count()):
                item = self.file_list_widget.item(i)
                if item:
                    resolved = self.resolve_image_path(item.text())
                    if resolved and resolved not in image_paths:
                        image_paths.append(resolved)

        if not image_paths and self.image_list:
            image_paths = list(self.image_list)

        if not image_paths:
            QtWidgets.QMessageBox.warning(self, "Eliminar Etiqueta", "No hay imágenes cargadas para procesar.")
            return

        yes, no = (
            QtWidgets.QMessageBox.StandardButton.Yes,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        msg = f"⚠️ ¿Deseas eliminar TODAS las instancias de la etiqueta '{label}' en las {len(image_paths)} imágenes de la lista?"
        if yes != QtWidgets.QMessageBox.warning(self, "Eliminar Etiqueta del Dataset", msg, yes | no, no):
            return

        modified_count = 0
        deleted_shapes_count = 0

        # Process current canvas shapes
        if self.canvas.shapes:
            to_remove = [s for s in self.canvas.shapes if getattr(s, "label", None) == label]
            if to_remove:
                for s in to_remove:
                    if s in self.canvas.shapes:
                        self.canvas.shapes.remove(s)
                    deleted_shapes_count += 1
                self.remove_labels(to_remove)
                self.set_dirty()
                if self.filename:
                    self.save_file()
                modified_count += 1

        # Process all JSON files on disk
        for img_path in image_paths:
            json_file = self.get_label_file_for_image(img_path)
            if json_file and osp.exists(json_file):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    shapes = data.get("shapes", [])
                    new_shapes = [s for s in shapes if s.get("label") != label]

                    if len(new_shapes) < len(shapes):
                        deleted_shapes_count += (len(shapes) - len(new_shapes))
                        data["shapes"] = new_shapes
                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        modified_count += 1
                except Exception as e:
                    logger.error("Error al eliminar etiqueta en JSON %s: %s", json_file, e)

        # Refresh canvas if current image was modified
        if self.filename:
            json_file = self.get_label_file_for_image(self.filename)
            if json_file and osp.exists(json_file):
                self.load_label_file(json_file)

        # Refresh active label filter if present
        active_filter = getattr(self, "_active_label_filter", None)
        if active_filter:
            self.filter_images_by_label(active_filter)

        msg = f"🗑️ Se eliminaron {deleted_shapes_count} objeto(s) con la etiqueta '{label}' en {modified_count} archivo(s)."
        self.statusBar().showMessage(msg, 5000)
        QtWidgets.QMessageBox.information(self, "Eliminar Etiqueta del Dataset", msg)

    def copy_shape(self):
        self.canvas.end_move(copy=True)
        for shape in self.canvas.selected_shapes:
            self.add_label(shape)
        self.label_list.clearSelection()
        self.set_dirty()

    def move_shape(self):
        self.canvas.end_move(copy=False)
        self.set_dirty()

    def open_saved_directories_dialog(self):
        """Open Saved Directories Manager Dialog."""
        dialog = DirectoryBookmarksDialog(self, self.saved_directories)
        if dialog.exec():
            self.saved_directories = dialog.saved_dirs
            self.settings.setValue("saved_directories", self.saved_directories)
            if dialog.selected_dir:
                self.import_image_folder(dialog.selected_dir)

    def open_folder_dialog(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if dirpath and osp.exists(dirpath):
            self.import_image_folder(dirpath)
        else:
            self.open_saved_directories_dialog()

    @property
    def image_list(self):
        lst = []
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            lst.append(item.text())
        return lst

    def import_dropped_image_files(self, image_files):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in image_files:
            if file in self.image_list or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.file_list_widget.addItem(item)

        if len(self.image_list) > 1:
            self.actions.open_next_image.setEnabled(True)
            self.actions.open_prev_image.setEnabled(True)

        self.open_next_image()

    def import_image_folder(self, dirpath, pattern=None, load=True):
        self.actions.open_next_image.setEnabled(True)
        self.actions.open_prev_image.setEnabled(True)

        if not self.may_continue() or not dirpath:
            return

        self.last_open_dir = dirpath
        dirpath_str = str(dirpath)
        if dirpath_str and dirpath_str not in self.saved_directories and osp.exists(dirpath_str):
            self.saved_directories.append(dirpath_str)
            self.settings.setValue("saved_directories", self.saved_directories)

        self.filename = None
        self.file_list_widget.clear()
        for filename in self.scan_all_images(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.file_list_widget.addItem(item)
        self.open_next_image(load=load)
        self.analyze_folder_dataset(dirpath)

    def update_unique_label_counts(self):
        """Actualiza el conteo por imagen y resalta las etiquetas activas en blanco."""
        if not hasattr(self, "unique_label_list") or not hasattr(self, "canvas"):
            return

        counts = {}
        for shape in self.canvas.shapes:
            if shape.label and shape.label not in [AutoLabelingMode.ADD, AutoLabelingMode.REMOVE, AutoLabelingMode.OBJECT]:
                counts[shape.label] = counts.get(shape.label, 0) + 1

        for row in range(self.unique_label_list.count()):
            item = self.unique_label_list.item(row)
            label = item.data(Qt.ItemDataRole.UserRole)
            if label:
                cnt = counts.get(label, 0)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, count=cnt, active_in_image=(cnt > 0)
                )

    def set_q_plus_plus_mode(self, enabled):
        """Enable/Disable Q(++) fast auto-labeling mode."""
        self.q_plus_plus_mode = enabled
        if enabled:
            label = self.get_selected_unique_label()
            label_info = f"'{label}'" if label else "Ninguno (Selecciona una de la lista)"
            self.statusBar().showMessage(
                f"⚡ Modo Q(++) ACTIVADO. Etiqueta activa para auto-labeling: {label_info}", 6000
            )
            if hasattr(self, "auto_labeling_widget") and self.auto_labeling_widget:
                if self.auto_labeling_widget.auto_labeling_mode == AutoLabelingMode.NONE:
                    self.auto_labeling_widget.set_auto_labeling_mode(
                        AutoLabelingMode.ADD, AutoLabelingMode.POINT
                    )
        else:
            self.statusBar().showMessage("Modo Q(++) Desactivado", 3000)

    def get_selected_unique_label(self):
        """Get currently selected label in unique_label_list or last selected label."""
        if hasattr(self, "unique_label_list"):
            items = self.unique_label_list.selectedItems()
            if items:
                label = items[0].data(Qt.ItemDataRole.UserRole)
                if label and label not in [
                    AutoLabelingMode.OBJECT,
                    AutoLabelingMode.ADD,
                    AutoLabelingMode.REMOVE,
                ]:
                    return label
        if hasattr(self, "_last_selected_unique_label") and self._last_selected_unique_label:
            return self._last_selected_unique_label
        return self.find_last_label()

    def on_unique_label_item_clicked(self, item):
        """Atajo directo: Clic en etiqueta activa el modo de dibujo de Polígono (W) salvo que estemos en Auto-Labeling."""
        if item:
            # Seleccionar etiqueta para dibujar
            label = item.data(Qt.ItemDataRole.UserRole)
            if label:
                self.label_dialog.add_label_history(label)
                self._last_selected_unique_label = label
                if getattr(self, "q_plus_plus_mode", False):
                    self.statusBar().showMessage(
                        f"⚡ Q(++) Etiqueta cambiada a: '{label}'", 4000
                    )
                    if hasattr(self, "auto_labeling_widget") and self.auto_labeling_widget:
                        if self.auto_labeling_widget.auto_labeling_mode == AutoLabelingMode.NONE:
                            self.auto_labeling_widget.set_auto_labeling_mode(
                                AutoLabelingMode.ADD, AutoLabelingMode.POINT
                            )
            # Solo cambiar a dibujo manual de polígono si NO estamos en Auto-Labeling ni en Q(++)
            is_auto = (
                hasattr(self, "auto_labeling_widget")
                and self.auto_labeling_widget
                and self.auto_labeling_widget.auto_labeling_mode != AutoLabelingMode.NONE
            )
            if not is_auto and not getattr(self, "q_plus_plus_mode", False):
                self.toggle_draw_mode(False, create_mode="polygon")

    def refine_current_polygons(self):
        """Motor de Perfeccionamiento de Anotaciones: Suaviza vértices y elimina ruido micro-poligonal."""
        if not self.canvas.shapes:
            self.statusBar().showMessage("⚠️ No hay polígonos en la imagen actual para perfeccionar.")
            return

        refined_count = 0
        removed_count = 0

        for shape in list(self.canvas.shapes):
            if len(shape.points) < 3:
                self.canvas.shapes.remove(shape)
                removed_count += 1
                continue

            pts = np.array([[p.x(), p.y()] for p in shape.points], dtype=np.float32)
            contour = pts.reshape((-1, 1, 2))

            # Filtrar micro-ruido poligonal (< 15 px de área)
            area = cv2.contourArea(contour)
            if area < 15.0:
                self.canvas.shapes.remove(shape)
                removed_count += 1
                continue

            # Algoritmo Ramer-Douglas-Peucker para contorno limpio y preciso
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= 3:
                new_points = [QtCore.QPointF(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                shape.points = new_points
                refined_count += 1

        self.canvas.update()
        self.set_dirty()
        self.update_unique_label_counts()

        msg = f"⚡ Perfeccionamiento Completado: {refined_count} polígonos optimizados, {removed_count} artefactos borrados."
        self.statusBar().showMessage(msg)
        QtWidgets.QMessageBox.information(self, "⚡ Motor de Perfeccionamiento", msg)

    def audit_dataset_health(self):
        """Auditoría de Calidad y Salud del Dataset."""
        if not self.last_open_dir or not osp.exists(self.last_open_dir):
            QtWidgets.QMessageBox.warning(self, "📊 Auditoría Dataset", "Por favor abre una carpeta primero.")
            return

        json_files = glob.glob(osp.join(self.last_open_dir, "*.json"))
        images = self.scan_all_images(self.last_open_dir)

        total_images = len(images)
        annotated_count = len(json_files)
        pct_annotated = (annotated_count / total_images * 100) if total_images > 0 else 0

        class_counts = {}
        total_polygons = 0

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for shape in data.get("shapes", []):
                        lbl = shape.get("label", "").strip()
                        if lbl:
                            class_counts[lbl] = class_counts.get(lbl, 0) + 1
                            total_polygons += 1
            except Exception:
                pass

        report_lines = [
            f"<b>📁 Carpeta:</b> {osp.basename(self.last_open_dir)}",
            f"<b>🖼️ Imágenes Totales:</b> {total_images}",
            f"<b>✅ Imágenes Anotadas:</b> {annotated_count} ({pct_annotated:.1f}%)",
            f"<b>🔺 Polígonos Totales:</b> {total_polygons}",
            "<br/><b>🏷️ Distribución de Clases:</b>"
        ]

        for cls_name, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / total_polygons * 100) if total_polygons > 0 else 0
            report_lines.append(f"• <b>{cls_name}</b>: {cnt} ({pct:.1f}%)")

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("📊 Auditoría de Salud del Dataset")
        msg_box.setText("<b>Informe de Calidad y Cobertura de Etiquetas</b>")
        msg_box.setInformativeText("<br/>".join(report_lines))
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.exec()

    def analyze_folder_dataset(self, dirpath):
        """Motor de análisis automático de dataset al cargar una carpeta (.json, .txt, labels.txt)"""
        if not dirpath or not osp.exists(dirpath):
            return

        json_files = []
        txt_files = []
        for root, _, files in os.walk(dirpath):
            for file in files:
                full_path = osp.join(root, file)
                if file.lower().endswith(".json"):
                    json_files.append(full_path)
                elif file.lower().endswith(".txt") and file.lower() not in ["labels.txt", "classes.txt"]:
                    txt_files.append(full_path)

        detected_classes = set()
        polygon_count = 0
        annotated_json_count = len(json_files)

        # Parse JSON files
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for shape in data.get("shapes", []):
                        lbl = shape.get("label", "").strip()
                        if lbl and lbl not in [AutoLabelingMode.ADD, AutoLabelingMode.REMOVE, AutoLabelingMode.OBJECT]:
                            detected_classes.add(lbl)
                            polygon_count += 1
            except Exception:
                pass

        # Parse labels.txt / classes.txt if present
        labels_txt_path = osp.join(dirpath, "labels.txt")
        if not osp.exists(labels_txt_path):
            labels_txt_path = osp.join(dirpath, "classes.txt")

        if osp.exists(labels_txt_path):
            try:
                with open(labels_txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        l = line.strip()
                        if l:
                            detected_classes.add(l)
            except Exception:
                pass

        # Register detected classes into unique_label_list
        for class_name in sorted(list(detected_classes)):
            if not self.unique_label_list.find_items_by_label(class_name):
                item = self.unique_label_list.create_item_from_label(class_name)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(class_name)
                self.unique_label_list.set_item_label(item, class_name, rgb)

        self.update_unique_label_counts()

        # Notification Popup & Status Summary
        if detected_classes or annotated_json_count > 0 or txt_files:
            classes_str = ", ".join(sorted(list(detected_classes))) if detected_classes else "Ninguna"
            summary_msg = f"📊 Análisis de Dataset: {annotated_json_count} JSONs | {len(txt_files)} TXTs | Clases: {classes_str}"
            self.statusBar().showMessage(summary_msg)

            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("📊 Motor de Análisis de Dataset")
            msg_box.setText("<b>Análisis Automático de Etiquetas Completado</b>")
            msg_box.setInformativeText(
                f"<b>Carpeta:</b> {osp.basename(dirpath)}<br/>"
                f"<b>Archivos JSON Anotados:</b> {annotated_json_count}<br/>"
                f"<b>Archivos TXT Detecciones:</b> {len(txt_files)}<br/>"
                f"<b>Polígonos Totales:</b> {polygon_count}<br/>"
                f"<b>Clases Detectadas:</b> {classes_str}<br/><br/>"
                f"<i>✓ Todas las clases han sido registradas automáticamente en el selector.</i>"
            )
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
            msg_box.exec()

    def eventFilter(self, source, event):
        """Intercept viewport right clicks on file_list_widget to preserve multi-selection."""
        if (
            hasattr(self, "file_list_widget")
            and (source == self.file_list_widget or source == self.file_list_widget.viewport())
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
        ):
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                item = self.file_list_widget.itemAt(event.pos())
                if item and item.isSelected():
                    # Consume press event so Qt preserves the multi-selection.
                    # Qt's customContextMenuRequested will still be triggered on release!
                    return True
        return super().eventFilter(source, event)

    def scan_all_images(self, folder_path):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
            if fmt.data().decode().lower() != "svg"
        ]

        images = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relative_path = osp.join(root, file)
                    images.append(relative_path)
        images = natsort.os_sorted(images)
        return images

    def toggle_auto_labeling_widget(self):
        """Toggle auto labeling widget visibility."""
        if self.auto_labeling_widget.isVisible():
            self.auto_labeling_widget.hide()
        else:
            self.auto_labeling_widget.show()

    def _reactivate_q_plus_plus(self):
        """Reactivate Q(++) mode safely after file load finishes."""
        if not getattr(self, "_is_loading_file", False) and self.filename:
            if hasattr(self, "auto_labeling_widget") and self.auto_labeling_widget:
                if hasattr(self.auto_labeling_widget, "button_q_plus_plus"):
                    self.auto_labeling_widget.button_q_plus_plus.setChecked(True)

    @pyqtSlot()
    def new_shapes_from_auto_labeling(self, auto_labeling_result):
        """Apply auto labeling results to the current image."""
        if not self.image or not self.image_path or getattr(self, "_is_loading_file", False):
            return
        # Descartar resultados desactualizados pertenecientes a una imagen anterior
        if hasattr(auto_labeling_result, "filename") and auto_labeling_result.filename:
            if not self.filename or osp.abspath(auto_labeling_result.filename) != osp.abspath(self.filename):
                logger.warning(
                    f"Descartando resultado de IA obsoleto para '{auto_labeling_result.filename}' "
                    f"(imagen actual es '{self.filename}')"
                )
                return
        elif not getattr(auto_labeling_result, "shapes", None):
            return
        # Clear existing shapes
        if auto_labeling_result.replace:
            self.load_shapes([], replace=True)
            self.label_list.clear()
            self.load_shapes(auto_labeling_result.shapes, replace=True)
        else:  # Just update existing shapes
            # Remove shapes with label AutoLabelingMode.OBJECT
            for shape in list(self.canvas.shapes):
                if shape.label == AutoLabelingMode.OBJECT:
                    try:
                        item = self.label_list.find_item_by_shape(shape)
                        if item:
                            self.label_list.remove_item(item)
                    except (ValueError, Exception):
                        pass
            self.load_shapes(auto_labeling_result.shapes, replace=False)

        # If Q(++) mode is active, automatically finalize and tag object synchronously
        if getattr(self, "q_plus_plus_mode", False) and not getattr(self, "_is_loading_file", False):
            try:
                self.finish_auto_labeling_object()
            except Exception as e:
                logger.error("Error in Q(++) auto-labeling completion: %s", e)

    def on_auto_labeling_started(self, message):
        """Block canvas input during auto labeling model load or inference"""
        self.canvas.set_loading(True, message)

    def on_auto_labeling_finished(self):
        """Unblock canvas input after auto labeling model load or inference"""
        self.canvas.set_loading(False)

    def clear_auto_labeling_marks(self):
        """Clear auto labeling marks from the current image and reset model predictor state."""
        if hasattr(self, "canvas") and self.canvas:
            if self.canvas.drawing():
                self.canvas.current = None
                self.canvas.drawing_polygon.emit(False)

        # Reset model manager marks array to prevent SAM predictor tensor mismatch crashes
        if hasattr(self, "auto_labeling_widget") and self.auto_labeling_widget:
            try:
                self.auto_labeling_widget.model_manager.set_auto_labeling_marks([])
            except Exception:
                pass

        # Clean up label list (using safe snapshot copy of shapes)
        for shape in list(self.canvas.shapes):
            if shape.label in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                try:
                    item = self.label_list.find_item_by_shape(shape)
                    if item:
                        self.label_list.remove_item(item)
                except Exception:
                    pass

        # Clean up unique label list
        for shape_label in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            for item in self.unique_label_list.find_items_by_label(shape_label):
                self.unique_label_list.takeItem(self.unique_label_list.row(item))

        # Remove shapes from the canvas
        self.canvas.shapes = [
            shape
            for shape in list(self.canvas.shapes)
            if shape.label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        self.canvas.update()

    def find_last_label(self):
        """
        Find the last label in the label list.
        Exclude labels for auto labeling.
        """

        # Get from dialog history
        last_label = self.label_dialog.get_last_label()
        if last_label:
            return last_label

        # Get selected label from the label list
        items = self.label_list.selected_items()
        if items:
            shape = items[0].data(Qt.ItemDataRole.UserRole)
            return shape.label

        # Get the last label from the label list
        for item in reversed(self.label_list):
            shape = item.data(Qt.ItemDataRole.UserRole)
            if shape.label not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                return shape.label

        # No label is found
        return ""

    def finish_auto_labeling_object(self):
        """Finish auto labeling object safely without recursive/concurrent re-entry crashes."""
        if getattr(self, "_is_finishing_auto_object", False) or getattr(self, "_is_loading_file", False) or not self.filename:
            return

        try:
            self._is_finishing_auto_object = True

            has_object = False
            for shape in list(self.canvas.shapes):
                if shape.label == AutoLabelingMode.OBJECT:
                    has_object = True
                    break

            # If there is no object, do nothing
            if not has_object:
                return

            # Ask a label for the object
            text, flags, group_id = "", {}, None
            last_label = self.find_last_label()
            if getattr(self, "q_plus_plus_mode", False):
                selected_label = self.get_selected_unique_label()
                if selected_label:
                    text = selected_label
                elif last_label:
                    text = last_label
                else:
                    text = "object"
            elif self._config["auto_use_last_label"] and last_label:
                text = last_label
            else:
                previous_text = self.label_dialog.edit.text()
                text, flags, group_id = self.label_dialog.pop_up(
                    text=self.find_last_label(),
                    flags={},
                    group_id=None,
                )
                if not text:
                    self.label_dialog.edit.setText(previous_text)
                    self.clear_auto_labeling_marks()
                    return

            if not text:
                text = "object"

            if not self.validate_label(text):
                self.error_message(
                    self.tr("Invalid label"),
                    self.tr("Invalid label '{}' with validation type '{}'").format(
                        text, self._config["validate_label"]
                    ),
                )
                return

            # Add to label history
            self.label_dialog.add_label_history(text)

            # Update label for the object
            updated_shapes = False
            for shape in list(self.canvas.shapes):
                if shape.label == AutoLabelingMode.OBJECT:
                    updated_shapes = True
                    shape.label = text
                    shape.flags = flags
                    shape.group_id = group_id
                    # Update unique label list
                    if not self.unique_label_list.find_items_by_label(shape.label):
                        unique_label_item = self.unique_label_list.create_item_from_label(
                            shape.label
                        )
                        self.unique_label_list.addItem(unique_label_item)
                        rgb = self._get_rgb_by_label(shape.label)
                        self.unique_label_list.set_item_label(
                            unique_label_item, shape.label, rgb
                        )

                    # Update label list
                    self._update_shape_color(shape)
                    try:
                        item = self.label_list.find_item_by_shape(shape)
                    except (ValueError, Exception):
                        item = None

                    if item:
                        if shape.group_id is None:
                            color = shape.fill_color.getRgb()[:3]
                            item.setText(
                                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                                    html.escape(shape.label), *color
                                )
                            )
                        else:
                            item.setText(f"{shape.label} ({shape.group_id})")
                    else:
                        text_label = shape.label if shape.group_id is None else f"{shape.label} ({shape.group_id})"
                        label_list_item = LabelListWidgetItem(text_label, shape)
                        self.label_list.add_item(label_list_item)

            # Clean up auto labeling objects & reset marks
            self.clear_auto_labeling_marks()

            # Update shape colors
            for shape in list(self.canvas.shapes):
                self._update_shape_color(shape)

            if updated_shapes:
                self.set_dirty()
                if getattr(self, "auto_next_image_mode", False) and not getattr(self, "_is_loading_file", False):
                    self.save_file()
                    self.open_next_image()
        finally:
            self._is_finishing_auto_object = False

    def set_text_editing(self, enable):
        """Set text editing."""
        if enable:
            # Enable text editing and set shape text from selected shape
            if len(self.canvas.selected_shapes) == 1:
                self.shape_text_label.setText(self.tr("Object Text"))
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(self.canvas.selected_shapes[0].text)
                self.shape_text_edit.textChanged.connect(self.shape_text_changed)
            else:
                self.shape_text_label.setText(self.tr("Image Text"))
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(self.other_data.get("image_text", ""))
                self.shape_text_edit.textChanged.connect(self.shape_text_changed)
            self.shape_text_edit.setDisabled(False)
        else:
            self.shape_text_edit.setDisabled(True)
            self.shape_text_label.setText(
                self.tr("Switch to Edit mode for text editing")
            )
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText("")
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)

    def export_annotations(self):
        """Open export dialog to export annotations to different formats."""
        # Get the current directory
        current_dir = None
        if self.filename:
            current_dir = osp.dirname(self.filename)
        elif self.output_dir:
            current_dir = self.output_dir

        # Create and show export dialog
        dialog = ExportDialog(self, current_dir)
        dialog.exec()

    def toggle_tools(self):
        """Toggle the tools panel visibility."""
        if hasattr(self.parent, "toggle_tools_panel"):
            self.parent.toggle_tools_panel()

    def reset_dock_layout(self):
        """Reset dock widget layout to default positions."""
        # Close all docks first
        self.shape_text_dock.close()
        self.flag_dock.close()
        self.label_dock.close()
        self.shape_dock.close()
        self.file_dock.close()
        self.tools_dock.close()

        # Re-add them in the desired order/position
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.tools_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.shape_text_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.shape_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.flag_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.label_dock
        )
        self.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.file_dock
        )

        # Show all docks
        self.tools_dock.show()
        self.file_dock.show()
        self.shape_dock.show()
        self.label_dock.show()
        self.flag_dock.hide()
        self.shape_text_dock.show()

        # Make sure tools dock is visible
        self.tools_dock.raise_()

        # Connect dock signals to save state when changed and update orientation
        self.tools_dock.dockLocationChanged.connect(self.on_tools_dock_location_changed)
        self.shape_text_dock.dockLocationChanged.connect(self.save_dock_state)
        self.flag_dock.dockLocationChanged.connect(self.save_dock_state)
        self.label_dock.dockLocationChanged.connect(self.save_dock_state)
        self.shape_dock.dockLocationChanged.connect(self.save_dock_state)
        self.file_dock.dockLocationChanged.connect(self.save_dock_state)

        # Also connect visibility changes
        self.tools_dock.visibilityChanged.connect(self.save_dock_state)
        self.shape_text_dock.visibilityChanged.connect(self.save_dock_state)
        self.flag_dock.visibilityChanged.connect(self.save_dock_state)
        self.label_dock.visibilityChanged.connect(self.save_dock_state)
        self.shape_dock.visibilityChanged.connect(self.save_dock_state)
        self.file_dock.visibilityChanged.connect(self.save_dock_state)

        # Apply a workaround to ensure proper sizes
        self.main_window.resizeDocks(
            [
                self.tools_dock,
                self.shape_text_dock,
                self.flag_dock,
                self.label_dock,
                self.shape_dock,
                self.file_dock,
            ],
            [40, 300, 300, 300, 300, 300],
            Qt.Orientation.Horizontal,
        )

        # Reset any saved dock state in config
        try:
            config = get_config()
            if (
                "ui" in config
                and isinstance(config["ui"], dict)
                and "dock_state" in config["ui"]
            ):
                del config["ui"]["dock_state"]
                save_config(config)
                logger.info("Previous dock state cleared from config")
        except Exception as e:
            logger.error(f"Error clearing dock state from config: {e}")

        # Wait a short time for layout to stabilize, then save new layout
        QtCore.QTimer.singleShot(100, self.save_dock_state)

        # Show a status message
        self.statusBar().showMessage(self.tr("Dock layout reset to default"), 5000)

    def set_theme(self, theme):
        """Set application theme"""
        # Update environment variable to override system theme detection
        if theme == "light":
            os.environ["DARK_MODE"] = "0"
        elif theme == "dark":
            os.environ["DARK_MODE"] = "1"
        else:  # system
            if "DARK_MODE" in os.environ:
                del os.environ["DARK_MODE"]

        # Save the theme setting to config
        self._config["theme"] = theme
        save_config(self._config)

        # Show dialog to restart application
        msg_box = QMessageBox()
        msg_box.setText(
            self.tr("Please restart the application to apply the theme change.")
        )
        msg_box.exec()

    def save_dock_state(self, force=False):
        """Save dock state to config with error handling.

        Args:
            force (bool): If True, save regardless of how much time has passed since the last save
        """
        try:
            # Use a minimum time interval between saves to prevent too frequent saving
            current_time = QtCore.QDateTime.currentMSecsSinceEpoch()
            if not force and hasattr(self, "_last_dock_save_time"):
                time_since_last_save = current_time - self._last_dock_save_time
                if time_since_last_save < 10000:  # Less than 10 seconds since last save
                    return  # Skip this save to prevent excessive config writes

            config = get_config()

            # Make sure UI configuration exists
            if "ui" not in config or not isinstance(config["ui"], dict):
                config["ui"] = {}

            # Get QByteArray state and convert to Base64 string
            byte_state = self.main_window.saveState()
            if byte_state.isEmpty():
                logger.warning("Cannot save empty dock state")
                return

            base64_state = byte_state.toBase64().data().decode()
            if not base64_state:
                logger.warning("Failed to encode dock state to Base64")
                return

            # Store in config and save
            config["ui"]["dock_state"] = base64_state
            save_config(config)
            self._last_dock_save_time = current_time
            logger.debug("Dock state saved successfully")

        except Exception as e:
            logger.error(f"Error saving dock state: {e}")

    def load_dock_state(self):
        """Load dock state from config with better error handling."""
        config = get_config()

        # Check if we have a valid dock state in config
        has_dock_state = (
            "ui" in config
            and isinstance(config["ui"], dict)
            and "dock_state" in config["ui"]
            and config["ui"]["dock_state"]
        )

        if not has_dock_state:
            logger.info("No saved dock state found, using default layout")
            return

        logger.info("Attempting to load dock state...")

        try:
            # Convert stored Base64 string back to QByteArray
            base64_str = config["ui"]["dock_state"]
            logger.debug(f"Encoded dock state: {base64_str[:30]}...")

            try:
                dock_state = QtCore.QByteArray.fromBase64(base64_str.encode())
                logger.debug(f"Decoded QByteArray size: {len(dock_state)}")
            except Exception as decode_error:
                logger.error(f"Failed to decode Base64 string: {decode_error}")
                raise decode_error

            # Make sure all dock widgets exist before restoring state
            all_docks_exist = all(
                [
                    hasattr(self, "tools_dock"),
                    hasattr(self, "shape_text_dock"),
                    hasattr(self, "flag_dock"),
                    hasattr(self, "label_dock"),
                    hasattr(self, "shape_dock"),
                    hasattr(self, "file_dock"),
                ]
            )

            if not all_docks_exist:
                logger.error(
                    "Cannot restore dock state - not all dock widgets are initialized"
                )
                return

            # Force all docks to be visible first
            self.tools_dock.setVisible(True)
            self.shape_text_dock.setVisible(True)
            self.flag_dock.setVisible(True)
            self.label_dock.setVisible(True)
            self.shape_dock.setVisible(True)
            self.file_dock.setVisible(True)

            # Try to restore state
            if self.main_window.restoreState(dock_state):
                logger.info("✓ Dock state loaded successfully")
                # Apply a workaround for proper dock resizing
                self.main_window.resizeDocks(
                    [
                        self.tools_dock,
                        self.shape_text_dock,
                        self.flag_dock,
                        self.label_dock,
                        self.shape_dock,
                        self.file_dock,
                    ],
                    [40, 300, 300, 300, 300, 300],
                    Qt.Orientation.Horizontal,
                )
            else:
                logger.warning("✗ Failed to restore dock state - incompatible layout")
                # Reset to default layout
                self.reset_dock_layout()
                return

        except Exception as e:
            logger.warning(f"✗ Error restoring dock state: {e}")
            # If there was an error, delete the invalid state
            if (
                "ui" in config
                and isinstance(config["ui"], dict)
                and "dock_state" in config["ui"]
            ):
                del config["ui"]["dock_state"]
                save_config(config)
                logger.info("Invalid dock state removed from config")

    def on_tools_dock_location_changed(self):
        """Handle tools dock location changes to adjust toolbar orientation."""
        # Get the current dock area of the tools dock
        area = self.main_window.dockWidgetArea(self.tools_dock)

        # If dock is moved to top or bottom areas, use horizontal layout
        if (
            area == Qt.DockWidgetArea.TopDockWidgetArea
            or area == Qt.DockWidgetArea.BottomDockWidgetArea
        ):
            self.tools.setOrientation(Qt.Orientation.Horizontal)
            # Adjust dock height for horizontal layout - including space for title bar
            self.tools_dock.setMinimumHeight(65)  # Increased to accommodate title bar
            self.tools_dock.setMaximumHeight(65)
            # Reset width constraints
            self.tools_dock.setMinimumWidth(0)
            self.tools_dock.setMaximumWidth(16777215)  # Qt's QWIDGETSIZE_MAX
        else:  # Otherwise (left, right, or floating), use vertical layout
            self.tools.setOrientation(Qt.Orientation.Vertical)
            # Adjust dock width for vertical layout
            self.tools_dock.setMinimumWidth(40)
            self.tools_dock.setMaximumWidth(40)
            # Reset height constraints
            self.tools_dock.setMinimumHeight(0)
            self.tools_dock.setMaximumHeight(16777215)  # Qt's QWIDGETSIZE_MAX

            # If floating, provide more reasonable dimensions
            if not area:  # Qt returns 0 for floating docks
                self.tools_dock.setMinimumWidth(0)
                self.tools_dock.setMaximumWidth(16777215)
                # Set a good default size for the floating toolbox
                self.tools_dock.resize(40, 300)
                # Ensure the toolbar is vertical in floating mode
                self.tools.setOrientation(Qt.Orientation.Vertical)

        # Force toolbar to update its layout
        self.tools.update()

        # Save the dock state
        self.save_dock_state()
