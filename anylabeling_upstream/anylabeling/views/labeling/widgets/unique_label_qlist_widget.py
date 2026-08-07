import html

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    # QT Overload
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.position().toPoint()).isValid():
            self.clearSelection()

    def find_items_by_label(self, label):
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == label:
                items.append(item)
        return items

    def create_item_from_label(self, label):
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, label)
        return item

    def set_item_label(self, item, label, color=None, count=0, active_in_image=False):
        qlabel = QtWidgets.QLabel()
        if color is None:
            color_str = "#0284c7"
        else:
            color_str = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        if count > 0 or active_in_image:
            text = f'<font color="{color_str}">●</font> <b><font color="#0f172a">{html.escape(label)}</font></b> <font color="#059669"><b>({count})</b></font>'
            qlabel.setStyleSheet("padding: 4px 8px; background-color: #e2e8f0; border-radius: 6px; border: 1px solid #cbd5e1;")
        else:
            text = f'<font color="{color_str}">●</font> <font color="#334155">{html.escape(label)}</font> <font color="#64748b">(0)</font>'
            qlabel.setStyleSheet("padding: 4px 8px; background-color: transparent;")

        qlabel.setText(text)
        qlabel.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        size_hint = qlabel.sizeHint()
        size_hint.setHeight(max(size_hint.height() + 10, 36))
        item.setSizeHint(size_hint)
        self.setItemWidget(item, qlabel)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            return

        label = item.data(Qt.ItemDataRole.UserRole)
        if not label:
            return

        menu = QtWidgets.QMenu(self)
        action_inspect = menu.addAction(f"🔍 Inspeccionar Objetos de '{label}'...")
        action_select_all = menu.addAction(f"🎯 Seleccionar Objetos de '{label}'")
        action_replace = menu.addAction(f"🔁 Reemplazar Clase '{label}'...")

        action = menu.exec(self.mapToGlobal(event.pos()))
        parent_widget = self.parent()
        while parent_widget and not hasattr(parent_widget, "select_all_by_class"):
            parent_widget = parent_widget.parent()

        if action == action_inspect and parent_widget:
            parent_widget.open_label_inspector(label)
        elif action == action_select_all and parent_widget:
            parent_widget.select_all_by_class(label)
        elif action == action_replace and parent_widget:
            parent_widget.open_global_replace_dialog()
