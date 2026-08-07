"""Defines toolbar for anylabeling, including"""

from PyQt6 import QtCore, QtWidgets

from anylabeling.styles import AppTheme


class ToolBar(QtWidgets.QToolBar):
    """Toolbar widget for labeling tool"""

    def __init__(self, title):
        super().__init__(title)
        layout = self.layout()
        margin = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*margin)
        self.setContentsMargins(*margin)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint
        )

        # Use theme system for styling
        self.setStyleSheet(
            f"""
            QToolBar {{
                background: {AppTheme.get_color("toolbar_bg")};
                padding: 4px;
                border-radius: 8px;
                border: 1px solid {AppTheme.get_color("border")};
                spacing: 4px;
            }}
            """
        )

    def add_action(self, action):
        """Add an action (button) to the toolbar"""
        if isinstance(action, QtWidgets.QWidgetAction):
            return super().addAction(action)
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())

        # Give Brain / Auto Labeling button a distinct, elegant Windows accent highlight
        if action and action.iconText() and ("Auto" in action.iconText() or "brain" in action.iconText().lower() or "auto" in action.iconText().lower()):
            btn.setToolTip(self.tr("Auto-Labeling (IA Segmentación - Cerebro)"))
            is_dark = AppTheme.is_dark_mode()
            border_col = "#4CC2FF" if is_dark else "#0067C0"
            bg_col = "rgba(0, 120, 212, 0.15)" if is_dark else "rgba(0, 103, 192, 0.08)"
            btn.setStyleSheet(f"""
                QToolButton {{
                    background-color: {bg_col};
                    border: 1.5px solid {border_col};
                    border-radius: 6px;
                    padding: 4px;
                    margin: 2px;
                }}
                QToolButton:hover {{
                    background-color: {border_col};
                    color: white;
                }}
            """)

        self.addWidget(btn)

        # Center alignment
        for i in range(self.layout().count()):
            if isinstance(self.layout().itemAt(i).widget(), QtWidgets.QToolButton):
                self.layout().itemAt(i).setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter
                )

        return True
