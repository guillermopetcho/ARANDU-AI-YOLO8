import os

import darkdetect
from PyQt6.QtGui import QColor, QPalette


class AppTheme:
    """
    Theme manager for the application
    Provides Windows 11 Fluent desktop styling for both light and dark themes
    """

    # Modern color palette
    PRIMARY_LIGHT = "#0067C0"  # Windows Segoe Accent Blue
    PRIMARY_DARK = "#0078D4"
    ACCENT_LIGHT = "#005A9E"
    ACCENT_DARK = "#4CC2FF"

    # Light theme colors (Windows 11 Fluent Light Palette)
    LIGHT = {
        "window": "#f3f3f3",
        "window_text": "#1b1b1b",
        "base": "#ffffff",
        "alternate_base": "#f9f9f9",
        "text": "#1b1b1b",
        "button": "#ffffff",
        "button_text": "#1b1b1b",
        "bright_text": "#000000",
        "highlight": "#0067c0",
        "highlighted_text": "#ffffff",
        "link": "#0067c0",
        "dark": "#d1d1d1",
        "mid": "#a0a0a0",
        "midlight": "#e5e5e5",
        "light": "#ffffff",
        # Custom colors
        "border": "#e5e5e5",
        "toolbar_bg": "#ffffff",
        "dock_title_bg": "#e9ecef",
        "dock_title_text": "#005a9e",
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "panel_bg": "#ffffff",
        "selection": "#0067c0",
    }

    # Dark theme colors (Windows 11 Fluent Dark Palette)
    DARK = {
        "window": "#202020",
        "window_text": "#ffffff",
        "base": "#2c2c2c",
        "alternate_base": "#323232",
        "text": "#ffffff",
        "button": "#2c2c2c",
        "button_text": "#ffffff",
        "bright_text": "#ffffff",
        "highlight": "#0078d4",
        "highlighted_text": "#ffffff",
        "link": "#4cc2ff",
        "dark": "#1a1a1a",
        "mid": "#505050",
        "midlight": "#383838",
        "light": "#383838",
        # Custom colors
        "border": "#383838",
        "toolbar_bg": "#272727",
        "dock_title_bg": "#2d2d2d",
        "dock_title_text": "#4cc2ff",
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "panel_bg": "#2c2c2c",
        "selection": "#0078d4",
    }

    @staticmethod
    def is_dark_mode():
        """Check if system is using dark mode or if it's set via environment variable"""
        if "DARK_MODE" in os.environ:
            return os.environ["DARK_MODE"] == "1"
        return False

    @staticmethod
    def get_color(color_name):
        """Get color based on current theme"""
        is_dark = AppTheme.is_dark_mode()
        colors = AppTheme.DARK if is_dark else AppTheme.LIGHT
        return colors.get(color_name, "#FFFFFF" if not is_dark else "#212121")

    @staticmethod
    def apply_theme(app):
        """Apply theme to entire application"""
        is_dark = AppTheme.is_dark_mode()
        colors = AppTheme.DARK if is_dark else AppTheme.LIGHT

        app.setStyle("Fusion")

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["window_text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["base"]))
        palette.setColor(
            QPalette.ColorRole.AlternateBase, QColor(colors["alternate_base"])
        )
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors["button"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["button_text"]))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(colors["bright_text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["highlight"]))
        palette.setColor(
            QPalette.ColorRole.HighlightedText, QColor(colors["highlighted_text"])
        )
        palette.setColor(QPalette.ColorRole.Link, QColor(colors["link"]))
        palette.setColor(QPalette.ColorRole.Dark, QColor(colors["dark"]))
        palette.setColor(QPalette.ColorRole.Mid, QColor(colors["mid"]))
        palette.setColor(QPalette.ColorRole.Midlight, QColor(colors["midlight"]))
        palette.setColor(QPalette.ColorRole.Light, QColor(colors["light"]))

        app.setPalette(palette)
        app.setStyleSheet(AppTheme.get_stylesheet())

    @staticmethod
    def get_stylesheet():
        """Get stylesheet for current theme"""
        is_dark = AppTheme.is_dark_mode()
        colors = AppTheme.DARK if is_dark else AppTheme.LIGHT

        return f"""
        * {{
            font-family: 'Segoe UI Variable Text', 'Segoe UI', 'Inter', 'SF Pro Text', 'Roboto', sans-serif;
            font-size: 12px;
        }}

        /* Main Window */
        QMainWindow {{
            background-color: {colors["window"]};
            color: {colors["window_text"]};
        }}

        /* Menus and Menu Bar */
        QMenuBar {{
            background-color: {colors["toolbar_bg"]};
            color: {colors["window_text"]};
            border-bottom: 1px solid {colors["border"]};
            padding: 2px 4px;
        }}

        QMenuBar::item {{
            background-color: transparent;
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: 500;
        }}

        QMenuBar::item:selected {{
            background-color: {colors["highlight"]};
            color: {colors["highlighted_text"]};
        }}

        QMenu {{
            background-color: {colors["base"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 4px;
        }}

        QMenu::item {{
            padding: 6px 24px 6px 12px;
            border-radius: 4px;
        }}

        QMenu::item:selected {{
            background-color: {colors["highlight"]};
            color: {colors["highlighted_text"]};
        }}

        QDockWidget {{
            color: {colors["window_text"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background-color: {colors["base"]};
        }}

        QDockWidget::title {{
            text-align: left;
            padding-left: 10px;
            padding-top: 6px;
            padding-bottom: 6px;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            margin: 0px;
            font-weight: 600;
            background-color: {colors["dock_title_bg"]};
            color: {colors["dock_title_text"]};
            border-bottom: 1px solid {colors["border"]};
        }}

        /* Tool Bar */
        QToolBar {{
            background-color: {colors["toolbar_bg"]};
            padding: 4px;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            spacing: 4px;
        }}

        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 5px;
            margin: 1px;
        }}

        QToolButton:hover {{
            background-color: {colors["alternate_base"]};
            border: 1px solid {colors["border"]};
        }}

        QToolButton:pressed {{
            background-color: {colors["midlight"]};
        }}

        QToolButton:checked {{
            background-color: {colors["highlight"]};
            color: {colors["highlighted_text"]};
        }}

        /* Push Buttons */
        QPushButton {{
            background-color: {colors["button"]};
            color: {colors["button_text"]};
            border: 1px solid {colors["border"]};
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 500;
        }}

        QPushButton:hover {{
            background-color: {colors["alternate_base"]};
            border: 1px solid {colors["highlight"]};
        }}

        QPushButton:pressed {{
            background-color: {colors["highlight"]};
            color: #ffffff;
        }}

        /* LineEdits and Inputs */
        QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
            background-color: {colors["base"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 6px;
            padding: 5px 8px;
        }}

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
            border: 2px solid {colors["highlight"]};
            padding: 4px 7px;
        }}

        /* Combo Box */
        QComboBox {{
            background-color: {colors["base"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 6px;
            padding: 5px 10px;
        }}

        QComboBox:hover {{
            border: 1px solid {colors["highlight"]};
        }}

        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: none;
        }}

        /* List Widgets */
        QListWidget {{
            background-color: {colors["base"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 6px;
            padding: 4px;
        }}

        QListWidget::item {{
            padding: 6px 10px;
            border-radius: 4px;
            margin: 1px 0px;
        }}

        QListWidget::item:selected {{
            background-color: {colors["highlight"]};
            color: #ffffff;
            font-weight: 600;
        }}

        QListWidget::item:hover:!selected {{
            background-color: {colors["alternate_base"]};
        }}

        /* Scroll Areas and Scroll Bars */
        QScrollArea {{
            background-color: {colors["window"]};
            border: none;
        }}

        QScrollBar:vertical {{
            background-color: transparent;
            width: 10px;
            margin: 2px;
            border-radius: 5px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {colors["mid"]};
            min-height: 24px;
            border-radius: 4px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {colors["highlight"]};
        }}

        QScrollBar:horizontal {{
            background-color: transparent;
            height: 10px;
            margin: 2px;
            border-radius: 5px;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {colors["mid"]};
            min-width: 24px;
            border-radius: 4px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background-color: {colors["highlight"]};
        }}

        QScrollBar::add-line, QScrollBar::sub-line {{
            width: 0px;
            height: 0px;
        }}

        /* Status Bar */
        QStatusBar {{
            background-color: {colors["toolbar_bg"]};
            color: {colors["window_text"]};
            border-top: 1px solid {colors["border"]};
            padding: 2px;
        }}

        QStatusBar::item {{
            border: none;
        }}

        /* Specific Widget Styling */
        #zoomWidget QToolButton {{
            margin: 0px 1px;
            padding: 2px;
        }}

        /* Auto Labeling Widget */
        #autoLabelingWidget {{
            background-color: {colors["panel_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 4px;
        }}

        #autoLabelingWidget QPushButton {{
            min-height: 26px;
        }}
        """

