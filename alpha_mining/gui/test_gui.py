"""
Test GUI launcher - No authentication required.
Launches the full GUI with mock/None generator for UI testing.
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import ttk

# Setup paths
gui_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(gui_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import GUI components
from alpha_mining.gui.theme import COLORS, FONTS, STYLES
from alpha_mining.gui.components.workflow_panel import WorkflowPanel
from alpha_mining.gui.components.dashboard import DashboardPanel
from alpha_mining.gui.components.config_panel import ConfigPanel
from alpha_mining.gui.components.monitor_panel import MonitorPanel
from alpha_mining.gui.components.database_panel import DatabasePanel

try:
    from alpha_mining.gui.components.evolution_panel import EvolutionPanel
    HAS_EVOLUTION = True
except Exception:
    HAS_EVOLUTION = False

try:
    from alpha_mining.core.config import ConfigManager
except Exception:
    ConfigManager = None


def main():
    logger.info("=" * 60)
    logger.info("🎨 TEST GUI - No Authentication (UI Design Testing)")
    logger.info("=" * 60)

    root = tk.Tk()
    root.title("⚡ GENERATION TWO ⚡ [TEST MODE - No Auth]")
    root.configure(bg=COLORS['bg_primary'])
    root.geometry("1400x900")

    # Style setup
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background=COLORS['bg_primary'], borderwidth=0)
    style.configure('TNotebook.Tab', background=COLORS['bg_panel'], foreground=COLORS['accent_cyan'])
    style.map('TNotebook.Tab', background=[('selected', COLORS['bg_secondary'])])

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Workflow tab (main tab being tested)
    workflow_panel = WorkflowPanel(notebook, generator=None)
    notebook.add(workflow_panel.frame, text="🚀 WORKFLOW")

    # Dashboard tab
    dashboard = DashboardPanel(notebook, update_callback=lambda: {})
    notebook.add(dashboard.frame, text="📊 DASHBOARD")

    # Evolution tab
    if HAS_EVOLUTION:
        evolution_panel = EvolutionPanel(notebook, evolution_callback=lambda o, n: None)
        notebook.add(evolution_panel.frame, text="🧬 EVOLUTION")

    # Config tab
    config_manager = ConfigManager() if ConfigManager else None
    config_panel = ConfigPanel(notebook, config_manager=config_manager, update_callback=lambda s, c: None)
    notebook.add(config_panel.frame, text="⚙️ CONFIG")

    # Monitor tab
    monitor_panel = MonitorPanel(notebook)
    notebook.add(monitor_panel.frame, text="📡 MONITOR")

    # Database tab
    database_panel = DatabasePanel(notebook, db_config_callback=lambda c: None)
    notebook.add(database_panel.frame, text="💾 DATABASE")

    # Add a test mode banner
    banner = tk.Label(
        root,
        text="🎨 TEST MODE — No Authentication — UI Design Testing Only",
        font=("Consolas", 11, "bold"),
        fg="#FF6600",
        bg=COLORS['bg_primary']
    )
    banner.pack(side=tk.TOP, fill=tk.X, pady=2)

    # Setup GUI logging
    class GUILogHandler(logging.Handler):
        def __init__(self, monitor):
            super().__init__()
            self.monitor = monitor

        def emit(self, record):
            msg = self.format(record)
            if self.monitor:
                self.monitor.add_log(record.levelname, msg)

    handler = GUILogHandler(monitor_panel)
    handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)

    logger.info("🎨 Test GUI ready — all panels loaded without authentication")
    logger.info("📋 API buttons will show errors (no session) — this is expected in test mode")

    root.mainloop()


if __name__ == "__main__":
    main()
