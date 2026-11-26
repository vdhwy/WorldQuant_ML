"""
Step 1: Data Fields Setup
Handles operator loading and data field fetching/loading
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
import os
import threading
import logging

import pandas as pd
import alpha_mining.lib as ace

from ...theme import COLORS, FONTS, STYLES

logger = logging.getLogger(__name__)


class Step1DataFields:
    """Step 1: Load Operators & Data Fields"""
    
    def __init__(self, parent_frame, workflow_panel):
        """
        Initialize Step 1
        
        Args:
            parent_frame: Parent frame to pack into
            workflow_panel: Reference to main WorkflowPanel for callbacks
        """
        self.parent_frame = parent_frame
        self.workflow = workflow_panel
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable container
        self._scroll_canvas = tk.Canvas(self.frame, bg=COLORS['bg_panel'], highlightthickness=0)
        self._scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self._scroll_canvas.yview)
        self.content_frame = tk.Frame(self._scroll_canvas, bg=COLORS['bg_panel'])
        
        self.content_frame.bind(
            '<Configure>',
            lambda e: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox('all'))
        )
        
        self._canvas_window = self._scroll_canvas.create_window((0, 0), window=self.content_frame, anchor='nw')
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)
        
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Make the inner frame expand to canvas width
        self._scroll_canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Enable mousewheel scrolling
        self.content_frame.bind('<Enter>', self._bind_mousewheel)
        self.content_frame.bind('<Leave>', self._unbind_mousewheel)
        
        self._create_widgets()
    
    def _on_canvas_configure(self, event):
        """Resize inner frame to match canvas width"""
        self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)
    
    def _bind_mousewheel(self, event):
        """Bind mousewheel scrolling when cursor enters the frame"""
        self._scroll_canvas.bind_all('<MouseWheel>', self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel scrolling when cursor leaves the frame"""
        self._scroll_canvas.unbind_all('<MouseWheel>')
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
    
    def _create_widgets(self):
        """Create Step 1 widgets"""
        tk.Label(
            self.content_frame,
            text="Step 1: Load Operators & Data Fields",
            font=FONTS['heading'],
            fg=COLORS['accent_cyan'],
            bg=COLORS['bg_panel']
        ).pack(pady=10)
        
        tk.Label(
            self.content_frame,
            text="First, fetch operators from the API or load from operatorRAW.json file.",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(pady=5)
        
        # Operator file input
        operator_frame = tk.Frame(self.content_frame, bg=COLORS['bg_secondary'], relief=tk.RAISED, bd=2)
        operator_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            operator_frame,
            text="⚙️ Operators (Required):",
            font=FONTS['default'],
            fg=COLORS['accent_yellow'],
            bg=COLORS['bg_secondary']
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Fetch from API button row
        api_fetch_frame = tk.Frame(operator_frame, bg=COLORS['bg_secondary'])
        api_fetch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fetch_operators_button = tk.Button(
            api_fetch_frame,
            text="🌐 FETCH FROM API",
            command=self._fetch_operators_from_api,
            **STYLES['button']
        )
        self.fetch_operators_button.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            api_fetch_frame,
            text="Fetch all operators directly from the WorldQuant Brain API",
            font=FONTS['default'],
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_secondary']
        ).pack(side=tk.LEFT, padx=5)
        
        # Separator
        tk.Label(
            operator_frame,
            text="— OR load from local file —",
            font=FONTS['default'],
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_secondary']
        ).pack(pady=3)
        
        operator_path_frame = tk.Frame(operator_frame, bg=COLORS['bg_secondary'])
        operator_path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.operator_path_var = tk.StringVar(value="")
        tk.Label(
            operator_path_frame,
            text="operatorRAW.json:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_secondary']
        ).pack(side=tk.LEFT, padx=5)
        
        operator_entry = tk.Entry(
            operator_path_frame,
            textvariable=self.operator_path_var,
            width=50,
            **STYLES['entry']
        )
        operator_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        tk.Button(
            operator_path_frame,
            text="Browse...",
            command=self._browse_operator_file,
            **STYLES['button']
        ).pack(side=tk.LEFT, padx=5)
        
        self.load_operator_button = tk.Button(
            operator_path_frame,
            text="📂 LOAD OPERATORS",
            command=self._load_operator_file,
            **STYLES['button']
        )
        self.load_operator_button.pack(side=tk.LEFT, padx=5)
        
        self.operator_status = tk.Label(
            operator_frame,
            text="⚠️ Operators not loaded",
            font=FONTS['default'],
            fg=COLORS['error'],
            bg=COLORS['bg_secondary']
        )
        self.operator_status.pack(pady=5)
        
        tk.Label(
            self.content_frame,
            text="Then, fetch or load data fields for your regions:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(pady=5)
        
        # Storage configuration
        storage_frame = tk.Frame(self.content_frame, bg=COLORS['bg_secondary'], relief=tk.RAISED, bd=2)
        storage_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            storage_frame,
            text="💾 Storage Configuration:",
            font=FONTS['default'],
            fg=COLORS['accent_yellow'],
            bg=COLORS['bg_secondary']
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        self.storage_type = tk.StringVar(value="json")
        
        storage_options_frame = tk.Frame(storage_frame, bg=COLORS['bg_secondary'])
        storage_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for text, value in [
            ("JSON Files (Default - cached per region)", "json"),
            ("Local SQLite Database", "sqlite"),
            ("Remote Database URL", "remote")
        ]:
            tk.Radiobutton(
                storage_options_frame,
                text=text,
                variable=self.storage_type,
                value=value,
                font=FONTS['default'],
                fg=COLORS['text_primary'],
                bg=COLORS['bg_secondary'],
                selectcolor=COLORS['bg_panel'],
                activebackground=COLORS['bg_secondary'],
                activeforeground=COLORS['accent_cyan']
            ).pack(anchor=tk.W, padx=5)
        
        # Database path/URL input
        self.storage_config_frame = tk.Frame(storage_frame, bg=COLORS['bg_secondary'])
        self.storage_config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.storage_path_var = tk.StringVar(value="")
        self.storage_url_var = tk.StringVar(value="")
        
        self.storage_path_label = tk.Label(
            self.storage_config_frame,
            text="Database Path:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_secondary']
        )
        
        self.storage_path_entry = tk.Entry(
            self.storage_config_frame,
            textvariable=self.storage_path_var,
            width=40,
            **STYLES['entry']
        )
        
        self.storage_browse_button = tk.Button(
            self.storage_config_frame,
            text="Browse...",
            command=self._browse_storage_path,
            **STYLES['button']
        )
        
        self.storage_url_label = tk.Label(
            self.storage_config_frame,
            text="Database URL:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_secondary']
        )
        
        self.storage_url_entry = tk.Entry(
            self.storage_config_frame,
            textvariable=self.storage_url_var,
            width=50,
            **STYLES['entry']
        )
        
        self.storage_type.trace('w', lambda *args: self._update_storage_config_visibility())
        self._update_storage_config_visibility()
        
        # Region selection
        region_frame = tk.Frame(self.content_frame, bg=COLORS['bg_panel'])
        region_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            region_frame,
            text="Regions:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(side=tk.LEFT, padx=5)
        
        self.region_vars = {}
        regions = ['USA', 'EUR', 'CHN', 'ASI', 'GLB', 'IND']
        for region in regions:
            var = tk.BooleanVar(value=True)
            self.region_vars[region] = var
            tk.Checkbutton(
                region_frame,
                text=region,
                variable=var,
                font=FONTS['default'],
                fg=COLORS['text_primary'],
                bg=COLORS['bg_panel'],
                selectcolor=COLORS['bg_secondary'],
                activebackground=COLORS['bg_panel'],
                activeforeground=COLORS['accent_cyan']
            ).pack(side=tk.LEFT, padx=5)

        # Dataset filters and fetch controls
        dataset_frame = tk.Frame(self.content_frame, bg=COLORS['bg_secondary'], relief=tk.RAISED, bd=2)
        dataset_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            dataset_frame,
            text="📦 Fetch Datasets with Filters:",
            font=FONTS['default'],
            fg=COLORS['accent_yellow'],
            bg=COLORS['bg_secondary']
        ).pack(anchor=tk.W, padx=5, pady=5)

        filter_frame = tk.Frame(dataset_frame, bg=COLORS['bg_secondary'])
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        self.dataset_category_var = tk.StringVar(value="")
        self.dataset_coverage_min_var = tk.StringVar(value="0.0")
        self.dataset_coverage_max_var = tk.StringVar(value="1.0")
        self.dataset_field_count_min_var = tk.StringVar(value="0")
        self.dataset_field_count_max_var = tk.StringVar(value="999")
        self.dataset_user_count_min_var = tk.StringVar(value="0")
        self.dataset_user_count_max_var = tk.StringVar(value="999")
        self.dataset_value_score_min_var = tk.StringVar(value="0.0")
        self.dataset_value_score_max_var = tk.StringVar(value="10.0")

        tk.Label(filter_frame, text="Category:", font=FONTS['default'], fg=COLORS['text_primary'], bg=COLORS['bg_secondary']).grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        tk.Entry(filter_frame, textvariable=self.dataset_category_var, width=20, **STYLES['entry']).grid(row=0, column=1, padx=5, pady=3)

        tk.Label(filter_frame, text="Coverage min/max:", font=FONTS['default'], fg=COLORS['text_primary'], bg=COLORS['bg_secondary']).grid(row=0, column=2, padx=5, pady=3, sticky=tk.W)
        tk.Entry(filter_frame, textvariable=self.dataset_coverage_min_var, width=8, **STYLES['entry']).grid(row=0, column=3, padx=5, pady=3)
        tk.Entry(filter_frame, textvariable=self.dataset_coverage_max_var, width=8, **STYLES['entry']).grid(row=0, column=4, padx=5, pady=3)

        tk.Label(filter_frame, text="Fields min/max:", font=FONTS['default'], fg=COLORS['text_primary'], bg=COLORS['bg_secondary']).grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        tk.Entry(filter_frame, textvariable=self.dataset_field_count_min_var, width=8, **STYLES['entry']).grid(row=1, column=1, padx=5, pady=3)
        tk.Entry(filter_frame, textvariable=self.dataset_field_count_max_var, width=8, **STYLES['entry']).grid(row=1, column=2, padx=5, pady=3)

        tk.Label(filter_frame, text="Users min/max:", font=FONTS['default'], fg=COLORS['text_primary'], bg=COLORS['bg_secondary']).grid(row=1, column=3, padx=5, pady=3, sticky=tk.W)
        tk.Entry(filter_frame, textvariable=self.dataset_user_count_min_var, width=8, **STYLES['entry']).grid(row=1, column=4, padx=5, pady=3)
        tk.Entry(filter_frame, textvariable=self.dataset_user_count_max_var, width=8, **STYLES['entry']).grid(row=1, column=5, padx=5, pady=3)

        tk.Label(filter_frame, text="Value score min/max:", font=FONTS['default'], fg=COLORS['text_primary'], bg=COLORS['bg_secondary']).grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        tk.Entry(filter_frame, textvariable=self.dataset_value_score_min_var, width=8, **STYLES['entry']).grid(row=2, column=1, padx=5, pady=3)
        tk.Entry(filter_frame, textvariable=self.dataset_value_score_max_var, width=8, **STYLES['entry']).grid(row=2, column=2, padx=5, pady=3)

        self.fetch_datasets_button = tk.Button(
            dataset_frame,
            text="📥 FETCH DATASETS",
            command=self._fetch_datasets,
            **STYLES['button']
        )
        self.fetch_datasets_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.dataset_status = tk.Label(
            dataset_frame,
            text="",
            font=FONTS['default'],
            fg=COLORS['accent_green'],
            bg=COLORS['bg_secondary']
        )
        self.dataset_status.pack(pady=5)

        self.dataset_output = tk.Text(self.content_frame, height=5, wrap=tk.WORD, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'], state=tk.DISABLED, bd=2, relief=tk.SUNKEN)
        self.dataset_output.pack(fill=tk.BOTH, padx=10, pady=(0,10))

        # Action buttons
        action_frame = tk.Frame(self.content_frame, bg=COLORS['bg_panel'])
        action_frame.pack(pady=10)
        
        self.load_button = tk.Button(
            action_frame,
            text="📂 LOAD DATA FIELDS",
            command=self._load_data_fields,
            **STYLES['button']
        )
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.fetch_datafields_api_button = tk.Button(
            action_frame,
            text="🌐 FETCH DATAFIELDS (API)",
            command=self._fetch_datafields_from_api,
            **STYLES['button']
        )
        self.fetch_datafields_api_button.pack(side=tk.LEFT, padx=5)
        
        self.fetch_status = tk.Label(
            self.content_frame,
            text="",
            font=FONTS['default'],
            fg=COLORS['accent_green'],
            bg=COLORS['bg_panel']
        )
        self.fetch_status.pack(pady=5)
        
        # "Proceed as last time" button
        proceed_frame = tk.Frame(self.content_frame, bg=COLORS['bg_panel'])
        proceed_frame.pack(pady=10)
        
        self.proceed_last_button = tk.Button(
            proceed_frame,
            text="⚡ PROCEED AS LAST TIME",
            command=self._proceed_as_last_time,
            font=FONTS['default'],
            fg=COLORS['accent_cyan'],
            bg=COLORS['bg_secondary'],
            relief=tk.RAISED,
            bd=2,
            padx=20,
            pady=5
        )
        self.proceed_last_button.pack(pady=5)
        
        # Check if last config exists
        config_file = Path.home() / ".generation_two" / "workflow_config.json"
        if config_file.exists():
            self.proceed_last_button.config(state=tk.NORMAL)
        else:
            self.proceed_last_button.config(state=tk.DISABLED)
    
    def _update_storage_config_visibility(self):
        """Update visibility of storage configuration inputs"""
        for widget in self.storage_config_frame.winfo_children():
            widget.pack_forget()
        
        storage_type = self.storage_type.get()
        
        if storage_type == "sqlite":
            self.storage_path_label.pack(side=tk.LEFT, padx=5)
            self.storage_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.storage_browse_button.pack(side=tk.LEFT, padx=5)
            if not self.storage_path_var.get():
                self.storage_path_var.set("generation_two_backtests.db")
        elif storage_type == "remote":
            self.storage_url_label.pack(side=tk.LEFT, padx=5)
            self.storage_url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def _browse_operator_file(self):
        """Browse for operatorRAW.json file"""
        filename = filedialog.askopenfilename(
            title="Select operatorRAW.json File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filename:
            self.operator_path_var.set(filename)
    
    def _fetch_operators_from_api(self):
        """Fetch all operators directly from the WorldQuant Brain API using get_operators"""
        if not self.workflow.generator or not self.workflow.generator.template_generator:
            messagebox.showerror("Error", "Generator not initialized. Please log in first.")
            return
        
        session = getattr(self.workflow.generator.template_generator, 'sess', None)
        if not session:
            messagebox.showerror("Error", "No authenticated session available. Please log in first.")
            return
        
        # Disable button and show progress
        self.fetch_operators_button.config(state=tk.DISABLED, text="⏳ Fetching...")
        self.operator_status.config(text="🔄 Fetching operators from API...", fg=COLORS['accent_yellow'])
        
        def fetch_thread():
            try:
                operators_df = ace.get_operators(session)
                if 'id' in operators_df.columns:
                    operators_df = operators_df.drop_duplicates(subset=['id'])
                elif 'name' in operators_df.columns:
                    operators_df = operators_df.drop_duplicates(subset=['name'])
                operators = operators_df.to_dict('records')
                
                if not operators:
                    self.frame.after(0, lambda: self._on_fetch_operators_error("No operators returned from API"))
                    return
                
                self.frame.after(0, lambda: self._on_fetch_operators_success(operators))
                
            except Exception as e:
                logger.error(f"Error fetching operators from API: {e}", exc_info=True)
                self.frame.after(0, lambda: self._on_fetch_operators_error(str(e)))
        
        threading.Thread(target=fetch_thread, daemon=True).start()
    
    def _on_fetch_operators_success(self, operators):
        """Handle successful operator fetch from API"""
        try:
            self._apply_operators(operators, source="API")
            self.fetch_operators_button.config(state=tk.NORMAL, text="🌐 FETCH FROM API")
        except Exception as e:
            self._on_fetch_operators_error(str(e))
    
    def _on_fetch_operators_error(self, error_msg):
        """Handle error during operator fetch from API"""
        self.fetch_operators_button.config(state=tk.NORMAL, text="🌐 FETCH FROM API")
        self.operator_status.config(
            text=f"❌ Failed to fetch operators: {error_msg}",
            fg=COLORS['error']
        )
        messagebox.showerror("Error", f"Failed to fetch operators from API:\n{error_msg}")
    
    def _apply_operators(self, operators, source="file"):
        """Apply loaded operators to the workflow (shared by file load and API fetch)"""
        # Update operator fetcher
        if self.workflow.generator and self.workflow.generator.template_generator:
            if self.workflow.generator.template_generator.operator_fetcher:
                self.workflow.generator.template_generator.operator_fetcher.operators = operators
                if self.workflow.generator.template_generator.search_engine:
                    self.workflow.generator.template_generator.search_engine.operators = operators
                    self.workflow.generator.template_generator.search_engine._build_indices()
                else:
                    from alpha_mining.data_fetcher import SmartSearchEngine
                    self.workflow.generator.template_generator.search_engine = SmartSearchEngine(
                        operators=operators,
                        data_fields={}
                    )
        
        self.workflow.operators_loaded = True
        self.operator_status.config(
            text=f"✅ Loaded {len(operators)} operators from {source}",
            fg=COLORS['accent_green']
        )
        
        self.workflow.operators = operators
        self.workflow.steps_completed.add(1)
        self.workflow._update_step_indicators()
        self.workflow._save_config()
        
        # Update Step 2 category combo with actual categories from the data
        self._update_category_combo(operators)
        
        logger.info(f"Loaded {len(operators)} operators from {source}")
    
    def _update_category_combo(self, operators):
        """Update the Step 2 category combo with actual categories from operators"""
        try:
            categories = sorted(set(
                op.get('category', '') for op in operators if op.get('category')
            ))
            all_categories = ["All"] + categories
            
            if hasattr(self.workflow, 'step2') and hasattr(self.workflow.step2, 'category_var'):
                # Find the category combobox widget
                for widget in self.workflow.step2.frame.winfo_children():
                    self._find_and_update_combo(widget, all_categories)
            elif hasattr(self.workflow, 'category_var'):
                # Legacy: directly on workflow
                for widget in self.workflow.frame.winfo_children():
                    self._find_and_update_combo(widget, all_categories)
        except Exception as e:
            logger.debug(f"Could not update category combo: {e}")
    
    def _find_and_update_combo(self, widget, categories):
        """Recursively find and update ttk.Combobox with category values"""
        if isinstance(widget, ttk.Combobox):
            widget.config(values=categories)
            return
        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                self._find_and_update_combo(child, categories)
    
    def _load_operator_file(self):
        """Load operators from user-specified file"""
        operator_path = self.operator_path_var.get()
        if not operator_path:
            messagebox.showwarning("Warning", "Please select operatorRAW.json file")
            return
        
        if not os.path.exists(operator_path):
            messagebox.showerror("Error", f"File not found: {operator_path}")
            return
        
        try:
            with open(operator_path, 'r', encoding='utf-8') as f:
                operators = json.load(f)
            
            if not operators or not isinstance(operators, list):
                messagebox.showerror("Error", "Invalid operator file format")
                return
            
            # Update cache_file path in operator fetcher
            if self.workflow.generator and self.workflow.generator.template_generator:
                if self.workflow.generator.template_generator.operator_fetcher:
                    self.workflow.generator.template_generator.operator_fetcher.cache_file = Path(operator_path)
            
            self._apply_operators(operators, source=operator_path)
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON file: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load operators: {e}")
            logger.error(f"Error loading operators: {e}", exc_info=True)
    
    def _browse_storage_path(self):
        """Browse for SQLite database file"""
        filename = filedialog.asksaveasfilename(
            title="Select SQLite Database",
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        if filename:
            self.storage_path_var.set(filename)

    def _resolve_default_universe(self, region: str) -> str:
        """Resolve the default universe for a region if the UI does not specify one."""
        try:
            from alpha_mining.core.region_config import get_default_universe
            return get_default_universe(region)
        except Exception:
            fallback = {
                'USA': 'TOP3000',
                'EUR': 'TOP2500',
                'CHN': 'TOP2000U',
                'ASI': 'MINVOL1M',
                'GLB': 'TOP3000',
                'IND': 'TOP500',
            }
            return fallback.get(region, 'TOP3000')

    def _apply_dataset_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply user-provided dataset filters to the fetched dataset DataFrame."""
        if df.empty:
            return df

        category = self.dataset_category_var.get().strip()
        if category:
            if 'category' in df.columns:
                df = df[df['category'].astype(str).str.contains(category, case=False, na=False)]
            elif 'category_id' in df.columns:
                df = df[df['category_id'].astype(str).str.contains(category, case=False, na=False)]

        def parse_float(var, default):
            try:
                return float(var.get())
            except Exception:
                return default

        def parse_int(var, default):
            try:
                return int(var.get())
            except Exception:
                return default

        coverage_min = parse_float(self.dataset_coverage_min_var, 0.0)
        coverage_max = parse_float(self.dataset_coverage_max_var, 1.0)
        field_min = parse_int(self.dataset_field_count_min_var, 0)
        field_max = parse_int(self.dataset_field_count_max_var, 999999)
        user_min = parse_int(self.dataset_user_count_min_var, 0)
        user_max = parse_int(self.dataset_user_count_max_var, 999999)
        value_min = parse_float(self.dataset_value_score_min_var, 0.0)
        value_max = parse_float(self.dataset_value_score_max_var, 10.0)

        if 'coverage' in df.columns:
            df = df[df['coverage'].between(coverage_min, coverage_max)]
        if 'fieldCount' in df.columns:
            df = df[df['fieldCount'].between(field_min, field_max)]
        if 'userCount' in df.columns:
            df = df[df['userCount'].between(user_min, user_max)]
        if 'valueScore' in df.columns:
            df = df[df['valueScore'].between(value_min, value_max)]

        return df

    def _fetch_datasets(self):
        """Fetch datasets from WorldQuant Brain using get_datasets and show results."""
        if not self.workflow.generator or not self.workflow.generator.template_generator:
            messagebox.showerror("Error", "Generator not initialized")
            return

        selected_regions = [r for r, var in self.region_vars.items() if var.get()]
        if not selected_regions:
            messagebox.showwarning("Warning", "Please select at least one region")
            return

        self.fetch_datasets_button.config(state=tk.DISABLED)
        self.dataset_status.config(text="Fetching datasets...", fg=COLORS['accent_yellow'])
        self.dataset_output.config(state=tk.NORMAL)
        self.dataset_output.delete('1.0', tk.END)
        self.dataset_output.config(state=tk.DISABLED)

        def fetch_thread():
            all_datasets = []
            session = self.workflow.generator.template_generator.sess

            for region in selected_regions:
                try:
                    universe = self._resolve_default_universe(region)
                    datasets = ace.get_datasets(
                        session,
                        instrument_type='EQUITY',
                        region=region,
                        delay=1,
                        universe=universe,
                        theme=False,
                    )
                    if datasets.empty:
                        continue
                    filtered = self._apply_dataset_filters(datasets)
                    filtered['region'] = region
                    all_datasets.append(filtered)
                except Exception as e:
                    logger.error(f"Failed to fetch datasets for {region}: {e}", exc_info=True)
                    self.workflow.frame.after(0, lambda msg=f"Failed to fetch datasets for {region}: {e}": self.dataset_status.config(text=msg, fg=COLORS['error']))

            if all_datasets:
                combined = pd.concat(all_datasets, ignore_index=True).drop_duplicates(subset=['id'])
                
                constants_dir = Path("constants")
                constants_dir.mkdir(exist_ok=True)
                cache_file = constants_dir / "datasets_cache.json"
                if cache_file.exists():
                    try:
                        existing = pd.read_json(cache_file)
                        combined = pd.concat([existing, combined], ignore_index=True).drop_duplicates(subset=['id'])
                    except Exception as e:
                        logger.error(f"Error loading existing dataset cache: {e}")
                
                combined.to_json(cache_file, orient="records", indent=4)
                
                self.fetched_dataset_ids = combined['id'].tolist()
                self.workflow.dataset_ids = self.fetched_dataset_ids
                top_ids = self.fetched_dataset_ids[:50]
                display_text = f"Found {len(self.fetched_dataset_ids)} datasets\n" + "\n".join(top_ids)
                self.workflow.frame.after(0, lambda: self._show_dataset_results(display_text, success=True))
            else:
                self.fetched_dataset_ids = []
                self.workflow.dataset_ids = []
                self.workflow.frame.after(0, lambda: self._show_dataset_results("No datasets found with the selected filters.", success=False))

            self.workflow.frame.after(0, lambda: self.fetch_datasets_button.config(state=tk.NORMAL))

        threading.Thread(target=fetch_thread, daemon=True).start()

    def _show_dataset_results(self, text: str, success: bool = True):
        self.dataset_output.config(state=tk.NORMAL)
        self.dataset_output.delete('1.0', tk.END)
        self.dataset_output.insert(tk.END, text)
        self.dataset_output.config(state=tk.DISABLED)
        self.dataset_status.config(
            text="Datasets fetched successfully" if success else "No datasets matched your filter",
            fg=COLORS['accent_green'] if success else COLORS['error']
        )
    
    def _fetch_datafields_from_api(self):
        """Fetch data fields directly from the WorldQuant Brain API using ace.get_datafields()"""
        if not self.workflow.generator or not self.workflow.generator.template_generator:
            messagebox.showerror("Error", "Generator not initialized. Please log in first.")
            return
        
        session = getattr(self.workflow.generator.template_generator, 'sess', None)
        if not session:
            messagebox.showerror("Error", "No authenticated session available. Please log in first.")
            return
        
        selected_regions = [r for r, var in self.region_vars.items() if var.get()]
        if not selected_regions:
            messagebox.showwarning("Warning", "Please select at least one region")
            return
        
        # Disable buttons while fetching
        self.fetch_datafields_api_button.config(state=tk.DISABLED, text="⏳ Fetching...")
        self.load_button.config(state=tk.DISABLED)
        self.fetch_status.config(text="🔄 Fetching datafields from API...", fg=COLORS['accent_yellow'])
        
        def fetch_thread():
            try:
                total_fields = 0
                successful_regions = []
                failed_regions = []
                
                for i, region in enumerate(selected_regions, 1):
                    try:
                        self.workflow.frame.after(0, lambda r=region, idx=i, total=len(selected_regions):
                            self.fetch_status.config(
                                text=f"🔄 Fetching datafields for {r}... ({idx}/{total})",
                                fg=COLORS['accent_yellow']
                            ))
                        
                        fetcher = self.workflow.generator.template_generator.data_field_fetcher
                        if not fetcher:
                            failed_regions.append((region, "Data field fetcher not initialized"))
                            continue
                        
                        fields = fetcher.fetch_data_fields(
                            region=region,
                            force_refresh=True,
                            try_all_universes=True
                        )
                        
                        if not fields:
                            failed_regions.append((region, "No datafields returned"))
                            continue
                            
                        datafields_df = pd.DataFrame(fields)
                        
                        constants_dir = Path("constants")
                        constants_dir.mkdir(exist_ok=True)
                        cache_file = constants_dir / f"datafields_cache_{region}.json"
                        if cache_file.exists():
                            try:
                                existing = pd.read_json(cache_file)
                                datafields_df = pd.concat([existing, datafields_df], ignore_index=True).drop_duplicates(subset=['id'])
                            except Exception as e:
                                logger.error(f"Error loading existing datafields cache for {region}: {e}")
                        
                        datafields_df.to_json(cache_file, orient="records", indent=4)
                        
                        fields = datafields_df.to_dict('records')
                        field_count = len(fields)
                        total_fields += field_count
                        successful_regions.append((region, field_count))
                        
                        # Store in the generator's data_field_fetcher
                        if (self.workflow.generator and 
                            self.workflow.generator.template_generator and
                            self.workflow.generator.template_generator.data_field_fetcher):
                            self.workflow.generator.template_generator.data_field_fetcher.data_fields[region] = fields
                        
                        # Also update search engine if available
                        if (self.workflow.generator and 
                            self.workflow.generator.template_generator and
                            self.workflow.generator.template_generator.search_engine):
                            if not hasattr(self.workflow.generator.template_generator.search_engine, 'data_fields') or \
                               not isinstance(self.workflow.generator.template_generator.search_engine.data_fields, dict):
                                self.workflow.generator.template_generator.search_engine.data_fields = {}
                            self.workflow.generator.template_generator.search_engine.data_fields[region] = fields
                        
                        self.workflow.frame.after(0, lambda r=region, c=field_count:
                            self.fetch_status.config(
                                text=f"✓ Fetched {c} datafields for {r}",
                                fg=COLORS['accent_green']
                            ))
                        
                    except Exception as e:
                        failed_regions.append((region, str(e)))
                        logger.error(f"Error fetching datafields for {region}: {e}", exc_info=True)
                        self.workflow.frame.after(0, lambda r=region, err=str(e):
                            self.fetch_status.config(
                                text=f"✗ Error for {r}: {err[:50]}",
                                fg=COLORS['error']
                            ))
                
                # Final summary
                if successful_regions:
                    summary = "Successfully fetched datafields from API:\n"
                    for region, count in successful_regions:
                        summary += f"  {region}: {count} datafields\n"
                    summary += f"\nTotal: {total_fields} datafields"
                    
                    if failed_regions:
                        summary += f"\n\nFailed regions:\n"
                        for region, error in failed_regions:
                            summary += f"  {region}: {error}\n"
                    
                    self.workflow.frame.after(0, lambda: self.fetch_status.config(
                        text=f"✅ Complete: {total_fields} datafields from {len(successful_regions)} region(s)",
                        fg=COLORS['accent_green']
                    ))
                    self.workflow.frame.after(0, lambda: messagebox.showinfo("Success", summary))
                    self.workflow.frame.after(0, lambda: self.workflow.steps_completed.add(0))
                    self.workflow.frame.after(0, self.workflow._update_step_indicators)
                    self.workflow.frame.after(0, self.workflow._save_config)
                else:
                    error_summary = "Failed to fetch datafields for all regions:\n"
                    for region, error in failed_regions:
                        error_summary += f"  {region}: {error}\n"
                    
                    self.workflow.frame.after(0, lambda: self.fetch_status.config(
                        text="❌ All fetches failed",
                        fg=COLORS['error']
                    ))
                    self.workflow.frame.after(0, lambda: messagebox.showerror("Error", error_summary))
                
            except Exception as e:
                error_msg = f"Unexpected error fetching datafields: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.workflow.frame.after(0, lambda: self.fetch_status.config(
                    text=f"❌ Error: {str(e)[:50]}",
                    fg=COLORS['error']
                ))
                self.workflow.frame.after(0, lambda: messagebox.showerror("Error", error_msg))
            finally:
                self.workflow.frame.after(0, lambda: self.fetch_datafields_api_button.config(
                    state=tk.NORMAL, text="🌐 FETCH DATAFIELDS (API)"
                ))
                self.workflow.frame.after(0, lambda: self.load_button.config(state=tk.NORMAL))
        
        threading.Thread(target=fetch_thread, daemon=True).start()
    
    def _load_data_fields(self):
        """Load data fields from cache/storage"""
        if not self.workflow.generator or not self.workflow.generator.template_generator:
            messagebox.showerror("Error", "Generator not initialized")
            return
        
        selected_regions = [r for r, var in self.region_vars.items() if var.get()]
        if not selected_regions:
            messagebox.showwarning("Warning", "Please select at least one region")
            return
        
        self.load_button.config(state=tk.DISABLED)
        self.fetch_status.config(text="Loading data fields from cache...", fg=COLORS['accent_yellow'])
        
        def load_thread():
            try:
                successful_regions = []
                failed_regions = []
                
                for i, region in enumerate(selected_regions, 1):
                    try:
                        self.workflow.frame.after(0, lambda r=region, idx=i, total=len(selected_regions): 
                            self.fetch_status.config(
                                text=f"Loading {r}... ({idx}/{total})",
                                fg=COLORS['accent_yellow']
                            ))
                        
                        # Check cache or database
                        cache_file = None
                        if self.workflow.generator.template_generator.data_field_fetcher:
                            cache_dir = self.workflow.generator.template_generator.data_field_fetcher.cache_dir
                            import glob
                            for pattern in [f"data_fields_cache_{region}_1.json", f"data_fields_cache_{region}_1_*.json"]:
                                matches = glob.glob(str(cache_dir / pattern))
                                if matches:
                                    cache_file = matches[0]
                                    break
                        
                        if cache_file and os.path.exists(cache_file):
                            try:
                                with open(cache_file, 'r', encoding='utf-8') as f:
                                    sample = f.read(5000)
                                    if sample.strip().startswith('['):
                                        file_size = os.path.getsize(cache_file)
                                        estimated_count = max(1, file_size // 200)
                                        if self.workflow.generator.template_generator.data_field_fetcher:
                                            if region not in self.workflow.generator.template_generator.data_field_fetcher.data_fields:
                                                self.workflow.generator.template_generator.data_field_fetcher.data_fields[region] = []
                                        successful_regions.append(region)
                            except Exception:
                                fields = self.workflow.generator.template_generator.get_data_fields_for_region(region, delay=1)
                                if fields and len(fields) > 0:
                                    successful_regions.append(region)
                                else:
                                    failed_regions.append(region)
                        else:
                            storage_type = self.storage_type.get()
                            if storage_type == "sqlite":
                                db_path = self.storage_path_var.get() or "generation_two_backtests.db"
                                if os.path.exists(db_path):
                                    try:
                                        import sqlite3
                                        conn = sqlite3.connect(db_path)
                                        cursor = conn.cursor()
                                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data_fields' LIMIT 1")
                                        if cursor.fetchone():
                                            cursor.execute("SELECT COUNT(*) FROM data_fields WHERE region=? LIMIT 1", (region,))
                                            count = cursor.fetchone()[0]
                                            if count > 0:
                                                if self.workflow.generator.template_generator.data_field_fetcher:
                                                    if region not in self.workflow.generator.template_generator.data_field_fetcher.data_fields:
                                                        self.workflow.generator.template_generator.data_field_fetcher.data_fields[region] = []
                                                successful_regions.append(region)
                                        conn.close()
                                    except Exception:
                                        failed_regions.append(region)
                                else:
                                    failed_regions.append(region)
                            else:
                                fields = self.workflow.generator.template_generator.get_data_fields_for_region(region, delay=1)
                                if fields and len(fields) > 0:
                                    successful_regions.append(region)
                                else:
                                    failed_regions.append(region)
                            
                    except Exception as e:
                        logger.error(f"Error loading fields for {region}: {e}", exc_info=True)
                        failed_regions.append(region)
                
                if successful_regions:
                    status_msg = f"✅ Loaded fields for: {', '.join(successful_regions)}"
                    if failed_regions:
                        status_msg += f"\n⚠️ Not found: {', '.join(failed_regions)}"
                    self.workflow.frame.after(0, lambda msg=status_msg: self.fetch_status.config(
                        text=msg,
                        fg=COLORS['accent_green'] if not failed_regions else COLORS['accent_yellow']
                    ))
                    if successful_regions:
                        self.workflow.steps_completed.add(0)
                        self.workflow.frame.after(0, self.workflow._update_step_indicators)
                else:
                    error_msg = f"❌ No cached fields found for any region.\nPlease use 'Fetch Data Fields' first."
                    self.workflow.frame.after(0, lambda msg=error_msg: self.fetch_status.config(
                        text=msg,
                        fg=COLORS['error']
                    ))
                
            except Exception as e:
                error_msg = f"Error loading data fields: {str(e)[:100]}"
                logger.error(error_msg, exc_info=True)
                self.workflow.frame.after(0, lambda msg=error_msg: self.fetch_status.config(
                    text=msg,
                    fg=COLORS['error']
                ))
                self.workflow.frame.after(0, lambda: messagebox.showerror("Error", error_msg))
            finally:
                self.workflow.frame.after(0, lambda: self.load_button.config(state=tk.NORMAL))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def _proceed_as_last_time(self):
        """Load last configuration and proceed automatically"""
        config_file = Path.home() / ".generation_two" / "workflow_config.json"
        if not config_file.exists():
            messagebox.showwarning("Warning", "No previous configuration found")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'operator_path' in config and config['operator_path']:
                self.operator_path_var.set(config['operator_path'])
                if os.path.exists(config['operator_path']):
                    self._load_operator_file()
            
            if 'storage_type' in config:
                self.storage_type.set(config['storage_type'])
            if 'storage_path' in config:
                self.storage_path_var.set(config['storage_path'])
            if 'storage_url' in config:
                self.storage_url_var.set(config['storage_url'])
            
            if 'regions' in config:
                for region, selected in config['regions'].items():
                    if region in self.region_vars:
                        self.region_vars[region].set(selected)
            
            self._update_storage_config_visibility()
            
            if self.workflow.operators_loaded:
                self._load_data_fields()
            
            messagebox.showinfo("Success", "Last configuration loaded! You can proceed to next steps.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load last configuration: {e}")
            logger.error(f"Error in proceed as last time: {e}", exc_info=True)
