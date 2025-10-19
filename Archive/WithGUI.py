import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import calendar
from datetime import datetime


class DeclineCurveAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Decline Curve Analysis - Professional Edition")
        self.root.geometry("1400x900")
        
        # Data variables
        self.df_all = None
        self.well_names = []
        self.current_well = None
        self.validation_message = ""  # Store validation message for Auto Select
        
        # Set style
        self.setup_style()
        
        # Create main layout
        self.create_layout()
        
    def setup_style(self):
        """Configure the application style"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = '#f0f0f0'
        panel_color = '#ffffff'
        accent_color = '#0078d4'
        
        style.configure('TFrame', background=bg_color)
        style.configure('Card.TFrame', background=panel_color, relief='raised')
        style.configure('TLabel', background=panel_color, font=('Segoe UI', 9))
        style.configure('Title.TLabel', font=('Segoe UI', 11, 'bold'), background=panel_color)
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), background=bg_color)
        style.configure('TButton', font=('Segoe UI', 9))
        style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'))
        
    def create_layout(self):
        """Create the main application layout"""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky='nsew')
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Configure grid
        main_container.columnconfigure(0, weight=0)  # Left panel
        main_container.columnconfigure(1, weight=1)  # Plot area
        main_container.rowconfigure(0, weight=0)     # Header
        main_container.rowconfigure(1, weight=1)     # Content
        
        # Header
        self.create_header(main_container)
        
        # Left control panel
        self.create_control_panel(main_container)
        
        # Right plot area
        self.create_plot_area(main_container)
        
    def create_header(self, parent):
        """Create application header"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="Decline Curve Analysis", 
                                style='Header.TLabel')
        title_label.pack(side='left')
        
        # Load data button
        load_btn = ttk.Button(header_frame, text="📁 Load Data", 
                             command=self.load_data, style='Accent.TButton')
        load_btn.pack(side='right', padx=5)
        
    def create_control_panel(self, parent):
        """Create left control panel with settings"""
        control_frame = ttk.Frame(parent, style='Card.TFrame', padding="15")
        control_frame.grid(row=1, column=0, sticky='ns', padx=(0, 10))
        
        # Well Selection Section
        well_frame = ttk.LabelFrame(control_frame, text="Well Selection", padding="10")
        well_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(well_frame, text="Select Well:").pack(anchor='w', pady=(0, 5))
        
        self.well_var = tk.StringVar()
        self.well_combo = ttk.Combobox(well_frame, textvariable=self.well_var, 
                                       state='readonly', width=25)
        self.well_combo.pack(fill='x', pady=(0, 5))
        self.well_combo.bind('<<ComboboxSelected>>', self.on_well_selected)
        
        # Analysis Settings Section
        settings_frame = ttk.LabelFrame(control_frame, text="Analysis Settings", padding="10")
        settings_frame.pack(fill='x', pady=(0, 15))
        
        # Outlier Threshold
        ttk.Label(settings_frame, text="Outlier Threshold (σ):").pack(anchor='w', pady=(0, 5))
        self.outlier_var = tk.DoubleVar(value=2.0)
        outlier_scale = ttk.Scale(settings_frame, from_=1.0, to=4.0, 
                                 variable=self.outlier_var, orient='horizontal')
        outlier_scale.pack(fill='x', pady=(0, 5))
        self.outlier_label = ttk.Label(settings_frame, text="2.0")
        self.outlier_label.pack(anchor='w', pady=(0, 10))
        self.outlier_var.trace('w', self.update_outlier_label)
        
        # Forecast Average Points
        ttk.Label(settings_frame, text="Forecast Avg Points:").pack(anchor='w', pady=(0, 5))
        self.forecast_var = tk.IntVar(value=2)
        forecast_spin = ttk.Spinbox(settings_frame, from_=0, to=12, 
                                   textvariable=self.forecast_var, width=10)
        forecast_spin.pack(anchor='w', pady=(0, 5))
        
        ttk.Label(settings_frame, text="(0=model, 1=last point, 2+=average)", 
                 font=('Segoe UI', 8)).pack(anchor='w', pady=(0, 10))
        
        # Detection Method
        ttk.Label(settings_frame, text="Start Detection:").pack(anchor='w', pady=(0, 5))
        self.method_var = tk.StringVar(value="auto_select")
        ttk.Radiobutton(settings_frame, text="Auto Select (Recommended)", 
                       variable=self.method_var, value="auto_select").pack(anchor='w')
        ttk.Radiobutton(settings_frame, text="Auto Period Detection", 
                       variable=self.method_var, value="auto").pack(anchor='w')
        ttk.Radiobutton(settings_frame, text="Decline Start Detection", 
                       variable=self.method_var, value="decline").pack(anchor='w')
        
        # Action Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        self.analyze_btn = ttk.Button(button_frame, text="🔍 Analyze", 
                                     command=self.run_analysis, state='disabled')
        self.analyze_btn.pack(fill='x', pady=(0, 5))
        
        export_btn = ttk.Button(button_frame, text="💾 Export Plot", 
                               command=self.export_plot, state='disabled')
        export_btn.pack(fill='x', pady=(0, 5))
        
        # Results Section
        results_frame = ttk.LabelFrame(control_frame, text="Analysis Results", padding="10")
        results_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        self.results_text = tk.Text(results_frame, height=12, width=30, 
                                   font=('Consolas', 9), wrap='word')
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def create_plot_area(self, parent):
        """Create right plot area"""
        plot_frame = ttk.Frame(parent, style='Card.TFrame', padding="10")
        plot_frame.grid(row=1, column=1, sticky='nsew')
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        # Initial plot
        self.ax.text(0.5, 0.5, 'Load data to begin analysis', 
                    ha='center', va='center', fontsize=14, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
    def update_outlier_label(self, *args):
        """Update outlier threshold label"""
        value = self.outlier_var.get()
        self.outlier_label.config(text=f"{value:.1f}")
        
    def load_data(self):
        """Load CSV data file"""
        filename = filedialog.askopenfilename(
            title="Select Production Data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.df_all = pd.read_csv(filename, low_memory=False)
                self.well_names = sorted(self.df_all['Well_Name'].unique())
                self.well_combo['values'] = self.well_names
                
                if self.well_names:
                    self.well_combo.current(0)
                    self.well_var.set(self.well_names[0])
                    self.analyze_btn['state'] = 'normal'
                    
                self.update_results(f"✓ Loaded {len(self.df_all)} records\n"
                                  f"✓ Found {len(self.well_names)} wells")
                messagebox.showinfo("Success", f"Loaded {len(self.well_names)} wells")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
                
    def on_well_selected(self, event=None):
        """Handle well selection change"""
        self.current_well = self.well_var.get()
        
    def update_results(self, text):
        """Update results text box"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        
    def run_analysis(self):
        """Run decline curve analysis"""
        if self.df_all is None or not self.current_well:
            messagebox.showwarning("Warning", "Please load data and select a well")
            return
        
        try:
            # Get parameters
            outlier_threshold = self.outlier_var.get()
            forecast_avg_points = self.forecast_var.get()
            method = self.method_var.get()
            
            # Run analysis
            results = self.perform_dca_analysis(
                self.df_all, 
                self.current_well,
                outlier_threshold,
                forecast_avg_points,
                method
            )
            
            if results:
                self.plot_results(results)
                self.display_results(results)
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze:\n{str(e)}")
            
    def perform_dca_analysis(self, df_all, well_name, outlier_threshold, 
                            forecast_avg_points, method):
        """Perform decline curve analysis - core logic"""
        
        df_well = df_all[df_all['Well_Name'] == well_name].copy()
        if df_well.empty:
            return None
            
        df_well = df_well.sort_values('Prod_Date')
        
        # Determine start index based on method
        if method == "auto_select":
            # Auto Select: Try decline method first, validate, fallback to auto if needed
            auto_idx = self.find_last_production_period(df_well)
            decline_idx = self.find_decline_start_last_major_drop(df_well)
            
            # Validate decline_idx
            df_well_sorted = df_well.sort_values('Prod_Date')
            rate_series = df_well_sorted['M_Oil_Prod'].values
            
            if decline_idx < len(rate_series):
                # Check if decline_idx is too early (rate at decline start < 80% of max rate after it)
                decline_rate = rate_series[decline_idx]
                future_max = np.max(rate_series[decline_idx:])
                
                if decline_rate < 0.8 * future_max:
                    # Decline idx appears too early, use auto instead
                    start_idx = auto_idx
                    method_label = "Auto Select → Auto Period Detection"
                    validation_msg = f"⚠ Decline detection too early (rate < 80% of future max)\n→ Using Auto Period Detection"
                else:
                    # Decline idx is appropriate
                    start_idx = decline_idx
                    method_label = "Auto Select → Decline Start Detection"
                    validation_msg = f"✓ Decline detection validated\n→ Using Decline Start Detection"
            else:
                # decline_idx out of range, use auto
                start_idx = auto_idx
                method_label = "Auto Select → Auto Period Detection"
                validation_msg = f"⚠ Decline index out of range\n→ Using Auto Period Detection"
            
            # Store validation message for display
            self.validation_message = validation_msg
            
        elif method == "auto":
            start_idx = self.find_last_production_period(df_well)
            method_label = "Auto Period Detection"
            self.validation_message = ""
        else:  # method == "decline"
            start_idx = self.find_decline_start_last_major_drop(df_well)
            method_label = "Decline Start Detection"
            self.validation_message = ""
        
        # Prepare data
        df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
        df_well['days_in_month'] = df_well['Prod_Date'].apply(
            lambda d: calendar.monthrange(d.year, d.month)[1]
        )
        df_well['oil_prod_daily'] = df_well['M_Oil_Prod'] / df_well['days_in_month']
        df_well['t'] = (df_well['Prod_Date'] - df_well['Prod_Date'].min()).dt.days // 30
        
        df_full = df_well.copy()
        q_original_full = df_well['oil_prod_daily'].values.copy()
        
        # Smooth full data
        n = len(q_original_full)
        w = min(7, n if n % 2 == 1 else n-1)
        w = max(3, w)
        try:
            q_smoothed_full = savgol_filter(q_original_full, window_length=w, polyorder=2)
        except:
            q_smoothed_full = q_original_full.copy()
        
        # Apply start index
        if start_idx > 0:
            df_well = df_well.iloc[start_idx:].reset_index(drop=True)
        
        # Fit with outlier filtering
        t = df_well['t'].values - df_well['t'].values[0]
        q = df_well['oil_prod_daily'].values
        
        mask, popt = self.filter_outliers_iterative(t, q, threshold=outlier_threshold)
        
        if popt is None:
            return None
        
        # Generate forecast
        last_date = df_well['Prod_Date'].max()
        forecast_dates = pd.date_range(start=last_date, periods=61, freq='ME')
        last_t = t[-1]
        forecast_t = np.arange(last_t, last_t + 61)
        forecast_q = self.arps_hyperbolic(forecast_t, *popt)
        
        # Adjust initial forecast rate
        if forecast_avg_points > 1:
            n_points = min(forecast_avg_points, np.sum(mask))
            initial_rate = np.mean(q[mask][-n_points:])
        elif forecast_avg_points == 0:
            initial_rate = forecast_q[0]
        else:
            initial_rate = q[mask][-1]
        
        if forecast_avg_points != 0:
            scaling_factor = initial_rate / forecast_q[0]
            forecast_q = forecast_q * scaling_factor
        
        # Calculate metrics
        q_pred = self.arps_hyperbolic(t, *popt)
        residuals = q - q_pred
        channel_width = np.std(residuals)
        
        return {
            'df_well': df_well,
            'df_full': df_full,
            't': t,
            'q': q,
            'q_original': q_original_full,
            'q_smoothed': q_smoothed_full,
            'mask': mask,
            'popt': popt,
            'q_pred': q_pred,
            'channel_width': channel_width,
            'forecast_dates': forecast_dates,
            'forecast_q': forecast_q,
            'method_label': method_label,
            'start_idx': start_idx,
            'initial_rate': initial_rate
        }
    
    def plot_results(self, results):
        """Plot analysis results"""
        self.ax.clear()
        
        df_well = results['df_well']
        df_full = results['df_full']
        q = results['q']
        mask = results['mask']
        q_pred = results['q_pred']
        channel_width = results['channel_width']
        
        # Original data
        self.ax.plot(df_full['Prod_Date'], results['q_original'], 'o', 
                    color='red', label='Original Data', alpha=0.6, markersize=5)
        
        # Smoothed data
        self.ax.plot(df_full['Prod_Date'], results['q_smoothed'], '--', 
                    color='red', label='Smoothed Data', linewidth=2)
        
        # Inliers
        self.ax.plot(df_well['Prod_Date'][mask], q[mask], 'o', 
                    color='blue', label='Decline Phase Data', markersize=6)
        
        # Outliers
        if np.sum(~mask) > 0:
            self.ax.plot(df_well['Prod_Date'][~mask], q[~mask], 'x', 
                        color='red', label='Outliers', markersize=10, markeredgewidth=2)
        
        # Confidence channel
        upper_bound = q_pred + channel_width
        lower_bound = np.maximum(q_pred - channel_width, 0)
        self.ax.fill_between(df_well['Prod_Date'], lower_bound, upper_bound,
                            color='green', alpha=0.2, 
                            label=f'±{channel_width:.1f} bbl/day Channel')
        
        # Model fit
        self.ax.plot(df_well['Prod_Date'], q_pred, '-', 
                    color='green', label='Arps Model', linewidth=2)
        
        # Forecast
        self.ax.plot(results['forecast_dates'], results['forecast_q'], '--', 
                    color='blue', label='5-Year Forecast', linewidth=2)
        
        self.ax.set_xlabel('Date', fontsize=10)
        self.ax.set_ylabel('Oil Production Rate (bbl/day)', fontsize=10)
        self.ax.set_title(f"Well: {self.current_well}\nMethod: {results['method_label']}", 
                         fontsize=12, fontweight='bold')
        self.ax.legend(loc='best', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(bottom=0)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def display_results(self, results):
        """Display analysis results in text box"""
        popt = results['popt']
        mask = results['mask']
        q = results['q']
        forecast_q = results['forecast_q']
        
        n_outliers = np.sum(~mask)
        cv = np.std(q[mask]) / np.mean(q[mask])
        
        quality = "✓ Excellent" if cv < 0.3 else "⚠ Moderate" if cv < 0.4 else "⚠ Poor"
        
        # Add validation message if Auto Select was used
        validation_section = ""
        if self.validation_message:
            validation_section = f"\nMETHOD SELECTION:\n{self.validation_message}\n"
        
        text = f"""=== ANALYSIS RESULTS ===

Well: {self.current_well}
Method: {results['method_label']}{validation_section}

FITTED PARAMETERS:
qi = {popt[0]:.2f} bbl/day
Di = {popt[1]:.4f} /day
b  = {popt[2]:.2f}

DATA QUALITY:
Data points: {len(q)}
Outliers removed: {n_outliers}
CV: {cv:.3f}
Quality: {quality}

FORECAST SUMMARY:
Initial rate: {forecast_q[0]:.2f} bbl/day
Rate after 5 yrs: {forecast_q[-1]:.2f} bbl/day
Total decline: {((forecast_q[0]-forecast_q[-1])/forecast_q[0]*100):.1f}%

Channel width: ±{results['channel_width']:.1f} bbl/day
"""
        self.update_results(text)
    
    def export_plot(self):
        """Export current plot"""
        if self.ax.lines:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("All files", "*.*")]
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Plot exported successfully")
    
    # Core DCA functions
    @staticmethod
    def arps_hyperbolic(t, qi, Di, b):
        return qi / (1 + b * Di * t) ** (1/b)
    
    def fit_arps_hyperbolic(self, t, q):
        p0 = [q[0], 0.1, 0.5]
        try:
            popt, pcov = curve_fit(
                self.arps_hyperbolic, t, q, p0=p0,
                bounds=([1e-6, 1e-6, 0], [np.inf, 2, 2])
            )
            return popt, pcov
        except:
            return None, None
    
    def filter_outliers_iterative(self, t, q, threshold=3, max_iterations=5):
        mask = np.ones(len(t), dtype=bool)
        
        for iteration in range(max_iterations):
            t_fit = t[mask]
            q_fit = q[mask]
            
            popt, pcov = self.fit_arps_hyperbolic(t_fit, q_fit)
            if popt is None:
                break
            
            q_pred = self.arps_hyperbolic(t, *popt)
            residuals = q - q_pred
            residuals_fit = residuals[mask]
            std_residuals = np.std(residuals_fit)
            new_mask = np.abs(residuals) <= threshold * std_residuals
            
            if np.array_equal(mask, new_mask):
                break
            
            mask = new_mask
        
        popt, pcov = self.fit_arps_hyperbolic(t[mask], q[mask])
        return mask, popt
    
    def find_last_production_period(self, df_well, min_production_months=12,
                                   surge_multiplier=2, gap_threshold=0.3, 
                                   lookback_window=20):
        df = df_well.sort_values('Prod_Date').copy()
        df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
        df['days_in_month'] = df['Prod_Date'].apply(
            lambda d: calendar.monthrange(d.year, d.month)[1]
        )
        
        q = (df['M_Oil_Prod'] / df['days_in_month']).values
        n = len(q)
        
        if n < min_production_months:
            return 0
        
        q_positive = q[q > 0]
        if len(q_positive) == 0:
            return 0
        
        median_rate = np.median(q_positive)
        mean_rate = np.mean(q_positive)
        shutin_threshold = median_rate * gap_threshold
        surge_threshold = mean_rate * surge_multiplier
        
        candidate_starts = []
        
        for i in range(n - min_production_months, lookback_window, -1):
            if q[i] < shutin_threshold and q[i-1] > shutin_threshold:
                candidate_starts.append({'idx': i, 'type': 'shut-in recovery'})
            
            if i >= lookback_window:
                avg_before = np.mean(q[max(0, i-lookback_window):i])
                if avg_before > 0 and q[i] > surge_threshold and q[i] > avg_before * surge_multiplier:
                    candidate_starts.append({'idx': i, 'type': 'production surge'})
        
        if candidate_starts:
            candidate_starts.sort(key=lambda x: x['idx'], reverse=True)
            return candidate_starts[0]['idx']
        
        split_point = n // 4
        early_median = np.median(q[:split_point])
        recent_median = np.median(q[split_point:])
        
        if early_median > recent_median * 2:
            return split_point
        
        return 0
    
    def find_decline_start_last_major_drop(self, df_well, smooth_window=20,
                                          min_decline_months=12, drop_frac=0.3):
        df = df_well.sort_values('Prod_Date').copy()
        df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
        df['days_in_month'] = df['Prod_Date'].apply(
            lambda d: calendar.monthrange(d.year, d.month)[1]
        )
        q = (df['M_Oil_Prod'] / df['days_in_month']).values.astype(float)
        n = len(q)
        
        if n < 10:
            return 0
        
        w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        w = min(w, n if n % 2 == 1 else n-1)
        try:
            q_smooth = savgol_filter(q, window_length=w, polyorder=2)
        except:
            q_smooth = q.copy()
        
        peaks, _ = find_peaks(q_smooth, distance=6)
        if len(peaks) == 0:
            return int(np.argmax(q_smooth))
        
        candidates = []
        for p in peaks:
            end = min(n, p + min_decline_months)
            if end <= p + 3:
                continue
            post_mean = np.mean(q_smooth[p+1:end])
            decline_ratio = (q_smooth[p] - post_mean) / q_smooth[p]
            if decline_ratio >= drop_frac:
                candidates.append(p)
        
        if not candidates:
            return int(np.argmax(q_smooth))
        
        return int(candidates[-1])


if __name__ == "__main__":
    root = tk.Tk()
    app = DeclineCurveAnalysisGUI(root)
    root.mainloop()