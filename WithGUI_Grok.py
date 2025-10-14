# Decline Curve Analysis Tool with GUI
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import calendar

# --- Arps model ---
def arps_hyperbolic(t, qi, Di, b):
    """Arps Hyperbolic model equation"""
    return qi / (1 + b * Di * t) ** (1/b)

def fit_arps_hyperbolic(t, q):
    """Fit Arps Hyperbolic model to production data"""
    p0 = [q[0], 0.1, 0.5]
    try:
        popt, pcov = curve_fit(
            arps_hyperbolic, t, q, p0=p0,
            bounds=([1e-6, 1e-6, 0], [np.inf, 2, 2])
        )
        return popt, pcov
    except Exception as e:
        print(f"Model fitting failed: {e}")
        return None, None

def filter_outliers_iterative(t, q, threshold=3, max_iterations=5):
    """
    Iteratively fit Arps model and remove outliers based on residuals.
    
    Parameters:
    -----------
    t : array
        Time values
    q : array
        Production rate values
    threshold : float
        Number of standard deviations for outlier threshold (default: 3)
    max_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    mask : boolean array
        True for inliers, False for outliers
    popt : array
        Final fitted parameters
    """
    mask = np.ones(len(t), dtype=bool)
    
    for iteration in range(max_iterations):
        # Fit model on current inliers
        t_fit = t[mask]
        q_fit = q[mask]
        
        popt, pcov = fit_arps_hyperbolic(t_fit, q_fit)
        if popt is None:
            print(f"Fitting failed at iteration {iteration}")
            break
            
        # Compute residuals for all points
        q_pred = arps_hyperbolic(t, *popt)
        residuals = q - q_pred
        
        # Compute std on current inliers only
        residuals_fit = residuals[mask]
        std_residuals = np.std(residuals_fit)
        
        # Identify outliers
        new_mask = np.abs(residuals) <= threshold * std_residuals
        
        # Check convergence
        if np.array_equal(mask, new_mask):
            print(f"Converged after {iteration + 1} iterations")
            break
            
        n_removed = np.sum(mask) - np.sum(new_mask)
        print(f"Iteration {iteration + 1}: Removed {n_removed} outliers, std = {std_residuals:.2f}")
        mask = new_mask
    
    # Final fit
    popt, pcov = fit_arps_hyperbolic(t[mask], q[mask])
    
    return mask, popt

def create_arps_plot(df_well, t, q, mask, popt, well_id, df_full, q_original, q_smoothed, outlier_threshold, forecast_avg_points=1, start_method=""):
    """Create matplotlib figure for Arps fit"""
    if popt is None:
        print("No fit available to plot.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    q_pred = arps_hyperbolic(t, *popt)
    
    # Calculate constant channel width as 2 times the average standard deviation
    residuals = q - q_pred
    avg_std = np.std(residuals)
    channel_width = avg_std 
    
    # Calculate upper and lower bounds with constant width
    upper_bound = q_pred + channel_width
    lower_bound = q_pred - channel_width
    
    # Ensure lower bound doesn't go below zero (oil production can't be negative)
    lower_bound = np.maximum(lower_bound, 0)
    
    # Generate 5-year forecast
    last_date = df_well['Prod_Date'].max()
    forecast_dates = pd.date_range(start=last_date, periods=61, freq='ME')  # 60 months + 1 for last historical point
    
    # Time points for forecast (continuing from last historical time)
    last_t = t[-1]
    forecast_t = np.arange(last_t, last_t + 61)
    
    # Calculate the forecast using original fitted parameters
    forecast_q = arps_hyperbolic(forecast_t, *popt)
    
    # Adjust the initial rate based on forecast_avg_points
    if forecast_avg_points > 1:
        # Use average of last n rates from the filtered data (inliers only)
        n_points = min(forecast_avg_points, np.sum(mask))
        initial_rate = np.mean(q[mask][-n_points:])
        print(f"\nUsing average of last {n_points} rates as initial forecast rate: {initial_rate:.2f} bbl/day")
    elif forecast_avg_points == 0:
        # Use the last point from the fitted model line
        initial_rate = forecast_q[0]  # First point of the forecast (which is the model value at t[-1])
        print(f"\nUsing last fitted model point as initial forecast rate: {initial_rate:.2f} bbl/day")
    else:
        # Use last historical rate
        initial_rate = q[mask][-1]
        print(f"\nUsing last historical rate as initial forecast rate: {initial_rate:.2f} bbl/day")
    
    # Adjust the forecast to start from the chosen initial rate while maintaining the same decline
    if forecast_avg_points != 0:  # Only adjust if not using the model point
        scaling_factor = initial_rate / forecast_q[0]
        forecast_q = forecast_q * scaling_factor
    
    # Original data (red circles)
    ax.plot(df_full['Prod_Date'], q_original, 'o', color='red', 
             label='Original Data', alpha=0.6, markersize=5)
    
    # Smoothed data (red dashed line)
    ax.plot(df_full['Prod_Date'], q_smoothed, '--', color='red', 
             label='Smoothed Data', linewidth=2)
    
    # Filtered/Decline data - inliers (blue circles)
    ax.plot(df_well['Prod_Date'][mask], q[mask], 'o', color='blue', 
             label='Decline Phase Data (Inliers)', markersize=6)
    
    # Outliers (red X marks)
    if np.sum(~mask) > 0:
        ax.plot(df_well['Prod_Date'][~mask], q[~mask], 'x', color='red', 
                 label='Outliers (Excluded)', markersize=10, markeredgewidth=2)
    
    # Constant width channel (clipped at zero)
    ax.fill_between(df_well['Prod_Date'], lower_bound, upper_bound,
                    color='green', alpha=0.2, label=f'±{channel_width:.1f} bbl/day Channel')
    
    # Arps model fit (solid line)
    ax.plot(df_well['Prod_Date'], q_pred, '-', color='green', 
             label='Arps Hyperbolic Model', linewidth=2)
    
    # Plot forecast
    ax.plot(forecast_dates, forecast_q, '--', color='blue',
             label='5-Year Forecast', linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Oil Production Rate (bbl/day)')
    title = f'Well {well_id}'
    if start_method:
        title += f'\nStart Index Method: {start_method}'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
    
    return fig

# --- Helper functions (unchanged) ---
def get_days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

def find_decline_start(
    df_well,
    rate_col='M_Oil_Prod',
    smooth_window=13,       
    polyorder=2,
    peak_prominence=0.15,   
    post_ma=12,             
    decline_frac=0.15,      
    post_slope_win=12,      
    persist_months=9        
):
    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    df['days_in_month'] = df['Prod_Date'].apply(get_days_in_month)
    q = (df[rate_col] / df['days_in_month']).values.astype(float)
    n = len(q)
    if n == 0:
        return 0

    w = int(smooth_window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    try:
        q_smooth = savgol_filter(q, window_length=w, polyorder=min(polyorder, w-1))
    except Exception:
        q_smooth = q.copy()

    max_q = np.nanmax(q_smooth) if np.nanmax(q_smooth) > 0 else 1.0
    prom_abs = peak_prominence * max_q
    peaks, props = find_peaks(q_smooth, prominence=prom_abs, distance=6)

    if len(peaks) == 0:
        return int(np.nanargmax(q_smooth))

    candidates = []
    for p in peaks:
        if p + 1 >= n:
            continue

        end_ma = min(n, p + 1 + post_ma)
        ma_after = np.mean(q_smooth[p+1:end_ma])

        drop_ok = ma_after <= q_smooth[p] * (1.0 - decline_frac)

        end_slope = min(n, p + 1 + post_slope_win)
        if end_slope - (p+1) >= 3:
            x = np.arange(0, end_slope - (p+1))
            y = q_smooth[p+1:end_slope]
            slope = linregress(x, y).slope
        else:
            slope = 0.0

        persist = False
        if p + persist_months < n:
            neg_count = 0
            total_checks = 0
            for start in range(p+1, min(n - persist_months + 1, p+1 + post_slope_win - persist_months + 1)):
                xs = np.arange(0, persist_months)
                ys = q_smooth[start:start+persist_months]
                s = linregress(xs, ys).slope
                total_checks += 1
                if s < 0:
                    neg_count += 1
            if total_checks > 0 and (neg_count / total_checks) >= 0.7:
                persist = True

        if drop_ok and (persist or slope < 0):
            candidates.append({
                'idx': p,
                'peak_val': q_smooth[p],
                'prom': props['prominences'][np.where(peaks == p)[0][0]] if 'prominences' in props else 0.0,
                'slope': slope
            })

    if not candidates:
        alt_candidates = []
        long_ma = int(post_ma * 1.5)
        for p in peaks:
            end_ma = min(n, p + 1 + long_ma)
            ma_after = np.mean(q_smooth[p+1:end_ma]) if end_ma > p+1 else q_smooth[p]
            if ma_after <= q_smooth[p] * (1.0 - (decline_frac * 0.8)):
                alt_candidates.append({'idx': p, 'peak_val': q_smooth[p]})
        if alt_candidates:
            best = max(alt_candidates, key=lambda x: x['peak_val'])
            return int(best['idx'])
        return int(peaks[np.argmax(q_smooth[peaks])])

    best = max(candidates, key=lambda x: (x['peak_val'], x['prom']))
    return int(best['idx'])

def find_decline_start_last_major_drop(
    df_well,
    rate_col='M_Oil_Prod',
    smooth_window=13,
    polyorder=2,
    min_decline_months=18,
    drop_frac=0.2
):
    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    df['days_in_month'] = df['Prod_Date'].apply(get_days_in_month)
    q = (df[rate_col] / df['days_in_month']).values.astype(float)
    n = len(q)
    if n < 10:
        return 0

    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    w = min(w, n if n % 2 == 1 else n-1)
    try:
        q_smooth = savgol_filter(q, window_length=w, polyorder=min(polyorder, w-1))
    except Exception:
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

    idx = candidates[-1]
    return int(idx)

def find_last_production_period(df_well, rate_col='M_Oil_Prod', 
                                 min_production_months=12,
                                 surge_multiplier=3.5,
                                 gap_threshold=0.25,
                                 lookback_window=20):
    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    df['days_in_month'] = df['Prod_Date'].apply(get_days_in_month)
    
    q = (df[rate_col] / df['days_in_month']).values
    n = len(q)
    
    if n < min_production_months:
        print(f"Warning: Only {n} months of data available")
        return 0
    
    q_positive = q[q > 0]
    if len(q_positive) == 0:
        return 0
        
    median_rate = np.median(q_positive)
    mean_rate = np.mean(q_positive)
    
    shutin_threshold = median_rate * gap_threshold
    surge_threshold = mean_rate * surge_multiplier
    
    print(f"\n=== Production Period Detection ===")
    print(f"Median production rate: {median_rate:.2f} bbl/day")
    print(f"Shut-in threshold: {shutin_threshold:.2f} bbl/day")
    print(f"Surge detection threshold: {surge_threshold:.2f} bbl/day")
    
    candidate_starts = []
    
    for i in range(n - min_production_months, lookback_window, -1):
        if q[i] < shutin_threshold and q[i-1] > shutin_threshold:
            candidate_starts.append({
                'idx': i,
                'type': 'shut-in recovery',
                'date': df.iloc[i]['Prod_Date'],
                'before': q[i-1],
                'after': q[i]
            })
            
        if i >= lookback_window:
            avg_before = np.mean(q[max(0, i-lookback_window):i])
            if avg_before > 0 and q[i] > surge_threshold and q[i] > avg_before * surge_multiplier:
                candidate_starts.append({
                    'idx': i,
                    'type': 'production surge',
                    'date': df.iloc[i]['Prod_Date'],
                    'before': avg_before,
                    'after': q[i]
                })
        
        if i >= lookback_window:
            recent_avg = np.mean(q[i-lookback_window:i])
            if recent_avg < shutin_threshold and q[i] > median_rate * 0.8:
                candidate_starts.append({
                    'idx': i,
                    'type': 'extended shut-in recovery',
                    'date': df.iloc[i]['Prod_Date'],
                    'before': recent_avg,
                    'after': q[i]
                })
    
    if candidate_starts:
        candidate_starts.sort(key=lambda x: x['idx'], reverse=True)
        
        print(f"\nDetected {len(candidate_starts)} potential production period changes:")
        for i, event in enumerate(candidate_starts[:5]):
            print(f"  {i+1}. {event['type'].title()} at {event['date'].date()} "
                  f"(idx={event['idx']}, before={event['before']:.1f}, after={event['after']:.1f})")
        
        selected = candidate_starts[0]
        print(f"\n✓ Selected start: {selected['date'].date()} ({selected['type']})")
        print(f"  Using {n - selected['idx']} months of data for analysis")
        
        return selected['idx']
    
    split_point = n // 4
    early_median = np.median(q[:split_point])
    recent_median = np.median(q[split_point:])
    
    if early_median > recent_median * 2:
        print(f"\nEarly production ({early_median:.1f}) much higher than recent ({recent_median:.1f})")
        print(f"Starting analysis from index {split_point} ({df.iloc[split_point]['Prod_Date'].date()})")
        return split_point
    
    print(f"\nNo major production changes detected. Using all available data.")
    return 0

# Modified run_arps_for_well_auto to return figure and results
def run_arps_for_well_auto(df_all, well_name, outlier_threshold=2, 
                           forecast_avg_points=6, manual_start_idx=None,
                           detect_period=True, start_method="", 
                           decline_params={}, period_params={}, filter_params={}):
    df_well = df_all[df_all['Well_Name'] == well_name].copy()
    if df_well.empty:
        return None, "No data found for well"
    
    df_well = df_well.sort_values('Prod_Date')
    
    if manual_start_idx is not None:
        start_idx = manual_start_idx
    elif detect_period:
        start_idx = find_last_production_period(
            df_well,
            min_production_months=period_params.get('min_production_months', 12),
            surge_multiplier=period_params.get('surge_multiplier', 2.0),
            gap_threshold=period_params.get('gap_threshold', 0.3),
            lookback_window=period_params.get('lookback_window', 6)
        )
    else:
        start_idx = 0
    
    df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
    df_well['days_in_month'] = df_well['Prod_Date'].apply(get_days_in_month)
    df_well['oil_prod_daily'] = df_well['M_Oil_Prod'] / df_well['days_in_month']
    df_well['t'] = (df_well['Prod_Date'] - df_well['Prod_Date'].min()).dt.days // 30

    df_full = df_well.copy()
    q_original_full = df_well['oil_prod_daily'].values.copy()
    
    n = len(q_original_full)
    w = min(7, n if n % 2 == 1 else n-1)
    w = max(3, w)
    try:
        q_smoothed_full = savgol_filter(q_original_full, window_length=w, polyorder=2)
    except Exception:
        q_smoothed_full = q_original_full.copy()

    if start_idx > 0:
        df_well = df_well.iloc[start_idx:].reset_index(drop=True)

    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    
    mask, popt = filter_outliers_iterative(
        t, q, 
        threshold=filter_params.get('threshold', outlier_threshold),
        max_iterations=filter_params.get('max_iterations', 5)
    )
    
    fig = create_arps_plot(df_well, t, q, mask, popt, well_name, df_full, 
                           q_original_full, q_smoothed_full, outlier_threshold, 
                           forecast_avg_points, start_method)
    
    results = ""
    if popt is not None:
        n_outliers = np.sum(~mask)
        results += f"\n=== Final Results for well {well_name} ===\n"
        results += f"Total outliers removed: {n_outliers} out of {len(mask)} points\n"
        results += f"Fitted parameters:\n"
        results += f"qi = {popt[0]:.2f}, Di = {popt[1]:.4f}, b = {popt[2]:.2f}\n"
        
        cv = np.std(q[mask]) / np.mean(q[mask])
        results += f"\nCoefficient of Variation: {cv:.3f}\n"
        if cv > 0.4:
            results += "⚠ WARNING: High variability detected. Forecast may be unreliable.\n"
        elif cv > 0.3:
            results += "⚠ CAUTION: Moderate variability. Review forecast carefully.\n"
        else:
            results += "✓ Good data quality for decline curve analysis.\n"
    
    return fig, results

# GUI Class
class DeclineCurveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Decline Curve Analysis Tool")
        self.geometry("1200x800")
        
        # Set theme
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background='#f0f0f0', foreground='#333333')
        style.configure('TLabel', background='#f0f0f0')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TNotebook.Tab', background='#dddddd', foreground='#333333')
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')])
        
        # Load data
        try:
            self.df_all = pd.read_csv('OFM202409.csv', low_memory=False)
            self.wells = sorted(self.df_all['Well_Name'].unique())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.destroy()
            return
        
        # Create main frames
        self.left_frame = ttk.Frame(self, width=300, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Well selection
        ttk.Label(self.left_frame, text="Select Well:").pack(anchor=tk.W, pady=5)
        self.well_var = tk.StringVar()
        self.well_combo = ttk.Combobox(self.left_frame, textvariable=self.well_var, values=self.wells, state='readonly')
        self.well_combo.pack(fill=tk.X, pady=5)
        self.well_combo.bind('<<ComboboxSelected>>', self.on_well_select)
        
        # Start method radio buttons
        ttk.Label(self.left_frame, text="Start Index Method:").pack(anchor=tk.W, pady=5)
        self.start_method_var = tk.StringVar(value="Auto Select")
        methods = ["Auto Select", "Auto-Detected Recent Period", "Smoothing Decline Analysis", "Manual"]
        for m in methods:
            ttk.Radiobutton(self.left_frame, text=m, variable=self.start_method_var, value=m).pack(anchor=tk.W)
        
        # Manual start date
        self.manual_date_frame = ttk.Frame(self.left_frame)
        self.manual_date_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.manual_date_frame, text="Manual Start Year:").pack(side=tk.LEFT)
        self.manual_year_var = tk.IntVar(value=2020)
        ttk.Entry(self.manual_date_frame, textvariable=self.manual_year_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.manual_date_frame, text="Month:").pack(side=tk.LEFT)
        self.manual_month_var = tk.IntVar(value=1)
        ttk.Entry(self.manual_date_frame, textvariable=self.manual_month_var, width=3).pack(side=tk.LEFT, padx=5)
        
        # Settings notebook
        self.settings_notebook = ttk.Notebook(self.left_frame)
        self.settings_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # General tab
        general_tab = ttk.Frame(self.settings_notebook, padding=5)
        self.settings_notebook.add(general_tab, text="General")
        
        self.outlier_threshold_var = tk.DoubleVar(value=2.0)
        ttk.Label(general_tab, text="Outlier Threshold:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(general_tab, textvariable=self.outlier_threshold_var).grid(row=0, column=1, pady=2)
        
        self.forecast_avg_points_var = tk.IntVar(value=2)
        ttk.Label(general_tab, text="Forecast Avg Points:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(general_tab, textvariable=self.forecast_avg_points_var).grid(row=1, column=1, pady=2)
        
        self.filter_max_iter_var = tk.IntVar(value=5)
        ttk.Label(general_tab, text="Max Outlier Iterations:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(general_tab, textvariable=self.filter_max_iter_var).grid(row=2, column=1, pady=2)
        
        # Decline Detection tab
        decline_tab = ttk.Frame(self.settings_notebook, padding=5)
        self.settings_notebook.add(decline_tab, text="Decline Detection")
        
        self.smooth_window_var = tk.IntVar(value=13)
        ttk.Label(decline_tab, text="Smooth Window:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.smooth_window_var).grid(row=0, column=1, pady=2)
        
        self.polyorder_var = tk.IntVar(value=2)
        ttk.Label(decline_tab, text="Polyorder:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.polyorder_var).grid(row=1, column=1, pady=2)
        
        self.peak_prominence_var = tk.DoubleVar(value=0.15)
        ttk.Label(decline_tab, text="Peak Prominence:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.peak_prominence_var).grid(row=2, column=1, pady=2)
        
        self.post_ma_var = tk.IntVar(value=12)
        ttk.Label(decline_tab, text="Post MA:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.post_ma_var).grid(row=3, column=1, pady=2)
        
        self.decline_frac_var = tk.DoubleVar(value=0.15)
        ttk.Label(decline_tab, text="Decline Frac:").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.decline_frac_var).grid(row=4, column=1, pady=2)
        
        self.post_slope_win_var = tk.IntVar(value=12)
        ttk.Label(decline_tab, text="Post Slope Win:").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.post_slope_win_var).grid(row=5, column=1, pady=2)
        
        self.persist_months_var = tk.IntVar(value=9)
        ttk.Label(decline_tab, text="Persist Months:").grid(row=6, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.persist_months_var).grid(row=6, column=1, pady=2)
        
        self.min_decline_months_var = tk.IntVar(value=18)
        ttk.Label(decline_tab, text="Min Decline Months:").grid(row=7, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.min_decline_months_var).grid(row=7, column=1, pady=2)
        
        self.drop_frac_var = tk.DoubleVar(value=0.2)
        ttk.Label(decline_tab, text="Drop Frac:").grid(row=8, column=0, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.drop_frac_var).grid(row=8, column=1, pady=2)
        
        # Period Detection tab
        period_tab = ttk.Frame(self.settings_notebook, padding=5)
        self.settings_notebook.add(period_tab, text="Period Detection")
        
        self.min_prod_months_var = tk.IntVar(value=12)
        ttk.Label(period_tab, text="Min Production Months:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.min_prod_months_var).grid(row=0, column=1, pady=2)
        
        self.surge_mult_var = tk.DoubleVar(value=2.0)
        ttk.Label(period_tab, text="Surge Multiplier:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.surge_mult_var).grid(row=1, column=1, pady=2)
        
        self.gap_thresh_var = tk.DoubleVar(value=0.3)
        ttk.Label(period_tab, text="Gap Threshold:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.gap_thresh_var).grid(row=2, column=1, pady=2)
        
        self.lookback_win_var = tk.IntVar(value=6)
        ttk.Label(period_tab, text="Lookback Window:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.lookback_win_var).grid(row=3, column=1, pady=2)
        
        # Run button
        self.run_btn = ttk.Button(self.left_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=10)
        
        # Results text
        self.results_text = tk.Text(self.left_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, pady=5)
        
        # Plot canvas
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = None
        self.canvas = None
        self.toolbar = None
    
    def on_well_select(self, event):
        pass  # Can add preview if needed
    
    def run_analysis(self):
        well_name = self.well_var.get()
        if not well_name:
            messagebox.showwarning("Warning", "Please select a well")
            return
        
        start_method = self.start_method_var.get()
        manual_start_idx = None
        detect_period = False
        
        # Gather parameters
        outlier_threshold = self.outlier_threshold_var.get()
        forecast_avg_points = self.forecast_avg_points_var.get()
        
        filter_params = {
            'threshold': outlier_threshold,
            'max_iterations': self.filter_max_iter_var.get()
        }
        
        decline_params = {
            'smooth_window': self.smooth_window_var.get(),
            'polyorder': self.polyorder_var.get(),
            'peak_prominence': self.peak_prominence_var.get(),
            'post_ma': self.post_ma_var.get(),
            'decline_frac': self.decline_frac_var.get(),
            'post_slope_win': self.post_slope_win_var.get(),
            'persist_months': self.persist_months_var.get(),
            'min_decline_months': self.min_decline_months_var.get(),
            'drop_frac': self.drop_frac_var.get()
        }
        
        period_params = {
            'min_production_months': self.min_prod_months_var.get(),
            'surge_multiplier': self.surge_mult_var.get(),
            'gap_threshold': self.gap_thresh_var.get(),
            'lookback_window': self.lookback_win_var.get()
        }
        
        df_well_temp = self.df_all[self.df_all['Well_Name'] == well_name].copy()
        df_well_temp['Prod_Date'] = pd.to_datetime(df_well_temp['Prod_Date'])
        df_well_sorted = df_well_temp.sort_values('Prod_Date')
        
        if start_method == "Manual":
            try:
                year = self.manual_year_var.get()
                month = self.manual_month_var.get()
                manual_date = pd.to_datetime(f"{year}-{month:02d}-01")
                manual_start_idx = np.searchsorted(df_well_sorted['Prod_Date'], manual_date)
                if manual_start_idx >= len(df_well_sorted):
                    messagebox.showwarning("Warning", "Manual date is after last production date. Using last index.")
                    manual_start_idx = len(df_well_sorted) - 1
            except Exception as e:
                messagebox.showerror("Error", f"Invalid manual date: {str(e)}")
                return
        elif start_method == "Auto-Detected Recent Period":
            detect_period = True
        elif start_method == "Smoothing Decline Analysis":
            manual_start_idx = find_decline_start_last_major_drop(
                df_well_sorted,
                smooth_window=decline_params['smooth_window'],
                polyorder=decline_params['polyorder'],
                min_decline_months=decline_params['min_decline_months'],
                drop_frac=decline_params['drop_frac']
            )
        elif start_method == "Auto Select":
            auto_idx = find_last_production_period(
                df_well_sorted,
                min_production_months=period_params['min_production_months'],
                surge_multiplier=period_params['surge_multiplier'],
                gap_threshold=period_params['gap_threshold'],
                lookback_window=period_params['lookback_window']
            )
            decline_idx = find_decline_start_last_major_drop(
                df_well_sorted,
                smooth_window=decline_params['smooth_window'],
                polyorder=decline_params['polyorder'],
                min_decline_months=decline_params['min_decline_months'],
                drop_frac=decline_params['drop_frac']
            )
            
            rate_series = df_well_sorted['M_Oil_Prod'].values
            if decline_idx < len(rate_series):
                decline_rate = rate_series[decline_idx]
                future_max = np.max(rate_series[decline_idx:])
                if decline_rate < 0.8 * future_max:
                    chosen_idx = auto_idx
                    method_label = "Auto-Detected Recent Period (fallback)"
                else:
                    chosen_idx = decline_idx
                    method_label = "Smoothing Decline Analysis"
            else:
                chosen_idx = auto_idx
                method_label = "Auto-Detected Recent Period (fallback)"
            
            manual_start_idx = chosen_idx
            start_method = method_label
        
        # Run the analysis
        fig, results = run_arps_for_well_auto(
            self.df_all, well_name, outlier_threshold=outlier_threshold,
            forecast_avg_points=forecast_avg_points, manual_start_idx=manual_start_idx,
            detect_period=detect_period, start_method=start_method,
            decline_params=decline_params, period_params=period_params,
            filter_params=filter_params
        )
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)
        
        if fig:
            # Clear previous plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            if self.toolbar:
                self.toolbar.destroy()
            
            self.figure = fig
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
            self.toolbar.update()
            self.toolbar.pack(fill=tk.X)
        else:
            messagebox.showerror("Error", results)

if __name__ == "__main__":
    app = DeclineCurveApp()
    app.mainloop()