
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
import random

# Parameter ranges
filter_ranges = {
    'threshold': (1.5, 3.5),
    'max_iterations': (3, 10)
}

decline_ranges = {
    'smooth_window': (5, 25),
    'polyorder': (1, 3),
    'peak_prominence': (0.05, 0.3),
    'post_ma': (6, 18),
    'decline_frac': (0.05, 0.3),
    'post_slope_win': (6, 24),
    'persist_months': (3, 12)
}

period_ranges = {
    'min_production_months': (6, 24),
    'surge_multiplier': (1.5, 4.0),
    'gap_threshold': (0.1, 0.5),
    'lookback_window': (3, 12)
}

int_keys = {
    'smooth_window', 'polyorder', 'post_ma', 'post_slope_win', 'persist_months',
    'min_production_months', 'lookback_window', 'max_iterations'
}

def sample_param(range_val, is_int=False):
    if isinstance(range_val, list):
        v = random.choice(range_val)
    else:
        v = random.uniform(range_val[0], range_val[1])
    if is_int:
        v = int(round(v))
    return v

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
    mask = np.ones(len(t), dtype=bool)
    
    for iteration in range(max_iterations):
        t_fit = t[mask]
        q_fit = q[mask]
        
        popt, pcov = fit_arps_hyperbolic(t_fit, q_fit)
        if popt is None:
            print(f"Fitting failed at iteration {iteration}")
            break
            
        q_pred = arps_hyperbolic(t, *popt)
        residuals = q - q_pred
        
        residuals_fit = residuals[mask]
        std_residuals = np.std(residuals_fit)
        
        new_mask = np.abs(residuals) <= threshold * std_residuals
        
        if np.array_equal(mask, new_mask):
            print(f"Converged after {iteration + 1} iterations")
            break
            
        n_removed = np.sum(mask) - np.sum(new_mask)
        print(f"Iteration {iteration + 1}: Removed {n_removed} outliers, std = {std_residuals:.2f}")
        mask = new_mask
    
    popt, pcov = fit_arps_hyperbolic(t[mask], q[mask])
    
    return mask, popt

def create_arps_plot(df_well, t, q, mask, popt, well_id, df_full, q_original, q_smoothed, outlier_threshold, forecast_avg_points=1, start_method=""):
    if popt is None:
        print("No fit available to plot.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    q_pred = arps_hyperbolic(t, *popt)
    
    residuals = q - q_pred
    avg_std = np.std(residuals)
    channel_width = avg_std 
    
    upper_bound = q_pred + channel_width
    lower_bound = q_pred - channel_width
    
    lower_bound = np.maximum(lower_bound, 0)
    
    last_date = df_well['Prod_Date'].max()
    forecast_dates = pd.date_range(start=last_date, periods=61, freq='ME')  
    
    last_t = t[-1]
    forecast_t = np.arange(last_t, last_t + 61)
    
    forecast_q = arps_hyperbolic(forecast_t, *popt)
    
    if forecast_avg_points > 1:
        n_points = min(forecast_avg_points, np.sum(mask))
        initial_rate = np.mean(q[mask][-n_points:])
        print(f"\nUsing average of last {n_points} rates as initial forecast rate: {initial_rate:.2f} bbl/day")
    elif forecast_avg_points == 0:
        initial_rate = forecast_q[0]  
        print(f"\nUsing last fitted model point as initial forecast rate: {initial_rate:.2f} bbl/day")
    else:
        initial_rate = q[mask][-1]
        print(f"\nUsing last historical rate as initial forecast rate: {initial_rate:.2f} bbl/day")
    
    if forecast_avg_points != 0:  
        scaling_factor = initial_rate / forecast_q[0]
        forecast_q = forecast_q * scaling_factor
    
    ax.plot(df_full['Prod_Date'], q_original, 'o', color='red', 
             label='Original Data', alpha=0.6, markersize=5)
    
    ax.plot(df_full['Prod_Date'], q_smoothed, '--', color='red', 
             label='Smoothed Data', linewidth=2)
    
    ax.plot(df_well['Prod_Date'][mask], q[mask], 'o', color='blue', 
             label='Decline Phase Data (Inliers)', markersize=6)
    
    if np.sum(~mask) > 0:
        ax.plot(df_well['Prod_Date'][~mask], q[~mask], 'x', color='red', 
                 label='Outliers (Excluded)', markersize=10, markeredgewidth=2)
    
    ax.fill_between(df_well['Prod_Date'], lower_bound, upper_bound,
                    color='green', alpha=0.2, label=f'±{channel_width:.1f} bbl/day Channel')
    
    ax.plot(df_well['Prod_Date'], q_pred, '-', color='green', 
             label='Arps Hyperbolic Model', linewidth=2)
    
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
    ax.set_ylim(bottom=0)  
    
    return fig

# --- Helper functions ---
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

def compute_metrics(df_all, well_name, manual_start_idx=0, filter_params={}):
    df_well = df_all[df_all['Well_Name'] == well_name].copy()
    if df_well.empty:
        return {'score': -np.inf}
    
    df_well = df_well.sort_values('Prod_Date')
    
    start_idx = manual_start_idx
    
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
        threshold=filter_params.get('threshold', 2.0),
        max_iterations=filter_params.get('max_iterations', 5)
    )
    
    if popt is None:
        return {'score': -np.inf}
    
    num_used = np.sum(mask)
    mean_q = np.mean(q[mask])
    cv = np.std(q[mask]) / mean_q if mean_q > 0 else np.inf
    score = num_used / (1 + cv)
    
    return {'score': score, 'cv': cv, 'num_used': num_used}

# Modified run_arps_for_well_auto
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
            rate_col='M_Oil_Prod',
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
        print(f"\n{'='*60}")
        print(f"Analysis starting from: {df_well.iloc[0]['Prod_Date'].date()}")
        print(f"Data points used: {len(df_well)}")
        print(f"{'='*60}\n")

    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    
    print("--- Starting iterative outlier filtering ---")
    mask, popt = filter_outliers_iterative(
        t, q, 
        threshold=filter_params.get('threshold', outlier_threshold),
        max_iterations=filter_params.get('max_iterations', 5)
    )
    
    if popt is not None:
        n_outliers = np.sum(~mask)
        print(f"\n=== Final Results for well {well_name} ===")
        print(f"Total outliers removed: {n_outliers} out of {len(mask)} points")
        print(f"Fitted parameters:")
        print(f"qi = {popt[0]:.2f}, Di = {popt[1]:.4f}, b = {popt[2]:.2f}")
        
        cv = np.std(q[mask]) / np.mean(q[mask])
        print(f"\nCoefficient of Variation: {cv:.3f}")
        if cv > 0.4:
            print("⚠ WARNING: High variability detected. Forecast may be unreliable.")
        elif cv > 0.3:
            print("⚠ CAUTION: Moderate variability. Review forecast carefully.")
        else:
            print("✓ Good data quality for decline curve analysis.")
    
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
        
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background='#f0f0f0', foreground='#333333')
        style.configure('TLabel', background='#f0f0f0')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TNotebook.Tab', background='#dddddd', foreground='#333333')
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')])
        
        try:
            self.df_all = pd.read_csv('OFM202409.csv', low_memory=False)
            self.wells = sorted(self.df_all['Well_Name'].unique())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.destroy()
            return
        
        self.left_frame = ttk.Frame(self, width=300, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(self.left_frame, text="Select Well:").pack(anchor=tk.W, pady=5)
        self.well_var = tk.StringVar()
        self.well_combo = ttk.Combobox(self.left_frame, textvariable=self.well_var, values=self.wells, state='readonly')
        self.well_combo.pack(fill=tk.X, pady=5)
        self.well_combo.bind('<<ComboboxSelected>>', self.on_well_select)
        
        ttk.Label(self.left_frame, text="Start Index Method:").pack(anchor=tk.W, pady=5)
        self.start_method_var = tk.StringVar(value="Auto Select")
        methods = ["Auto Select", "Auto-Detected Recent Period", "Smoothing Decline Analysis", "Manual"]
        for m in methods:
            ttk.Radiobutton(self.left_frame, text=m, variable=self.start_method_var, value=m).pack(anchor=tk.W)
        
        self.manual_date_frame = ttk.Frame(self.left_frame)
        self.manual_date_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.manual_date_frame, text="Manual Start Year:").pack(side=tk.LEFT)
        self.manual_year_var = tk.IntVar(value=2020)
        ttk.Entry(self.manual_date_frame, textvariable=self.manual_year_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.manual_date_frame, text="Month:").pack(side=tk.LEFT)
        self.manual_month_var = tk.IntVar(value=1)
        ttk.Entry(self.manual_date_frame, textvariable=self.manual_month_var, width=3).pack(side=tk.LEFT, padx=5)
        
        self.settings_notebook = ttk.Notebook(self.left_frame)
        self.settings_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
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
        
        self.run_btn = ttk.Button(self.left_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=10)
        
        self.results_text = tk.Text(self.left_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, pady=5)
        
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = None
        self.canvas = None
        self.toolbar = None
    
    def on_well_select(self, event):
        pass
    
    def run_analysis(self):
        well_name = self.well_var.get()
        if not well_name:
            messagebox.showwarning("Warning", "Please select a well")
            return
        
        start_method = self.start_method_var.get()
        manual_start_idx = None
        method_label = start_method
        
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
        
        outlier_threshold = self.outlier_threshold_var.get()
        forecast_avg_points = self.forecast_avg_points_var.get()
        
        # Optimize parameters
        if start_method == "Auto Select":
            relevant_ranges = {**decline_ranges, **period_ranges, **filter_ranges}
            n_trials = 50
        elif start_method == "Auto-Detected Recent Period":
            relevant_ranges = {**period_ranges, **filter_ranges}
            n_trials = 50
        elif start_method == "Smoothing Decline Analysis":
            relevant_ranges = {**decline_ranges, **filter_ranges}
            n_trials = 50
        elif start_method == "Manual":
            relevant_ranges = filter_ranges
            n_trials = 20
        else:
            relevant_ranges = {}
            n_trials = 0
        
        best_score = -np.inf
        best_sampled = None
        
        for _ in range(n_trials):
            sampled = {param: sample_param(r, param in int_keys) for param, r in relevant_ranges.items()}
            
            if start_method == "Auto Select":
                sampled_decline = {k: sampled[k] for k in decline_ranges}
                sampled_period = {k: sampled[k] for k in period_ranges}
                sampled_filter = {k: sampled[k] for k in filter_ranges}
                auto_idx = find_last_production_period(df_well_sorted, **sampled_period)
                decline_idx = find_decline_start(df_well_sorted, **sampled_decline)
            elif start_method == "Auto-Detected Recent Period":
                sampled_period = {k: sampled[k] for k in period_ranges}
                sampled_filter = {k: sampled[k] for k in filter_ranges}
                start_idx = find_last_production_period(df_well_sorted, **sampled_period)
            elif start_method == "Smoothing Decline Analysis":
                sampled_decline = {k: sampled[k] for k in decline_ranges}
                sampled_filter = {k: sampled[k] for k in filter_ranges}
                start_idx = find_decline_start(df_well_sorted, **sampled_decline)
            elif start_method == "Manual":
                sampled_filter = {k: sampled[k] for k in filter_ranges}
                start_idx = manual_start_idx
            
            if start_method == "Auto Select":
                rate_series = df_well_sorted['M_Oil_Prod'].values
                if decline_idx < len(rate_series):
                    decline_rate = rate_series[decline_idx]
                    future_max = np.max(rate_series[decline_idx:])
                    if decline_rate < 0.8 * future_max:
                        chosen_idx = auto_idx
                        temp_label = "Auto-Detected Recent Period (fallback)"
                    else:
                        chosen_idx = decline_idx
                        temp_label = "Smoothing Decline Analysis"
                else:
                    chosen_idx = auto_idx
                    temp_label = "Auto-Detected Recent Period (fallback)"
                metrics = compute_metrics(self.df_all, well_name, manual_start_idx=chosen_idx, filter_params=sampled_filter)
            else:
                metrics = compute_metrics(self.df_all, well_name, manual_start_idx=start_idx, filter_params=sampled_filter)
            
            score = metrics['score']
            if score > best_score:
                best_score = score
                best_sampled = sampled
                if start_method == "Auto Select":
                    best_chosen_idx = chosen_idx
                    method_label = temp_label
                else:
                    best_start_idx = start_idx
        
        # Run final with best params
        if best_sampled is None:
            # No optimization, use defaults
            decline_params = {
                'smooth_window': self.smooth_window_var.get(),
                'polyorder': self.polyorder_var.get(),
                'peak_prominence': self.peak_prominence_var.get(),
                'post_ma': self.post_ma_var.get(),
                'decline_frac': self.decline_frac_var.get(),
                'post_slope_win': self.post_slope_win_var.get(),
                'persist_months': self.persist_months_var.get()
            }
            period_params = {
                'min_production_months': self.min_prod_months_var.get(),
                'surge_multiplier': self.surge_mult_var.get(),
                'gap_threshold': self.gap_thresh_var.get(),
                'lookback_window': self.lookback_win_var.get()
            }
            filter_params = {
                'threshold': self.outlier_threshold_var.get(),
                'max_iterations': self.filter_max_iter_var.get()
            }
            if start_method == "Auto Select":
                auto_idx = find_last_production_period(df_well_sorted, **period_params)
                decline_idx = find_decline_start(df_well_sorted, **decline_params)
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
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=chosen_idx,
                    detect_period=False, start_method=method_label,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params
                )
            elif start_method == "Auto-Detected Recent Period":
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=None,
                    detect_period=True, start_method=start_method,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params
                )
            elif start_method == "Smoothing Decline Analysis":
                start_idx = find_decline_start(df_well_sorted, **decline_params)
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=start_idx,
                    detect_period=False, start_method=start_method,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params
                )
            elif start_method == "Manual":
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=manual_start_idx,
                    detect_period=False, start_method=start_method,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params
                )
        else:
            # Use best
            results_extra = f"\nBest parameters found: {best_sampled}"
            if start_method == "Auto Select":
                best_decline = {k: best_sampled[k] for k in decline_ranges}
                best_period = {k: best_sampled[k] for k in period_ranges}
                best_filter = {k: best_sampled[k] for k in filter_ranges}
                auto_idx = find_last_production_period(df_well_sorted, **best_period)
                decline_idx = find_decline_start(df_well_sorted, **best_decline)
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
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=chosen_idx,
                    detect_period=False, start_method=method_label,
                    decline_params=best_decline, period_params=best_period,
                    filter_params=best_filter
                )
            elif start_method == "Auto-Detected Recent Period":
                best_period = {k: best_sampled[k] for k in period_ranges}
                best_filter = {k: best_sampled[k] for k in filter_ranges}
                start_idx = find_last_production_period(df_well_sorted, **best_period)
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=start_idx,
                    detect_period=False, start_method=start_method,
                    decline_params={}, period_params=best_period,
                    filter_params=best_filter
                )
            elif start_method == "Smoothing Decline Analysis":
                best_decline = {k: best_sampled[k] for k in decline_ranges}
                best_filter = {k: best_sampled[k] for k in filter_ranges}
                start_idx = find_decline_start(df_well_sorted, **best_decline)
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=start_idx,
                    detect_period=False, start_method=start_method,
                    decline_params=best_decline, period_params={},
                    filter_params=best_filter
                )
            elif start_method == "Manual":
                best_filter = {k: best_sampled[k] for k in filter_ranges}
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=manual_start_idx,
                    detect_period=False, start_method=start_method,
                    decline_params={}, period_params={},
                    filter_params=best_filter
                )
            results += results_extra
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)
        
        if fig:
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