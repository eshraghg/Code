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
from bayes_opt import BayesianOptimization

# Default parameter ranges for optimization
DEFAULT_FILTER_RANGES = {
    'threshold': (1.5, 3.5)
}

# Fixed constant for outlier iterations
MAX_OUTLIER_ITERATIONS = 10

DEFAULT_DECLINE_RANGES = {
    'smooth_window': (5, 25),
    'polyorder': (1, 3),
    'peak_prominence': (0.05, 0.3),
    'post_ma': (6, 18),
    'decline_frac': (0.05, 0.3),
    'post_slope_win': (6, 24),
    'persist_months': (3, 12)
}

DEFAULT_PERIOD_RANGES = {
    'min_production_months': (6, 24),
    'surge_multiplier': (1.5, 4.0),
    'gap_threshold': (0.1, 0.5),
    'lookback_window': (3, 12)
}

int_keys = {
    'smooth_window', 'polyorder', 'post_ma', 'post_slope_win', 'persist_months',
    'min_production_months', 'lookback_window'
}

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

def filter_outliers_iterative(t, q, threshold=3, max_iterations=MAX_OUTLIER_ITERATIONS):
    mask = np.ones(len(t), dtype=bool)
    n = len(t)
    
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
        
        last_non_outlier_idx = -1
        for i in range(n - 1, -1, -1):
            if new_mask[i]:
                last_non_outlier_idx = i
                break
        
        if last_non_outlier_idx >= 0 and last_non_outlier_idx < n - 1:
            has_trailing_outliers = not new_mask[n - 1]
            if has_trailing_outliers:
                for i in range(last_non_outlier_idx + 1, n):
                    new_mask[i] = True
                print(f"Iteration {iteration + 1}: Protected {n - last_non_outlier_idx - 1} trailing data points from outlier removal")
        
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
    forecast_months = 60
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    
    last_t = t[-1]
    forecast_t = last_t + np.arange(1, forecast_months + 1)
    
    qi_original, Di, b = popt
    
    forecast_q = arps_hyperbolic(forecast_t, qi_original, Di, b)
    
    if forecast_avg_points > 1:
        n_points = min(forecast_avg_points, np.sum(mask))
        initial_rate = np.mean(q[mask][-n_points:])
        fitted_rate_at_last_t = arps_hyperbolic(last_t, qi_original, Di, b)
        print(f"\nUsing average of last {n_points} rates: {initial_rate:.2f} bbl/day")
        print(f"Fitted model rate at last point: {fitted_rate_at_last_t:.2f} bbl/day")
        
        rate_difference = initial_rate - fitted_rate_at_last_t
        forecast_q += rate_difference
        
        forecast_q = np.maximum(forecast_q, 0)
        
        print(f"Applied vertical shift of {rate_difference:.2f} bbl/day to match initial rate")
        
    elif forecast_avg_points == 0:
        initial_rate = arps_hyperbolic(last_t, qi_original, Di, b)
        print(f"\nUsing fitted model forecast (no adjustment)")
        print(f"Initial forecast rate: {initial_rate:.2f} bbl/day")
        
    else:
        initial_rate = q[mask][-1]
        fitted_rate_at_last_t = arps_hyperbolic(last_t, qi_original, Di, b)
        print(f"\nUsing last historical rate: {initial_rate:.2f} bbl/day")
        print(f"Fitted model rate at last point: {fitted_rate_at_last_t:.2f} bbl/day")
        
        rate_difference = initial_rate - fitted_rate_at_last_t
        forecast_q += rate_difference
        
        forecast_q = np.maximum(forecast_q, 0)
        
        print(f"Applied vertical shift of {rate_difference:.2f} bbl/day to match last historical rate")
    
    print(f"Forecast parameters: qi={qi_original:.2f}, Di={Di:.4f}, b={b:.2f}")
    
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

def calculate_fit_quality(t, q, mask, popt):
    if popt is None:
        return 0
    
    q_pred = arps_hyperbolic(t[mask], *popt)
    q_actual = q[mask]
    
    ss_res = np.sum((q_actual - q_pred) ** 2)
    ss_tot = np.sum((q_actual - np.mean(q_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((q_actual - q_pred) / q_actual)) * 100 if np.all(q_actual > 0) else float('inf')
    mape_score = max(0, 1 - min(mape, 100) / 100)
    
    qi, Di, b = popt
    decline_reasonableness = 1.0
    if Di > 2.0 or Di < 0.01:
        decline_reasonableness = 0.1
    elif b > 1.5:
        decline_reasonableness = 0.7
    
    return 0.5 * r_squared + 0.3 * mape_score + 0.2 * decline_reasonableness

def calculate_trend_consistency(t, q, mask, popt):
    if popt is None:
        return 0
    
    q_actual = q[mask]
    t_actual = t[mask]
    
    if len(q_actual) >= 6:
        recent_q = q_actual[-6:]
        recent_t = t_actual[-6:]
        actual_slope = linregress(recent_t, recent_q).slope
    else:
        actual_slope = linregress(t_actual, q_actual).slope
    
    q_pred = arps_hyperbolic(t_actual, *popt)
    expected_slope = linregress(t_actual, q_pred).slope
    
    slope_ratio = abs(actual_slope / expected_slope) if expected_slope != 0 else 1
    consistency = 1.0 / (1.0 + abs(slope_ratio - 1.0))
    
    return consistency

def compute_metrics(df_all, well_name, manual_start_idx=0, filter_params=None, decline_params=None, period_params=None):
    df_well = df_all[df_all['Well_Name'] == well_name].copy()
    if df_well.empty:
        return {'score': -np.inf}
    
    df_well = df_well.sort_values('Prod_Date')
    start_idx = manual_start_idx
    
    df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
    df_well['days_in_month'] = df_well['Prod_Date'].apply(get_days_in_month)
    df_well['oil_prod_daily'] = df_well['M_Oil_Prod'] / df_well['days_in_month']
    df_well['t'] = (df_well['Prod_Date'] - df_well['Prod_Date'].min()).dt.days // 30

    if start_idx > 0:
        df_well = df_well.iloc[start_idx:].reset_index(drop=True)

    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    
    mask, popt = filter_outliers_iterative(
        t, q, 
        threshold=filter_params.get('threshold', 2.0)
    )
    
    if popt is None:
        return {'score': -np.inf}
    
    num_used = np.sum(mask)
    total_points = len(mask)
    mean_q = np.mean(q[mask])
    cv = np.std(q[mask]) / mean_q if mean_q > 0 else np.inf
    
    base_score = num_used / (1 + cv*4)
    
    data_utilization = num_used / total_points
    fit_quality = calculate_fit_quality(t, q, mask, popt)
    trend_consistency = calculate_trend_consistency(t, q, mask, popt)
    
    enhanced_score = (
        0.3 * base_score +
        0.2 * data_utilization +
        0.3 * fit_quality +
        0.2 * trend_consistency
    )
    
    return {
        'score': enhanced_score,
        'base_score': base_score,
        'cv': cv,
        'num_used': num_used,
        'total_points': total_points,
        'data_utilization': data_utilization,
        'fit_quality': fit_quality,
        'trend_consistency': trend_consistency,
        'popt': popt,
        'mask': mask
    }

def run_arps_for_well_auto(df_all, well_name, outlier_threshold=2, 
                           forecast_avg_points=6, manual_start_idx=None,
                           detect_period=True, start_method="", 
                           decline_params=None, period_params=None, filter_params=None):
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
        threshold=filter_params.get('threshold', outlier_threshold)
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

class DeclineCurveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Decline Curve Analysis Tool")
        self.geometry("1400x900")
        
        self.filter_ranges = DEFAULT_FILTER_RANGES.copy()
        self.decline_ranges = DEFAULT_DECLINE_RANGES.copy()
        self.period_ranges = DEFAULT_PERIOD_RANGES.copy()
        
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
            
            self.districts = sorted([d for d in self.df_all['District'].dropna().unique() if str(d).strip()])
            self.fields_by_district = {}
            self.wells_by_field = {}
            
            for district in self.districts:
                district_data = self.df_all[self.df_all['District'] == district]
                fields = sorted([f for f in district_data['Field'].dropna().unique() if str(f).strip()])
                self.fields_by_district[district] = fields
                
                for field in fields:
                    field_data = district_data[district_data['Field'] == field]
                    wells = sorted([w for w in field_data['Well_Name'].dropna().unique() if str(w).strip()])
                    self.wells_by_field[f"{district}|{field}"] = wells
            
            self.current_district = self.districts[0] if self.districts else None
            self.current_field = self.fields_by_district.get(self.current_district, [None])[0] if self.current_district else None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.destroy()
            return
        
        self.left_frame = ttk.Frame(self, width=350, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(self.left_frame, text="Select District:").pack(anchor=tk.W, pady=5)
        self.district_var = tk.StringVar(value=self.current_district if self.current_district else "")
        self.district_combo = ttk.Combobox(self.left_frame, textvariable=self.district_var, 
                                          values=self.districts, state='readonly')
        self.district_combo.pack(fill=tk.X, pady=5)
        self.district_combo.bind('<<ComboboxSelected>>', self.on_district_select)
        
        ttk.Label(self.left_frame, text="Select Field:").pack(anchor=tk.W, pady=5)
        self.field_var = tk.StringVar(value=self.current_field if self.current_field else "")
        initial_fields = self.fields_by_district.get(self.current_district, []) if self.current_district else []
        self.field_combo = ttk.Combobox(self.left_frame, textvariable=self.field_var, 
                                       values=initial_fields, state='readonly')
        self.field_combo.pack(fill=tk.X, pady=5)
        self.field_combo.bind('<<ComboboxSelected>>', self.on_field_select)
        
        ttk.Label(self.left_frame, text="Select Well:").pack(anchor=tk.W, pady=5)
        self.well_var = tk.StringVar()
        initial_wells = self.wells_by_field.get(f"{self.current_district}|{self.current_field}", []) if self.current_district and self.current_field else []
        self.well_combo = ttk.Combobox(self.left_frame, textvariable=self.well_var, 
                                      values=initial_wells, state='readonly')
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
        
        self.optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.left_frame, text="Optimize Parameters", variable=self.optimize_var).pack(anchor=tk.W, pady=5)
        
        ttk.Button(self.left_frame, text="Configure Optimization Ranges", command=self.open_range_config).pack(pady=5)
        
        self.settings_notebook = ttk.Notebook(self.left_frame)
        self.settings_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.create_general_tab()
        self.create_decline_tab()
        self.create_period_tab()
        
        self.run_btn = ttk.Button(self.left_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=10)
        
        self.results_text = tk.Text(self.left_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, pady=5)
        
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = None
        self.canvas = None
        self.toolbar = None
    
    def create_general_tab(self):
        general_tab = ttk.Frame(self.settings_notebook, padding=5)
        self.settings_notebook.add(general_tab, text="General")
        
        self.threshold_var = tk.DoubleVar(value=2.0)
        self.threshold_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_tab, text="", variable=self.threshold_opt_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(general_tab, text="Outlier Threshold:").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(general_tab, textvariable=self.threshold_var, width=10).grid(row=0, column=2, pady=2)
        
        self.forecast_avg_points_var = tk.IntVar(value=2)
        ttk.Label(general_tab, text="Forecast Avg Points:").grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(general_tab, textvariable=self.forecast_avg_points_var, width=10).grid(row=1, column=2, pady=2)
    
    def create_decline_tab(self):
        decline_tab = ttk.Frame(self.settings_notebook, padding=5)
        self.settings_notebook.add(decline_tab, text="Decline Detection")
        
        self.smooth_window_var = tk.IntVar(value=13)
        self.smooth_window_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.smooth_window_opt_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Smooth Window:").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.smooth_window_var, width=10).grid(row=0, column=2, pady=2)
        
        self.polyorder_var = tk.IntVar(value=2)
        self.polyorder_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.polyorder_opt_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Polyorder:").grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.polyorder_var, width=10).grid(row=1, column=2, pady=2)
        
        self.peak_prominence_var = tk.DoubleVar(value=0.15)
        self.peak_prominence_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.peak_prominence_opt_var).grid(row=2, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Peak Prominence:").grid(row=2, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.peak_prominence_var, width=10).grid(row=2, column=2, pady=2)
        
        self.post_ma_var = tk.IntVar(value=12)
        self.post_ma_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.post_ma_opt_var).grid(row=3, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Post MA:").grid(row=3, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.post_ma_var, width=10).grid(row=3, column=2, pady=2)
        
        self.decline_frac_var = tk.DoubleVar(value=0.15)
        self.decline_frac_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.decline_frac_opt_var).grid(row=4, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Decline Frac:").grid(row=4, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.decline_frac_var, width=10).grid(row=4, column=2, pady=2)
        
        self.post_slope_win_var = tk.IntVar(value=12)
        self.post_slope_win_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.post_slope_win_opt_var).grid(row=5, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Post Slope Win:").grid(row=5, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.post_slope_win_var, width=10).grid(row=5, column=2, pady=2)
        
        self.persist_months_var = tk.IntVar(value=9)
        self.persist_months_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(decline_tab, text="", variable=self.persist_months_opt_var).grid(row=6, column=0, sticky=tk.W)
        ttk.Label(decline_tab, text="Persist Months:").grid(row=6, column=1, sticky=tk.W)
        ttk.Entry(decline_tab, textvariable=self.persist_months_var, width=10).grid(row=6, column=2, pady=2)
    
    def create_period_tab(self):
        period_tab = ttk.Frame(self.settings_notebook, padding=5)
        self.settings_notebook.add(period_tab, text="Period Detection")
        
        self.min_production_months_var = tk.IntVar(value=12)
        self.min_production_months_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(period_tab, text="", variable=self.min_production_months_opt_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(period_tab, text="Min Production Months:").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.min_production_months_var, width=10).grid(row=0, column=2, pady=2)
        
        self.surge_multiplier_var = tk.DoubleVar(value=2.0)
        self.surge_multiplier_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(period_tab, text="", variable=self.surge_multiplier_opt_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(period_tab, text="Surge Multiplier:").grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.surge_multiplier_var, width=10).grid(row=1, column=2, pady=2)
        
        self.gap_threshold_var = tk.DoubleVar(value=0.3)
        self.gap_threshold_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(period_tab, text="", variable=self.gap_threshold_opt_var).grid(row=2, column=0, sticky=tk.W)
        ttk.Label(period_tab, text="Gap Threshold:").grid(row=2, column=1, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.gap_threshold_var, width=10).grid(row=2, column=2, pady=2)
        
        self.lookback_window_var = tk.IntVar(value=6)
        self.lookback_window_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(period_tab, text="", variable=self.lookback_window_opt_var).grid(row=3, column=0, sticky=tk.W)
        ttk.Label(period_tab, text="Lookback Window:").grid(row=3, column=1, sticky=tk.W)
        ttk.Entry(period_tab, textvariable=self.lookback_window_var, width=10).grid(row=3, column=2, pady=2)
    
    def open_range_config(self):
        config_window = tk.Toplevel(self)
        config_window.title("Configure Optimization Ranges")
        config_window.geometry("600x700")
        
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        filter_frame = ttk.Frame(notebook, padding=10)
        notebook.add(filter_frame, text="Filter Ranges")
        
        filter_vars = {}
        row = 0
        for param, (min_val, max_val) in self.filter_ranges.items():
            ttk.Label(filter_frame, text=f"{param}:").grid(row=row, column=0, sticky=tk.W, pady=5)
            ttk.Label(filter_frame, text="Min:").grid(row=row, column=1, padx=5)
            min_var = tk.DoubleVar(value=min_val) if param not in int_keys else tk.IntVar(value=int(min_val))
            ttk.Entry(filter_frame, textvariable=min_var, width=10).grid(row=row, column=2, padx=5)
            ttk.Label(filter_frame, text="Max:").grid(row=row, column=3, padx=5)
            max_var = tk.DoubleVar(value=max_val) if param not in int_keys else tk.IntVar(value=int(max_val))
            ttk.Entry(filter_frame, textvariable=max_var, width=10).grid(row=row, column=4, padx=5)
            filter_vars[param] = (min_var, max_var)
            row += 1
        
        decline_frame = ttk.Frame(notebook, padding=10)
        notebook.add(decline_frame, text="Decline Ranges")
        
        decline_vars = {}
        row = 0
        for param, (min_val, max_val) in self.decline_ranges.items():
            ttk.Label(decline_frame, text=f"{param}:").grid(row=row, column=0, sticky=tk.W, pady=5)
            ttk.Label(decline_frame, text="Min:").grid(row=row, column=1, padx=5)
            min_var = tk.DoubleVar(value=min_val) if param not in int_keys else tk.IntVar(value=int(min_val))
            ttk.Entry(decline_frame, textvariable=min_var, width=10).grid(row=row, column=2, padx=5)
            ttk.Label(decline_frame, text="Max:").grid(row=row, column=3, padx=5)
            max_var = tk.DoubleVar(value=max_val) if param not in int_keys else tk.IntVar(value=int(max_val))
            ttk.Entry(decline_frame, textvariable=max_var, width=10).grid(row=row, column=4, padx=5)
            decline_vars[param] = (min_var, max_var)
            row += 1
        
        period_frame = ttk.Frame(notebook, padding=10)
        notebook.add(period_frame, text="Period Ranges")
        
        period_vars = {}
        row = 0
        for param, (min_val, max_val) in self.period_ranges.items():
            ttk.Label(period_frame, text=f"{param}:").grid(row=row, column=0, sticky=tk.W, pady=5)
            ttk.Label(period_frame, text="Min:").grid(row=row, column=1, padx=5)
            min_var = tk.DoubleVar(value=min_val) if param not in int_keys else tk.IntVar(value=int(min_val))
            ttk.Entry(period_frame, textvariable=min_var, width=10).grid(row=row, column=2, padx=5)
            ttk.Label(period_frame, text="Max:").grid(row=row, column=3, padx=5)
            max_var = tk.DoubleVar(value=max_val) if param not in int_keys else tk.IntVar(value=int(max_val))
            ttk.Entry(period_frame, textvariable=max_var, width=10).grid(row=row, column=4, padx=5)
            period_vars[param] = (min_var, max_var)
            row += 1
        
        button_frame = ttk.Frame(config_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_ranges():
            try:
                for param, (min_var, max_var) in filter_vars.items():
                    self.filter_ranges[param] = (min_var.get(), max_var.get())
                for param, (min_var, max_var) in decline_vars.items():
                    self.decline_ranges[param] = (min_var.get(), max_var.get())
                for param, (min_var, max_var) in period_vars.items():
                    self.period_ranges[param] = (min_var.get(), max_var.get())
                messagebox.showinfo("Success", "Optimization ranges updated successfully!")
                config_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update ranges: {str(e)}")
        
        def reset_ranges():
            for param, (min_val, max_val) in DEFAULT_FILTER_RANGES.items():
                filter_vars[param][0].set(min_val)
                filter_vars[param][1].set(max_val)
            for param, (min_val, max_val) in DEFAULT_DECLINE_RANGES.items():
                decline_vars[param][0].set(min_val)
                decline_vars[param][1].set(max_val)
            for param, (min_val, max_val) in DEFAULT_PERIOD_RANGES.items():
                period_vars[param][0].set(min_val)
                period_vars[param][1].set(max_val)
        
        ttk.Button(button_frame, text="Save", command=save_ranges).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=reset_ranges).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=config_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def on_district_select(self, event):
        district = self.district_var.get()
        if district:
            self.current_district = district
            fields = self.fields_by_district.get(district, [])
            self.field_combo['values'] = fields
            if fields:
                self.field_var.set(fields[0])
                self.current_field = fields[0]
                self.on_field_select(None)
            else:
                self.field_var.set("")
                self.well_combo['values'] = []
                self.well_var.set("")
    
    def on_field_select(self, event):
        district = self.district_var.get()
        field = self.field_var.get()
        if district and field:
            self.current_field = field
            wells = self.wells_by_field.get(f"{district}|{field}", [])
            self.well_combo['values'] = wells
            if wells:
                self.well_var.set(wells[0])
            else:
                self.well_var.set("")
    
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
        
        outlier_threshold = self.threshold_var.get()
        forecast_avg_points = self.forecast_avg_points_var.get()
        
        filter_params = {
            'threshold': self.threshold_var.get()
        }
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
            'min_production_months': self.min_production_months_var.get(),
            'surge_multiplier': self.surge_multiplier_var.get(),
            'gap_threshold': self.gap_threshold_var.get(),
            'lookback_window': self.lookback_window_var.get()
        }
        
        optimize = self.optimize_var.get()
        
        if optimize:
            pbounds = {}
            fixed_params = {}
            
            param_to_opt_var = {
                'threshold': 'threshold_opt_var',
                'smooth_window': 'smooth_window_opt_var',
                'polyorder': 'polyorder_opt_var',
                'peak_prominence': 'peak_prominence_opt_var',
                'post_ma': 'post_ma_opt_var',
                'decline_frac': 'decline_frac_opt_var',
                'post_slope_win': 'post_slope_win_opt_var',
                'persist_months': 'persist_months_opt_var',
                'min_production_months': 'min_production_months_opt_var',
                'surge_multiplier': 'surge_multiplier_opt_var',
                'gap_threshold': 'gap_threshold_opt_var',
                'lookback_window': 'lookback_window_opt_var'
            }
            
            fixed_params.update(filter_params)
            fixed_params.update(decline_params)
            fixed_params.update(period_params)
            
            # Combine all parameter ranges
            relevant_ranges = {
                **self.filter_ranges,
                **self.decline_ranges,
                **self.period_ranges
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
                n_iter = 15
            elif start_method == "Auto-Detected Recent Period":
                chosen_idx = find_last_production_period(df_well_sorted, **period_params)
                n_iter = 15
            elif start_method == "Smoothing Decline Analysis":
                chosen_idx = find_decline_start(df_well_sorted, **decline_params)
                n_iter = 15
            elif start_method == "Manual":
                chosen_idx = manual_start_idx
                n_iter = 10
            else:
                chosen_idx = 0
                n_iter = 0
            
            for param in relevant_ranges:
                try:
                    opt_var = param_to_opt_var.get(param)
                    if not opt_var:
                        continue
                    if hasattr(self, opt_var) and getattr(self, opt_var).get():
                        pbounds[param] = relevant_ranges[param]
                except AttributeError:
                    pass
            
            if pbounds:
                def objective_function(**params):
                    for param in int_keys:
                        if param in params:
                            params[param] = int(round(params[param]))
                    
                    sampled_filter = {k: params.get(k, fixed_params[k]) for k in self.filter_ranges}
                    sampled_decline = {k: params.get(k, fixed_params[k]) for k in self.decline_ranges}
                    sampled_period = {k: params.get(k, fixed_params[k]) for k in self.period_ranges}
                    
                    # Determine start_idx based on start_method
                    if start_method == "Auto Select":
                        auto_idx = find_last_production_period(df_well_sorted, **sampled_period)
                        decline_idx = find_decline_start(df_well_sorted, **sampled_decline)
                        rate_series = df_well_sorted['M_Oil_Prod'].values
                        if decline_idx < len(rate_series):
                            decline_rate = rate_series[decline_idx]
                            future_max = np.max(rate_series[decline_idx:])
                            if decline_rate < 0.8 * future_max:
                                start_idx = auto_idx
                            else:
                                start_idx = decline_idx
                        else:
                            start_idx = auto_idx
                    elif start_method == "Auto-Detected Recent Period":
                        start_idx = find_last_production_period(df_well_sorted, **sampled_period)
                    elif start_method == "Smoothing Decline Analysis":
                        start_idx = find_decline_start(df_well_sorted, **sampled_decline)
                    elif start_method == "Manual":
                        start_idx = manual_start_idx
                    else:
                        start_idx = 0
                    
                    metrics = compute_metrics(
                        self.df_all, well_name, manual_start_idx=start_idx,
                        filter_params=sampled_filter,
                        decline_params=sampled_decline,
                        period_params=sampled_period
                    )
                    return metrics['score']
                
                try:
                    optimizer = BayesianOptimization(
                        f=objective_function,
                        pbounds=pbounds,
                        random_state=1,
                        verbose=0
                    )
                    optimizer.maximize(init_points=5, n_iter=n_iter)
                    best_params = optimizer.max['params']
                    best_score = optimizer.max['target']
                    
                    for param in int_keys:
                        if param in best_params:
                            best_params[param] = int(round(best_params[param]))
                    
                    # Update GUI and parameter dictionaries only for optimized parameters
                    optimized_params = []
                    for param in best_params:
                        opt_var = param_to_opt_var.get(param)
                        if opt_var and hasattr(self, opt_var) and getattr(self, opt_var).get():
                            var_name = f"{param}_var"
                            if hasattr(self, var_name):
                                value = best_params[param]
                                getattr(self, var_name).set(round(value, 2) if param not in int_keys else int(value))
                                optimized_params.append(param)
                            if param in self.filter_ranges:
                                filter_params[param] = best_params[param]
                            elif param in self.decline_ranges:
                                decline_params[param] = best_params[param]
                            elif param in self.period_ranges:
                                period_params[param] = best_params[param]
                    
                    fig, results = run_arps_for_well_auto(
                        self.df_all, well_name, outlier_threshold=outlier_threshold,
                        forecast_avg_points=forecast_avg_points, manual_start_idx=chosen_idx,
                        detect_period=False, start_method=method_label,
                        decline_params=decline_params, period_params=period_params,
                        filter_params=filter_params
                    )
                    results += f"\n\nOptimization Results:\nOptimized parameters: {optimized_params}\nBest score: {best_score:.2f}"
                except Exception as e:
                    messagebox.showerror("Error", f"Optimization failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=chosen_idx,
                    detect_period=False, start_method=method_label,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params
                )
                results += "\n\nNo parameters optimized (all optimization checkboxes unchecked)."
        else:
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
            results += "\n\nOptimization disabled."
        
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