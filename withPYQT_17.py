
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QCheckBox, QLineEdit, QTextEdit,
                             QTabWidget, QGroupBox, QGridLayout, QRadioButton, QButtonGroup,
                             QMessageBox, QSplitter, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QFont, QPalette, QColor, QCursor
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import calendar
from bayes_opt import BayesianOptimization

# Default parameter ranges for optimization
DEFAULT_FILTER_RANGES = {
    'threshold': (1.5, 2.2)
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

def fit_arps_hyperbolic(t, q, fixed_qi=None):
    """Fit Arps Hyperbolic model to production data"""
    if fixed_qi is not None:
        def model_fixed_qi(t, Di, b):
            return arps_hyperbolic(t, fixed_qi, Di, b)
        p0 = [0.1, 0.5]
        try:
            popt_2, pcov_2 = curve_fit(
                model_fixed_qi, t, q, p0=p0,
                bounds=([1e-6, 0], [2, 2])
            )
            popt = [fixed_qi, popt_2[0], popt_2[1]]
            pcov = np.zeros((3, 3))
            pcov[1:, 1:] = pcov_2
            return popt, pcov
        except Exception as e:
            print(f"Model fitting with fixed qi failed: {e}")
            return None, None
    else:
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

def filter_outliers_iterative(t, q, threshold=3, max_iterations=MAX_OUTLIER_ITERATIONS, fixed_qi=None):
    mask = np.ones(len(t), dtype=bool)
    n = len(t)
    if n < 3:
        print(f"Data has only {n} points, skipping outlier removal to ensure at least 3 points.")
        popt, pcov = fit_arps_hyperbolic(t, q, fixed_qi=fixed_qi)
        return mask, popt
    for iteration in range(max_iterations):
        t_fit = t[mask]
        q_fit = q[mask]
        popt, pcov = fit_arps_hyperbolic(t_fit, q_fit, fixed_qi=fixed_qi)
        if popt is None:
            print(f"Fitting failed at iteration {iteration}")
            break
        q_pred = arps_hyperbolic(t, *popt)
        residuals = q - q_pred
        residuals_fit = residuals[mask]
        std_residuals = np.std(residuals_fit)
        new_mask = np.abs(residuals) <= threshold * std_residuals
        if np.sum(new_mask) < 3:
            print(f"Iteration {iteration + 1}: Would result in {np.sum(new_mask)} points, skipping outlier removal.")
            popt, pcov = fit_arps_hyperbolic(t[mask], q[mask], fixed_qi=fixed_qi)
            return mask, popt
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
    popt, pcov = fit_arps_hyperbolic(t[mask], q[mask], fixed_qi=fixed_qi)
    return mask, popt

def create_arps_plot(df_well, t, q, mask, popt, well_id, df_full, q_original, q_smoothed, 
                     outlier_threshold, forecast_avg_points=1, start_method="", 
                     show_outliers=True, show_pre_decline=True, show_forecast=True, 
                     show_smoothed=False, show_channel=False, forecast_duration=60,
                     forecast_offset=0):
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
    forecast_months = forecast_duration
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1 + int(max(0, forecast_offset))),
                                   periods=forecast_months, freq='MS')
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
    decline_start_idx = len(q_original) - len(q)
    
    # Create continuous date range for smooth model curve
    first_date = df_well['Prod_Date'].min()
    last_historical_date = df_well['Prod_Date'].max()
    all_dates = pd.date_range(start=first_date, end=last_historical_date, freq='MS')
    
    # Create continuous time array for model prediction
    t_continuous = np.array([(d - first_date).days // 30 for d in all_dates])
    q_pred_continuous = arps_hyperbolic(t_continuous, *popt)
    
    if show_pre_decline and decline_start_idx > 0:
        ax.plot(df_full['Prod_Date'].iloc[:decline_start_idx], 
                q_original[:decline_start_idx], 'o', color='red', 
                label='Pre-Decline Data', alpha=0.6, markersize=5)
    if show_smoothed:
        if show_pre_decline and decline_start_idx > 0:
            ax.plot(df_full['Prod_Date'].iloc[:decline_start_idx], 
                    q_smoothed[:decline_start_idx], '--', color='red', 
                    label='Smoothed Pre-Decline', linewidth=2, alpha=0.7)
        ax.plot(df_full['Prod_Date'].iloc[decline_start_idx:], 
                q_smoothed[decline_start_idx:], '--', color='orange', 
                label='Smoothed Decline Phase', linewidth=2)
    ax.plot(df_well['Prod_Date'][mask], q[mask], 'o', color='blue', 
            label='Decline Phase Data (Inliers)', markersize=6)
    if show_outliers and np.sum(~mask) > 0:
        ax.plot(df_well['Prod_Date'][~mask], q[~mask], 'x', color='red', 
                label='Outliers (Excluded)', markersize=10, markeredgewidth=2)
    
    # Plot continuous model curve for all months
    ax.plot(all_dates, q_pred_continuous, '-', color='green', 
            label='Arps Hyperbolic Model', linewidth=2)
    
    if show_channel:
        # Create continuous channel bounds
        upper_bound_continuous = q_pred_continuous + channel_width
        lower_bound_continuous = np.maximum(q_pred_continuous - channel_width, 0)
        ax.fill_between(all_dates, lower_bound_continuous, upper_bound_continuous,
                        color='green', alpha=0.2, label=f'±{channel_width:.1f} bbl/day Channel')
    if show_forecast:
        forecast_years = forecast_duration / 12
        if forecast_years in list(range(1, 31)):
            forecast_label = f'{int(forecast_years)}-Year Forecast'
        else:
            forecast_label = f'{forecast_duration}-Month Forecast'
        ax.plot(forecast_dates, forecast_q, '--', color='blue',
                label=forecast_label, linewidth=2)
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

    # === Penalize very low Di ===
    Di_penalty = 1.0
    if Di < 0.001:
        Di_penalty = 0.01
    elif Di < 0.01:
        Di_penalty = 0.3
    elif Di < 0.05:
        Di_penalty = 0.7

    decline_reasonableness = 1.0
    if Di > 2.0:
        decline_reasonableness = 0.1
    elif b > 1.5:
        decline_reasonableness = 0.7

    return (0.5 * r_squared + 0.3 * mape_score + 0.2 * decline_reasonableness) * Di_penalty

def calculate_trend_consistency(t, q, mask, popt):
    q_actual = q[mask]
    t_actual = t[mask]
    if len(q_actual) < 3:
        return 0
    if len(q_actual) >= 6:
        recent_q = q_actual[-6:]
        recent_t = t_actual[-6:]
        actual_slope = linregress(recent_t, recent_q).slope
    else:
        actual_slope = linregress(t_actual, q_actual).slope
    q_pred = arps_hyperbolic(t_actual, *popt)
    expected_slope = linregress(t_actual, q_pred).slope
    if np.isnan(actual_slope) or np.isnan(expected_slope):
        return 0
    slope_ratio = abs(actual_slope / expected_slope) if expected_slope != 0 else 1
    consistency = 1.0 / (1.0 + abs(slope_ratio - 1.0))
    return consistency

def compute_metrics(df_all, well_name, manual_start_idx=0, filter_params=None, decline_params=None, period_params=None, ignore_gaps=False, fixed_qi=None):
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
    if not ignore_gaps:
        if len(df_well) > 1:
            date_diffs = df_well['Prod_Date'].diff()
            large_gaps = date_diffs > pd.Timedelta(days=365 * 10)
            if large_gaps.any():
                last_gap_idx = np.where(large_gaps.values)[0][-1]
                points_after_gap = len(df_well) - (last_gap_idx + 1)
                if points_after_gap >= 10:
                    gap_date_before = df_well.iloc[last_gap_idx]['Prod_Date']
                    gap_date_after = df_well.iloc[last_gap_idx + 1]['Prod_Date']
                    gap_years = (gap_date_after - gap_date_before).days / 365.25
                    print(f"\n{'!'*60}")
                    print(f"LARGE DATA GAP DETECTED:")
                    print(f"Gap of {gap_years:.1f} years between:")
                    print(f"  {gap_date_before.date()} and {gap_date_after.date()}")
                    print(f"Points after gap: {points_after_gap}")
                    print(f"Excluding first {last_gap_idx + 1} points before the gap")
                    print(f"{'!'*60}\n")
                    df_well = df_well.iloc[last_gap_idx + 1:].reset_index(drop=True)
    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    mask, popt = filter_outliers_iterative(
        t, q, 
        threshold=filter_params.get('threshold', 2.0),
        fixed_qi=fixed_qi
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
        0.1 * data_utilization +
        0.3 * fit_quality +
        0.3 * trend_consistency
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
                           decline_params=None, period_params=None, filter_params=None,
                           show_outliers=True, show_pre_decline=True, 
                           show_forecast=True, show_smoothed=False, show_channel=False, 
                           forecast_duration=60, forecast_offset=0, fixed_qi=None):
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
    if len(df_well) > 1:
        date_diffs = df_well['Prod_Date'].diff()
        large_gaps = date_diffs > pd.Timedelta(days=365 * 10)
        print(f"\n--- Checking for large data gaps ---")
        print(f"Start method: {start_method if start_method else 'auto-detect'}")
        print(f"Detect period enabled: {detect_period}")
        print(f"Large gaps found: {large_gaps.any()}")
        print(f"--- End of gap check ---\n")
        if start_method not in ["Manual Start Date", "Manual Start Date & Initial Rate (Qi)"] and large_gaps.any():
            last_gap_idx = np.where(large_gaps.values)[0][-1]
            points_after_gap = len(df_well) - (last_gap_idx + 1)
            if points_after_gap >= 10:
                gap_date_before = df_well.iloc[last_gap_idx]['Prod_Date']
                gap_date_after = df_well.iloc[last_gap_idx + 1]['Prod_Date']
                gap_years = (gap_date_after - gap_date_before).days / 365.25
                print(f"\n{'!'*60}")
                print(f"LARGE DATA GAP DETECTED:")
                print(f"Gap of {gap_years:.1f} years between:")
                print(f"  {gap_date_before.date()} and {gap_date_after.date()}")
                print(f"Points after gap: {points_after_gap}")
                print(f"Excluding first {last_gap_idx + 1} points before the gap")
                print(f"{'!'*60}\n")
                df_well = df_well.iloc[last_gap_idx + 1:].reset_index(drop=True)
                start_idx += last_gap_idx + 1
        else:
            print(f"\nManual start selected — using all available points after {df_well.iloc[0]['Prod_Date'].date()} (ignoring gaps).")
    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    print("--- Starting iterative outlier filtering ---")
    mask, popt = filter_outliers_iterative(
        t, q, 
        threshold=filter_params.get('threshold', outlier_threshold),
        fixed_qi=fixed_qi
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
                           forecast_avg_points, start_method,
                           show_outliers, show_pre_decline, show_forecast, 
                           show_smoothed, show_channel, forecast_duration, forecast_offset)
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

class DeclineCurveApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decline Curve Analysis Tool")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #2c3e50;
                font-size: 11px;
            }
            QComboBox, QLineEdit {
                padding: 4px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                min-height: 20px;
                max-height: 28px;
            }
            QComboBox:hover, QLineEdit:focus {
                border: 1px solid #3498db;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 26px;
                max-height: 32px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2c3e50;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #d5dbdb;
            }
            QCheckBox {
                spacing: 5px;
                max-height: 24px;
            }
            QRadioButton {
                spacing: 5px;
                max-height: 24px;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #fdfefe;
                padding: 4px;
            }
        """)
        self.filter_ranges = DEFAULT_FILTER_RANGES.copy()
        self.decline_ranges = DEFAULT_DECLINE_RANGES.copy()
        self.period_ranges = DEFAULT_PERIOD_RANGES.copy()
        
        try:
            self.df_all = pd.read_csv('OFM202409.csv', low_memory=False)
            self.districts = sorted([d for d in self.df_all['District'].dropna().unique() if str(d).strip()])
            self.fields_by_district = {}
            self.formations_by_field = {}
            self.wells_by_formation = {}
            for district in self.districts:
                district_data = self.df_all[self.df_all['District'] == district]
                fields = sorted([f for f in district_data['Field'].dropna().unique() if str(f).strip()])
                self.fields_by_district[district] = fields
                for field in fields:
                    field_data = district_data[district_data['Field'] == field]
                    formations = sorted([fm for fm in field_data['Alias_Formation'].dropna().unique() if str(fm).strip()])
                    self.formations_by_field[f"{district}|{field}"] = formations
                    for formation in formations:
                        formation_data = field_data[field_data['Alias_Formation'] == formation]
                        wells = sorted([w for w in formation_data['Well_Name'].dropna().unique() if str(w).strip()])
                        self.wells_by_formation[f"{district}|{field}|{formation}"] = wells
            self.current_district = self.districts[0] if self.districts else None
            self.current_field = self.fields_by_district.get(self.current_district, [None])[0] if self.current_district else None
            self.current_formation = None
            self.initializing = True  # Flag to prevent reset during initialization
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            sys.exit(1)
            return

        # Create main container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)
        
        # Top frame for field/reservoir/well selection
        selection_group = QGroupBox("Well Selection")
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(6)
        selection_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        selection_group.setLayout(selection_layout)
        
        selection_layout.addWidget(QLabel("District:"))
        self.district_combo = QComboBox()
        self.district_combo.addItems(self.districts)
        self.district_combo.setMinimumWidth(150)
        if self.current_district:
            self.district_combo.setCurrentText(self.current_district)
        self.district_combo.currentTextChanged.connect(self.on_district_select)
        selection_layout.addWidget(self.district_combo)

        selection_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(150)
        initial_fields = self.fields_by_district.get(self.current_district, []) if self.current_district else []
        self.field_combo.addItems(initial_fields)
        if self.current_field:
            self.field_combo.setCurrentText(self.current_field)
        self.field_combo.currentTextChanged.connect(self.on_field_select)
        selection_layout.addWidget(self.field_combo)

        selection_layout.addWidget(QLabel("Formation:"))
        self.formation_combo = QComboBox()
        self.formation_combo.setMinimumWidth(150)
        initial_formations = self.formations_by_field.get(f"{self.current_district}|{self.current_field}", []) if self.current_district and self.current_field else []
        self.formation_combo.addItems(initial_formations)
        self.formation_combo.currentTextChanged.connect(self.on_formation_select)
        selection_layout.addWidget(self.formation_combo)

        selection_layout.addWidget(QLabel("Well:"))
        self.well_combo = QComboBox()
        self.well_combo.setMinimumWidth(150)
        self.well_combo.currentTextChanged.connect(self.on_well_select)
        selection_layout.addWidget(self.well_combo)
        
        # Auto-select first items in each combo based on alphabetical order
        if initial_formations:
            self.formation_combo.setCurrentIndex(0)
            self.on_formation_select(initial_formations[0])
        
        selection_layout.addStretch()
        
        main_layout.addWidget(selection_group)
        
        # Bottom splitter for chart and controls
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_widget.setMinimumWidth(420)
        left_widget.setMaximumWidth(500)
        
        # Chart frame
        self.canvas_frame = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_frame)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.canvas_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Start Date Selection Method
        start_method_group = QGroupBox("Start Date Selection Method")
        start_method_layout = QVBoxLayout()
        start_method_group.setLayout(start_method_layout)
        left_layout.addWidget(start_method_group)
        
        self.start_method_buttons = QButtonGroup()
        self.auto_select_radio = QRadioButton("Auto Select")
        self.auto_select_radio.setChecked(True)
        self.auto_select_radio.toggled.connect(lambda checked: self.update_vertical_line_from_inputs() if checked else None)
        self.start_method_buttons.addButton(self.auto_select_radio, 0)
        start_method_layout.addWidget(self.auto_select_radio)

        # Professional group for Manual Start Date
        self.manual_frame = QGroupBox("Manual Start Date")
        manual_layout = QVBoxLayout()
        self.manual_frame.setLayout(manual_layout)
        
        self.manual_start_radio = QRadioButton("Manual Start Date")
        self.manual_start_radio.toggled.connect(lambda checked: self.update_vertical_line_from_inputs() if checked else None)
        self.start_method_buttons.addButton(self.manual_start_radio, 1)
        manual_layout.addWidget(self.manual_start_radio)
        
        manual_date_layout = QHBoxLayout()
        manual_date_layout.addWidget(QLabel("Year:"))
        self.manual_year_edit = QLineEdit("2020")
        self.manual_year_edit.setMaximumWidth(60)
        self.manual_year_edit.textChanged.connect(self.update_vertical_line_from_inputs)
        manual_date_layout.addWidget(self.manual_year_edit)
        manual_date_layout.addWidget(QLabel("Month:"))
        self.manual_month_edit = QLineEdit("1")
        self.manual_month_edit.setMaximumWidth(40)
        self.manual_month_edit.textChanged.connect(self.update_vertical_line_from_inputs)
        manual_date_layout.addWidget(self.manual_month_edit)
        self.select_start_btn = QPushButton("Select on Chart")
        self.select_start_btn.clicked.connect(self.enable_chart_click_selection)
        manual_date_layout.addWidget(self.select_start_btn)
        manual_date_layout.addStretch()
        manual_layout.addLayout(manual_date_layout)
        
        start_method_layout.addWidget(self.manual_frame)

        # Professional group for Manual Start Date & Initial Rate (Qi)
        self.qi_frame = QGroupBox("Manual Start Date & Initial Rate (Qi)")
        qi_layout = QVBoxLayout()
        self.qi_frame.setLayout(qi_layout)
        
        self.qi_radio = QRadioButton("Manual Start Date & Initial Rate (Qi)")
        self.qi_radio.toggled.connect(lambda checked: self.update_vertical_line_from_inputs() if checked else None)
        self.start_method_buttons.addButton(self.qi_radio, 2)
        qi_layout.addWidget(self.qi_radio)
        
        qi_date_layout = QHBoxLayout()
        qi_date_layout.addWidget(QLabel("Year:"))
        self.qi_year_edit = QLineEdit("2020")
        self.qi_year_edit.setMaximumWidth(60)
        self.qi_year_edit.textChanged.connect(self.update_vertical_line_from_inputs)
        qi_date_layout.addWidget(self.qi_year_edit)
        qi_date_layout.addWidget(QLabel("Month:"))
        self.qi_month_edit = QLineEdit("1")
        self.qi_month_edit.setMaximumWidth(40)
        self.qi_month_edit.textChanged.connect(self.update_vertical_line_from_inputs)
        qi_date_layout.addWidget(self.qi_month_edit)
        qi_date_layout.addStretch()
        qi_layout.addLayout(qi_date_layout)
        
        qi_rate_layout = QHBoxLayout()
        qi_rate_layout.addWidget(QLabel("Initial Rate (Qi) bbl/day:"))
        self.qi_value_edit = QLineEdit("")
        self.qi_value_edit.setMaximumWidth(100)
        qi_rate_layout.addWidget(self.qi_value_edit)
        self.select_qi_btn = QPushButton("Select on Chart")
        self.select_qi_btn.clicked.connect(self.enable_qi_click_selection)
        qi_rate_layout.addWidget(self.select_qi_btn)
        qi_rate_layout.addStretch()
        qi_layout.addLayout(qi_rate_layout)
        
        start_method_layout.addWidget(self.qi_frame)

        # Optimization controls
        opt_layout = QHBoxLayout()
        self.optimize_check = QCheckBox("Optimize Parameters")
        self.optimize_check.setChecked(True)
        opt_layout.addWidget(self.optimize_check)
        opt_config_btn = QPushButton("Configure Optimization Ranges")
        opt_config_btn.clicked.connect(self.open_range_config)
        opt_layout.addWidget(opt_config_btn)
        left_layout.addLayout(opt_layout)

        # Tabs for settings
        self.settings_tabs = QTabWidget()
        left_layout.addWidget(self.settings_tabs)
        self.create_general_tab()
        self.create_decline_tab()
        self.create_period_tab()
        self.create_plot_options_tab()

        # Button frame for Run Analysis and Reset
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-size: 12px;
                padding: 6px;
                min-height: 28px;
                max-height: 32px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        button_layout.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_parameters)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                font-size: 12px;
                padding: 6px;
                min-height: 28px;
                max-height: 32px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        button_layout.addWidget(self.reset_btn)
        left_layout.addLayout(button_layout)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        left_layout.addWidget(self.results_text)

        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.current_analysis_params = None
        self.current_well_name = None
        self.click_selection_enabled = False
        self.qi_selection_enabled = False
        self.click_cid = None
        self.qi_click_cid = None
        self.key_cid = None
        self.fixed_qi = None
        self.original_cursor = None
        self.event_filter_installed = False
        self.vertical_line = None  # Store reference to vertical line on chart
        
        # Initialization complete, enable well selection reset behavior
        self.initializing = False
        
        # Display raw data chart for the first well on startup
        initial_well = self.well_combo.currentText()
        if initial_well:
            self.display_raw_data(initial_well)

    def closeEvent(self, event):
        event.accept()
    
    def remove_vertical_line(self):
        """Remove the vertical line from the chart if it exists"""
        if self.vertical_line is not None and self.figure is not None:
            try:
                self.vertical_line.remove()
                self.canvas.draw()
            except:
                pass  # Line may have already been removed
            self.vertical_line = None
    
    def update_vertical_line_from_inputs(self):
        """Add or update vertical line based on current manual date inputs"""
        if self.figure is None or self.canvas is None:
            return
        
        # Only show vertical line if in manual mode
        if self.auto_select_radio.isChecked():
            self.remove_vertical_line()
            return
        
        try:
            # Get the appropriate year and month based on which manual mode is selected
            if self.manual_start_radio.isChecked():
                year = int(self.manual_year_edit.text())
                month = int(self.manual_month_edit.text())
            elif self.qi_radio.isChecked():
                year = int(self.qi_year_edit.text())
                month = int(self.qi_month_edit.text())
            else:
                return
            
            # Create the date from inputs
            selected_date = pd.to_datetime(f"{year}-{month:02d}-01")
            
            # Remove existing line
            self.remove_vertical_line()
            
            # Add new vertical line at the selected date
            ax = self.figure.axes[0]
            self.vertical_line = ax.axvline(x=selected_date, color='red', linestyle='--', linewidth=2, label='Selected Start Date')
            ax.legend()
            self.canvas.draw()
        except (ValueError, AttributeError):
            # Invalid date input, don't draw line
            pass
    
    def eventFilter(self, obj, event):
        """Event filter to catch ESC key press globally"""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                if self.click_selection_enabled or self.qi_selection_enabled:
                    self.disable_selection_mode()
                    return True
        return super().eventFilter(obj, event)

    def enable_chart_click_selection(self):
        if self.figure is None:
            QMessageBox.warning(self, "Warning", "Please run an analysis first to display a chart.")
            return
        self.click_selection_enabled = True
        self.qi_selection_enabled = False
        if self.click_cid:
            self.figure.canvas.mpl_disconnect(self.click_cid)
        if self.qi_click_cid:
            self.figure.canvas.mpl_disconnect(self.qi_click_cid)
        if self.key_cid:
            self.figure.canvas.mpl_disconnect(self.key_cid)
        
        # Store original cursor and change to crosshair
        self.original_cursor = self.canvas.cursor()
        self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        
        # Set focus to canvas to receive keyboard events
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        
        # Install event filter to catch ESC key (only if not already installed)
        if not self.event_filter_installed:
            self.canvas.installEventFilter(self)
            self.event_filter_installed = True
        
        self.click_cid = self.figure.canvas.mpl_connect('button_press_event', self.on_chart_click)
        self.key_cid = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_chart_click(self, event):
        if not self.click_selection_enabled or event.inaxes is None:
            return
        clicked_date = event.xdata
        if np.isnan(clicked_date):
            return
        date_clicked = pd.to_datetime(clicked_date, unit='D')
        well_name = self.well_combo.currentText()
        if not well_name:
            return
        df_well = self.df_all[self.df_all['Well_Name'] == well_name].copy()
        df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
        df_well = df_well.sort_values('Prod_Date')
        idx = np.searchsorted(df_well['Prod_Date'], date_clicked)
        if idx >= len(df_well):
            idx = len(df_well) - 1
        elif idx < 0:
            idx = 0
        selected_date = df_well.iloc[idx]['Prod_Date']
        self.manual_year_edit.setText(str(selected_date.year))
        self.manual_month_edit.setText(str(selected_date.month))
        self.manual_start_radio.setChecked(True)
        
        # Remove any existing vertical line
        self.remove_vertical_line()
        
        # Add a vertical line at the selected date
        ax = self.figure.axes[0]
        self.vertical_line = ax.axvline(x=selected_date, color='red', linestyle='--', linewidth=2, label='Selected Start Date')
        ax.legend()
        self.canvas.draw()
        
        self.disable_selection_mode()

    def enable_qi_click_selection(self):
        if self.figure is None:
            QMessageBox.warning(self, "Warning", "Please run an analysis first to display a chart.")
            return
        self.qi_selection_enabled = True
        self.click_selection_enabled = False
        if self.click_cid:
            self.figure.canvas.mpl_disconnect(self.click_cid)
        if self.qi_click_cid:
            self.figure.canvas.mpl_disconnect(self.qi_click_cid)
        if self.key_cid:
            self.figure.canvas.mpl_disconnect(self.key_cid)
        
        # Store original cursor and change to crosshair
        self.original_cursor = self.canvas.cursor()
        self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        
        # Set focus to canvas to receive keyboard events
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        
        # Install event filter to catch ESC key (only if not already installed)
        if not self.event_filter_installed:
            self.canvas.installEventFilter(self)
            self.event_filter_installed = True
        
        self.qi_click_cid = self.figure.canvas.mpl_connect('button_press_event', self.on_qi_chart_click)
        self.key_cid = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_qi_chart_click(self, event):
        if not self.qi_selection_enabled or event.inaxes is None:
            return
        clicked_date = event.xdata
        clicked_rate = event.ydata
        if np.isnan(clicked_date) or np.isnan(clicked_rate) or clicked_rate <= 0:
            QMessageBox.warning(self, "Invalid Point", "Please click on a valid data point with positive rate.")
            return
        date_clicked = pd.to_datetime(clicked_date, unit='D')
        well_name = self.well_combo.currentText()
        if not well_name:
            return
        df_well = self.df_all[self.df_all['Well_Name'] == well_name].copy()
        df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
        df_well = df_well.sort_values('Prod_Date')
        idx = np.searchsorted(df_well['Prod_Date'], date_clicked)
        if idx >= len(df_well):
            idx = len(df_well) - 1
        elif idx < 0:
            idx = 0
        selected_date = df_well.iloc[idx]['Prod_Date']
        self.qi_year_edit.setText(str(selected_date.year))
        self.qi_month_edit.setText(str(selected_date.month))
        self.fixed_qi = float(clicked_rate)
        self.qi_value_edit.setText(f"{clicked_rate:.2f}")
        self.qi_radio.setChecked(True)
        
        # Remove any existing vertical line
        self.remove_vertical_line()
        
        # Add a vertical line at the selected date
        ax = self.figure.axes[0]
        self.vertical_line = ax.axvline(x=selected_date, color='red', linestyle='--', linewidth=2, label='Selected Start Date')
        ax.legend()
        self.canvas.draw()
        
        self.disable_selection_mode()
    
    def on_key_press(self, event):
        """Handle key press events, particularly ESC to cancel selection"""
        if event.key == 'escape':
            self.disable_selection_mode()
    
    def disable_selection_mode(self):
        """Disable selection mode and restore original cursor"""
        self.click_selection_enabled = False
        self.qi_selection_enabled = False
        
        # Disconnect all event handlers
        if self.click_cid:
            self.figure.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        if self.qi_click_cid:
            self.figure.canvas.mpl_disconnect(self.qi_click_cid)
            self.qi_click_cid = None
        if self.key_cid:
            self.figure.canvas.mpl_disconnect(self.key_cid)
            self.key_cid = None
        
        # Note: We don't remove the event filter here anymore
        # It stays installed on the canvas and just checks the flags
        # This way it works for subsequent clicks
        
        # Restore original cursor
        if self.canvas and self.original_cursor is not None:
            self.canvas.setCursor(self.original_cursor)
        elif self.canvas:
            self.canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def create_general_tab(self):
        general_tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        general_tab.setLayout(main_layout)
        
        label_width = 180  # Slightly wider for longer labels in General tab
        checkbox_width = 24  # Fixed width for checkbox area to ensure alignment
        
        # Row 1 - Outlier Threshold (with checkbox)
        row1 = QHBoxLayout()
        row1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.threshold_opt_check = QCheckBox()
        self.threshold_opt_check.setChecked(True)
        self.threshold_opt_check.setFixedWidth(checkbox_width)
        row1.addWidget(self.threshold_opt_check)
        label1 = QLabel("Outlier Threshold:")
        label1.setFixedWidth(label_width)
        row1.addWidget(label1)
        self.threshold_edit = QLineEdit("2.0")
        self.threshold_edit.setMaximumWidth(100)
        row1.addWidget(self.threshold_edit)
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Row 2 - Forecast Avg Points (no checkbox, add spacer)
        row2 = QHBoxLayout()
        row2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        spacer2 = QLabel("")  # Spacer to align with checkbox width
        spacer2.setFixedWidth(checkbox_width)
        row2.addWidget(spacer2)
        label2 = QLabel("Forecast Avg Points:")
        label2.setFixedWidth(label_width)
        row2.addWidget(label2)
        self.forecast_avg_points_edit = QLineEdit("0")
        self.forecast_avg_points_edit.setMaximumWidth(100)
        row2.addWidget(self.forecast_avg_points_edit)
        row2.addStretch()
        main_layout.addLayout(row2)
        
        # Row 3 - Forecast Duration
        row3 = QHBoxLayout()
        row3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        spacer3 = QLabel("")
        spacer3.setFixedWidth(checkbox_width)
        row3.addWidget(spacer3)
        label3 = QLabel("Forecast Duration (months):")
        label3.setFixedWidth(label_width)
        row3.addWidget(label3)
        self.forecast_duration_edit = QLineEdit("60")
        self.forecast_duration_edit.setMaximumWidth(100)
        row3.addWidget(self.forecast_duration_edit)
        row3.addStretch()
        main_layout.addLayout(row3)
        
        # Row 4 - Forecast Offset
        row4 = QHBoxLayout()
        row4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        spacer4 = QLabel("")
        spacer4.setFixedWidth(checkbox_width)
        row4.addWidget(spacer4)
        label4 = QLabel("Forecast Offset (months):")
        label4.setFixedWidth(label_width)
        row4.addWidget(label4)
        self.forecast_offset_edit = QLineEdit("0")
        self.forecast_offset_edit.setMaximumWidth(100)
        row4.addWidget(self.forecast_offset_edit)
        row4.addStretch()
        main_layout.addLayout(row4)
        
        main_layout.addStretch()
        self.settings_tabs.addTab(general_tab, "General")

    def create_decline_tab(self):
        decline_tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        decline_tab.setLayout(main_layout)
        
        label_width = 150  # Fixed width for labels to align input fields
        
        # Row 1
        row1 = QHBoxLayout()
        row1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.smooth_window_opt_check = QCheckBox()
        self.smooth_window_opt_check.setChecked(True)
        row1.addWidget(self.smooth_window_opt_check)
        label1 = QLabel("Smooth Window:")
        label1.setFixedWidth(label_width)
        row1.addWidget(label1)
        self.smooth_window_edit = QLineEdit("13")
        self.smooth_window_edit.setMaximumWidth(100)
        row1.addWidget(self.smooth_window_edit)
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Row 2
        row2 = QHBoxLayout()
        row2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.polyorder_opt_check = QCheckBox()
        self.polyorder_opt_check.setChecked(True)
        row2.addWidget(self.polyorder_opt_check)
        label2 = QLabel("Polyorder:")
        label2.setFixedWidth(label_width)
        row2.addWidget(label2)
        self.polyorder_edit = QLineEdit("2")
        self.polyorder_edit.setMaximumWidth(100)
        row2.addWidget(self.polyorder_edit)
        row2.addStretch()
        main_layout.addLayout(row2)
        
        # Row 3
        row3 = QHBoxLayout()
        row3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.peak_prominence_opt_check = QCheckBox()
        self.peak_prominence_opt_check.setChecked(True)
        row3.addWidget(self.peak_prominence_opt_check)
        label3 = QLabel("Peak Prominence:")
        label3.setFixedWidth(label_width)
        row3.addWidget(label3)
        self.peak_prominence_edit = QLineEdit("0.15")
        self.peak_prominence_edit.setMaximumWidth(100)
        row3.addWidget(self.peak_prominence_edit)
        row3.addStretch()
        main_layout.addLayout(row3)
        
        # Row 4
        row4 = QHBoxLayout()
        row4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.post_ma_opt_check = QCheckBox()
        self.post_ma_opt_check.setChecked(True)
        row4.addWidget(self.post_ma_opt_check)
        label4 = QLabel("Post MA:")
        label4.setFixedWidth(label_width)
        row4.addWidget(label4)
        self.post_ma_edit = QLineEdit("12")
        self.post_ma_edit.setMaximumWidth(100)
        row4.addWidget(self.post_ma_edit)
        row4.addStretch()
        main_layout.addLayout(row4)
        
        # Row 5
        row5 = QHBoxLayout()
        row5.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.decline_frac_opt_check = QCheckBox()
        self.decline_frac_opt_check.setChecked(True)
        row5.addWidget(self.decline_frac_opt_check)
        label5 = QLabel("Decline Frac:")
        label5.setFixedWidth(label_width)
        row5.addWidget(label5)
        self.decline_frac_edit = QLineEdit("0.15")
        self.decline_frac_edit.setMaximumWidth(100)
        row5.addWidget(self.decline_frac_edit)
        row5.addStretch()
        main_layout.addLayout(row5)
        
        # Row 6
        row6 = QHBoxLayout()
        row6.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.post_slope_win_opt_check = QCheckBox()
        self.post_slope_win_opt_check.setChecked(True)
        row6.addWidget(self.post_slope_win_opt_check)
        label6 = QLabel("Post Slope Win:")
        label6.setFixedWidth(label_width)
        row6.addWidget(label6)
        self.post_slope_win_edit = QLineEdit("12")
        self.post_slope_win_edit.setMaximumWidth(100)
        row6.addWidget(self.post_slope_win_edit)
        row6.addStretch()
        main_layout.addLayout(row6)
        
        # Row 7
        row7 = QHBoxLayout()
        row7.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.persist_months_opt_check = QCheckBox()
        self.persist_months_opt_check.setChecked(True)
        row7.addWidget(self.persist_months_opt_check)
        label7 = QLabel("Persist Months:")
        label7.setFixedWidth(label_width)
        row7.addWidget(label7)
        self.persist_months_edit = QLineEdit("9")
        self.persist_months_edit.setMaximumWidth(100)
        row7.addWidget(self.persist_months_edit)
        row7.addStretch()
        main_layout.addLayout(row7)
        
        main_layout.addStretch()
        self.settings_tabs.addTab(decline_tab, "Decline Detection")

    def create_period_tab(self):
        period_tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        period_tab.setLayout(main_layout)
        
        label_width = 150  # Fixed width for labels to align input fields
        
        # Row 1
        row1 = QHBoxLayout()
        row1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.min_production_months_opt_check = QCheckBox()
        self.min_production_months_opt_check.setChecked(True)
        row1.addWidget(self.min_production_months_opt_check)
        label1 = QLabel("Min Production Months:")
        label1.setFixedWidth(label_width)
        row1.addWidget(label1)
        self.min_production_months_edit = QLineEdit("12")
        self.min_production_months_edit.setMaximumWidth(100)
        row1.addWidget(self.min_production_months_edit)
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Row 2
        row2 = QHBoxLayout()
        row2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.surge_multiplier_opt_check = QCheckBox()
        self.surge_multiplier_opt_check.setChecked(True)
        row2.addWidget(self.surge_multiplier_opt_check)
        label2 = QLabel("Surge Multiplier:")
        label2.setFixedWidth(label_width)
        row2.addWidget(label2)
        self.surge_multiplier_edit = QLineEdit("2.0")
        self.surge_multiplier_edit.setMaximumWidth(100)
        row2.addWidget(self.surge_multiplier_edit)
        row2.addStretch()
        main_layout.addLayout(row2)
        
        # Row 3
        row3 = QHBoxLayout()
        row3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.gap_threshold_opt_check = QCheckBox()
        self.gap_threshold_opt_check.setChecked(True)
        row3.addWidget(self.gap_threshold_opt_check)
        label3 = QLabel("Gap Threshold:")
        label3.setFixedWidth(label_width)
        row3.addWidget(label3)
        self.gap_threshold_edit = QLineEdit("0.3")
        self.gap_threshold_edit.setMaximumWidth(100)
        row3.addWidget(self.gap_threshold_edit)
        row3.addStretch()
        main_layout.addLayout(row3)
        
        # Row 4
        row4 = QHBoxLayout()
        row4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lookback_window_opt_check = QCheckBox()
        self.lookback_window_opt_check.setChecked(True)
        row4.addWidget(self.lookback_window_opt_check)
        label4 = QLabel("Lookback Window:")
        label4.setFixedWidth(label_width)
        row4.addWidget(label4)
        self.lookback_window_edit = QLineEdit("6")
        self.lookback_window_edit.setMaximumWidth(100)
        row4.addWidget(self.lookback_window_edit)
        row4.addStretch()
        main_layout.addLayout(row4)
        
        main_layout.addStretch()
        self.settings_tabs.addTab(period_tab, "Period Detection")

    def create_plot_options_tab(self):
        plot_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        plot_tab.setLayout(layout)
        
        self.show_outliers_check = QCheckBox("Show Outliers")
        self.show_outliers_check.setChecked(True)
        self.show_outliers_check.stateChanged.connect(self.update_plot_options)
        layout.addWidget(self.show_outliers_check)
        
        self.show_pre_decline_check = QCheckBox("Show Pre-Decline Points")
        self.show_pre_decline_check.setChecked(True)
        self.show_pre_decline_check.stateChanged.connect(self.update_plot_options)
        layout.addWidget(self.show_pre_decline_check)
        
        self.show_forecast_check = QCheckBox("Show Forecast")
        self.show_forecast_check.setChecked(True)
        self.show_forecast_check.stateChanged.connect(self.update_plot_options)
        layout.addWidget(self.show_forecast_check)
        
        self.show_smoothed_check = QCheckBox("Show Smoothed Data")
        self.show_smoothed_check.setChecked(False)
        self.show_smoothed_check.stateChanged.connect(self.update_plot_options)
        layout.addWidget(self.show_smoothed_check)
        
        self.show_channel_check = QCheckBox("Show Channel")
        self.show_channel_check.setChecked(False)
        self.show_channel_check.stateChanged.connect(self.update_plot_options)
        layout.addWidget(self.show_channel_check)
        
        layout.addStretch()
        self.settings_tabs.addTab(plot_tab, "Plot Options")

    def update_plot_options(self):
        if self.current_analysis_params is None or self.current_well_name is None:
            return
        if self.figure is None:
            return
        try:
            show_outliers = self.show_outliers_check.isChecked()
            show_pre_decline = self.show_pre_decline_check.isChecked()
            show_forecast = self.show_forecast_check.isChecked()
            show_smoothed = self.show_smoothed_check.isChecked()
            show_channel = self.show_channel_check.isChecked()
            current_offset = max(0, min(120, int(self.forecast_offset_edit.text())))
            fig, results = run_arps_for_well_auto(
                self.df_all, self.current_well_name,
                outlier_threshold=self.current_analysis_params['outlier_threshold'],
                forecast_avg_points=self.current_analysis_params['forecast_avg_points'],
                manual_start_idx=self.current_analysis_params['manual_start_idx'],
                detect_period=self.current_analysis_params['detect_period'],
                start_method=self.current_analysis_params['start_method'],
                decline_params=self.current_analysis_params['decline_params'],
                period_params=self.current_analysis_params['period_params'],
                filter_params=self.current_analysis_params['filter_params'],
                show_outliers=show_outliers,
                show_pre_decline=show_pre_decline,
                show_forecast=show_forecast,
                show_smoothed=show_smoothed,
                show_channel=show_channel,
                forecast_duration=self.current_analysis_params.get('forecast_duration', 60),
                forecast_offset=current_offset,
                fixed_qi=getattr(self, 'fixed_qi', None)
            )
            if fig:
                if self.canvas:
                    self.canvas_layout.removeWidget(self.canvas)
                    self.canvas.deleteLater()
                if self.toolbar:
                    self.canvas_layout.removeWidget(self.toolbar)
                    self.toolbar.deleteLater()
                self.figure = fig
                self.canvas = FigureCanvasQTAgg(fig)
                self.toolbar = NavigationToolbar2QT(self.canvas, self.canvas_frame)
                self.canvas_layout.addWidget(self.toolbar)
                self.canvas_layout.addWidget(self.canvas)
                
                # Reset event filter flag for new canvas
                self.event_filter_installed = False
                
                # Redraw vertical line if in manual mode
                self.update_vertical_line_from_inputs()
        except Exception as e:
            print(f"Error updating plot options: {e}")

    def open_range_config(self):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        
        config_dialog = QDialog(self)
        config_dialog.setWindowTitle("Configure Optimization Ranges")
        config_dialog.setMinimumSize(600, 700)
        layout = QVBoxLayout()
        config_dialog.setLayout(layout)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Filter tab
        filter_widget = QWidget()
        filter_layout = QGridLayout()
        filter_widget.setLayout(filter_layout)
        filter_vars = {}
        row = 0
        for param, (min_val, max_val) in self.filter_ranges.items():
            filter_layout.addWidget(QLabel(f"{param}:"), row, 0)
            filter_layout.addWidget(QLabel("Min:"), row, 1)
            min_edit = QLineEdit(str(min_val))
            min_edit.setMaximumWidth(100)
            filter_layout.addWidget(min_edit, row, 2)
            filter_layout.addWidget(QLabel("Max:"), row, 3)
            max_edit = QLineEdit(str(max_val))
            max_edit.setMaximumWidth(100)
            filter_layout.addWidget(max_edit, row, 4)
            filter_vars[param] = (min_edit, max_edit)
            row += 1
        tabs.addTab(filter_widget, "Filter Ranges")
        
        # Decline tab
        decline_widget = QWidget()
        decline_layout = QGridLayout()
        decline_widget.setLayout(decline_layout)
        decline_vars = {}
        row = 0
        for param, (min_val, max_val) in self.decline_ranges.items():
            decline_layout.addWidget(QLabel(f"{param}:"), row, 0)
            decline_layout.addWidget(QLabel("Min:"), row, 1)
            min_edit = QLineEdit(str(min_val))
            min_edit.setMaximumWidth(100)
            decline_layout.addWidget(min_edit, row, 2)
            decline_layout.addWidget(QLabel("Max:"), row, 3)
            max_edit = QLineEdit(str(max_val))
            max_edit.setMaximumWidth(100)
            decline_layout.addWidget(max_edit, row, 4)
            decline_vars[param] = (min_edit, max_edit)
            row += 1
        tabs.addTab(decline_widget, "Decline Ranges")
        
        # Period tab
        period_widget = QWidget()
        period_layout = QGridLayout()
        period_widget.setLayout(period_layout)
        period_vars = {}
        row = 0
        for param, (min_val, max_val) in self.period_ranges.items():
            period_layout.addWidget(QLabel(f"{param}:"), row, 0)
            period_layout.addWidget(QLabel("Min:"), row, 1)
            min_edit = QLineEdit(str(min_val))
            min_edit.setMaximumWidth(100)
            period_layout.addWidget(min_edit, row, 2)
            period_layout.addWidget(QLabel("Max:"), row, 3)
            max_edit = QLineEdit(str(max_val))
            max_edit.setMaximumWidth(100)
            period_layout.addWidget(max_edit, row, 4)
            period_vars[param] = (min_edit, max_edit)
            row += 1
        tabs.addTab(period_widget, "Period Ranges")
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset to Default")
        cancel_btn = QPushButton("Cancel")
        
        def save_ranges():
            try:
                for param, (min_edit, max_edit) in filter_vars.items():
                    self.filter_ranges[param] = (float(min_edit.text()), float(max_edit.text()))
                for param, (min_edit, max_edit) in decline_vars.items():
                    self.decline_ranges[param] = (float(min_edit.text()), float(max_edit.text()))
                for param, (min_edit, max_edit) in period_vars.items():
                    self.period_ranges[param] = (float(min_edit.text()), float(max_edit.text()))
                QMessageBox.information(config_dialog, "Success", "Optimization ranges updated successfully!")
                config_dialog.accept()
            except Exception as e:
                QMessageBox.critical(config_dialog, "Error", f"Failed to update ranges: {str(e)}")
        
        def reset_ranges():
            for param, (min_val, max_val) in DEFAULT_FILTER_RANGES.items():
                filter_vars[param][0].setText(str(min_val))
                filter_vars[param][1].setText(str(max_val))
            for param, (min_val, max_val) in DEFAULT_DECLINE_RANGES.items():
                decline_vars[param][0].setText(str(min_val))
                decline_vars[param][1].setText(str(max_val))
            for param, (min_val, max_val) in DEFAULT_PERIOD_RANGES.items():
                period_vars[param][0].setText(str(min_val))
                period_vars[param][1].setText(str(max_val))
        
        save_btn.clicked.connect(save_ranges)
        reset_btn.clicked.connect(reset_ranges)
        cancel_btn.clicked.connect(config_dialog.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        config_dialog.exec()

    def on_district_select(self, district):
        if district:
            self.current_district = district
            fields = self.fields_by_district.get(district, [])
            self.field_combo.clear()
            self.field_combo.addItems(fields)
            if fields:
                self.current_field = fields[0]
                self.field_combo.setCurrentIndex(0)
                self.on_field_select(fields[0])
            else:
                self.formation_combo.clear()
                self.well_combo.clear()

    def on_field_select(self, field):
        district = self.district_combo.currentText()
        if district and field:
            self.current_field = field
            formations = self.formations_by_field.get(f"{district}|{field}", [])
            self.formation_combo.clear()
            self.formation_combo.addItems(formations)
            if formations:
                self.current_formation = formations[0]
                self.formation_combo.setCurrentIndex(0)
                self.on_formation_select(formations[0])
            else:
                self.well_combo.clear()

    def on_formation_select(self, formation):
        district = self.district_combo.currentText()
        field = self.field_combo.currentText()
        if district and field and formation:
            self.current_formation = formation
            wells = self.wells_by_formation.get(f"{district}|{field}|{formation}", [])
            self.well_combo.clear()
            self.well_combo.addItems(wells)
            if wells:
                self.well_combo.setCurrentIndex(0)
    
    def on_well_select(self, well):
        """Reset parameters when well selection changes"""
        # Don't reset during initialization
        if hasattr(self, 'initializing') and self.initializing:
            return
        if well:  # Only reset if a well is actually selected
            self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all parameters to defaults except district/field/formation/well filters"""
        # Reset start method to Auto Select
        self.auto_select_radio.setChecked(True)
        
        # Reset general parameters to defaults
        self.threshold_edit.setText("2.0")
        self.threshold_opt_check.setChecked(True)
        self.forecast_avg_points_edit.setText("0")
        self.forecast_duration_edit.setText("60")
        self.forecast_offset_edit.setText("0")
        
        # Reset decline detection parameters to defaults
        self.smooth_window_edit.setText("13")
        self.smooth_window_opt_check.setChecked(True)
        self.polyorder_edit.setText("2")
        self.polyorder_opt_check.setChecked(True)
        self.peak_prominence_edit.setText("0.15")
        self.peak_prominence_opt_check.setChecked(True)
        self.post_ma_edit.setText("12")
        self.post_ma_opt_check.setChecked(True)
        self.decline_frac_edit.setText("0.15")
        self.decline_frac_opt_check.setChecked(True)
        self.post_slope_win_edit.setText("12")
        self.post_slope_win_opt_check.setChecked(True)
        self.persist_months_edit.setText("9")
        self.persist_months_opt_check.setChecked(True)
        
        # Reset period detection parameters to defaults
        self.min_production_months_edit.setText("12")
        self.min_production_months_opt_check.setChecked(True)
        self.surge_multiplier_edit.setText("2.0")
        self.surge_multiplier_opt_check.setChecked(True)
        self.gap_threshold_edit.setText("0.3")
        self.gap_threshold_opt_check.setChecked(True)
        self.lookback_window_edit.setText("6")
        self.lookback_window_opt_check.setChecked(True)
        
        # Reset plot options - show outliers, forecast, and pre-decline by default
        self.show_outliers_check.setChecked(True)
        self.show_pre_decline_check.setChecked(True)
        self.show_forecast_check.setChecked(True)
        self.show_smoothed_check.setChecked(False)
        self.show_channel_check.setChecked(False)
        
        # Reset optimization checkbox
        self.optimize_check.setChecked(True)
        
        # Reset manual date inputs
        self.manual_year_edit.setText("2020")
        self.manual_month_edit.setText("1")
        self.qi_year_edit.setText("2020")
        self.qi_month_edit.setText("1")
        self.qi_value_edit.setText("")
        
        # Reset fixed_qi
        self.fixed_qi = None
        
        # Clear results text
        self.results_text.clear()
        
        # Display raw data if well is selected
        well_name = self.well_combo.currentText()
        if well_name:
            self.display_raw_data(well_name)
    
    def display_raw_data(self, well_name):
        """Display only raw production data points in black without any analysis"""
        df_well = self.df_all[self.df_all['Well_Name'] == well_name].copy()
        if df_well.empty:
            return
        
        df_well = df_well.sort_values('Prod_Date')
        df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
        df_well['days_in_month'] = df_well['Prod_Date'].apply(get_days_in_month)
        df_well['oil_prod_daily'] = df_well['M_Oil_Prod'] / df_well['days_in_month']
        
        # Create simple plot with only raw data
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df_well['Prod_Date'], df_well['oil_prod_daily'], 'o', color='black', 
                label='Production Data', markersize=6)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Oil Production Rate (bbl/day)')
        ax.set_title(f'Well {well_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Display the plot
        if self.canvas:
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        if self.toolbar:
            self.canvas_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
        
        self.figure = fig
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.canvas_frame)
        self.canvas_layout.addWidget(self.toolbar)
        self.canvas_layout.addWidget(self.canvas)
        
        # Reset event filter flag for new canvas
        self.event_filter_installed = False
        
        # Clear analysis params since this is just raw data
        self.current_well_name = None
        self.current_analysis_params = None

    def run_analysis(self):
        well_name = self.well_combo.currentText()
        if not well_name:
            QMessageBox.warning(self, "Warning", "Please select a well")
            return

        # Get start method
        if self.auto_select_radio.isChecked():
            start_method = "Auto Select"
        elif self.manual_start_radio.isChecked():
            start_method = "Manual Start Date"
        else:
            start_method = "Manual Start Date & Initial Rate (Qi)"
            
        manual_start_idx = None
        method_label = start_method
        df_well_temp = self.df_all[self.df_all['Well_Name'] == well_name].copy()
        df_well_temp['Prod_Date'] = pd.to_datetime(df_well_temp['Prod_Date'])
        df_well_sorted = df_well_temp.sort_values('Prod_Date')

        if start_method != "Manual Start Date & Initial Rate (Qi)":
            self.fixed_qi = None

        if start_method in ["Manual Start Date", "Manual Start Date & Initial Rate (Qi)"]:
            try:
                if start_method == "Manual Start Date":
                    year = int(self.manual_year_edit.text())
                    month = int(self.manual_month_edit.text())
                else:
                    year = int(self.qi_year_edit.text())
                    month = int(self.qi_month_edit.text())
                    # Read Qi value from entry field
                    qi_str = self.qi_value_edit.text().strip()
                    if qi_str:
                        try:
                            self.fixed_qi = float(qi_str)
                        except ValueError:
                            QMessageBox.critical(self, "Error", f"Invalid Initial Rate (Qi) value: {qi_str}")
                            return
                    else:
                        QMessageBox.warning(self, "Warning", "Please enter an Initial Rate (Qi) value or click on the chart to select one.")
                        return
                manual_date = pd.to_datetime(f"{year}-{month:02d}-01")
                manual_start_idx = np.searchsorted(df_well_sorted['Prod_Date'], manual_date)
                if manual_start_idx >= len(df_well_sorted):
                    QMessageBox.warning(self, "Warning", "Manual date is after last production date. Using last index.")
                    manual_start_idx = len(df_well_sorted) - 1
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Invalid manual date: {str(e)}")
                return

        outlier_threshold = float(self.threshold_edit.text())
        forecast_avg_points = int(self.forecast_avg_points_edit.text())
        forecast_duration = int(self.forecast_duration_edit.text())
        forecast_offset = max(0, min(120, int(self.forecast_offset_edit.text())))

        filter_params = {'threshold': float(self.threshold_edit.text())}
        decline_params = {
            'smooth_window': int(self.smooth_window_edit.text()),
            'polyorder': int(self.polyorder_edit.text()),
            'peak_prominence': float(self.peak_prominence_edit.text()),
            'post_ma': int(self.post_ma_edit.text()),
            'decline_frac': float(self.decline_frac_edit.text()),
            'post_slope_win': int(self.post_slope_win_edit.text()),
            'persist_months': int(self.persist_months_edit.text())
        }
        period_params = {
            'min_production_months': int(self.min_production_months_edit.text()),
            'surge_multiplier': float(self.surge_multiplier_edit.text()),
            'gap_threshold': float(self.gap_threshold_edit.text()),
            'lookback_window': int(self.lookback_window_edit.text())
        }

        show_outliers = self.show_outliers_check.isChecked()
        show_pre_decline = self.show_pre_decline_check.isChecked()
        show_forecast = self.show_forecast_check.isChecked()
        show_smoothed = self.show_smoothed_check.isChecked()
        show_channel = self.show_channel_check.isChecked()
        optimize = self.optimize_check.isChecked()

        pbounds = {}
        param_to_opt_check = {
            'threshold': 'threshold_opt_check',
            'smooth_window': 'smooth_window_opt_check',
            'polyorder': 'polyorder_opt_check',
            'peak_prominence': 'peak_prominence_opt_check',
            'post_ma': 'post_ma_opt_check',
            'decline_frac': 'decline_frac_opt_check',
            'post_slope_win': 'post_slope_win_opt_check',
            'persist_months': 'persist_months_opt_check',
            'min_production_months': 'min_production_months_opt_check',
            'surge_multiplier': 'surge_multiplier_opt_check',
            'gap_threshold': 'gap_threshold_opt_check',
            'lookback_window': 'lookback_window_opt_check'
        }
        relevant_ranges = {**self.filter_ranges, **self.decline_ranges, **self.period_ranges}
        for param in relevant_ranges:
            opt_check = param_to_opt_check.get(param)
            if opt_check and hasattr(self, opt_check) and getattr(self, opt_check).isChecked():
                pbounds[param] = relevant_ranges[param]

        if optimize:
            fixed_params = {**filter_params, **decline_params, **period_params}
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
            elif start_method in ["Manual Start Date", "Manual Start Date & Initial Rate (Qi)"]:
                chosen_idx = manual_start_idx
                n_iter = 10
            else:
                chosen_idx = 0
                n_iter = 0

            if pbounds:
                def objective_function(**params):
                    for param in int_keys:
                        if param in params:
                            params[param] = int(round(params[param]))
                    sampled_filter = {k: params.get(k, fixed_params[k]) for k in self.filter_ranges}
                    sampled_decline = {k: params.get(k, fixed_params[k]) for k in self.decline_ranges}
                    sampled_period = {k: params.get(k, fixed_params[k]) for k in self.period_ranges}

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
                    elif start_method in ["Manual Start Date", "Manual Start Date & Initial Rate (Qi)"]:
                        start_idx = manual_start_idx
                    else:
                        start_idx = 0

                    metrics = compute_metrics(
                        self.df_all, well_name, manual_start_idx=start_idx,
                        filter_params=sampled_filter,
                        decline_params=sampled_decline,
                        period_params=sampled_period,
                        ignore_gaps=(start_method in ["Manual Start Date", "Manual Start Date & Initial Rate (Qi)"]),
                        fixed_qi=self.fixed_qi if start_method == "Manual Start Date & Initial Rate (Qi)" else None
                    )

                    if metrics['popt'] is not None:
                        Di = metrics['popt'][1]
                        if Di < 0.001:
                            metrics['score'] *= 0.01
                        elif Di < 0.01:
                            metrics['score'] *= 0.3
                        elif Di < 0.05:
                            metrics['score'] *= 0.7

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
                    optimized_params = []
                    for param in best_params:
                        opt_check = param_to_opt_check.get(param)
                        if opt_check and hasattr(self, opt_check) and getattr(self, opt_check).isChecked():
                            edit_name = f"{param}_edit"
                            if hasattr(self, edit_name):
                                value = best_params[param]
                                if param in int_keys:
                                    getattr(self, edit_name).setText(str(int(value)))
                                else:
                                    getattr(self, edit_name).setText(str(round(value, 2)))
                                optimized_params.append(param)
                            if param in self.filter_ranges:
                                filter_params[param] = best_params[param]
                            elif param in self.decline_ranges:
                                decline_params[param] = best_params[param]
                            elif param in self.period_ranges:
                                period_params[param] = best_params[param]
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Optimization failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return

                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=chosen_idx,
                    detect_period=False, start_method=method_label,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params,
                    show_outliers=show_outliers, show_pre_decline=show_pre_decline,
                    show_forecast=show_forecast, show_smoothed=show_smoothed,
                    show_channel=show_channel, forecast_duration=forecast_duration,
                    forecast_offset=forecast_offset,
                    fixed_qi=self.fixed_qi if start_method == "Manual Start Date & Initial Rate (Qi)" else None
                )
            else:
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=chosen_idx,
                    detect_period=False, start_method=method_label,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params,
                    show_outliers=show_outliers, show_pre_decline=show_pre_decline,
                    show_forecast=show_forecast, show_smoothed=show_smoothed,
                    show_channel=show_channel, forecast_duration=forecast_duration,
                    forecast_offset=forecast_offset,
                    fixed_qi=self.fixed_qi if start_method == "Manual Start Date & Initial Rate (Qi)" else None
                )
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
                    filter_params=filter_params,
                    show_outliers=show_outliers, show_pre_decline=show_pre_decline,
                    show_forecast=show_forecast, show_smoothed=show_smoothed,
                    show_channel=show_channel, forecast_duration=forecast_duration,
                    forecast_offset=forecast_offset,
                    fixed_qi=None
                )
            elif start_method in ["Manual Start Date", "Manual Start Date & Initial Rate (Qi)"]:
                fig, results = run_arps_for_well_auto(
                    self.df_all, well_name, outlier_threshold=outlier_threshold,
                    forecast_avg_points=forecast_avg_points, manual_start_idx=manual_start_idx,
                    detect_period=False, start_method=start_method,
                    decline_params=decline_params, period_params=period_params,
                    filter_params=filter_params,
                    show_outliers=show_outliers, show_pre_decline=show_pre_decline,
                    show_forecast=show_forecast, show_smoothed=show_smoothed,
                    show_channel=show_channel, forecast_duration=forecast_duration,
                    forecast_offset=forecast_offset,
                    fixed_qi=self.fixed_qi if start_method == "Manual Start Date & Initial Rate (Qi)" else None
                )

        self.results_text.clear()
        self.results_text.setText(results)

        if fig:
            if self.canvas:
                self.canvas_layout.removeWidget(self.canvas)
                self.canvas.deleteLater()
            if self.toolbar:
                self.canvas_layout.removeWidget(self.toolbar)
                self.toolbar.deleteLater()
            self.figure = fig
            self.canvas = FigureCanvasQTAgg(fig)
            self.toolbar = NavigationToolbar2QT(self.canvas, self.canvas_frame)
            self.canvas_layout.addWidget(self.toolbar)
            self.canvas_layout.addWidget(self.canvas)
            
            # Reset event filter flag for new canvas
            self.event_filter_installed = False
            
            self.current_well_name = well_name
            self.current_analysis_params = {
                'outlier_threshold': outlier_threshold,
                'forecast_avg_points': forecast_avg_points,
                'manual_start_idx': chosen_idx if 'chosen_idx' in locals() else manual_start_idx,
                'detect_period': False,
                'start_method': method_label,
                'decline_params': decline_params,
                'period_params': period_params,
                'filter_params': filter_params,
                'forecast_duration': forecast_duration,
                'forecast_offset': forecast_offset
            }
            
            # Redraw vertical line if in manual mode
            self.update_vertical_line_from_inputs()
        else:
            QMessageBox.critical(self, "Error", results)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeclineCurveApp()
    window.show()
    sys.exit(app.exec())