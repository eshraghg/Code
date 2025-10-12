# Load data
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


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

def plot_arps_fit(df_well, t, q, mask, popt, well_id, df_full, q_original, q_smoothed, outlier_threshold, forecast_avg_points=1):
    """Plot actual production vs Arps Hyperbolic model with outliers marked and constant-width channel"""
    if popt is None:
        print("No fit available to plot.")
        return
    
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
    
    plt.figure(figsize=(12, 7))
    
    # Original data (red circles)
    plt.plot(df_full['Prod_Date'], q_original, 'o', color='red', 
             label='Original Data', alpha=0.6, markersize=5)
    
    # Smoothed data (red dashed line)
    plt.plot(df_full['Prod_Date'], q_smoothed, '--', color='red', 
             label='Smoothed Data', linewidth=2)
    
    # Filtered/Decline data - inliers (blue circles)
    plt.plot(df_well['Prod_Date'][mask], q[mask], 'o', color='blue', 
             label='Decline Phase Data (Inliers)', markersize=6)
    
    # Outliers (red X marks)
    if np.sum(~mask) > 0:
        plt.plot(df_well['Prod_Date'][~mask], q[~mask], 'x', color='red', 
                 label='Outliers (Excluded)', markersize=10, markeredgewidth=2)
    
    # Constant width channel (clipped at zero)
    plt.fill_between(df_well['Prod_Date'], lower_bound, upper_bound,
                    color='green', alpha=0.2, label=f'±{channel_width:.1f} bbl/day Channel')
    
    # Arps model fit (solid line)
    plt.plot(df_well['Prod_Date'], q_pred, '-', color='green', 
             label='Arps Hyperbolic Model', linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_dates, forecast_q, '--', color='blue',
             label='5-Year Forecast', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Oil Production Rate (bbl/day)')
    plt.title(f'Well {well_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.show()
    
    # Print forecast summary
    print("\n=== Forecast Summary ===")
    print(f"Last historical rate: {forecast_q[0]:.2f} bbl/day")
    print(f"Forecasted rate after 5 years: {forecast_q[-1]:.2f} bbl/day")
    print(f"Decline over 5 years: {((forecast_q[0] - forecast_q[-1])/forecast_q[0]*100):.1f}%")


# --- Helper ---
def get_days_in_month(date):
    import calendar
    return calendar.monthrange(date.year, date.month)[1]

def find_decline_start(
    df_well,
    rate_col='M_Oil_Prod',
    smooth_window=13,       # months for Savitzky-Golay (odd)
    polyorder=2,
    peak_prominence=0.15,   # relative to max for peak detection
    post_ma=12,             # months to compute forward moving-average decline
    decline_frac=0.15,      # required fractional drop from peak to post_ma mean (15%)
    post_slope_win=12,      # months window to compute slope after peak
    persist_months=9        # how many months slope must remain negative (persistence)
):
    """
    Return index (position in df_well sorted by Prod_Date) of detected decline start.
    The returned index refers to position in the sorted dataframe (0..n-1).
    """
    import calendar
    from scipy.signal import find_peaks
    from scipy.stats import linregress

    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    df['days_in_month'] = df['Prod_Date'].apply(lambda d: calendar.monthrange(d.year, d.month)[1])
    q = (df[rate_col] / df['days_in_month']).values.astype(float)
    n = len(q)
    if n == 0:
        return 0

    # ensure odd and <= n
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

    # peak detection (prominence relative to max)
    max_q = np.nanmax(q_smooth) if np.nanmax(q_smooth) > 0 else 1.0
    prom_abs = peak_prominence * max_q
    peaks, props = find_peaks(q_smooth, prominence=prom_abs, distance=6)

    # If no peaks, fallback to global max
    if len(peaks) == 0:
        return int(np.nanargmax(q_smooth))

    candidates = []
    for p in peaks:
        # require some data after peak
        if p + 1 >= n:
            continue

        # compute forward moving average over next post_ma months (or until end)
        end_ma = min(n, p + 1 + post_ma)
        ma_after = np.mean(q_smooth[p+1:end_ma])

        # fractional drop requirement
        drop_ok = ma_after <= q_smooth[p] * (1.0 - decline_frac)

        # slope over next post_slope_win months (use original q_smooth to avoid tiny noise)
        end_slope = min(n, p + 1 + post_slope_win)
        if end_slope - (p+1) >= 3:
            x = np.arange(0, end_slope - (p+1))
            y = q_smooth[p+1:end_slope]
            slope = linregress(x, y).slope
        else:
            slope = 0.0

        # check persistence: is slope negative for several forward windows?
        persist = False
        if p + persist_months < n:
            # compute slopes for sliding windows of length persist_months across the next post_slope_win months
            neg_count = 0
            total_checks = 0
            for start in range(p+1, min(n - persist_months + 1, p+1 + post_slope_win - persist_months + 1)):
                xs = np.arange(0, persist_months)
                ys = q_smooth[start:start+persist_months]
                s = linregress(xs, ys).slope
                total_checks += 1
                if s < 0:
                    neg_count += 1
            # require that at least 70% of checked windows have negative slope
            if total_checks > 0 and (neg_count / total_checks) >= 0.7:
                persist = True

        # accept candidate if drop and persistent negative slope (or strong single-window negative slope)
        if drop_ok and (persist or slope < 0):
            candidates.append({
                'idx': p,
                'peak_val': q_smooth[p],
                'prom': props['prominences'][np.where(peaks == p)[0][0]] if 'prominences' in props else 0.0,
                'slope': slope
            })

    # If no candidate passed the sustained-decline test, fallback: choose the highest peak but
    # require that after it a *long* downward trend exists (more lenient)
    if not candidates:
        # evaluate each peak for longer-term drop (post_ma*1.5)
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
        # fallback to absolute peak index
        return int(peaks[np.argmax(q_smooth[peaks])])

    # choose the best candidate: highest peak value (tie-break by prominence)
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
    """
    Detect the *last major decline* start:
    - Smooths data
    - Finds all peaks
    - Picks the last peak before a long sustained drop (>min_decline_months) 
      with >=drop_frac decline in average rate.
    """
    import calendar
    from scipy.signal import find_peaks

    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    df['days_in_month'] = df['Prod_Date'].apply(lambda d: calendar.monthrange(d.year, d.month)[1])
    q = (df[rate_col] / df['days_in_month']).values.astype(float)
    n = len(q)
    if n < 10:
        return 0

    # Smooth
    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    w = min(w, n if n % 2 == 1 else n-1)
    try:
        q_smooth = savgol_filter(q, window_length=w, polyorder=min(polyorder, w-1))
    except Exception:
        q_smooth = q.copy()

    # Find peaks (any local max)
    peaks, _ = find_peaks(q_smooth, distance=6)
    if len(peaks) == 0:
        return int(np.argmax(q_smooth))

    candidates = []
    for p in peaks:
        # look ahead window
        end = min(n, p + min_decline_months)
        if end <= p + 3:
            continue
        post_mean = np.mean(q_smooth[p+1:end])
        decline_ratio = (q_smooth[p] - post_mean) / q_smooth[p]
        if decline_ratio >= drop_frac:
            candidates.append(p)

    # If no clear candidates, pick the global max
    if not candidates:
        return int(np.argmax(q_smooth))

    # pick the *last* such candidate (we want last major decline)
    idx = candidates[-1]
    return int(idx)


# --- Main Arps workflow with outlier filtering ---
def find_last_production_period(df_well, rate_col='M_Oil_Prod', 
                                 min_production_months=12,
                                 surge_multiplier=3.5,
                                 gap_threshold=0.25,
                                 lookback_window=10):
    """
    Detect the start of the last stable production period by identifying
    major shut-ins, workovers, or production surges.
    
    Parameters:
    -----------
    df_well : DataFrame
        Well production data
    rate_col : str
        Column name for production rate
    min_production_months : int
        Minimum months of continuous production to consider valid
    surge_multiplier : float
        Multiplier to detect production surges (e.g., 2.5 means 250% increase)
    gap_threshold : float
        Fraction of median production below which is considered shut-in
    lookback_window : int
        Number of months to look back when detecting changes
        
    Returns:
    --------
    int : Index position where last stable production period starts
    """
    import calendar
    
    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    df['days_in_month'] = df['Prod_Date'].apply(
        lambda d: calendar.monthrange(d.year, d.month)[1]
    )
    
    # Daily rate for better comparison
    q = (df[rate_col] / df['days_in_month']).values
    n = len(q)
    
    if n < min_production_months:
        print(f"Warning: Only {n} months of data available")
        return 0
    
    # Calculate rolling statistics
    q_positive = q[q > 0]
    if len(q_positive) == 0:
        return 0
        
    median_rate = np.median(q_positive)
    mean_rate = np.mean(q_positive)
    
    # Define thresholds
    shutin_threshold = median_rate * gap_threshold
    surge_threshold = mean_rate * surge_multiplier
    
    print(f"\n=== Production Period Detection ===")
    print(f"Median production rate: {median_rate:.2f} bbl/day")
    print(f"Shut-in threshold: {shutin_threshold:.2f} bbl/day")
    print(f"Surge detection threshold: {surge_threshold:.2f} bbl/day")
    
    # Work backwards from the end to find the last major event
    candidate_starts = []
    
    for i in range(n - min_production_months, lookback_window, -1):
        # Check for shut-in (very low production)
        if q[i] < shutin_threshold and q[i-1] > shutin_threshold:
            candidate_starts.append({
                'idx': i,
                'type': 'shut-in recovery',
                'date': df.iloc[i]['Prod_Date'],
                'before': q[i-1],
                'after': q[i]
            })
            
        # Check for major production surge (workover/stimulation)
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
        
        # Check for recovery from extended low production
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
    
    # If events found, use the most recent one
    if candidate_starts:
        # Sort by index (most recent first)
        candidate_starts.sort(key=lambda x: x['idx'], reverse=True)
        
        print(f"\nDetected {len(candidate_starts)} potential production period changes:")
        for i, event in enumerate(candidate_starts[:5]):  # Show top 5
            print(f"  {i+1}. {event['type'].title()} at {event['date'].date()} "
                  f"(idx={event['idx']}, before={event['before']:.1f}, after={event['after']:.1f})")
        
        # Use the most recent event
        selected = candidate_starts[0]
        print(f"\n✓ Selected start: {selected['date'].date()} ({selected['type']})")
        print(f"  Using {n - selected['idx']} months of data for analysis")
        
        return selected['idx']
    
    # No major events detected - check if early data is significantly different
    # Split data into first 25% and last 75%
    split_point = n // 4
    early_median = np.median(q[:split_point])
    recent_median = np.median(q[split_point:])
    
    # If early production is much higher (>2x), skip it
    if early_median > recent_median * 2:
        print(f"\nEarly production ({early_median:.1f}) much higher than recent ({recent_median:.1f})")
        print(f"Starting analysis from index {split_point} ({df.iloc[split_point]['Prod_Date'].date()})")
        return split_point
    
    print(f"\nNo major production changes detected. Using all available data.")
    return 0


# Modified version of your main function
def run_arps_for_well_auto(df_all, well_name, outlier_threshold=2, 
                           forecast_avg_points=6, manual_start_idx=None,
                           detect_period=True):
    """
    Run Arps analysis with automatic detection of last production period
    
    Parameters:
    -----------
    df_all : DataFrame
        Full dataset
    well_name : str
        Well name to analyze
    outlier_threshold : float
        Standard deviations for outlier detection
    forecast_avg_points : int
        Number of points to average for forecast start
    manual_start_idx : int or None
        If provided, overrides automatic detection
    detect_period : bool
        If True, automatically detect last production period
    """
    import calendar
    
    df_well = df_all[df_all['Well_Name'] == well_name].copy()
    if df_well.empty:
        print(f"No data found for well '{well_name}'")
        return
    
    df_well = df_well.sort_values('Prod_Date')
    
    # Determine start index
    if manual_start_idx is not None:
        start_idx = manual_start_idx
        print(f"Using manual start index: {start_idx}")
    elif detect_period:
        start_idx = find_last_production_period(
            df_well,
            min_production_months=12,
            surge_multiplier=2.0,
            gap_threshold=0.3,
            lookback_window=6
        )
    else:
        start_idx = 0
    
    # Continue with your existing workflow
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
    except Exception:
        q_smoothed_full = q_original_full.copy()

    # Apply start index
    if start_idx > 0:
        df_well = df_well.iloc[start_idx:].reset_index(drop=True)
        print(f"\n{'='*60}")
        print(f"Analysis starting from: {df_well.iloc[0]['Prod_Date'].date()}")
        print(f"Data points used: {len(df_well)}")
        print(f"{'='*60}\n")

    # Fit with outlier filtering
    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    
    # Assess if we have enough data
    if len(q) < 12:
        print(f"ERROR: Insufficient data points ({len(q)}). Need at least 12 months.")
        return
    
    print("--- Starting iterative outlier filtering ---")
    mask, popt = filter_outliers_iterative(t, q, threshold=outlier_threshold)
    
    if popt is not None:
        n_outliers = np.sum(~mask)
        print(f"\n=== Final Results for well {well_name} ===")
        print(f"Total outliers removed: {n_outliers} out of {len(mask)} points")
        print(f"Fitted parameters:")
        print(f"qi = {popt[0]:.2f}, Di = {popt[1]:.4f}, b = {popt[2]:.2f}")
        
        # Assess data quality
        cv = np.std(q[mask]) / np.mean(q[mask])
        print(f"\nCoefficient of Variation: {cv:.3f}")
        if cv > 0.4:
            print("⚠ WARNING: High variability detected. Forecast may be unreliable.")
        elif cv > 0.3:
            print("⚠ CAUTION: Moderate variability. Review forecast carefully.")
        else:
            print("✓ Good data quality for decline curve analysis.")
    
    plot_arps_fit(df_well, t, q, mask, popt, well_name, df_full, 
                  q_original_full, q_smoothed_full, outlier_threshold, 
                  forecast_avg_points)

# --- Usage ---
df_all = pd.read_csv('OFM202409.csv', low_memory=False)



# Choose a well to analyze
well_name = df_all['Well_Name'].unique()[11]
df_well_temp = df_all[df_all['Well_Name'] == well_name]

# Method 1: Automatic production period detection
auto_idx = find_last_production_period(df_well_temp)

# Method 2: Decline start detection (your existing method)
decline_idx = find_decline_start_last_major_drop(
    df_well_temp,
    smooth_window=20,
    min_decline_months=6,
    drop_frac=0.9
)

print(f"\nAuto-detected period start: index {auto_idx}")
print(f"Decline curve start: index {decline_idx}")
print(f"Recommendation: Use index {max(auto_idx, decline_idx)} for most conservative analysis")

# Run with the chosen index
run_arps_for_well_auto(
    df_all, 
    well_name, 
    manual_start_idx=max(auto_idx, decline_idx),
    outlier_threshold=3,
    forecast_avg_points=6
)