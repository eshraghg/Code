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


def plot_arps_fit(df_well, t, q, mask, popt, well_id, df_full, q_original, q_smoothed):
    """Plot actual production vs Arps Hyperbolic model with outliers marked"""
    if popt is None:
        print("No fit available to plot.")
        return
    
    q_pred = arps_hyperbolic(t, *popt)
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
    
    # Arps model fit (solid line)
    plt.plot(df_well['Prod_Date'], q_pred, '-', color='green', 
             label='Arps Hyperbolic Model', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Oil Production Rate (bbl/day)')
    plt.title(f'Arps Hyperbolic Fit for Well {well_id} (with Outlier Filtering)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Helper ---
def get_days_in_month(date):
    import calendar
    return calendar.monthrange(date.year, date.month)[1]


def find_decline_start(
    df_well,
    rate_col='M_Oil_Prod',
    min_window=7,
    polyorder=2,
    peak_prominence=0.2,
    min_decline_len=3,
    decline_ma_window=3,
    drop_frac=0.05
):
    """
    Return index (position in df_well sorted by Prod_Date) of detected decline start.
    """
    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    import calendar
    df['days_in_month'] = df['Prod_Date'].apply(lambda d: calendar.monthrange(d.year, d.month)[1])
    df['oil_prod_daily'] = df[rate_col] / df['days_in_month']

    q = df['oil_prod_daily'].values.astype(float)
    n = len(q)
    if n == 0:
        return 0

    w = min_window if min_window % 2 == 1 else min_window + 1
    w = min(w, n if n % 2 == 1 else n-1)
    if w < 3:
        w = 3

    try:
        q_smooth = savgol_filter(q, window_length=w, polyorder=min(polyorder, w-1))
    except Exception:
        q_smooth = q.copy()

    max_q = np.nanmax(q_smooth) if np.nanmax(q_smooth) > 0 else 1.0
    prom_abs = peak_prominence * max_q

    peaks, props = find_peaks(q_smooth, prominence=prom_abs, distance=1)

    if len(peaks) == 0:
        idx = int(np.nanargmax(q_smooth))
        return idx

    candidates = []
    for p in peaks:
        if p + min_decline_len >= n:
            continue

        consecutive_down = 0
        for i in range(p, min(n-1, p + min_decline_len + 5)):
            if q_smooth[i+1] < q_smooth[i]:
                consecutive_down += 1

        ma_after = np.mean(q_smooth[p+1 : min(n, p+1 + decline_ma_window)])
        peak_val = q_smooth[p]
        drop_ok = ma_after < peak_val * (1.0 - drop_frac)

        if consecutive_down >= min_decline_len or drop_ok:
            prom = props['prominences'][np.where(peaks == p)[0][0]]
            candidates.append({'idx': p, 'peak_val': peak_val, 'prom': prom})

    if not candidates:
        best = peaks[np.argmax(q_smooth[peaks])]
        return int(best)

    best = max(candidates, key=lambda x: (x['peak_val'], x['prom']))
    return int(best['idx'])


# --- Main Arps workflow with outlier filtering ---
def run_arps_for_well(df_all, well_name, start_idx=0, outlier_threshold=2):
    """Prepare data, fit Arps Hyperbolic model with outlier filtering, and plot"""
    import calendar
    df_well = df_all[df_all['Well_Name'] == well_name].copy()
    if df_well.empty:
        print(f"No data found for well '{well_name}'")
        return

    df_well = df_well.sort_values('Prod_Date')
    df_well['Prod_Date'] = pd.to_datetime(df_well['Prod_Date'])
    df_well['days_in_month'] = df_well['Prod_Date'].apply(
        lambda d: calendar.monthrange(d.year, d.month)[1]
    )
    df_well['oil_prod_daily'] = df_well['M_Oil_Prod'] / df_well['days_in_month']
    df_well['t'] = (df_well['Prod_Date'] - df_well['Prod_Date'].min()).dt.days // 30

    df_full = df_well.copy()
    q_original_full = df_well['oil_prod_daily'].values.copy()
    
    # Compute smoothed data for full dataset
    n = len(q_original_full)
    w = 7 if 7 % 2 == 1 else 8
    w = min(w, n if n % 2 == 1 else n-1)
    if w < 3:
        w = 3
    try:
        q_smoothed_full = savgol_filter(q_original_full, window_length=w, polyorder=2)
    except Exception:
        q_smoothed_full = q_original_full.copy()

    # Apply given start index
    if start_idx > 0 and start_idx < len(df_well):
        df_well = df_well.iloc[start_idx:].reset_index(drop=True)
        print(f"Using decline data starting from index {start_idx}: {df_well.iloc[0]['Prod_Date'].date()}")

    # Fit with outlier filtering
    t = df_well['t'].values - df_well['t'].values[0]
    q = df_well['oil_prod_daily'].values
    
    print("\n--- Starting iterative outlier filtering ---")
    mask, popt = filter_outliers_iterative(t, q, threshold=outlier_threshold)
    
    if popt is not None:
        n_outliers = np.sum(~mask)
        print(f"\n=== Final Results for well {well_name} ===")
        print(f"Total outliers removed: {n_outliers} out of {len(mask)} points")
        print(f"Fitted parameters:")
        print(f"qi = {popt[0]:.2f}, Di = {popt[1]:.4f}, b = {popt[2]:.2f}")
    
    plot_arps_fit(df_well, t, q, mask, popt, well_name, df_full, q_original_full, q_smoothed_full)


# --- Usage ---
df_all = pd.read_csv('OFM202409.csv', low_memory=False)

# Step 1: detect decline start
well_name = df_all['Well_Name'].unique()[11]
start_idx = find_decline_start(df_all[df_all['Well_Name'] == well_name])

# Step 2: run Arps analysis with outlier filtering (threshold=3 means 3×std)
run_arps_for_well(df_all, well_name, start_idx=start_idx, outlier_threshold=2)