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


def plot_arps_fit(df_well, t, q, popt, well_id, df_full, q_original, q_smoothed):
    """Plot actual production vs Arps Hyperbolic model with original and smoothed data"""
    if popt is None:
        print("No fit available to plot.")
        return
    q_pred = arps_hyperbolic(t, *popt)
    plt.figure(figsize=(10, 6))
    
    # Original data (red circles)
    plt.plot(df_full['Prod_Date'], q_original, 'o', color='red', 
             label='Original Data', alpha=0.6, markersize=5)
    
    # Smoothed data (red dashed line)
    plt.plot(df_full['Prod_Date'], q_smoothed, '--', color='red', 
             label='Smoothed Data', linewidth=2)
    
    # Filtered/Decline data (blue circles)
    plt.plot(df_well['Prod_Date'], q, 'o', color='blue', 
             label='Decline Phase Data', markersize=6)
    
    # Arps model fit (solid line)
    plt.plot(df_well['Prod_Date'], q_pred, '-', color='green', 
             label='Arps Hyperbolic Model', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Oil Production Rate (bbl/day)')
    plt.title(f'Arps Hyperbolic Fit for Well {well_id}')
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
    min_window=7,      # smoothing window (odd)
    polyorder=2,       # savgol poly order
    peak_prominence=0.2,   # relative prominence threshold (fraction of max)
    min_decline_len=3,     # minimum consecutive months showing decline after peak
    decline_ma_window=3,   # small MA to test post-peak decline
    drop_frac=0.05         # require post-peak mean to be lower than peak*(1-drop_frac)
):
    """
    Return index (position in df_well sorted by Prod_Date) of detected decline start.
    Strategy:
     - compute daily production
     - smooth it
     - detect peaks with prominence
     - for each peak verify a sustained fall after the peak by testing consecutive negative slopes
       or falling moving-average and that the following mean drops by drop_frac
     - choose the most prominent/highest candidate peak (so major early peaks win over small late spikes)
    """
    df = df_well.sort_values('Prod_Date').copy()
    df['Prod_Date'] = pd.to_datetime(df['Prod_Date'])
    # days in month -> daily rate
    import calendar
    df['days_in_month'] = df['Prod_Date'].apply(lambda d: calendar.monthrange(d.year, d.month)[1])
    df['oil_prod_daily'] = df[rate_col] / df['days_in_month']

    q = df['oil_prod_daily'].values.astype(float)
    n = len(q)
    if n == 0:
        return 0

    # ensure odd window and <= n
    w = min_window if min_window % 2 == 1 else min_window + 1
    w = min(w, n if n % 2 == 1 else n-1)
    if w < 3:
        w = 3

    # smooth
    try:
        q_smooth = savgol_filter(q, window_length=w, polyorder=min(polyorder, w-1))
    except Exception:
        q_smooth = q.copy()

    # peak detection: use absolute prominence scaled by max(q_smooth)
    max_q = np.nanmax(q_smooth) if np.nanmax(q_smooth) > 0 else 1.0
    prom_abs = peak_prominence * max_q

    peaks, props = find_peaks(q_smooth, prominence=prom_abs, distance=1)

    if len(peaks) == 0:
        # fallback: use global max
        idx = int(np.nanargmax(q_smooth))
        return idx

    candidates = []
    for p in peaks:
        # require at least min_decline_len points after peak to evaluate
        if p + min_decline_len >= n:
            continue

        # check consecutive negative slope or falling moving average
        consecutive_down = 0
        for i in range(p, min(n-1, p + min_decline_len + 5)):
            if q_smooth[i+1] < q_smooth[i]:
                consecutive_down += 1
            else:
                # reset small interruptions? optional: keep counting but break if too many rises
                pass

        # moving average check (post-peak)
        ma_after = np.mean(q_smooth[p+1 : min(n, p+1 + decline_ma_window)])
        peak_val = q_smooth[p]
        drop_ok = ma_after < peak_val * (1.0 - drop_frac)

        if consecutive_down >= min_decline_len or drop_ok:
            # record candidate with its peak height and prominence
            prom = props['prominences'][np.where(peaks == p)[0][0]]
            candidates.append({'idx': p, 'peak_val': peak_val, 'prom': prom})

    if not candidates:
        # Last resort: choose highest detected peak (avoid tiny late peak)
        best = peaks[np.argmax(q_smooth[peaks])]
        return int(best)

    # prefer the candidate with largest peak_val * prom (or choose peak_val)
    best = max(candidates, key=lambda x: (x['peak_val'], x['prom']))
    return int(best['idx'])


# --- (2) Main Arps workflow ---
def run_arps_for_well(df_all, well_name, start_idx=0):
    """Prepare data, fit Arps Hyperbolic model, and plot starting from given index"""
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

    # Store original data and full dataframe before filtering
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

    # --- Apply given start index ---
    if start_idx > 0 and start_idx < len(df_well):
        df_well = df_well.iloc[start_idx:].reset_index(drop=True)
        print(f"Using decline data starting from index {start_idx}: {df_well.iloc[0]['Prod_Date'].date()}")

    # --- Fit and plot ---
    t = df_well['t'].values - df_well['t'].values[0]  # reset time to zero
    q = df_well['oil_prod_daily'].values
    popt, pcov = fit_arps_hyperbolic(t, q)
    if popt is not None:
        print(f"Fitted parameters for well {well_name}:")
        print(f"qi = {popt[0]:.2f}, Di = {popt[1]:.4f}, b = {popt[2]:.2f}")
    
    # Pass full dataframe and original/smoothed data to plot
    plot_arps_fit(df_well, t, q, popt, well_name, df_full, q_original_full, q_smoothed_full)


# --- Example usage ---
df_all = pd.read_csv('OFM202409.csv', low_memory=False)

# Step 1: detect decline start
well_name = df_all['Well_Name'].unique()[55]
start_idx = find_decline_start(df_all[df_all['Well_Name'] == well_name])

# Step 2: run Arps analysis starting from that index
run_arps_for_well(df_all, well_name, start_idx=start_idx)