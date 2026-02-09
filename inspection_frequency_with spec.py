# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:47:12 2026

@author: chad
"""

import pandas as pd
import numpy as np

# =========================
# Inputs / Outputs
# =========================
data_path = r"C:\Users\chad\Documents\AI\tour_data_export.xlsx"
std_path  = r"C:\Users\chad\Documents\AI\tour_standards.xlsx"
output_path = r"C:\Users\chad\Documents\AI\inspection_frequency_and_matrix_with_standards.xlsx"

# =========================
# Settings you can tune
# =========================
GROUP_COLS = ["Part Number", "Department", "Machine", "Cavity"]
SAFETY_FACTOR_DEFAULT = 2.0

# In your export, subgroup members often live in Data 1..Data 5
# We will automatically use Data 1..Data n where n = subgroup size.
# =========================

# d2 constants for estimating sigma from Rbar (AIAG-style)
D2 = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
    9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407,
    15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735
}

# =========================
# Helper functions
# =========================
def nelson_rule_flags(series: pd.Series) -> pd.Series:
    """
    Basic Nelson-like rules on a time-ordered series with respect to its mean and std.
    Returns a boolean flag per point if any rule triggers.

    Implements (common set):
      R1: 1 point beyond 3σ
      R2: 9 points in a row on same side of mean
      R3: 6 points in a row increasing or decreasing
      R4: 14 points alternating up/down
      R5: 2 of 3 points beyond 2σ on same side
      R6: 4 of 5 points beyond 1σ on same side
      R7: 15 points in a row within 1σ
      R8: 8 points in a row outside 1σ
    """
    x = series.astype(float).reset_index(drop=True)
    mu = x.mean()
    sigma = x.std(ddof=1)
    if not np.isfinite(sigma) or sigma <= 0 or len(x) < 20:
        return pd.Series([False]*len(x), index=series.index)

    z = (x - mu) / sigma
    flags = np.zeros(len(x), dtype=bool)

    # R1: beyond 3 sigma
    flags |= (np.abs(z) > 3).to_numpy()

    # R2: 9 in a row same side of mean
    side = np.sign(x - mu).to_numpy()
    run = 0
    last = 0
    for i, s in enumerate(side):
        if s == 0:
            run = 0
            last = 0
        elif s == last:
            run += 1
        else:
            run = 1
            last = s
        if run >= 9:
            flags[i] = True

    # R3: 6 in a row increasing or decreasing (5 consecutive diffs)
    inc = 0
    dec = 0
    for i in range(1, len(x)):
        if x[i] > x[i-1]:
            inc += 1
            dec = 0
        elif x[i] < x[i-1]:
            dec += 1
            inc = 0
        else:
            inc = 0
            dec = 0
        if inc >= 5 or dec >= 5:
            flags[i] = True

    # R4: 14 points alternating
    if len(x) >= 15:
        alt = 0
        for i in range(2, len(x)):
            if (x[i] > x[i-1] and x[i-1] < x[i-2]) or (x[i] < x[i-1] and x[i-1] > x[i-2]):
                alt += 1
            else:
                alt = 0
            if alt >= 13:
                flags[i] = True

    # R5: 2 of 3 beyond 2σ same side
    for i in range(2, len(x)):
        window = z[i-2:i+1]
        if ((window > 2).sum() >= 2) or ((window < -2).sum() >= 2):
            flags[i] = True

    # R6: 4 of 5 beyond 1σ same side
    for i in range(4, len(x)):
        window = z[i-4:i+1]
        if ((window > 1).sum() >= 4) or ((window < -1).sum() >= 4):
            flags[i] = True

    # R7: 15 in a row within 1σ
    for i in range(14, len(x)):
        window = np.abs(z[i-14:i+1])
        if (window < 1).all():
            flags[i] = True

    # R8: 8 in a row outside 1σ
    for i in range(7, len(x)):
        window = np.abs(z[i-7:i+1])
        if (window > 1).all():
            flags[i] = True

    return pd.Series(flags, index=series.index)

def flags_per_week(time_series: pd.Series, flag_series: pd.Series) -> float:
    tmin = time_series.min()
    tmax = time_series.max()
    span_days = (tmax - tmin).total_seconds() / 86400.0
    weeks = max(span_days / 7.0, 1e-9)
    return float(flag_series.sum() / weeks)

def estimate_sigma_from_subgroups(means: pd.Series, ranges: pd.Series, n: int, chart_type: str) -> float:
    """
    - If n==1 or Range chart is 'Moving Range' => sigma ≈ MRbar / d2(2)
    - Else => sigma ≈ Rbar / d2(n)
    """
    if n == 1 or (isinstance(chart_type, str) and "moving" in chart_type.lower()):
        mr = means.diff().abs()
        mrbar = mr.dropna().mean()
        return float(mrbar / D2[2]) if np.isfinite(mrbar) and mrbar > 0 else np.nan

    rbar = ranges.dropna().mean()
    d2 = D2.get(int(n), np.nan)
    return float(rbar / d2) if np.isfinite(rbar) and rbar > 0 and np.isfinite(d2) else np.nan

def compute_cpk(mu, sigma, lsl, usl):
    """Cp/Cpk from sigma and specs."""
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan, np.nan
    if np.isfinite(usl) and np.isfinite(lsl):
        cp = (usl - lsl) / (6*sigma)
        cpu = (usl - mu) / (3*sigma)
        cpl = (mu - lsl) / (3*sigma)
        cpk = min(cpu, cpl)
        return cp, cpk
    # one-sided spec fallback
    cpu = (usl - mu) / (3*sigma) if np.isfinite(usl) else np.nan
    cpl = (mu - lsl) / (3*sigma) if np.isfinite(lsl) else np.nan
    return np.nan, (cpu if np.isfinite(cpu) else cpl)

def trend_based_drift_time_hours(times: pd.Series, y: pd.Series, target, lsl, usl, lookback=20) -> float:
    """
    Drift-time estimate:
      - compute local slope (units/hr) on rolling windows
      - use median absolute slope as typical drift speed
      - drift time = distance from center to nearest spec / drift speed
    """
    tt = pd.to_datetime(times)
    yy = y.astype(float)
    if len(yy) < lookback + 5:
        return 72.0

    t_hours = (tt - tt.min()).dt.total_seconds() / 3600.0

    slopes = []
    for i in range(lookback, len(yy)):
        xw = t_hours.iloc[i-lookback:i].to_numpy()
        yw = yy.iloc[i-lookback:i].to_numpy()
        if np.any(~np.isfinite(yw)) or np.any(~np.isfinite(xw)):
            continue
        if xw[-1] - xw[0] <= 0:
            continue
        xm = xw.mean()
        ym = yw.mean()
        denom = ((xw - xm)**2).sum()
        if denom <= 0:
            continue
        slope = ((xw - xm)*(yw - ym)).sum() / denom
        if np.isfinite(slope):
            slopes.append(slope)

    if len(slopes) < 10:
        return 72.0

    med_abs_slope = float(np.median(np.abs(slopes)))
    if not np.isfinite(med_abs_slope) or med_abs_slope < 1e-9:
        return 72.0

    center = target if np.isfinite(target) else float(yy.mean())
    dists = []
    if np.isfinite(usl): dists.append(abs(usl - center))
    if np.isfinite(lsl): dists.append(abs(center - lsl))
    if not dists:
        return 72.0

    dist_to_spec = float(min(dists))
    drift = dist_to_spec / med_abs_slope
    return float(np.clip(drift, 0.5, 168.0))

def risk_level(cpk, nelson_wk, fail_rate):
    """Simple risk scoring based on capability + instability + failures."""
    if (np.isfinite(cpk) and cpk < 1.0) or (np.isfinite(fail_rate) and fail_rate >= 0.01) or (np.isfinite(nelson_wk) and nelson_wk >= 5):
        return 3
    if (np.isfinite(cpk) and cpk < 1.33) or (np.isfinite(fail_rate) and fail_rate >= 0.001) or (np.isfinite(nelson_wk) and nelson_wk >= 1):
        return 2
    return 1

def recommendation_from_hours(h: float) -> str:
    if not np.isfinite(h): return "Unknown"
    if h <= 0.5:  return "Every 30 min"
    if h <= 1:    return "Hourly"
    if h <= 2:    return "Every 2 Hours"
    if h <= 4:    return "Every 4 Hours"
    if h <= 8:    return "Every Shift"
    if h <= 24:   return "Once per Day"
    return "Less than daily"

# =========================
# Load data + standards
# =========================
df = pd.read_excel(data_path)
std = pd.read_excel(std_path)

df.columns = [c.strip() for c in df.columns]
std.columns = [c.strip() for c in std.columns]

df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
df = df.dropna(subset=["Date/Time"])

# Merge standards to data by Part Number (your Part Number already includes the characteristic)
dfm = df.merge(std, on="Part Number", how="left", suffixes=("", "_std"))

DATA_COLS = sorted([c for c in dfm.columns if c.startswith("Data ")],
                   key=lambda x: int(x.split()[-1]))

# =========================
# Build inspection matrix
# =========================
rows = []

for key, g in dfm.groupby(GROUP_COLS, dropna=False):
    part, dept, machine, cavity = key
    gg = g.sort_values("Date/Time").copy()
    if len(gg) < 15:
        continue

    # subgroup size: prefer standards, else data
    if "Subgroup size_std" in gg.columns and gg["Subgroup size_std"].notna().any():
        n = gg["Subgroup size_std"].dropna().iloc[0]
    else:
        n = gg["Subgroup size"].dropna().iloc[0] if "Subgroup size" in gg.columns and gg["Subgroup size"].notna().any() else 1

    try:
        n = int(n)
        n = max(1, min(n, len(DATA_COLS)))
    except:
        n = 1

    chart_type = gg["Range chart"].dropna().iloc[0] if "Range chart" in gg.columns and gg["Range chart"].notna().any() else ""

    # subgroup mean/range from Data 1..Data n on each row
    vals = gg[DATA_COLS[:n]].apply(pd.to_numeric, errors="coerce")
    subgroup_mean = vals.mean(axis=1)
    subgroup_range = (vals.max(axis=1) - vals.min(axis=1)) if n > 1 else pd.Series(np.nan, index=gg.index)

    # pull specs/target
    lsl_v = pd.to_numeric(gg["Lo Spec"], errors="coerce").dropna()
    usl_v = pd.to_numeric(gg["Hi Spec"], errors="coerce").dropna()
    tgt_v = pd.to_numeric(gg["Target x"], errors="coerce").dropna()

    lsl = float(lsl_v.iloc[0]) if len(lsl_v) else np.nan
    usl = float(usl_v.iloc[0]) if len(usl_v) else np.nan
    target = float(tgt_v.iloc[0]) if len(tgt_v) else np.nan

    # sigma estimate from MR or R
    sigma = estimate_sigma_from_subgroups(subgroup_mean, subgroup_range, n, chart_type)
    mu = float(subgroup_mean.mean())

    cp, cpk = compute_cpk(mu, sigma, lsl, usl)

    # Nelson flags computed from the subgroup means sequence
    rule_flags = nelson_rule_flags(subgroup_mean)
    nel_wk = flags_per_week(gg["Date/Time"], rule_flags)

    # failure rate from export
    fail_rate = float((pd.to_numeric(gg.get("Real-time failures", 0), errors="coerce").fillna(0) > 0).mean())

    # drift time toward nearest spec based on observed trend speed
    drift_h = trend_based_drift_time_hours(gg["Date/Time"], subgroup_mean, target, lsl, usl, lookback=20)

    # risk score
    risk = risk_level(cpk, nel_wk, fail_rate)
    safety = SAFETY_FACTOR_DEFAULT

    # recommended interval (hrs)
    interval_h = drift_h / (risk * safety)
    if np.isfinite(nel_wk):
        interval_h *= (1.0 / (1.0 + nel_wk / 5.0))
    interval_h = float(np.clip(interval_h, 0.25, 168.0))

    # actual current cadence
    deltas = pd.to_datetime(gg["Date/Time"]).diff().dt.total_seconds().dropna() / 3600.0
    current_med_h = float(deltas.median()) if len(deltas) else np.nan

    rows.append({
        "Part Number": part,
        "Characteristic (Std Description)": gg["Description"].dropna().iloc[0] if "Description" in gg.columns and gg["Description"].notna().any() else "",
        "Department": dept,
        "Machine": machine,
        "Cavity": cavity,
        "Chart type": chart_type,
        "Subgroup size": n,
        "N subgroups": int(len(gg)),
        "Mean (X̄)": mu,
        "Sigma estimate": sigma,
        "Lo Spec": lsl,
        "Hi Spec": usl,
        "Target": target,
        "Cp": cp,
        "Cpk": cpk,
        "Nelson flags per week (calc)": nel_wk,
        "Failure rate": fail_rate,
        "Estimated drift time to spec (hrs)": drift_h,
        "Risk level (1-3)": risk,
        "Safety factor": safety,
        "Calculated inspection interval (hrs)": interval_h,
        "Recommended frequency": recommendation_from_hours(interval_h),
        "Current median interval (hrs)": current_med_h
    })

matrix = pd.DataFrame(rows).sort_values(
    ["Risk level (1-3)", "Calculated inspection interval (hrs)", "Cpk"],
    ascending=[False, True, True]
).reset_index(drop=True)

# simplified operational view
freq = matrix[[
    "Part Number", "Characteristic (Std Description)", "Department", "Machine", "Cavity",
    "Risk level (1-3)", "Calculated inspection interval (hrs)", "Recommended frequency",
    "Cpk", "Nelson flags per week (calc)", "Failure rate", "Estimated drift time to spec (hrs)",
    "Current median interval (hrs)"
]].copy()

# standards coverage tab
coverage = dfm[["Part Number"]].drop_duplicates().merge(
    std[["Part Number","Description","Lo Spec","Hi Spec","Target x","Subgroup size","Range chart"]],
    on="Part Number", how="left"
)

# export
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    freq.to_excel(writer, index=False, sheet_name="Inspection_Frequency")
    matrix.to_excel(writer, index=False, sheet_name="Inspection_Matrix")
    coverage.to_excel(writer, index=False, sheet_name="Standards_Coverage")

print(f"Saved: {output_path}")
