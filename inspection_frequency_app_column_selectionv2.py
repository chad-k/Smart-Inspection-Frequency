# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 10:51:53 2026

@author: chad
"""

# app.py
# Smart Inspection Frequency – Streamlit Dashboard
# ULTIMATE FLEXIBILITY: User selects ALL important columns
#
# Sidebar has:
# 1. Grouping columns selection
# 2. Measurement columns selection
# 3. Specification columns selection (Lo Spec, Hi Spec, Target, Description, etc.)

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Inspection Frequency", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
REPO_EXPORT_PATH = DATA_DIR / "tour_data_export.xlsx"
REPO_STANDARDS_PATH = DATA_DIR / "tour_standards.xlsx"

SAFETY_FACTOR_DEFAULT = 2.0

D2 = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
    9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407,
    15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735
}

# =========================
# Helper functions
# =========================
def nelson_rule_flags(series: pd.Series) -> pd.Series:
    """Nelson control chart rules detection."""
    x = series.astype(float).reset_index(drop=True)
    mu = x.mean()
    sigma = x.std(ddof=1)

    if not np.isfinite(sigma) or sigma <= 0 or len(x) < 20:
        return pd.Series([False] * len(x), index=series.index)

    z = (x - mu) / sigma
    flags = np.zeros(len(x), dtype=bool)

    flags |= (np.abs(z) > 3).to_numpy()

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

    inc = 0
    dec = 0
    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            inc += 1
            dec = 0
        elif x[i] < x[i - 1]:
            dec += 1
            inc = 0
        else:
            inc = 0
            dec = 0
        if inc >= 6 or dec >= 6:
            flags[i] = True

    if len(x) >= 15:
        alt = 0
        for i in range(2, len(x)):
            if (x[i] > x[i - 1] and x[i - 1] < x[i - 2]) or (x[i] < x[i - 1] and x[i - 1] > x[i - 2]):
                alt += 1
            else:
                alt = 0
            if alt >= 13:
                flags[i] = True

    for i in range(2, len(x)):
        window = z[i - 2:i + 1]
        if ((window > 2).sum() >= 2) or ((window < -2).sum() >= 2):
            flags[i] = True

    for i in range(4, len(x)):
        window = z[i - 4:i + 1]
        if ((window > 1).sum() >= 4) or ((window < -1).sum() >= 4):
            flags[i] = True

    for i in range(14, len(x)):
        window = np.abs(z[i - 14:i + 1])
        if (window < 1).all():
            flags[i] = True

    for i in range(7, len(x)):
        window = np.abs(z[i - 7:i + 1])
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
    if n == 1 or (isinstance(chart_type, str) and "moving" in chart_type.lower()):
        mr = means.diff().abs()
        mrbar = mr.dropna().mean()
        return float(mrbar / D2[2]) if np.isfinite(mrbar) and mrbar > 0 else np.nan

    rbar = ranges.dropna().mean()
    d2 = D2.get(int(n), np.nan)
    return float(rbar / d2) if np.isfinite(rbar) and rbar > 0 and np.isfinite(d2) else np.nan


def compute_cpk(mu, sigma, lsl, usl):
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan, np.nan
    if np.isfinite(usl) and np.isfinite(lsl):
        cp = (usl - lsl) / (6 * sigma)
        cpu = (usl - mu) / (3 * sigma)
        cpl = (mu - lsl) / (3 * sigma)
        cpk = min(cpu, cpl)
        return cp, cpk
    cpu = (usl - mu) / (3 * sigma) if np.isfinite(usl) else np.nan
    cpl = (mu - lsl) / (3 * sigma) if np.isfinite(lsl) else np.nan
    return np.nan, (cpu if np.isfinite(cpu) else cpl)


def trend_based_drift_time_hours(times: pd.Series, y: pd.Series, target, lsl, usl, lookback=20) -> float:
    tt = pd.to_datetime(times)
    yy = y.astype(float)

    if len(yy) < lookback + 5:
        return 72.0

    t_hours = (tt - tt.min()).dt.total_seconds() / 3600.0

    slopes = []
    for i in range(lookback, len(yy)):
        xw = t_hours.iloc[i - lookback:i].to_numpy()
        yw = yy.iloc[i - lookback:i].to_numpy()
        if np.any(~np.isfinite(yw)) or np.any(~np.isfinite(xw)):
            continue
        if xw[-1] - xw[0] <= 0:
            continue
        xm = xw.mean()
        ym = yw.mean()
        denom = ((xw - xm) ** 2).sum()
        if denom <= 0:
            continue
        slope = ((xw - xm) * (yw - ym)).sum() / denom
        if np.isfinite(slope):
            slopes.append(slope)

    if len(slopes) < 10:
        return 72.0

    med_abs_slope = float(np.median(np.abs(slopes)))
    if not np.isfinite(med_abs_slope) or med_abs_slope < 1e-9:
        return 72.0

    center = target if np.isfinite(target) else float(yy.mean())

    dists = []
    if np.isfinite(usl):
        dists.append(abs(usl - center))
    if np.isfinite(lsl):
        dists.append(abs(center - lsl))
    if not dists:
        return 72.0

    dist_to_spec = float(min(dists))
    drift = dist_to_spec / med_abs_slope
    return float(np.clip(drift, 0.5, 168.0))


def risk_level(cpk, nelson_wk, fail_rate):
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


@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


def build_outputs(df: pd.DataFrame, std: pd.DataFrame, safety_factor: float, 
                  data_columns: list, group_columns: list, spec_columns: dict):
    """
    Build inspection frequency matrices.
    
    spec_columns: dict with keys:
        - lo_spec_col: column name for lower spec
        - hi_spec_col: column name for upper spec
        - target_col: column name for target
        - description_col: column name for description
        - subgroup_size_col: column name for subgroup size
        - range_chart_col: column name for chart type
    """
    df = df.copy()
    std = std.copy()

    df.columns = [c.strip() for c in df.columns]
    std.columns = [c.strip() for c in std.columns]

    if "Date/Time" not in df.columns:
        raise ValueError("Export file must contain a 'Date/Time' column.")
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df = df.dropna(subset=["Date/Time"])

    if "Part Number" not in df.columns or "Part Number" not in std.columns:
        raise ValueError("Both files must contain 'Part Number'.")

    dfm = df.merge(std, on="Part Number", how="left", suffixes=("", "_std"))

    missing_group = [c for c in group_columns if c not in dfm.columns]
    if missing_group:
        raise ValueError(f"Grouping columns not found: {missing_group}")

    missing_data = [c for c in data_columns if c not in dfm.columns]
    if missing_data:
        raise ValueError(f"Measurement columns not found: {missing_data}")

    DATA_COLS = data_columns
    GROUP_COLS = group_columns

    if not DATA_COLS:
        raise ValueError("No measurement columns selected.")
    if not GROUP_COLS:
        raise ValueError("No grouping columns selected.")

    rows = []
    excluded_groups = []

    for key, g in dfm.groupby(GROUP_COLS, dropna=False):
        if len(GROUP_COLS) == 1:
            key_dict = {GROUP_COLS[0]: key}
        else:
            key_dict = {col: val for col, val in zip(GROUP_COLS, key)}
        
        gg = g.sort_values("Date/Time").copy()

        if len(gg) < 15:
            excluded_groups.append({
                **key_dict,
                "Subgroup count": len(gg),
                "Reason": f"Only {len(gg)} subgroups (minimum: 15)"
            })
            continue

        # Get subgroup size
        n = 1
        if spec_columns.get("subgroup_size_col") and spec_columns["subgroup_size_col"] in gg.columns:
            sz = pd.to_numeric(gg[spec_columns["subgroup_size_col"]], errors="coerce").dropna()
            if len(sz):
                n = int(sz.iloc[0])
        try:
            n = max(1, min(n, len(DATA_COLS)))
        except Exception:
            n = 1

        # Get chart type
        chart_type = ""
        if spec_columns.get("range_chart_col") and spec_columns["range_chart_col"] in gg.columns:
            ct = gg[spec_columns["range_chart_col"]].dropna()
            if len(ct):
                chart_type = str(ct.iloc[0])

        vals = gg[DATA_COLS[:n]].apply(pd.to_numeric, errors="coerce")
        subgroup_mean = vals.mean(axis=1)
        subgroup_range = (vals.max(axis=1) - vals.min(axis=1)) if n > 1 else pd.Series(np.nan, index=gg.index)

        # Get spec limits from user-selected columns
        lsl = np.nan
        usl = np.nan
        target = np.nan

        if spec_columns.get("lo_spec_col") and spec_columns["lo_spec_col"] in gg.columns:
            lsl_v = pd.to_numeric(gg[spec_columns["lo_spec_col"]], errors="coerce").dropna()
            lsl = float(lsl_v.iloc[0]) if len(lsl_v) else np.nan

        if spec_columns.get("hi_spec_col") and spec_columns["hi_spec_col"] in gg.columns:
            usl_v = pd.to_numeric(gg[spec_columns["hi_spec_col"]], errors="coerce").dropna()
            usl = float(usl_v.iloc[0]) if len(usl_v) else np.nan

        if spec_columns.get("target_col") and spec_columns["target_col"] in gg.columns:
            tgt_v = pd.to_numeric(gg[spec_columns["target_col"]], errors="coerce").dropna()
            target = float(tgt_v.iloc[0]) if len(tgt_v) else np.nan

        sigma = estimate_sigma_from_subgroups(subgroup_mean, subgroup_range, n, chart_type)
        mu = float(subgroup_mean.mean())

        cp, cpk = compute_cpk(mu, sigma, lsl, usl)

        rule_flags = nelson_rule_flags(subgroup_mean)
        nel_wk = flags_per_week(gg["Date/Time"], rule_flags)

        # Handle Real-time failures column safely
        if "Real-time failures" in gg.columns:
            failures = pd.to_numeric(gg["Real-time failures"], errors="coerce").fillna(0)
            fail_rate = float((failures > 0).mean())
        else:
            fail_rate = 0.0

        drift_h = trend_based_drift_time_hours(gg["Date/Time"], subgroup_mean, target, lsl, usl, lookback=20)

        risk = risk_level(cpk, nel_wk, fail_rate)

        interval_h = drift_h / (risk * safety_factor)
        if np.isfinite(nel_wk):
            interval_h *= (1.0 / (1.0 + nel_wk / 5.0))
        interval_h = float(np.clip(interval_h, 0.25, 168.0))

        deltas = pd.to_datetime(gg["Date/Time"]).diff().dt.total_seconds().dropna() / 3600.0
        current_med_h = float(deltas.median()) if len(deltas) else np.nan

        # Get description
        description = ""
        if spec_columns.get("description_col") and spec_columns["description_col"] in gg.columns:
            desc = gg[spec_columns["description_col"]].dropna()
            if len(desc):
                description = str(desc.iloc[0])

        row = {
            **key_dict,
            "Characteristic (Std Description)": description,
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
            "Safety factor": float(safety_factor),
            "Calculated inspection interval (hrs)": interval_h,
            "Recommended frequency": recommendation_from_hours(interval_h),
            "Current median interval (hrs)": current_med_h
        }

        rows.append(row)

    matrix = pd.DataFrame(rows)
    if not matrix.empty:
        matrix = matrix.sort_values(
            ["Risk level (1-3)", "Calculated inspection interval (hrs)", "Cpk"],
            ascending=[False, True, True]
        ).reset_index(drop=True)

    freq_cols = list(group_columns) + [
        "Characteristic (Std Description)",
        "Risk level (1-3)", "Calculated inspection interval (hrs)", "Recommended frequency",
        "Cpk", "Nelson flags per week (calc)", "Failure rate",
        "Estimated drift time to spec (hrs)", "Current median interval (hrs)"
    ]
    freq = matrix[freq_cols].copy() if not matrix.empty else pd.DataFrame(columns=freq_cols)

    coverage = dfm[["Part Number"]].drop_duplicates()

    return dfm, freq, matrix, coverage, pd.DataFrame(excluded_groups)


# ============================================================================
# STREAMLIT UI - COMPLETE COLUMN SELECTION
# ============================================================================
st.title("Smart Inspection Frequency")
st.caption("Select all columns explicitly • Maximum transparency")

with st.sidebar:
    st.header("📥 Load Data")
    mode = st.radio("Data source", ["Use GitHub repo files", "Upload Excel files"], index=0)

    try:
        if mode == "Upload Excel files":
            export_up = st.file_uploader("Upload tour_data_export.xlsx", type=["xlsx"])
            std_up = st.file_uploader("Upload tour_standards.xlsx", type=["xlsx"])
            
            if not export_up or not std_up:
                st.error("Please upload both files")
                st.stop()
            
            df = read_excel_bytes(export_up.getvalue())
            std = read_excel_bytes(std_up.getvalue())
        else:
            if not REPO_EXPORT_PATH.exists() or not REPO_STANDARDS_PATH.exists():
                st.error("Repo files not found")
                st.stop()
            
            df = pd.read_excel(REPO_EXPORT_PATH)
            std = pd.read_excel(REPO_STANDARDS_PATH)
    except Exception as e:
        st.error(f"File load failed: {e}")
        st.stop()

    export_cols = sorted([c.strip() for c in df.columns])
    std_cols = sorted([c.strip() for c in std.columns])
    
    # Show available columns
    with st.expander("📋 Available columns"):
        st.write("**Export file:**")
        st.code(", ".join(export_cols))
        st.write("**Standards file:**")
        st.code(", ".join(std_cols))

    # ========================================================================
    # GROUPING COLUMNS
    # ========================================================================
    st.header("🔍 Grouping Columns")
    st.write("Which columns identify each process?")
    
    default_group = [c for c in export_cols if c in ["Part Number", "Department", "Machine", "Cavity"]]
    selected_group_cols = st.multiselect(
        label="Select grouping columns",
        options=export_cols,
        default=default_group,
        help="Each unique combination gets its own inspection frequency."
    )
    
    if not selected_group_cols:
        st.warning("⚠️ Select at least one grouping column")
        st.stop()
    
    st.info(f"✅ Grouping by: {', '.join(selected_group_cols)}")

    # ========================================================================
    # MEASUREMENT COLUMNS
    # ========================================================================
    st.header("📊 Measurement Columns")
    st.write("Which columns contain the actual measurement data?")
    
    default_data = [c for c in export_cols if c.startswith("Data ")]
    selected_data_cols = st.multiselect(
        label="Select measurement columns",
        options=export_cols,
        default=default_data,
        help="Columns with subgroup measurements"
    )
    
    if not selected_data_cols:
        st.warning("⚠️ Select at least one measurement column")
        st.stop()
    
    st.success(f"✅ Using {len(selected_data_cols)} measurement columns")

    # ========================================================================
    # SPECIFICATION COLUMNS (from standards file)
    # ========================================================================
    st.header("📏 Specification Columns")
    st.write("Which columns contain specification limits? (Optional - leave blank if not available)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lo_spec_col = st.selectbox(
            "Lower spec limit (Lo Spec)",
            options=[None] + std_cols,
            help="Column with lower specification limits"
        )
        
        target_col = st.selectbox(
            "Target value",
            options=[None] + std_cols,
            help="Column with target/nominal values"
        )
        
        description_col = st.selectbox(
            "Description/Characteristic",
            options=[None] + std_cols,
            help="Column with characteristic descriptions"
        )
    
    with col2:
        hi_spec_col = st.selectbox(
            "Upper spec limit (Hi Spec)",
            options=[None] + std_cols,
            help="Column with upper specification limits"
        )
        
        subgroup_size_col = st.selectbox(
            "Subgroup size",
            options=[None] + std_cols,
            help="Column with subgroup sample sizes"
        )
        
        range_chart_col = st.selectbox(
            "Range chart type",
            options=[None] + std_cols,
            help="Column with control chart types"
        )
    
    st.info("✅ Specification columns selected")

    # ========================================================================
    # TUNING
    # ========================================================================
    st.header("⚙️ Tuning")
    safety_factor = st.number_input(
        "Safety factor",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.25
    )

    st.header("▶️ Run")
    run_btn = st.button("Build inspection frequency", type="primary")

if not run_btn:
    st.stop()

# Build spec_columns dict from user selections
spec_columns = {
    "lo_spec_col": lo_spec_col,
    "hi_spec_col": hi_spec_col,
    "target_col": target_col,
    "description_col": description_col,
    "subgroup_size_col": subgroup_size_col,
    "range_chart_col": range_chart_col,
}

with st.spinner("Computing inspection frequency..."):
    try:
        dfm, freq, matrix, coverage, excluded_df = build_outputs(
            df, std, safety_factor,
            data_columns=selected_data_cols,
            group_columns=selected_group_cols,
            spec_columns=spec_columns
        )
    except Exception as e:
        st.error(f"❌ Computation failed: {e}")
        st.stop()

# Display results
c1, c2, c3 = st.columns(3)
c1.metric("Rows in export", f"{len(df):,}")
c2.metric("Rows in standards", f"{len(std):,}")
c3.metric("Matrix rows produced", f"{len(matrix):,}")

if not excluded_df.empty:
    st.warning(f"⚠️ {len(excluded_df)} groups excluded (fewer than 15 subgroups)")
    with st.expander("View excluded groups"):
        st.dataframe(excluded_df, use_container_width=True)

if matrix.empty:
    st.error("❌ No groups with sufficient data (15+ subgroups) found.")
    st.stop()

tabs = st.tabs(["Inspection_Frequency", "Inspection_Matrix", "Standards_Coverage", "Raw Preview"])

with tabs[0]:
    st.subheader("Inspection_Frequency")
    st.dataframe(freq, use_container_width=True)

with tabs[1]:
    st.subheader("Inspection_Matrix")
    st.dataframe(matrix, use_container_width=True)

with tabs[2]:
    st.subheader("Standards_Coverage")
    st.dataframe(coverage, use_container_width=True)

with tabs[3]:
    st.subheader("Raw Export Preview (first 200 rows)")
    st.dataframe(df.head(200), use_container_width=True)

# Download
st.subheader("📥 Download Excel Output")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    freq.to_excel(writer, index=False, sheet_name="Inspection_Frequency")
    matrix.to_excel(writer, index=False, sheet_name="Inspection_Matrix")
    coverage.to_excel(writer, index=False, sheet_name="Standards_Coverage")
    if not excluded_df.empty:
        excluded_df.to_excel(writer, index=False, sheet_name="Excluded_Groups")

st.download_button(
    "Download inspection_frequency_and_matrix.xlsx",
    data=out.getvalue(),
    file_name="inspection_frequency_and_matrix.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
