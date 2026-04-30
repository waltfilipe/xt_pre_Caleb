import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard")

# ==========================
# CSS
# ==========================
st.markdown(
    """
    <style>
    .small-metric { padding: 6px 8px; }
    .small-metric .label {
      font-size: 12px; color: #ffffff; margin-bottom: 3px; opacity: 0.95;
    }
    .small-metric .value {
      font-size: 18px; font-weight: 600; color: #ffffff;
    }
    .small-metric .delta {
      font-size: 11px; color: #e6e6e6; margin-top: 4px;
    }
    .stats-section-title {
      font-size: 14px; font-weight: 600; margin-bottom: 6px; color: #ffffff;
    }
    .streamlit-expanderHeader { color: #ffffff !important; }
    .streamlit-expander { background: rgba(255,255,255,0.02); }
    .filter-panel {
      background: linear-gradient(168deg, rgba(30,39,56,0.92) 0%, rgba(22,28,40,0.97) 100%);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 24px 18px 20px 18px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25), 0 1px 4px rgba(0,0,0,0.12);
      backdrop-filter: blur(6px);
    }
    .filter-panel h3 {
      font-size: 15px; color: #c8d6e5; letter-spacing: 0.5px; margin-bottom: 8px;
    }
    .filter-panel .filter-divider {
      border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 14px 0;
    }
    .stSubheader { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def small_metric(label: str, value: str, delta: str | None = None):
    html = f"""
    <div class="small-metric">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
    """
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ==========================
# Title
# ==========================
st.title("Pass Map Dashboard")

# ==========================
# Configuration / Constants
# ==========================
FIELD_X, FIELD_Y       = 120.0, 80.0
HALF_LINE_X            = FIELD_X / 2          # 60.0
FINAL_THIRD_LINE_X     = 80.0

LANE_LEFT_MIN  = 53.33
LANE_RIGHT_MAX = 26.67

LATERAL_MIN_DIST = 12.0

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_SUCCESS    = "#c8c8c8"   # light grey  – successful, non-progressive
COLOR_PROGRESSIVE = "#2F80ED"  # blue        – progressive (successful)
COLOR_FAIL        = "#E07070"  # light red   – failed
ALPHA_SUCCESS     = 0.07       # very transparent for plain successful passes

FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

# ── Position → match mapping ─────────────────────────────────────────────────
POSITION_BY_MATCH: dict[str, str] = {
    "Vs Sacramento United": "LCB",
    "Vs Capital City":      "LCB",
    "Vs Wake FC":           "RCB",
}

# ==========================
# DATA
# Format: (type, x_start, y_start, x_end, y_end, foot)
#   type : "PASS WON" | "PASS LOST"
#   foot : "strong" | "weak"   (dominant foot → "strong foot" in football English)
# ==========================
matches_data = {
    "Vs Sacramento United": [
        # ── Completed – Strong Foot ───────────────────────────────────────
        ("PASS WON",  6.14, 26.04, 15.28, 25.70, "strong"),
        ("PASS WON", 28.08, 22.05, 28.42, 58.78, "strong"),
        ("PASS WON", 42.05, 20.22, 49.19,  4.59, "strong"),
        ("PASS WON", 46.54, 17.89, 66.98,  2.60, "strong"),
        ("PASS WON", 49.36, 11.41, 63.66, 12.07, "strong"),
        ("PASS WON", 63.99, 26.87, 75.63, 10.08, "strong"),
        ("PASS WON", 71.14, 44.99, 52.35, 44.49, "strong"),
        ("PASS WON", 39.65,  4.70, 22.43, 30.09, "strong"),
        ("PASS WON", 34.96, 21.74, 32.52, 56.35, "strong"),
        ("PASS WON", 41.74, 23.65, 56.52, 50.96, "strong"),
        ("PASS WON", 48.17, 56.17, 69.91, 54.61, "strong"),
        ("PASS WON", 55.48, 65.74, 41.04, 74.09, "strong"),
        ("PASS WON", 58.61, 36.35, 56.87, 65.39, "strong"),
        ("PASS WON", 55.83, 22.09, 55.30, 36.52, "strong"),
        ("PASS WON", 62.43, 37.57, 78.61,  7.48, "strong"),
        ("PASS WON", 52.67, 10.30, 46.43, 33.22, "strong"),
        ("PASS WON", 77.60,  5.72, 83.83, 15.62, "strong"),
        ("PASS WON", 73.75, 12.87, 91.17,  7.37, "strong"),
        ("PASS WON", 68.25, 19.28, 72.28,  7.00, "strong"),
        ("PASS WON", 64.95, 18.73, 82.18, 35.05, "strong"),
        ("PASS WON", 69.72, 26.98, 83.10, 40.92, "strong"),
        ("PASS WON", 80.17, 18.92, 65.68, 41.28, "strong"),
        ("PASS WON", 77.42, 22.58, 71.00, 42.02, "strong"),
        ("PASS WON", 73.20, 19.65, 75.22, 39.27, "strong"),
        ("PASS WON", 34.40, 19.89, 43.71,  6.92, "strong"),
        ("PASS WON", 37.06, 19.05, 46.87,  7.92, "strong"),
        ("PASS WON", 38.06, 23.21, 38.72, 53.46, "strong"),
        ("PASS WON", 54.18, 32.35, 54.68, 55.46, "strong"),
        ("PASS WON", 48.53, 19.22, 72.63, 56.29, "strong"),
        ("PASS WON", 52.02, 18.39, 83.77, 11.08, "strong"),
        ("PASS WON", 75.13, 23.21, 78.12, 39.34, "strong"),
        ("PASS WON", 78.78, 19.72, 81.44, 47.98, "strong"),
        ("PASS WON", 81.94, 22.88, 88.09, 43.66, "strong"),
        ("PASS WON", 40.05, 17.23, 47.03, 11.57, "strong"),
        ("PASS WON", 46.20, 16.06, 53.52,  5.09, "strong"),
        ("PASS WON", 37.39, 18.39, 52.35, 27.37, "strong"),
        ("PASS WON", 84.60, 23.88, 94.91,  7.92, "strong"),
        ("PASS WON", 76.12, 20.38, 81.44, 41.00, "strong"),
        ("PASS WON", 66.48, 31.69, 84.94, 40.50, "strong"),
        ("PASS WON", 58.17, 38.17, 75.29, 15.56, "strong"),
        ("PASS WON", 67.31, 27.70, 88.43, 23.54, "strong"),
        ("PASS WON", 83.77, 21.38, 81.11, 41.16, "strong"),
        ("PASS WON",  2.48,  4.26, 12.29, 16.56, "strong"),
        ("PASS WON", 14.62,  2.76,  2.82, 22.21, "strong"),
        ("PASS WON", 45.37, 15.06, 43.88, 53.96, "strong"),
        ("PASS WON", 59.83, 36.18, 30.41, 38.01, "strong"),
        ("PASS WON", 45.04, 18.22, 54.68, 32.35, "strong"),
        ("PASS WON", 55.18, 32.02, 64.65, 15.23, "strong"),
        ("PASS WON", 51.02, 19.89, 65.82,  2.76, "strong"),
        ("PASS WON", 76.12, 19.72, 81.61, 30.36, "strong"),
        ("PASS WON", 90.59, 10.74, 88.59, 27.70, "strong"),
        ("PASS WON", 82.61, 22.55, 97.74,  5.76, "strong"),
        ("PASS WON", 62.33, 36.84, 69.81, 46.15, "strong"),
        # ── Incomplete – Strong Foot ──────────────────────────────────────
        ("PASS LOST", 82.94, 31.19, 95.74, 35.01, "strong"),
        ("PASS LOST", 84.27, 35.68, 75.63, 34.18, "strong"),
        ("PASS LOST", 51.36, 12.07, 66.65, 19.05, "strong"),
        ("PASS LOST", 13.95, 22.21, 56.84, 38.84, "strong"),
        # ── Completed – Weak Foot ─────────────────────────────────────────
        ("PASS WON",  0.65, 11.57, 12.12,  2.26, "weak"),
        ("PASS WON", 11.13, 22.71,  2.65, 40.17, "weak"),
        ("PASS WON", 51.85, 13.07, 66.48, 12.41, "weak"),
        ("PASS WON", 36.39, 21.71, 51.52, 17.56, "weak"),
        ("PASS WON", 46.87, 17.23, 53.68, 25.54, "weak"),
        ("PASS WON", 39.05, 17.39, 38.56, 49.64, "weak"),
        ("PASS WON", 59.34, 17.39, 53.35, 37.01, "weak"),
        ("PASS WON", 54.68, 18.39, 56.18, 38.84, "weak"),
        ("PASS WON", 72.97, 20.38, 63.82, 47.15, "weak"),
        ("PASS WON", 73.80, 23.21, 78.62, 50.81, "weak"),
        # ── Incomplete – Weak Foot ────────────────────────────────────────
        ("PASS LOST", 50.52, 22.05, 70.31, 16.23, "weak"),
    ],

    "Vs Capital City": [
        # ── Completed – Strong Foot ───────────────────────────────────────
        ("PASS WON", 15.12, 11.41, 26.42,  1.93, "strong"),
        ("PASS WON", 15.12,  9.25, 14.62, 39.83, "strong"),
        ("PASS WON", 14.78, 25.54, 14.78, 51.14, "strong"),
        ("PASS WON", 21.60, 19.55,  6.31, 35.35, "strong"),
        ("PASS WON", 33.57, 21.55, 39.55, 10.24, "strong"),
        ("PASS WON", 34.23, 22.71, 37.06, 50.97, "strong"),
        ("PASS WON", 36.89, 25.21, 12.96, 33.52, "strong"),
        ("PASS WON", 27.25, 34.18, 52.69, 24.71, "strong"),
        ("PASS WON", 36.89, 27.86, 56.34, 19.39, "strong"),
        ("PASS WON", 59.17, 38.17, 70.64, 41.83, "strong"),
        ("PASS WON", 55.35, 38.17, 77.29, 28.20, "strong"),
        ("PASS WON", 64.65, 26.20, 81.94, 10.41, "strong"),
        ("PASS WON", 82.11, 14.23, 93.25, 13.74, "strong"),
        ("PASS WON", 75.46, 26.37, 74.79, 51.80, "strong"),
        ("PASS WON", 78.78, 32.85, 84.94, 27.53, "strong"),
        ("PASS WON", 72.30, 48.98, 87.26, 49.97, "strong"),
        ("PASS WON", 13.95, 24.04, 20.27, 11.74, "strong"),
        ("PASS WON", 23.93, 27.37, 23.26, 16.06, "strong"),
        ("PASS WON", 20.77, 27.53, 29.25, 57.29, "strong"),
        ("PASS WON", 52.52, 10.91, 31.41, 32.02, "strong"),
        ("PASS WON", 53.68, 18.39, 44.37, 37.01, "strong"),
        ("PASS WON", 55.01, 22.55, 68.15, 11.57, "strong"),
        ("PASS WON", 57.84, 24.37, 82.94,  6.09, "strong"),
        ("PASS WON", 63.66, 23.88, 87.43,  6.42, "strong"),
        ("PASS WON", 71.97, 12.24, 63.32, 39.67, "strong"),
        ("PASS WON", 63.16, 40.50, 51.19, 61.11, "strong"),
        ("PASS WON", 66.48, 39.67, 81.28, 41.00, "strong"),
        ("PASS WON", 62.66, 45.98, 76.12, 31.85, "strong"),
        # ── Incomplete – Strong Foot ──────────────────────────────────────
        ("PASS LOST", 52.35, 30.52, 92.25, 21.22, "strong"),
        ("PASS LOST", 59.83, 37.84, 78.12, 36.01, "strong"),
        ("PASS LOST", 53.02, 45.98, 68.81, 39.67, "strong"),
        ("PASS LOST", 22.93, 46.82, 94.41, 78.23, "strong"),
        # ── Completed – Weak Foot ─────────────────────────────────────────
        ("PASS WON", 57.34, 30.36, 55.01, 55.13, "weak"),
        ("PASS WON", 63.49, 44.82, 76.62, 39.83, "weak"),
    ],

    "Vs Wake FC": [
        # ── Completed – Strong Foot ───────────────────────────────────────
        ("PASS WON", 39.05, 13.40, 27.25, 27.70, "strong"),
        ("PASS WON", 13.45, 52.30, 35.56, 52.47, "strong"),
        ("PASS WON", 29.75, 61.11, 12.29, 41.50, "strong"),
        ("PASS WON", 36.56, 55.46, 37.23, 27.20, "strong"),
        ("PASS WON", 37.06, 25.04, 56.34, 50.97, "strong"),
        ("PASS WON", 53.02, 48.15, 70.31, 29.19, "strong"),
        ("PASS WON", 71.64, 53.80, 43.21, 39.83, "strong"),
        ("PASS WON", 32.24, 59.95, 56.34, 57.79, "strong"),
        ("PASS WON", 41.05, 63.44, 51.85, 73.41, "strong"),
        ("PASS WON", 45.54, 74.91, 77.62, 74.74, "strong"),
        ("PASS WON", 71.97, 75.57, 76.79, 69.76, "strong"),
        ("PASS WON", 82.28, 70.09, 72.97, 61.11, "strong"),
        ("PASS WON", 72.63, 52.14, 41.88, 40.17, "strong"),
        ("PASS WON", 78.29, 50.64, 77.29, 23.71, "strong"),
        ("PASS WON", 89.09, 42.49, 79.78, 41.99, "strong"),
        ("PASS WON", 78.62, 47.15, 85.77, 24.54, "strong"),
        ("PASS WON", 34.23, 21.38, 11.46, 34.51, "strong"),
        ("PASS WON", 23.10, 43.99, 31.91, 44.16, "strong"),
        ("PASS WON", 18.44, 51.64, 28.91, 76.07, "strong"),
        ("PASS WON", 58.67, 37.01, 63.16, 18.56, "strong"),
        ("PASS WON", 37.06, 55.46, 22.76, 46.15, "strong"),
        ("PASS WON", 37.56, 50.14, 35.23, 32.35, "strong"),
        ("PASS WON", 41.55, 53.46, 40.72, 30.03, "strong"),
        ("PASS WON", 42.71, 35.68, 57.51, 24.87, "strong"),
        ("PASS WON", 42.55, 49.31, 79.45, 46.32, "strong"),
        ("PASS WON", 48.86, 58.29, 73.30, 59.28, "strong"),
        ("PASS WON", 61.33, 61.11, 79.78, 48.15, "strong"),
        ("PASS WON", 65.15, 73.58, 66.65, 35.35, "strong"),
        ("PASS WON", 55.35, 36.01, 68.15,  9.08, "strong"),
        ("PASS WON", 73.96, 53.13, 86.76, 53.96, "strong"),
        ("PASS WON", 80.61, 60.11, 93.25, 69.76, "strong"),
        ("PASS WON", 52.19, 52.80, 65.65, 68.59, "strong"),
        ("PASS WON", 47.70, 73.75, 59.00, 74.58, "strong"),
        ("PASS WON", 40.72, 33.52, 25.76, 43.32, "strong"),
        ("PASS WON", 26.75, 50.31, 36.06, 39.67, "strong"),
        ("PASS WON", 27.09, 51.14, 40.88, 45.49, "strong"),
        ("PASS WON", 27.92, 57.95, 32.07, 26.70, "strong"),
        ("PASS WON", 20.10, 56.46, 33.40, 49.81, "strong"),
        ("PASS WON", 31.24, 64.44,  5.31, 36.18, "strong"),
        ("PASS WON", 26.59, 56.96, 33.40, 75.91, "strong"),
        ("PASS WON", 20.10, 67.76, 36.39, 57.79, "strong"),
        ("PASS WON",  8.14, 64.94, 15.95, 78.07, "strong"),
        ("PASS WON",  0.49, 71.25, 20.10, 72.58, "strong"),
        ("PASS WON",  5.31, 72.25, 18.28, 57.45, "strong"),
        # ── Incomplete – Strong Foot ──────────────────────────────────────
        ("PASS LOST", 15.45, 44.32, 23.43, 44.65, "strong"),
        ("PASS LOST", 13.45, 48.98, 41.71, 56.12, "strong"),
        ("PASS LOST", 17.44, 66.93, 27.92, 54.96, "strong"),
        ("PASS LOST", 42.88, 48.81, 65.82, 40.17, "strong"),
        ("PASS LOST", 68.98, 16.39, 46.87, 33.85, "strong"),
        # ── Completed – Weak Foot ─────────────────────────────────────────
        ("PASS WON", 22.26, 15.73, 30.58, 37.51, "weak"),
        ("PASS WON",  8.80, 56.79,  2.65, 35.35, "weak"),
        ("PASS WON", 21.93, 54.46, 28.42, 75.24, "weak"),
        ("PASS WON", 29.91, 50.31, 30.74, 64.44, "weak"),
        ("PASS WON", 40.22, 51.80, 39.89, 70.92, "weak"),
        ("PASS WON", 36.06, 19.05, 55.35, 16.06, "weak"),
        ("PASS WON", 51.52, 21.71, 59.50,  8.75, "weak"),
        ("PASS WON", 51.19, 30.03, 71.47,  6.25, "weak"),
        ("PASS WON", 84.10, 40.66, 76.12, 46.65, "weak"),
        # ── Incomplete – Weak Foot ────────────────────────────────────────
        ("PASS LOST", 21.27, 55.79, 34.90, 76.74, "weak"),
    ],
}

# ==========================
# Helpers
# ==========================
def classify_pass_direction(x_start: float, y_start: float,
                             x_end: float,   y_end: float) -> str:
    """
    Returns: 'forward' | 'backward' | 'lateral_left' | 'lateral_right'

    Lateral left  = pass moving toward y=80 (left touchline when attacking L���R).
    Lateral right = pass moving toward y=0  (right touchline when attacking L→R).
    """
    dx   = x_end - x_start
    dy   = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))

    if angle_deg <= 45.0:
        return "forward"
    elif angle_deg >= 135.0:
        return "backward"
    else:
        if dist > LATERAL_MIN_DIST:
            # dy > 0 → toward y=80 (left touchline)
            return "lateral_left" if dy > 0 else "lateral_right"
        # short ambiguous pass: classify by horizontal component
        return "forward" if dx >= 0 else "backward"


def progressive_pass(x_start: float, x_end: float) -> bool:
    """
    Wyscout progressive pass definition:
      - Both halves own:       ≥ 30 m closer to opponent's goal
      - Cross half-way line:   ≥ 15 m closer
      - Both halves opponent:  ≥ 10 m closer

    'Closer' is measured along the x-axis to the opponent's goal line (x=FIELD_X).
    """
    dist_start = FIELD_X - x_start   # x-distance from start to opponent's goal
    dist_end   = FIELD_X - x_end     # x-distance from end   to opponent's goal
    closer_by  = dist_start - dist_end  # positive = closer after the pass

    start_own  = x_start < HALF_LINE_X
    end_own    = x_end   < HALF_LINE_X
    start_opp  = not start_own
    end_opp    = not end_own

    if start_own and end_own:
        return closer_by >= 30.0
    if (start_own and end_opp) or (start_opp and end_own):
        return closer_by >= 15.0
    if start_opp and end_opp:
        return closer_by >= 10.0
    return False


# ==========================
# Build DataFrames
# ==========================
dfs_by_match: dict = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(
        events,
        columns=["type", "x_start", "y_start", "x_end", "y_end", "foot"],
    )
    dfm["match"]    = match_name
    dfm["position"] = POSITION_BY_MATCH[match_name]
    dfm["number"]   = np.arange(1, len(dfm) + 1)
    dfm["is_won"]   = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"]  = np.where(dfm["is_won"], "completed", "incomplete")
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r.x_start, r.y_start, r.x_end, r.y_end),
        axis=1,
    )
    dfm["is_forward"]       = dfm["direction"] == "forward"
    dfm["is_backward"]      = dfm["direction"] == "backward"
    dfm["is_lateral_left"]  = dfm["direction"] == "lateral_left"
    dfm["is_lateral_right"] = dfm["direction"] == "lateral_right"
    dfm["is_lateral"]       = dfm["is_lateral_left"] | dfm["is_lateral_right"]
    dfm["is_progressive"]   = dfm.apply(
        lambda r: progressive_pass(r.x_start, r.x_end), axis=1
    )
    dfm["pass_distance"] = np.sqrt(
        (dfm.x_end - dfm.x_start) ** 2 + (dfm.y_end - dfm.y_start) ** 2
    )
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data: dict = {"All Matches": df_all}
full_data.update(dfs_by_match)


# ==========================
# Stats
# ==========================
def _dir_stats(sub: pd.DataFrame, total: int):
    """Return direction counts + percentages for a foot sub-dataframe."""
    fwd  = int(sub["is_forward"].sum())
    bwd  = int(sub["is_backward"].sum())
    ll   = int(sub["is_lateral_left"].sum())
    lr   = int(sub["is_lateral_right"].sum())
    n    = len(sub) if len(sub) else 1
    return {
        "fwd": fwd, "fwd_pct": round(fwd / n * 100, 1),
        "bwd": bwd, "bwd_pct": round(bwd / n * 100, 1),
        "ll":  ll,  "ll_pct":  round(ll  / n * 100, 1),
        "lr":  lr,  "lr_pct":  round(lr  / n * 100, 1),
    }


def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        empty = {k: 0 for k in [
            "total_passes", "completed_passes", "incomplete_passes", "accuracy_pct",
            "avg_distance",
            "strong_total", "strong_completed", "strong_incomplete",
            "strong_accuracy_pct", "strong_avg_dist",
            "strong_prog_total", "strong_prog_completed",
            "strong_fwd", "strong_fwd_pct",
            "strong_bwd", "strong_bwd_pct",
            "strong_ll",  "strong_ll_pct",
            "strong_lr",  "strong_lr_pct",
            "weak_total", "weak_completed", "weak_incomplete",
            "weak_accuracy_pct", "weak_avg_dist", "weak_tendency_pct",
            "weak_prog_total", "weak_prog_completed",
            "weak_fwd", "weak_fwd_pct",
            "weak_bwd", "weak_bwd_pct",
            "weak_ll",  "weak_ll_pct",
            "weak_lr",  "weak_lr_pct",
            "prog_total", "prog_completed", "prog_accuracy_pct", "prog_pct_of_total",
            "forward_total",  "pct_forward",
            "backward_total", "pct_backward",
            "lateral_left_total",  "pct_lateral_left",
            "lateral_right_total", "pct_lateral_right",
        ]}
        return empty

    completed = int(df["is_won"].sum())
    accuracy  = completed / total * 100.0
    avg_dist  = float(df["pass_distance"].mean())

    strong = df[df["foot"] == "strong"]
    weak   = df[df["foot"] == "weak"]

    # ── Strong foot ───────────────────────────────────────────────────────────
    st_total  = len(strong)
    st_comp   = int(strong["is_won"].sum())
    st_incomp = st_total - st_comp
    st_acc    = st_comp / st_total * 100.0 if st_total else 0.0
    st_dist   = float(strong["pass_distance"].mean()) if st_total else 0.0
    st_prog_t = int(strong["is_progressive"].sum())
    st_prog_c = int((strong["is_progressive"] & strong["is_won"]).sum())
    st_dir    = _dir_stats(strong, st_total)

    # ── Weak foot ─────────────────────────────────────────────────────────────
    wk_total  = len(weak)
    wk_comp   = int(weak["is_won"].sum())
    wk_incomp = wk_total - wk_comp
    wk_acc    = wk_comp / wk_total * 100.0 if wk_total else 0.0
    wk_dist   = float(weak["pass_distance"].mean()) if wk_total else 0.0
    wk_tend   = wk_total / total * 100.0
    wk_prog_t = int(weak["is_progressive"].sum())
    wk_prog_c = int((weak["is_progressive"] & weak["is_won"]).sum())
    wk_dir    = _dir_stats(weak, wk_total)

    # ── Progressive overall ───────────────────────────────────────────────────
    prog_total = int(df["is_progressive"].sum())
    prog_comp  = int((df["is_progressive"] & df["is_won"]).sum())
    prog_acc   = prog_comp / prog_total * 100.0 if prog_total else 0.0
    prog_pct   = prog_total / total * 100.0

    # ── Overall direction ─────────────────────────────────────────────────────
    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    ll  = int(df["is_lateral_left"].sum())
    lr  = int(df["is_lateral_right"].sum())

    return {
        "total_passes":       total,
        "completed_passes":   completed,
        "incomplete_passes":  total - completed,
        "accuracy_pct":       round(accuracy, 2),
        "avg_distance":       round(avg_dist, 2),
        # strong foot
        "strong_total":           st_total,
        "strong_completed":       st_comp,
        "strong_incomplete":      st_incomp,
        "strong_accuracy_pct":    round(st_acc, 2),
        "strong_avg_dist":        round(st_dist, 2),
        "strong_prog_total":      st_prog_t,
        "strong_prog_completed":  st_prog_c,
        **{f"strong_{k}": v for k, v in st_dir.items()},
        # weak foot
        "weak_total":           wk_total,
        "weak_completed":       wk_comp,
        "weak_incomplete":      wk_incomp,
        "weak_accuracy_pct":    round(wk_acc, 2),
        "weak_avg_dist":        round(wk_dist, 2),
        "weak_tendency_pct":    round(wk_tend, 2),
        "weak_prog_total":      wk_prog_t,
        "weak_prog_completed":  wk_prog_c,
        **{f"weak_{k}": v for k, v in wk_dir.items()},
        # progressive overall
        "prog_total":           prog_total,
        "prog_completed":       prog_comp,
        "prog_accuracy_pct":    round(prog_acc, 2),
        "prog_pct_of_total":    round(prog_pct, 2),
        # overall direction
        "forward_total":        fwd,
        "pct_forward":          round(fwd / total * 100, 2),
        "backward_total":       bwd,
        "pct_backward":         round(bwd / total * 100, 2),
        "lateral_left_total":   ll,
        "pct_lateral_left":     round(ll / total * 100, 2),
        "lateral_right_total":  lr,
        "pct_lateral_right":    round(lr / total * 100, 2),
    }


# ==========================
# Draw: Pass Map
# ==========================
def draw_pass_map(df: pd.DataFrame, title: str):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
        line_alpha=0.95,
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.0, alpha=0.18)
    ax.axvline(x=HALF_LINE_X, color="#ffffff", linewidth=0.6, alpha=0.10, linestyle="--")

    for _, row in df.iterrows():
        is_won  = bool(row["is_won"])
        is_prog = bool(row["is_progressive"])

        if not is_won:
            color, alpha = COLOR_FAIL, 0.70
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.86
        else:
            # Plain completed pass: very light grey, very transparent
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"],   row["y_end"],
            color=color, width=1.55,
            headwidth=2.25, headlength=2.25,
            ax=ax, zorder=3, alpha=alpha,
        )
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=45, marker="o", color=color,
            edgecolors="white", linewidths=0.8,
            ax=ax, zorder=6, alpha=alpha,
        )

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)

    legend_elements = [
        Line2D([0], [0], color=COLOR_SUCCESS,    lw=2.5, label="Completed",   alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5, label="Progressive", alpha=0.90),
        Line2D([0], [0], color=COLOR_FAIL,        lw=2.5, label="Incomplete",  alpha=0.90),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize="x-small", labelspacing=0.5, borderpad=0.5,
    )
    for txt in legend.get_texts():
        txt.set_color("white")
    legend.get_frame().set_alpha(0.92)

    fig.patches.append(FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc",
    ))
    fig.text(0.5, 0.02, "Attacking Direction", ha="center", va="center",
             fontsize=9, color="#cccccc")

    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf), ax, fig


# ==========================
# Draw: Corridor Heatmap
# ==========================
def draw_corridor_heatmap(df: pd.DataFrame,
                           title: str = "Zone Heatmap — Completed Passes"):
    df_success = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)

    corridors = {
        "left":   (LANE_LEFT_MIN,  FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right":  (0.0,            LANE_RIGHT_MAX),
    }
    counts: dict = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            mask = (
                (df_success["x_end"] >= x0_) & (df_success["x_end"] < x1_)
                & (df_success["y_end"] >= y0)  & (df_success["y_end"] < y1)
            )
            arr[i] = int(mask.sum())
        counts[cname] = arr

    all_vals  = np.concatenate([counts[c] for c in counts])
    vmax      = max(1, int(all_vals.max()))
    cmap      = LinearSegmentedColormap.from_list(
        "white_red",
        ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"],
    )
    norm      = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)

    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            value     = counts[cname][i]
            ax.add_patch(Rectangle(
                (x0_, y0), x1_ - x0_, y1 - y0,
                facecolor=cmap(norm(value)),
                edgecolor=(1.0, 1.0, 1.0, 0.12),
                linewidth=0.6, alpha=0.95, zorder=2,
            ))
            text_color = "#000000" if value <= threshold else "#ffffff"
            fontw      = "700" if value >= vmax * 0.5 else "600"
            ax.text((x0_ + x1_) / 2, (y0 + y1) / 2, str(value),
                    ha="center", va="center",
                    color=text_color, fontsize=11, fontweight=fontw, zorder=4)

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    ax.axhline(y=LANE_LEFT_MIN,  color="#ffffff", linewidth=0.5,
               alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", linewidth=0.5,
               alpha=0.15, linestyle="--", zorder=3)

    fig.patches.append(FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc",
    ))
    fig.text(0.5, 0.02, "Attacking Direction", ha="center", va="center",
             fontsize=9, color="#cccccc")
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf), ax, fig


# ==========================
# Draw: Top Zone Connection Mini-Maps
# ==========================
def _top_zone_transitions(df_s: pd.DataFrame, top_k: int = 3):
    x_bins = np.linspace(0.0, FIELD_X, 7)
    y_bins = np.array([0.0, LANE_RIGHT_MAX, LANE_LEFT_MIN, FIELD_Y])
    if df_s.empty:
        return [], x_bins, y_bins

    sx = np.clip(np.searchsorted(x_bins, df_s["x_start"].to_numpy(), side="right") - 1, 0, 5)
    sy = np.clip(np.searchsorted(y_bins, df_s["y_start"].to_numpy(), side="right") - 1, 0, 2)
    ex = np.clip(np.searchsorted(x_bins, df_s["x_end"].to_numpy(),   side="right") - 1, 0, 5)
    ey = np.clip(np.searchsorted(y_bins, df_s["y_end"].to_numpy(),   side="right") - 1, 0, 2)

    transitions: dict = defaultdict(int)
    for a, b, c, d in zip(sx, sy, ex, ey):
        if int(a) == int(c) and int(b) == int(d):
            continue
        transitions[(int(a), int(b), int(c), int(d))] += 1

    links = sorted(transitions.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return links, x_bins, y_bins


def draw_top_connection_minimaps(
    df: pd.DataFrame,
    top_k: int = 3,
    title: str = "Top Zone Connections — Completed Passes",
):
    df_s = df[df["is_won"]].copy()
    links, x_bins, y_bins = _top_zone_transitions(df_s, top_k=top_k)

    x_cent = (x_bins[:-1] + x_bins[1:]) / 2.0
    y_cent = (y_bins[:-1] + y_bins[1:]) / 2.0
    max_cnt = max([v for _, v in links], default=1) if links else 1

    fig, axes = plt.subplots(1, top_k, figsize=(FIG_W * 1.65, FIG_H * 0.82), dpi=FIG_DPI)
    if top_k == 1:
        axes = [axes]
    fig.set_facecolor("#1a1a2e")
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#ffffff", line_alpha=0.90)

    for idx, ax in enumerate(axes):
        pitch.draw(ax=ax)
        ax.axhline(y=LANE_LEFT_MIN,  color="#ffffff", lw=0.4, alpha=0.12, linestyle="--")
        ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.4, alpha=0.12, linestyle="--")

        if idx >= len(links):
            ax.set_title("—", fontsize=9, color="#dbeafe", pad=4)
            continue

        (ix0, iy0, ix1, iy1), cnt = links[idx]
        x0, y0 = float(x_cent[ix0]), float(y_cent[iy0])
        x1, y1 = float(x_cent[ix1]), float(y_cent[iy1])
        rel   = cnt / max_cnt
        color = plt.cm.Blues(0.40 + 0.55 * rel)
        lw    = 1.2 + 4.2 * rel
        alpha = 0.35 + 0.60 * rel

        ax.add_patch(Rectangle(
            (x_bins[ix0], y_bins[iy0]),
            x_bins[ix0 + 1] - x_bins[ix0], y_bins[iy0 + 1] - y_bins[iy0],
            facecolor=(0.20, 0.45, 0.95, 0.18),
            edgecolor=(1, 1, 1, 0.18), lw=0.6, zorder=2,
        ))
        ax.add_patch(Rectangle(
            (x_bins[ix1], y_bins[iy1]),
            x_bins[ix1 + 1] - x_bins[ix1], y_bins[iy1 + 1] - y_bins[iy1],
            facecolor=(0.02, 0.70, 0.55, 0.18),
            edgecolor=(1, 1, 1, 0.18), lw=0.6, zorder=2,
        ))

        if ix0 == ix1 and iy0 == iy1:
            ax.scatter([x0], [y0], s=40 + 80 * rel, c=[color], marker="o",
                       edgecolors="white", linewidths=0.5, alpha=alpha, zorder=5)
        else:
            rad = float(np.clip(
                0.10 * np.sign((ix1 - ix0) + 0.4 * (iy1 - iy0)), -0.30, 0.30,
            ))
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=10 + 9 * rel,
                lw=lw, color=color, alpha=alpha, zorder=4,
            ))

        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{cnt}",
                color="#e5efff", fontsize=9, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          fc=(0.06, 0.09, 0.14, 0.80), ec="none"))
        ax.set_title(f"#{idx + 1}  ·  {cnt}×", fontsize=9, color="#dbeafe", pad=4)

    fig.suptitle(title, fontsize=11, color="#ffffff", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf), axes, fig


# ==========================
# Layout
# ==========================
st.caption("Click the origin dot on the pass map to inspect an event.")

col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

# ── Filters ──────────────────────────────────────────────────────────────────
with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

    # Position filter (LCB / RCB)
    st.markdown("### 📍 Position")
    position_sel = st.radio(
        "Filter by position",
        ["All Positions", "LCB", "RCB"],
        index=0,
    )
    st.markdown(
        "<div style='font-size:11px; color:#94a3b8; margin-top:-6px; margin-bottom:4px;'>"
        "LCB: Sac. United, Capital City &nbsp;|&nbsp; RCB: Wake FC"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

    # Build position-filtered match list
    if position_sel == "All Positions":
        avail_matches = list(dfs_by_match.keys())
    else:
        avail_matches = [m for m, p in POSITION_BY_MATCH.items() if p == position_sel]

    if avail_matches:
        pos_df_all = pd.concat(
            [dfs_by_match[m] for m in avail_matches], ignore_index=True
        )
    else:
        pos_df_all = df_all.iloc[0:0].copy()

    pos_full_data: dict = {"All Matches": pos_df_all}
    pos_full_data.update({m: dfs_by_match[m] for m in avail_matches})

    # Match selector
    st.markdown("### 🏟️ Match")
    selected_match = st.selectbox(
        "Choose the match", list(pos_full_data.keys()), index=0
    )

    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

    # Pass filter
    st.markdown("### 🎯 Pass Filter")
    pass_filter = st.radio(
        "Filter passes",
        [
            "All Passes",
            "Strong Foot Only",
            "Weak Foot Only",
            "Completed Only",
            "Incomplete Only",
            "Progressive Only",
        ],
        index=0,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("heat_selection", None),
    ("last_match",    selected_match),
    ("last_filter",   pass_filter),
    ("last_position", position_sel),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state["last_position"] != position_sel:
    st.session_state["heat_selection"] = None
    st.session_state["last_position"]  = position_sel
if st.session_state["last_match"] != selected_match:
    st.session_state["heat_selection"] = None
    st.session_state["last_match"]     = selected_match
if st.session_state["last_filter"] != pass_filter:
    st.session_state["heat_selection"] = None
    st.session_state["last_filter"]    = pass_filter

# ── Field column ──────────────────────────────────────────────────────────────
with col_field:
    df_base = pos_full_data[selected_match].copy()

    if pass_filter == "Strong Foot Only":
        df_base = df_base[df_base["foot"] == "strong"].reset_index(drop=True)
    elif pass_filter == "Weak Foot Only":
        df_base = df_base[df_base["foot"] == "weak"].reset_index(drop=True)
    elif pass_filter == "Completed Only":
        df_base = df_base[df_base["is_won"]].reset_index(drop=True)
    elif pass_filter == "Incomplete Only":
        df_base = df_base[~df_base["is_won"]].reset_index(drop=True)
    elif pass_filter == "Progressive Only":
        df_base = df_base[df_base["is_progressive"]].reset_index(drop=True)
    else:
        df_base = df_base.reset_index(drop=True)

    DISPLAY_WIDTH = 780

    # Reserve pass-map placeholder (renders above heatmap visually)
    pass_map_placeholder = st.empty()

    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown(
        '<h4 style="color:#ffffff; margin:6px 0 6px 0;">Zone Heatmap</h4>',
        unsafe_allow_html=True,
    )
    heat_img, hax, hfig = draw_corridor_heatmap(df_base)
    heat_click = streamlit_image_coordinates(heat_img, width=DISPLAY_WIDTH)

    if heat_click is not None:
        rw, rh = heat_img.size
        px = heat_click["x"] * (rw / heat_click["width"])
        py = heat_click["y"] * (rh / heat_click["height"])
        fx, fy = hax.transData.inverted().transform((px, rh - py))
        x_bins_h = np.linspace(0.0, FIELD_X, 7)
        ix = max(0, min(5, np.searchsorted(x_bins_h, fx, side="right") - 1))
        x0_h, x1_h = x_bins_h[ix], x_bins_h[ix + 1]
        if fy >= LANE_LEFT_MIN:
            cname, y0_h, y1_h = "left",   LANE_LEFT_MIN,  FIELD_Y
        elif fy < LANE_RIGHT_MAX:
            cname, y0_h, y1_h = "right",  0.0,            LANE_RIGHT_MAX
        else:
            cname, y0_h, y1_h = "center", LANE_RIGHT_MAX, LANE_LEFT_MIN
        st.session_state["heat_selection"] = {
            "ix": int(ix), "corridor": cname,
            "x0": float(x0_h), "x1": float(x1_h),
            "y0": float(y0_h), "y1": float(y1_h),
        }
    plt.close(hfig)

    # ── Zone Connection Mini-Maps ─────────────────────────────────────────────
    st.markdown(
        '<h4 style="color:#ffffff; margin:14px 0 4px 0;">Top Zone Connections</h4>',
        unsafe_allow_html=True,
    )
    mini_img, _, mini_fig = draw_top_connection_minimaps(df_base, top_k=3)
    st.image(mini_img, use_container_width=True)
    plt.close(mini_fig)

    # ── Pass Map (fills placeholder) ──────────────────────────────────────────
    with pass_map_placeholder.container():
        st.markdown(
            '<h4 style="color:#ffffff; margin:0 0 6px 0;">Pass Map</h4>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Zone Filter", key="clear_heat_filter"):
            st.session_state["heat_selection"] = None

        df_to_draw = df_base
        if st.session_state["heat_selection"] is not None:
            sel = st.session_state["heat_selection"]
            df_to_draw = df_base[
                (df_base["x_end"] >= sel["x0"]) & (df_base["x_end"] < sel["x1"])
                & (df_base["y_end"] >= sel["y0"]) & (df_base["y_end"] < sel["y1"])
            ].reset_index(drop=True)

        img_obj, ax, fig = draw_pass_map(
            df_to_draw, title=f"Pass Map — {selected_match}"
        )
        click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)

    # Process map click
    selected_pass = None
    if click is not None:
        rw, rh = img_obj.size
        px = click["x"] * (rw / click["width"])
        py = click["y"] * (rh / click["height"])
        fx, fy = ax.transData.inverted().transform((px, rh - py))
        df_sel = df_to_draw.copy()
        df_sel["_dist"] = np.sqrt(
            (df_sel["x_start"] - fx) ** 2 + (df_sel["y_start"] - fy) ** 2
        )
        candidates = df_sel[df_sel["_dist"] < 5.0].sort_values("_dist")
        if not candidates.empty:
            selected_pass = candidates.iloc[0]
    plt.close(fig)

    # Zone filter info bar
    if st.session_state["heat_selection"] is not None:
        sel = st.session_state["heat_selection"]
        sel_mask = (
            (df_base["x_end"] >= sel["x0"]) & (df_base["x_end"] < sel["x1"])
            & (df_base["y_end"] >= sel["y0"]) & (df_base["y_end"] < sel["y1"])
        )
        st.markdown(
            f"<div style='color:#ffffff; margin-top:6px;'>"
            f"<strong>Zone filter active:</strong> channel "
            f"<code>{sel['corridor']}</code>, column #{sel['ix']+1} "
            f"— {int(sel_mask.sum())} passes</div>",
            unsafe_allow_html=True,
        )

    # ── Selected Event ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Selected Event")
    if selected_pass is None:
        st.info("Click an origin dot on the pass map to inspect an event.")
    else:
        foot_label = (
            "Weak Foot" if selected_pass["foot"] == "weak" else "Strong Foot"
        )
        prog_tag = "  ·  ✅ Progressive" if selected_pass["is_progressive"] else ""
        st.success(
            f"Pass #{int(selected_pass['number'])} — {selected_pass['type']} "
            f"| {foot_label}{prog_tag}"
        )
        c1, c2 = st.columns(2)
        c1.write(f"**Origin:** ({selected_pass['x_start']:.2f}, {selected_pass['y_start']:.2f})")
        c2.write(f"**Destination:** ({selected_pass['x_end']:.2f}, {selected_pass['y_end']:.2f})")

        dir_label_map = {
            "forward":       "⬆️ Forward",
            "backward":      "⬇️ Backward",
            "lateral_left":  "◀️ Lateral Left",
            "lateral_right": "▶️ Lateral Right",
        }
        t1, t2 = st.columns(2)
        t1.write(f"**Direction:** {dir_label_map.get(selected_pass['direction'], selected_pass['direction'])}")
        t2.write(f"**Foot:** {foot_label}")
        st.metric("Pass Distance", f"{selected_pass['pass_distance']:.1f} m")

    # ── Full data table ───────────────────────────────────────────────────────
    with st.expander("📊 Full Pass Data Table"):
        display_cols = [
            "number", "type", "foot", "outcome", "direction",
            "x_start", "y_start", "x_end", "y_end", "pass_distance",
            "is_forward", "is_backward", "is_lateral_left", "is_lateral_right",
            "is_progressive",
        ]
        st.dataframe(
            df_to_draw[display_cols].style.format({
                "x_start":       "{:.2f}",
                "y_start":       "{:.2f}",
                "x_end":         "{:.2f}",
                "y_end":         "{:.2f}",
                "pass_distance": "{:.1f}",
            }),
            use_container_width=True,
            height=400,
        )


# ── Stats column ──────────────────────────────────────────────────────────────
with col_stats:
    s = compute_stats(df_to_draw)

    # ── General ──────────────────────────────────────────────────────────────
    with st.expander("📋 General Statistics", expanded=True):
        st.markdown('<div class="stats-section-title">Overview</div>',
                    unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1: small_metric("Total Passes",  f"{s['total_passes']}")
        with r2: small_metric("Completed",     f"{s['completed_passes']}")
        with r3: small_metric("Accuracy",      f"{s['accuracy_pct']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">🦶 Strong Foot</div>',
                    unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1: small_metric("Total",     f"{s['strong_total']}")
        with d2: small_metric("Completed", f"{s['strong_completed']}")
        with d3: small_metric("Accuracy",  f"{s['strong_accuracy_pct']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">🦵 Weak Foot</div>',
                    unsafe_allow_html=True)
        w1, w2, w3 = st.columns(3)
        with w1: small_metric("Total",     f"{s['weak_total']}")
        with w2: small_metric("Completed", f"{s['weak_completed']}")
        with w3: small_metric("Accuracy",  f"{s['weak_accuracy_pct']:.1f}%")

    # ── Advanced ─────────────────────────────────────────────────────────────
    with st.expander("🔬 Advanced Statistics", expanded=False):

        # Weak foot tendency
        st.markdown(
            '<div class="stats-section-title">🦵 Weak Foot Tendency</div>',
            unsafe_allow_html=True,
        )
        tf1, tf2 = st.columns(2)
        with tf1:
            small_metric(
                "Tendency",
                f"{s['weak_tendency_pct']:.1f}%",
                delta=f"{s['weak_total']} of {s['total_passes']} passes",
            )
        with tf2:
            small_metric(
                "Weak Foot Accuracy",
                f"{s['weak_accuracy_pct']:.1f}%",
                delta=f"{s['weak_completed']} completed / {s['weak_incomplete']} incomplete",
            )

        # Average distance + progressive per foot
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="stats-section-title">📏 Average Pass Distance</div>',
            unsafe_allow_html=True,
        )
        dst1, dst2 = st.columns(2)
        with dst1:
            small_metric(
                "Strong Foot (avg)",
                f"{s['strong_avg_dist']:.1f} m",
                delta=f"Accuracy: {s['strong_accuracy_pct']:.1f}%",
            )
        with dst2:
            small_metric(
                "Weak Foot (avg)",
                f"{s['weak_avg_dist']:.1f} m",
                delta=f"Accuracy: {s['weak_accuracy_pct']:.1f}%",
            )

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="stats-section-title">🔵 Progressive Passes per Foot</div>',
            unsafe_allow_html=True,
        )
        pp1, pp2 = st.columns(2)
        with pp1:
            small_metric(
                "Strong Foot",
                f"{s['strong_prog_total']}",
                delta=f"{s['strong_prog_completed']} completed",
            )
        with pp2:
            small_metric(
                "Weak Foot",
                f"{s['weak_prog_total']}",
                delta=f"{s['weak_prog_completed']} completed",
            )

        # Direction — Strong Foot
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="stats-section-title">🦶 Pass Direction — Strong Foot</div>',
            unsafe_allow_html=True,
        )
        dd1, dd2 = st.columns(2)
        with dd1:
            small_metric("⬆️ Forward",
                         f"{s['strong_fwd']}",
                         delta=f"{s['strong_fwd_pct']:.0f}% of strong foot passes")
        with dd2:
            small_metric("⬇️ Backward",
                         f"{s['strong_bwd']}",
                         delta=f"{s['strong_bwd_pct']:.0f}%")
        dd3, dd4 = st.columns(2)
        with dd3:
            small_metric("◀️ Lateral Left",
                         f"{s['strong_ll']}",
                         delta=f"{s['strong_ll_pct']:.0f}%")
        with dd4:
            small_metric("▶️ Lateral Right",
                         f"{s['strong_lr']}",
                         delta=f"{s['strong_lr_pct']:.0f}%")

        # Direction — Weak Foot
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="stats-section-title">🦵 Pass Direction — Weak Foot</div>',
            unsafe_allow_html=True,
        )
        dw1, dw2 = st.columns(2)
        with dw1:
            small_metric("⬆️ Forward",
                         f"{s['weak_fwd']}",
                         delta=f"{s['weak_fwd_pct']:.0f}% of weak foot passes")
        with dw2:
            small_metric("⬇️ Backward",
                         f"{s['weak_bwd']}",
                         delta=f"{s['weak_bwd_pct']:.0f}%")
        dw3, dw4 = st.columns(2)
        with dw3:
            small_metric("◀️ Lateral Left",
                         f"{s['weak_ll']}",
                         delta=f"{s['weak_ll_pct']:.0f}%")
        with dw4:
            small_metric("▶️ Lateral Right",
                         f"{s['weak_lr']}",
                         delta=f"{s['weak_lr_pct']:.0f}%")

        # Progressive Passes (overall)
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="stats-section-title">🔵 Progressive Passes (Wyscout)</div>',
            unsafe_allow_html=True,
        )
        p1, p2, p3, p4 = st.columns(4)
        with p1: small_metric("Total",      f"{s['prog_total']}")
        with p2: small_metric("Completed",  f"{s['prog_completed']}")
        with p3: small_metric("Accuracy",   f"{s['prog_accuracy_pct']:.1f}%")
        with p4: small_metric("% of Total", f"{s['prog_pct_of_total']:.1f}%")

    st.divider()
    st.caption(
        "Grey = Completed  ·  🔵 Blue = Progressive  ·  🔴 Red = Incomplete  |  "
        "Lateral Left = toward left touchline (y↑) when attacking left→right."
    )
