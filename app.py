# app.py

from __future__ import annotations

import datetime as dt
import re

import altair as alt
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from scipy.spatial import ConvexHull
from streamlit_folium import st_folium

# -------------------------------------------------
# IMPORTS FROM BACKEND
# -------------------------------------------------
try:
    from Crime_buster_backend import train_and_predict_for_grids, generate_police_stats
except ImportError:
    try:
        from Crime_buster_backend import train_and_predict_for_grids, generate_police_stats
    except ImportError:
        st.error("Backend file not found. Please ensure 'crime_buster_backend.py' exists.")
        st.stop()

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="🚨 Crime Buster",
    layout="wide",
    page_icon="🚨",
)

# ---------------- Session State ----------------
if "login" not in st.session_state:
    st.session_state["login"] = False
if "role" not in st.session_state:
    st.session_state["role"] = "Public"
if "username" not in st.session_state:
    st.session_state["username"] = None


# ---------------- Header ----------------
st.markdown(
    """
<div style='text-align:center; background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom: 20px;'>
    <h1 style='color:#1f77b4; margin:0;'>🚨 Crime Buster</h1>
    <p style='color:#555; margin-top:5px;'>Advanced Hotspot Prediction & Analytics</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------

def logout() -> None:
    st.session_state["login"] = False
    st.session_state["role"] = "Public"
    st.session_state["username"] = None
    st.rerun()

police_credentials = {
    "police1": "patrol123",
    "police2": "patrol456",
}

def dms_to_decimal(dms_str):
    """Robust parser for both Decimal and DMS coordinates."""
    if pd.isna(dms_str): return None
    dms_str = str(dms_str).strip()
    try:
        return float(dms_str)
    except ValueError:
        pass
    match = re.match(r"(\d+)°(\d+)'(\d+(\.\d+)?)\"?\s*([NSEW])?", dms_str)
    if match:
        deg = float(match.group(1))
        mn = float(match.group(2))
        sec = float(match.group(3))
        direction = match.group(5)
        val = deg + (mn / 60.0) + (sec / 3600.0)
        if direction in ['S', 'W']: val *= -1
        return val
    return None

@st.cache_data(show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> str:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 16}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "CrimeBusterApp/1.0"}, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("display_name", "Unknown place")
    except Exception:
        pass
    return "Unknown place"

@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> tuple[float | None, float | None]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "CrimeBusterApp/1.0"}, timeout=5)
        data = resp.json()
        if resp.status_code == 200 and data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

@st.cache_data(show_spinner=True)
def run_hotspot_model(df, min_grid, max_grid, step_grid):
    return train_and_predict_for_grids(df, min_grid, max_grid, step_grid)

@st.cache_data(show_spinner=False)
def compute_police_stats(df, hotspots, crime_type_col):
    return generate_police_stats(df, hotspots, crime_type_col)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎛 Controls")

    # Login
    if not st.session_state["login"] or st.session_state["role"] != "Police/Admin":
        st.info("Public Mode Active")
        with st.expander("👮 Police Login"):
            p_user = st.text_input("Username", key="p_user")
            p_pass = st.text_input("Password", type="password", key="p_pass")
            if st.button("Login"):
                if p_user in police_credentials and p_pass == police_credentials[p_user]:
                    st.session_state["login"] = True
                    st.session_state["role"] = "Police/Admin"
                    st.session_state["username"] = p_user
                    st.success("Success!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    else:
        st.success(f"👮 Officer: {st.session_state['username']}")
        if st.button("Logout"):
            logout()

    st.markdown("---")
    
    uploaded_file = st.file_uploader(" Upload Data", type=["csv", "xlsx", "xls"])
    
    st.subheader("Grid Settings")
    min_grid = st.slider("Min Grid (km)", 0.1, 2.0, 0.5, step=0.1)
    max_grid = st.slider("Max Grid (km)", min_grid, 2.0, 2.0, step=0.1)
    step_grid = st.slider("Step Size (km)", 0.1, 1.0, 0.5, step=0.1)

    st.subheader("Risk Thresholds")
    high_risk = st.slider("High Risk (>)", 0.5, 1.0, 0.7, step=0.05)
    med_risk = st.slider("Med Risk (>)", 0.1, high_risk, 0.4, step=0.05)

    st.subheader("Filters")
    date_filter_enabled = st.checkbox("Filter by Date")
    start_date = st.date_input("Start") if date_filter_enabled else None
    end_date = st.date_input("End") if date_filter_enabled else None

    st.markdown("---")
    search_query = st.text_input("Go to Location")
    search_btn = st.button("Search")
    show_heatmap = st.checkbox("Show Heatmap Layer", value=True)


# ---------------- Main App Logic ----------------

if not uploaded_file:
    st.info(" Welcome! Please upload a crime data file (CSV or Excel) to start.")
    st.stop()

# ==========================================
# 1. SMART DATA LOADER
# ==========================================
try:
    filename = uploaded_file.name.lower()
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            st.info("Make sure you have 'openpyxl' installed: `pip install openpyxl`")
            st.stop()
    else:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',') 
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1') 
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')

except Exception as e:
    st.error(f" Critical Error: Could not read file. Details: {e}")
    st.stop()


# 2. Normalize Columns
colmap = {}
for col in df.columns:
    c_lower = str(col).lower().strip()
    if "date" in c_lower: colmap[col] = "datetime"
    elif "lat" in c_lower: colmap[col] = "latitude"
    elif "lon" in c_lower: colmap[col] = "longitude"
    elif "primary" in c_lower and "type" in c_lower: pass
        
df = df.rename(columns=colmap)

# 3. ROBUST COORDINATE CONVERSION
if "latitude" in df.columns:
    df["latitude"] = df["latitude"].apply(dms_to_decimal)
if "longitude" in df.columns:
    df["longitude"] = df["longitude"].apply(dms_to_decimal)

# 4. Detect Crime Type Column
crime_type_col = None
priority_cols = ["Primary type", "CrimeHead_Name", "Crime Type", "Type"]
for p in priority_cols:
    if p in df.columns:
        crime_type_col = p
        break
if not crime_type_col:
    candidates = [c for c in df.columns if "type" in str(c).lower() or "desc" in str(c).lower() or "category" in str(c).lower()]
    if len(candidates) > 0: crime_type_col = candidates[0]

# 5. PARSE DATETIME
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")

# 6. Check Requirements & Clean
if "latitude" not in df.columns or "longitude" not in df.columns:
    st.error(f" Column mismatch. Found: {list(df.columns)}. \n\nNeed: 'latitude', 'longitude' columns.")
    st.stop()

# Outlier Filtering (Fixes "Infinite Blue Line")
df = df.dropna(subset=["latitude", "longitude"])
df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
df = df.dropna(subset=["latitude", "longitude"]).copy()

mid_lat = df["latitude"].median()
mid_lon = df["longitude"].median()
# Filter points > 1 degree away from median
mask = (
    (df["latitude"] >= mid_lat - 1.0) & (df["latitude"] <= mid_lat + 1.0) &
    (df["longitude"] >= mid_lon - 1.0) & (df["longitude"] <= mid_lon + 1.0)
)
df = df[mask]

if df.empty:
    st.error("No valid location data near the main cluster!")
    st.stop()

# 7. Filters
if crime_type_col:
    all_types = sorted(df[crime_type_col].dropna().astype(str).unique())
    selected_types = st.multiselect("Filter by Crime Type", all_types, default=all_types)
    if selected_types:
        df = df[df[crime_type_col].isin(selected_types)]

if date_filter_enabled and start_date and end_date and "datetime" in df.columns:
    df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]

st.success(f" Loaded {len(df)} records.")

# ---------------- Model Execution ----------------
with st.spinner(" Analyzing crime patterns..."):
    results = run_hotspot_model(df, min_grid, max_grid, step_grid)

available_grids = sorted(results.keys())
selected_grid = st.selectbox("Select Grid Resolution", available_grids, format_func=lambda x: f"{x} km")

hotspots = results[selected_grid]["hotspots"]
auc_score = results[selected_grid]["auc"]


# ---------------- Dashboard ----------------
st.markdown("---")
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Predictive Insights")
    st.metric("Model Accuracy (AUC)", f"{auc_score:.2f}")
    
    st.markdown("##### High Risk Areas")
    st.dataframe(
        hotspots[["grid_id", "risk_score", "latitude", "longitude"]].head(15),
        height=400,
        width="stretch"
    )
    
    csv_data = hotspots.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Prediction CSV", csv_data, "hotspots.csv", "text/csv")

with col_right:
    st.subheader("Risk Map")
    map_center = [df["latitude"].median(), df["longitude"].median()]
    if search_btn and search_query:
        lat, lon = geocode_place(search_query)
        if lat: map_center = [lat, lon]

    m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
    
    if show_heatmap:
        heat_data = df[["latitude", "longitude"]].values.tolist()
        if heat_data: HeatMap(heat_data, radius=13, blur=18).add_to(m)

    if len(df) > 3:
        try:
            sample = df[["latitude", "longitude"]].sample(min(500, len(df))).to_numpy()
            hull = ConvexHull(sample)
            folium.Polygon(locations=sample[hull.vertices].tolist(), color="blue", weight=2, fill=True, fill_opacity=0.05).add_to(m)
        except: pass

    marker_cluster = MarkerCluster().add_to(m)
    valid_hotspots = hotspots[(hotspots["latitude"].between(mid_lat-1, mid_lat+1)) & (hotspots["longitude"].between(mid_lon-1, mid_lon+1))]
    for _, row in valid_hotspots.iterrows():
        if row["risk_score"] < med_risk: continue
        color = "#d32f2f" if row["risk_score"] >= high_risk else "#f57c00"
        folium.CircleMarker([row["latitude"], row["longitude"]], radius=8 + (row["risk_score"]*5), color=color, fill=True, fill_opacity=0.7, popup=f"Risk: {row['risk_score']:.2f}").add_to(marker_cluster)

    st_folium(m, height=600)

# ---------------- Police Analytics (Advanced) ----------------
if st.session_state["login"] and st.session_state["role"] == "Police/Admin":
    st.markdown("---")
    st.header("🔐 Police Intelligence Dashboard")
    
    stats = compute_police_stats(df, hotspots, crime_type_col)
    
    if "datetime" in df.columns:
        df["day_name"] = df["datetime"].dt.day_name()
        df["month_name"] = df["datetime"].dt.month_name()
        df["year"] = df["datetime"].dt.year
        
    # --- Row 1: The Big Picture ---
    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        st.subheader("Crime Hotspots by Region")
        reg_col = next((c for c in df.columns if "region" in c.lower() or "station" in c.lower()), None)
        if reg_col:
            c1 = alt.Chart(df).mark_bar().encode(
                y=alt.Y(f"{reg_col}:N", sort='-x', title="Region / Station"),
                x=alt.X("count():Q", title="Total Incidents"),
                color=alt.Color("count():Q", legend=None, scale=alt.Scale(scheme="blues")),
                tooltip=[reg_col, "count()"]
            ).properties(height=350)
            st.altair_chart(c1, width="stretch")
        else:
            st.info("No 'Region' column found.")

    with r1_col2:
        st.subheader("📈 Trend Comparison (Year-over-Year)")
        ts = stats.get("timeseries")
        if ts is not None and not ts.empty:
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            ts_chart = ts.copy()
            ts_chart["month"] = ts_chart["datetime"].dt.month_name()
            ts_chart["year"] = ts_chart["datetime"].dt.year
            
            base = alt.Chart(ts_chart).encode(
                x=alt.X("month:O", sort=month_order, title="Month"),
                y=alt.Y("sum(count):Q", title="Incidents"),
                color=alt.Color("year:N", title="Year", scale=alt.Scale(scheme="category10")),
                tooltip=["year:N", "month:O", "sum(count):Q"]
            )
            st.altair_chart((base.mark_line(point=True) + base.mark_area(opacity=0.1)).properties(height=350).interactive(), width="stretch")
        else:
            st.info("No time-series data.")

    # --- Row 2: Tactical Analysis ---
    st.markdown("### 🧩 Tactical Analysis Matrix")
    t1_col, t2_col = st.columns(2)
    
    with t1_col:
        st.markdown("**1. Crime Matrix (Region vs Type)**")
        if reg_col and crime_type_col:
            c3 = alt.Chart(df).mark_rect().encode(
                x=alt.X(f"{reg_col}:N", title="Region"),
                y=alt.Y(f"{crime_type_col}:N", title="Crime Type"),
                color=alt.Color("count():Q", title="Count", scale=alt.Scale(scheme="reds")),
                tooltip=[reg_col, crime_type_col, "count()"]
            ).properties(height=400)
            st.altair_chart(c3, width="stretch")
        else:
            st.warning("Need Region/Type columns.")

    with t2_col:
        # NEW: Day of Week Analysis (Replaces Hourly)
        st.markdown("**2. Weekly Crime Pattern**")
        st.caption("Incidents by Day of the Week")
        if "datetime" in df.columns and crime_type_col:
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            c4 = alt.Chart(df).mark_bar().encode(
                x=alt.X("day_name:O", sort=days_order, title="Day"),
                y=alt.Y("count():Q", title="Total Incidents"),
                color=alt.Color(f"{crime_type_col}:N", title="Crime Type", scale=alt.Scale(scheme="category20")),
                tooltip=["day_name", f"{crime_type_col}", "count():Q"]
            ).properties(height=400)
            st.altair_chart(c4, width="stretch")

    # --- Extra Row: Major Trends ---
    st.markdown("### 📈 Major Crime Trends")
    st.caption("Yearly trend of the Top 5 most frequent crimes")
    if "datetime" in df.columns and crime_type_col:
        top_5 = df[crime_type_col].value_counts().head(5).index.tolist()
        c_extra = alt.Chart(df[df[crime_type_col].isin(top_5)]).mark_line(point=True).encode(
            x=alt.X("year(datetime):O", title="Year"),
            y=alt.Y("count():Q", title="Annual Cases"),
            color=alt.Color(f"{crime_type_col}:N", title="Crime Type"),
            tooltip=[f"{crime_type_col}", "year(datetime)", "count()"]
        ).properties(height=300)
        st.altair_chart(c_extra, width="stretch")

    # --- Row 3: Strategic Overview ---
    st.markdown("### 🍩 Strategic Overview")
    r3_col1, r3_col2 = st.columns(2)
    
    with r3_col1:
        st.subheader("Crime Type Distribution")
        if crime_type_col:
            top_crimes = df[crime_type_col].value_counts().head(10).index.tolist()
            df_top = df[df[crime_type_col].isin(top_crimes)]
            total = len(df_top)
            
            base = alt.Chart(df_top).encode(theta=alt.Theta("count():Q", stack=True))
            pie = base.mark_arc(innerRadius=70, stroke="#fff").encode(
                color=alt.Color(f"{crime_type_col}:N", legend=alt.Legend(orient="bottom", columns=2), scale=alt.Scale(scheme="tableau20")),
                order=alt.Order("count():Q", sort="descending"),
                tooltip=[f"{crime_type_col}", "count():Q"]
            )
            text = base.mark_text(radius=140).encode(
                text=alt.Text("count():Q"), order=alt.Order("count():Q", sort="descending"), color=alt.value("black")
            )
            center = alt.Chart(pd.DataFrame({'t':[str(total)]})).mark_text(size=24, color="#333").encode(text='t')
            st.altair_chart((pie + text + center).properties(height=400), width="stretch")

    with r3_col2:
        st.markdown("**2. Weekly Crime Rhythm (Bubble Plot)**")
        st.caption("Larger bubbles = More crimes on that day")
        
        if "datetime" in df.columns and crime_type_col:
            # Ensure day_name exists (just in case)
            if "day_name" not in df.columns:
                df["day_name"] = df["datetime"].dt.day_name()
                
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # The "Punch Card" Chart
            c4 = alt.Chart(df).mark_circle().encode(
                # X-Axis: Days of the week
                x=alt.X("day_name:O", sort=days_order, title=None),
                
                # Y-Axis: Crime Types
                y=alt.Y(f"{crime_type_col}:N", title=None),
                
                # Bubble Size: Bigger = More Crimes
                size=alt.Size("count():Q", legend=None, scale=alt.Scale(range=[50, 1000])),
                
                # Bubble Color: Darker = More Crimes (Magma scheme looks very modern)
                color=alt.Color("count():Q", legend=None, scale=alt.Scale(scheme="magma")),
                
                tooltip=[
                    alt.Tooltip("day_name", title="Day"),
                    f"{crime_type_col}",
                    alt.Tooltip("count():Q", title="Incidents")
                ]
            ).properties(height=400)
            
            st.altair_chart(c4, width="stretch")
        else:
            st.warning("Datetime column missing - Temporal charts disabled.")
