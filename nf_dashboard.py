import streamlit as st
import requests
import firebase_admin
from firebase_admin import db, credentials
import pandas as pd
import numpy as np
from math import radians, sin, cos, atan2, sqrt, isnan
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
from typing import Dict, Any, Tuple


#naming convention for course flight vs endurance flight
LIVE_REFRESH = 100 #app refresh when live (ms)
STANDARD_REFRESH = 5000 #app refresh for archive (ms)
TEAMS = [str(i) for i in range(1, 10)]      # "1".."9"
FLIGHTS = [str(i) for i in range(1, 5)]     # "1".."4"
ENFORCE_SORT = False
COLUMNS = [
    "millis","accX","accY","accZ","gyroX","gyroY","gyroZ", "magX","magY","magZ",
    "latitude","longitude", "gpsAltitude","Speed","SatCount","batteryVoltage", "roll", "pitch", "yaw"
]
GAUSS_COLS = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "magX", "magY", "magZ"]
GPS_COLS   = ["latitude", "longitude"] #without gpsAltitude?

interval = STANDARD_REFRESH
st_autorefresh(interval=interval, key="app_refresh")
# Seed /live as a MAP of teams -> flights (all empty objects)

cred = credentials.Certificate(r"D:\Neues Fliegen\Datalogger\Dashboard\Firebase DB\datalogger-nfc25-firebase-adminsdk-fbsvc-9fb825c058.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://datalogger-nfc25-default-rtdb.europe-west1.firebasedatabase.app"
    })

#---------------- FILTERING METHODS --------------------------------------------
class RealTimeGaussianFilter:
    def __init__(self, window_size=5, sigma=1.0): # Assign default values; can be overriden
        self.buffer = [] 
        self.weights =  self._compute_weights(window_size, sigma)

    def _compute_weights(self, window_size, sigma): # Calculate bell curve
        x = np.arange(-(window_size // 2), window_size // 2 + 1)
        weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return weights / weights.sum() # Return normalised weights
    
    def update(self, new_sample): 
        self.buffer.append(float(new_sample))

        if len(self.buffer) > len(self.weights):
            self.buffer.pop(0)  # Keep buffer size = window_size; remove oldest sample
        
        if len(self.buffer) < len(self.weights):
            return new_sample  # Not enough data yet, return same exact sample
        else:
            return float(np.round(np.dot(np.array(self.buffer), self.weights), 4))

class GPSFilter:
    def __init__(self, window_size=5): # Assign default values; can be overriden
        self.gps_readings = []
        self.satellite_count = []
        self.weights = np.empty(window_size, dtype=float)

    def _compute_weights(self): # Calculate weights for softmax filter
        np_satellite_count = np.array(self.satellite_count) # Typecast self.satellite_count as a numpy array
        exp_weights = np.exp(np_satellite_count - np.max(np_satellite_count)) # Subtract max 
        return exp_weights / np.sum(exp_weights)  # Return normalised weights

    def update(self, new_gps, new_sat_count):
        self.gps_readings.append(float(new_gps))
        self.satellite_count.append(int(new_sat_count))

        if len(self.gps_readings) > len(self.weights) and len(self.satellite_count) > len(self.weights):
            self.gps_readings.pop(0)  # Keep gps_readings size = window_size; remove oldest sample
            self.satellite_count.pop(0) 
        
        if len(self.gps_readings) < len(self.weights):
            return new_gps # Not enough data points; Return raw unfiltered value
        else:
            self.weights[:] = self._compute_weights()
            return float(np.round(np.dot(np.array(self.gps_readings), self.weights), 4))

#---------------- RETRIEVING DATA AND PROCESSING -------------------------------
def _iter_children(node):
    """Yield (key, value) for either a dict or a list (skipping None)."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k), v
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if v is not None:
                yield str(i), v

def init_firebase():
    cred = credentials.Certificate(r"D:\Neues Fliegen\Datalogger\Dashboard\Firebase DB\datalogger-nfc25-firebase-adminsdk-fbsvc-9fb825c058.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': r"https://datalogger-nfc25-default-rtdb.europe-west1.firebasedatabase.app"
        }) 

def to_df_from_node(node) -> pd.DataFrame:
    """
    Expect a FLIGHT node that is a dict of "<millis>" -> [values...] (or dict-row).
    """
    if not isinstance(node, dict):
        return pd.DataFrame(columns=COLUMNS + ["t_s"])

    n = len(COLUMNS)
    rows = []

    for k, v in node.items():
        if k in ("MODE", "mode"):
            continue

        if isinstance(v, (list, tuple)):
            row = list(v[:n]) + [None] * max(0, n - len(v))
            if COLUMNS[0] == "millis" and (row[0] is None or (isinstance(row[0], float) and pd.isna(row[0]))):
                try: row[0] = int(float(k))
                except: pass
            rows.append(row)

        elif isinstance(v, dict):
            row = [v.get(c) for c in COLUMNS]
            if COLUMNS[0] == "millis" and (row[0] is None or (isinstance(row[0], float) and pd.isna(row[0]))):
                try: row[0] = int(float(k))
                except: pass
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=COLUMNS + ["t_s"])

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["millis"] = pd.to_numeric(df["millis"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["millis"])
    if df.empty:
        return pd.DataFrame(columns=COLUMNS + ["t_s"])

    df["millis"] = df["millis"].astype("int64")
    base = int(df["millis"].min())
    df["t_s"] = (df["millis"] - base) / 1000.0
    if ENFORCE_SORT:
        df = df.sort_values("millis", kind="stable").reset_index(drop=True)
    return df

def flight_key(team: str, flight: str) -> str:
    return f"{team}-{flight}"

def ensure_state():
    if "flights" not in st.session_state:
        st.session_state["flights"] = {}  # key -> {"raw": df, "processed": df, "last_seen_millis": int}
    if "selected_flight_key" not in st.session_state:
        st.session_state["selected_flight_key"] = flight_key("1", "1")

def set_flight_state(key: str, raw_df: pd.DataFrame):
    filters_for_flight(key)
    processed = process_batch(raw_df)
    last_seen = int(processed["millis"].max()) if ("millis" in processed.columns and processed["millis"].notna().any()) else -1
    st.session_state["flights"][key] = {
        "raw": raw_df,
        "processed": processed,
        "last_seen_millis": last_seen,
    }

def filters_for_flight(key: str):
    fb = st.session_state.setdefault("filters_by_flight", {})
    if key not in fb:
        init_filters()                  # builds st.session_state["filters"]
        fb[key] = st.session_state["filters"]
    st.session_state["filters"] = fb[key]  # point to this flight's filters

def append_flight_state(key: str, df_new_raw: pd.DataFrame):
    if df_new_raw.empty:
        return
    filters_for_flight(key)
    proc_new = process_batch(df_new_raw)
    cur = st.session_state["flights"].get(key, None)
    if cur is None:
        set_flight_state(key, df_new_raw)
        return
    combined = pd.concat([cur["processed"], proc_new], ignore_index=True)
    if "millis" in combined.columns:
        combined = combined.dropna(subset=["millis"])
        combined = combined.drop_duplicates(subset=["millis"], keep="last")
        if ENFORCE_SORT:
            combined = combined.sort_values("millis", kind="stable")
        combined = combined.reset_index(drop=True)
        base = int(combined["millis"].min())
        combined["t_s"] = (combined["millis"] - base) / 1000.0

    st.session_state["flights"][key]["processed"] = combined
    st.session_state["flights"][key]["raw"] = combined.copy()
    st.session_state["flights"][key]["last_seen_millis"] = (
        int(combined["millis"].max()) if "millis" in combined.columns and combined["millis"].notna().any() else -1
    )

def init_filters():
    if "filters" not in st.session_state:
        st.session_state["filters"] = {}
        for c in GAUSS_COLS:
            st.session_state["filters"][c] = RealTimeGaussianFilter(window_size=5, sigma=1.0)
        for c in GPS_COLS:
            st.session_state["filters"][c] = GPSFilter(window_size=5)

def initial_load_all_flights():
    all_live = db.reference("/live").get() or {}
    found_any = False

    for team_key, team_node in _iter_children(all_live):
        for fl_key, flight_node in _iter_children(team_node):
            key = flight_key(str(team_key), str(fl_key))  # e.g., "1-1"
            df = to_df_from_node(flight_node)
            set_flight_state(key, df)
            found_any = True

    if not found_any:
        st.info("No flights found under /live yet.")

def fetch_selected_incremental(selected_key: str):
    """Only refresh the selected flight, pulling any rows with millis > last_seen."""
    if "-" not in selected_key:
        return
    team, fl = selected_key.split("-", 1)
    path = f"/live/{team}/{fl}"
    node = db.reference(path).get() or {}
    df_all = to_df_from_node(node)

    last_seen = st.session_state["flights"].get(selected_key, {}).get("last_seen_millis", -1)
    if "millis" in df_all.columns and df_all["millis"].notna().any() and last_seen >= 0:
        df_new = df_all[df_all["millis"] > last_seen].copy()
    else:
        # If millis is missing or last_seen is unknown, conservatively treat as no new rows
        df_new = pd.DataFrame(columns=df_all.columns)

    append_flight_state(selected_key, df_new)

def distanceBetween(lat1, lon1, lat2, lon2) -> float:
    try:
        lat1 = float(lat1); lon1 = float(lon1)
        lat2 = float(lat2); lon2 = float(lon2)
    except (TypeError, ValueError):
        return 0.0
    
    if any(isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    if not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180 and
            -90 <= lat2 <= 90 and -180 <= lon2 <= 180):
        return 0.0
    if (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0):
        return 0.0
    
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    phi1, phi2 = radians(lat1), radians(lat2)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    
    return 2 * 6_371_000.0 * atan2(sqrt(a), sqrt(1 - a))

def append_cum_distance(df, lat_col, lon_col):
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)

    lat_prev = np.roll(lat, 1); lat_prev[0] = lat[0]
    lon_prev = np.roll(lon, 1); lon_prev[0] = lon[0]

    vec_dist = np.vectorize(distanceBetween, otypes=[float])
    seg_dist = vec_dist(lat_prev, lon_prev, lat, lon)
    seg_dist[0] = 0.0

    df["segmentDistance"]   = seg_dist
    df["distanceTravelled"] = np.cumsum(seg_dist)

def recompute_all_distance():
    df = st.session_state.get("processed_df")
    if df is None or df.empty:
        return
    # Prefer filtered columns if available
    lat_col = "latitude_f" if "latitude_f" in df.columns else "latitude"
    lon_col = "longitude_f" if "longitude_f" in df.columns else "longitude"
    if lat_col not in df.columns or lon_col not in df.columns:
        return
    #df.sort_values("millis", kind="stable", inplace=True, ignore_index=True)
    append_cum_distance(df, lat_col, lon_col)


#---------------- PLOTTING DATA --------------------------------------------

def display_acc(selected_flight):
    df = pd.DataFrame(selected_flight)
    df_melted = df.melt(id_vars=["t_s"], value_vars=["accX", "accY", "accZ"], var_name="Axis", value_name="Acceleration")

    fig = px.line(
        df_melted,
        x="t_s",
        y="Acceleration",
        color="Axis",
        title="Acceleration",
        labels={"t_s": "Time (s)", "Acceleration": "G"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_speed(selected_flight):
    df = pd.DataFrame(selected_flight)

    fig = px.line(
        df,
        x="t_s",
        y="Speed",
        title="Speed",
        labels={"t_s" : "Time (s)", "Speed" : "m/s"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_rpy(selected_flight): #calcuate rpy
    df = pd.DataFrame(selected_flight)
    df_melted = df.melt(id_vars=["t_s"], value_vars=["gyroX", "gyroY", "gyroZ"], var_name="Axis", value_name="Attitude")

    fig = px.line(
        df_melted,
        x="t_s",
        y="Attitude",
        color="Axis",
        title="Roll, Pitch, Yaw",
        labels={"t_s": "Time (s)", "Attitude": "degrees"},
    )
    fig.update_yaxes(range=[-180, 180])
    st.plotly_chart(fig, use_container_width=True)

def display_gyro(df):
    df_melted = df.melt(id_vars=["t_s"], value_vars=["gyroX", "gyroY", "gyroZ"], var_name="Axis", value_name="Angular Acceleration")

    fig = px.line(
        df_melted,
        x="t_s",
        y="Angular Acceleration",
        color="Axis",
        title="Angular Acceleration",
        labels={"t_s": "Time (s)", "Angular Acceleration": "degrees / sec"},
    )
    fig.update_yaxes(range=[-90, 90])
    st.plotly_chart(fig, use_container_width=True)

def display_altitude(df):
    fig = px.line(
        df,
        x="t_s",
        y="gpsAltitude",
        title="Altitude",
        labels={"t_s" : "Time (s)", "gpsAltitude" : "Altitude (m)"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- “current-value only” widgets ----------------
def display_singleAcc(selected_flight):

    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("No accelerometer data yet."); return

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest   # for the Δ value

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Acc X (g)", f"{latest['accX']:.2f}", f"{latest['accX']-prev['accX']:+.2f}")
    with c2:
        st.metric("Acc Y (g)", f"{latest['accY']:.2f}", f"{latest['accY']-prev['accY']:+.2f}")
    with c3:
        st.metric("Acc Z (g)", f"{latest['accZ']:.2f}", f"{latest['accZ']-prev['accZ']:+.2f}")

def display_singleRPY(selected_flight):

    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("No attitude data yet."); return

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Roll (°)",  f"{latest['gyroX']:.1f}", f"{latest['gyroX']-prev['gyroX']:+.1f}")
    with c2:
        st.metric("Pitch (°)", f"{latest['gyroY']:.1f}", f"{latest['gyroY']-prev['gyroY']:+.1f}")
    with c3:
        st.metric("Yaw (°)",   f"{latest['gyroZ']:.1f}", f"{latest['gyroZ']-prev['gyroZ']:+.1f}")

def display_singleGyro(df):
        
    if df.empty:
        st.info("No gyroscope data yet."); return

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("GyroX (°/s)",  f"{latest['gyroX']:.1f}", f"{latest['gyroX']-prev['gyroX']:+.1f}")
    with c2:
        st.metric("GyroY (°/s)", f"{latest['gyroY']:.1f}", f"{latest['gyroY']-prev['gyroY']:+.1f}")
    with c3:
        st.metric("GyroZ (°/s)",   f"{latest['gyroZ']:.1f}", f"{latest['gyroZ']-prev['gyroZ']:+.1f}")

def display_singleSpeed(selected_flight):
    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("No speed data yet."); return

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    st.metric("Speed",  f"{latest['Speed']:.1f}", f"{latest['Speed']-prev['Speed']:+.1f}")

def display_singleAltitude(df):
    if df.empty:
        st.info("No altitude data yet."); return
    
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    st.metric("Altitude (m)",  f"{latest['gpsAltitude']:.1f}", f"{latest['gpsAltitude']-prev['gpsAltitude']:+.1f}")

def display_map(selected_flight):
    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("No valid coordinates yet.")
        return
    else:
        df = df[["latitude", "longitude"]].tail(1)
    st.map(data=df, latitude=None, longitude=None, color=None, size=1, zoom=17, use_container_width=True, width=None, height=None)

def display_distance_travelled(df):
    last_row = df.iloc[-1]

    last_lat = last_row["latitude"]      # or "latitude_f" if you use the filtered one
    last_lon = last_row["longitude"]     # or "longitude_f"

    last_dist = last_row["distanceTravelled"]

    st.markdown(f"Last point: {last_lat}, {last_lon} | **Total distance: {last_dist:.2f} meters**")

def display_battery(selected_flight):
    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("no battery data")
        
    st.write("Battery status: ", df.iloc[-1]['batteryVoltage'], "V")

# --- Display live flight ---
def display_live_flight(selected_live):
    state = st.session_state["flights"].get(selected_live, None)
    if state is state["processed"].empty:
        st.warning("This flight is empty.")
        return
    df = state["processed"]          # <-- DataFrame
    col1, col2, col3 = st.columns(3)
    with col1:
        display_singleSpeed(df)
        display_speed(df)

    with col2:
        display_singleAcc(df)
        display_acc(df)    

    with col3:
        display_singleGyro(df)
        display_gyro(df)

    col4, col5 = st.columns(2)
    with col4:
        display_singleRPY(df)
        display_rpy(df)
    with col5:
        display_singleAltitude(df)
        display_altitude(df)

    display_map(df)
    
    display_distance_travelled(df)
    display_battery(df)

# --- Streamlit functions - Front end ---

def show_live_dashboard():
    st.set_page_config(layout="wide")
    st.header("Dashboard (Live)")
    
    options = [flight_key(team, fl) for team in TEAMS for fl in FLIGHTS]
    selected = st.selectbox(
        "Select Team / Flight",
        options=options,
        index=options.index(st.session_state["selected_flight_key"]) if st.session_state["selected_flight_key"] in options else 0,
        key="selected_flight_key",
    )

    # --- Only update the selected flight ---
    fetch_selected_incremental(selected)

    # --- Display selected flight ---
    display_live_flight(selected)

def main():
    init_firebase()
    ensure_state()

    # Initial one-time load for all flights (creates empty frames for missing/empty nodes)
    if not st.session_state["flights"]:
        initial_load_all_flights()

    show_live_dashboard()

if __name__ == "__main__":
    main()
    
























