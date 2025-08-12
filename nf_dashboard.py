import streamlit as st
import firebase_admin
from firebase_admin import db, credentials
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go


#naming convention for course flight vs endurance flight
LIVE_REFRESH = 100 #app refresh when live (ms)
STANDARD_REFRESH = 5000 #app refresh for archive (ms)
DATA_REFRESH = 0.5 #dataframe update rate (ms)
DATA_FREQUENCY = 5 #data points per second

interval = STANDARD_REFRESH
st_autorefresh(interval=interval, key="app_refresh")

firebase_creds = st.secrets["firebase"]
cred = credentials.Certificate(dict(firebase_creds))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': r"https://datalogger-nfc25-default-rtdb.europe-west1.firebasedatabase.app"
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
COLUMNS = [
    "millis","accX","accY","accZ","gyroX","gyroY","gyroZ", "magX","magY","magZ",
    "latitude","longitude", "gpsAltitude","Speed","SatCount","batteryVoltage"
]
GAUSS_COLS = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "magX", "magY", "magZ"]
GPS_COLS   = ["latitude", "longitude"] #without gpsAltitude?

def to_raw_df(data):
    if isinstance(data, dict):
        data = list(data.values())

    good = [p for p in data if isinstance(p, dict) and isinstance(p.get("data"), list) and len(p["data"]) == len(COLUMNS)]
    if not good:
        return pd.DataFrame(columns=COLUMNS + ["mode", "ts"])

    df = pd.DataFrame([p["data"] for p in good], columns=COLUMNS)
    df["mode"] = [p.get("mode", "REALTIME") for p in good]

    df["millis"] = pd.to_numeric(df["millis"], errors="coerce").round().astype("Int64")
    df = df.dropna(subset=["millis"])
    df["millis"] = df["millis"].astype("int64")

    if "SatCount" in df.columns:
        df["SatCount"] = pd.to_numeric(df["SatCount"], errors="coerce").round().astype("Int64")

    df["ts"] = pd.to_datetime(df["millis"], unit="ms", utc=True)
    df.sort_values("millis", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def init_filters():
    if "filters" not in st.session_state:
        st.session_state["filters"] = {}
        for c in GAUSS_COLS:
            st.session_state["filters"][c] = RealTimeGaussianFilter(window_size=5, sigma=1.0)
        for c in GPS_COLS:
            st.session_state["filters"][c] = GPSFilter(window_size=5)

def process_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return raw_df

    df_out = raw_df.sort_values("millis", kind="stable").reset_index(drop=True).copy()

    # --- Gaussian-filtered sensor columns (stateful, per column) ---
    for col in GAUSS_COLS:
        filt = st.session_state["filters"].get(col)
        if filt is None or col not in df_out.columns:
            continue
        out = []
        for v in df_out[col].to_numpy():
            out.append(filt.update(float(v)))
        df_out[f"{col}_f"] = out

    # --- GPS-filtered columns: each uses satellite count as weights ---
    sat_col = "satCount" if "satCount" in df_out.columns else ("SatCount" if "SatCount" in df_out.columns else None)
    if sat_col is not None:
        for col in GPS_COLS:
            filt = st.session_state["filters"].get(col)
            if filt is None or col not in df_out.columns:
                continue
            out = []
            for val, sc in zip(df_out[col].to_numpy(), df_out[sat_col].to_numpy()):
                out.append(filt.update(float(val), int(sc)))
            df_out[f"{col}_f"] = out
    else:
        # If there is no satellite count, fall back to raw (or skip)
        for col in GPS_COLS:
            if col in df_out.columns:
                df_out[f"{col}_f"] = df_out[col].to_numpy()

    return df_out

def append_state(new_raw, new_processed):
    st.session_state["raw_df"] = (
        pd.concat([st.session_state["raw_df"], new_raw], ignore_index=True)
        if "raw_df" in st.session_state else new_raw
    )
    st.session_state["processed_df"] = (
        pd.concat([st.session_state["processed_df"], new_processed], ignore_index=True)
        if "processed_df" in st.session_state else new_processed
    )
    st.session_state["last_seen_millis"] = int(st.session_state["raw_df"]["millis"].max())
    if not st.session_state["raw_df"].empty:
        st.session_state["last_seen_millis"] = int(st.session_state["raw_df"]["millis"].max())

def fetch_new_data(node: str, last_seen_millis: int, limit=5000):
    ref = db.reference(node)
    snap = (ref.order_by_child("millis")
                .start_at(last_seen_millis + 1)  # strictly greater than last seen
                .limit_to_first(limit)
                .get())
    return snap or {}

#if "archived_flights_df" not in st.session_state:
#    st.session_state["archived_flights_df"] = {
#        k: g.copy()
#        for k, g in st.session_state["raw_df_archive"].groupby("flight_name")
#    }

#if "live_flights_df" not in st.session_state:
#    st.session_state["live_flights_df"] = {
#        k: g.copy()
#        for k, g in st.session_state["raw_df_live"].groupby("flight_name")
#    }
#archive_keys = list(st.session_state["archived_flights_df"].keys())
#live_keys = list(st.session_state["live_flights_df"].keys())


# --- Getting initial load ---
if "raw_df" not in st.session_state:
    ref = db.reference('/live')
    data = (ref.get() or {})
    init_filters()
    df_raw = to_raw_df(data)
    if not df_raw.empty:
        df_processed = process_batch(df_raw)              # uses stateful filters
        append_state(df_raw, df_processed)

#--- Fetch new data ---
new_data = fetch_new_data('/live', st.session_state.get("last_seen_millis", -1))
df_new_raw = to_raw_df(new_data)               # just the new rows
if not df_new_raw.empty:
    df_new_processed = process_batch(df_new_raw)
    append_state(df_new_raw, df_new_processed)  

#---------------- PLOTTING DATA --------------------------------------------

def display_acc(selected_flight):
    df = pd.DataFrame(selected_flight)
    df_melted = df.melt(id_vars=["millis"], value_vars=["accX", "accY", "accZ"], var_name="Axis", value_name="Acceleration")

    fig = px.line(
        df_melted,
        x="millis",
        y="Acceleration",
        color="Axis",
        title="Acceleration",
        labels={"millis": "Time (s)", "Acceleration": "G"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_speed(selected_flight):
    df = pd.DataFrame(selected_flight)

    fig = px.line(
        df,
        x="millis",
        y="Speed",
        title="Speed",
        labels={"millis" : "Time (s)", "Speed" : "m/s"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_rpy(selected_flight): #calcuate rpy
    df = pd.DataFrame(selected_flight)
    df_melted = df.melt(id_vars=["millis"], value_vars=["gyroX", "gyroY", "gyroZ"], var_name="Axis", value_name="Attitude")

    fig = px.line(
        df_melted,
        x="millis",
        y="Attitude",
        color="Axis",
        title="Roll, Pitch, Yaw",
        labels={"millis": "Time (s)", "Attitude": "degrees"},
    )
    fig.update_yaxes(range=[-180, 180])
    st.plotly_chart(fig, use_container_width=True)

def display_map(selected_flight):
    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("No valid coordinates yet.")
        return
    else:
        df = df[["latitude", "longitude"]].tail(1)
    st.map(data=df, latitude=None, longitude=None, color=None, size=1, zoom=20, use_container_width=True, width=None, height=None)

def display_distance_travelled(selected_flight):
    df = pd.DataFrame(selected_flight)
    df["distances"] = 0.0  # initialize column with floats
    distance_travelled = 0.0

    for i in range(len(df) - 1):
        lat1, lon1 = df.loc[i, "latitude"], df.loc[i, "longitude"]
        lat2, lon2 = df.loc[i + 1, "latitude"], df.loc[i + 1, "longitude"]

        if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
            distance_travelled += geodesic((lat1, lon1), (lat2, lon2)).meters
    
    df.loc[i + 1, "distances"] = distance_travelled
    st.write("Total distance travelled: ", distance_travelled)


    df = pd.DataFrame(selected_flight)

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

def display_singleSpeed(selected_flight):
    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("No speed data yet."); return

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    st.metric("Speed",  f"{latest['Speed']:.1f}", f"{latest['Speed']-prev['Speed']:+.1f}")

def display_battery(selected_flight):
    df = pd.DataFrame(selected_flight)
    if df.empty:
        st.info("no battery data")
        
    st.write("Battery status: ", df.iloc[-1]['batteryVoltage'], "V")

# --- Display live flight ---
def display_live_flight(selected_live):
    col1, col2, col3 = st.columns(3)
    with col1:
        display_singleSpeed(selected_live)
        display_speed(selected_live)

    with col2:
        display_singleAcc(selected_live)
        display_acc(selected_live)    

    with col3:
        display_singleRPY(selected_live)
        display_rpy(selected_live)
    
    display_map(selected_live)
    
    display_distance_travelled(selected_live)

    display_battery(selected_live)

def display_archive(selected_archive):
    col1, col2, col3 = st.columns(3)
    with col1:
        display_speed(selected_archive) 

    with col2:
        display_acc(selected_archive)

    with col3:
        display_rpy(selected_archive)
    display_distance_travelled(selected_archive)


# --- Streamlit functions - Front end ---

def show_live_dashboard():
    
    #if not get_live_flights(): #standby page
    #    st.subheader("🛬  Stand-by — waiting for next flight")
    #    st.info("Drop a new CSV into Firebase to begin streaming.")
    #    st_autorefresh(interval=STANDARD_REFRESH, key="standard_refresh")
    #else:
    #    selected_live = st.selectbox("Watch flight:", live_keys)
    
    display_live_flight(st.session_state['processed_df']) #change later
    
tab1, tab2 = st.tabs(["Live Flight", "Archive"])
selected_tab = st.session_state.get("selected_tab", "Live")
if tab1: #refresh faster for live dashboard
    interval = LIVE_REFRESH
else:
    interval = STANDARD_REFRESH


st.set_page_config(layout="wide")
with tab1:
    st.header("Dashboard (Live)")
    show_live_dashboard()
    

with tab2:
    st.header("Archived Flights")
    
    #selected_archive = st.selectbox("Choose a flight", archive_keys)
    #if selected_archive is not None:
    #    display_archive(archived_flights_df[selected_archive])




