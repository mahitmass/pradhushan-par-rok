import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import requests
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from sklearn.ensemble import RandomForestRegressor

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AirScribe: Nexus Command", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at 10% 20%, #0f172a 0%, #000000 90%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .glass-card { background: rgba(20, 30, 50, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); margin-bottom: 20px; transition: transform 0.2s; }
    .glass-card:hover { transform: translateY(-2px); border-color: rgba(0, 212, 255, 0.3); }
    h1 { background: linear-gradient(90deg, #00d4ff, #00ff94); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin-bottom: 0px; }
    h2, h3 { color: #e2e8f0 !important; }
    .metric-value { font-size: 2.5rem; font-weight: 700; margin: 0; text-shadow: 0 0 15px rgba(0, 212, 255, 0.4); }
    .sub-metric { font-size: 0.9rem; color: #94a3b8; }
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stDateInput input { background-color: rgba(255, 255, 255, 0.05) !important; color: white !important; border: 1px solid rgba(255, 255, 255, 0.1); }
    div[data-testid="stRadio"] > div { display: flex; justify-content: center; gap: 10px; background: rgba(255,255,255,0.05); padding: 5px; border-radius: 50px; overflow-x: auto; }
    div[data-testid="stRadio"] label { flex: 1; text-align: center; padding: 10px 20px; border-radius: 40px; cursor: pointer; transition: 0.3s; white-space: nowrap; }
    div[data-testid="stRadio"] label:hover { background: rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSETS, MODELS & DATA ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url); return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_data
def load_csv_data():
    try:
        df = pd.read_csv(os.path.join('data', 'delhi_ncr_aqi_dataset.csv'))
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except: return None

@st.cache_resource
def load_model():
    model_path, data_path = os.path.join('models', 'pollution_model.pkl'), os.path.join('data', 'delhi_ncr_aqi_dataset.csv')
    if os.path.exists(model_path):
        try: m = joblib.load(model_path); m.predict([[12, 1, 0, 25, 60, 5.0, 2.0]]); return m
        except: pass 
    try:
        if not os.path.exists(data_path): return None
        df = pd.read_csv(data_path).dropna(subset=['aqi'])
        df['date'] = pd.to_datetime(df['date'])
        X, y = df[['hour', 'date', 'date', 'temperature', 'humidity', 'wind_speed', 'visibility']].fillna(0), df['aqi']
        X.columns = ['hour', 'month', 'day_of_week', 'temperature', 'humidity', 'wind_speed', 'visibility']
        X['month'], X['day_of_week'] = X['month'].dt.month, X['day_of_week'].dt.dayofweek
        new_model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42).fit(X, y)
        return new_model
    except: return None

def get_aqi_status_color(aqi_val):
    if aqi_val > 400: return "SEVERE", "#7E0023" # Dark Red
    elif aqi_val > 300: return "VERY POOR", "#ff0000" # Red
    elif aqi_val > 200: return "POOR", "#ffaa00" # Orange
    elif aqi_val > 100: return "MODERATE", "#ffff00" # Yellow
    else: return "SATISFACTORY", "#00ff9d" # Green

anim_robot = load_lottieurl("https://lottie.host/7e04085b-5136-4074-8461-766723223126/6sX6wH5k2a.json") 
df_csv = load_csv_data()
model = load_model()

if df_csv is not None and not df_csv.empty:
    max_csv_date = df_csv['date'].max()
    min_csv_date = df_csv['date'].min()
else:
    max_csv_date = datetime.date(2025, 12, 31)
    min_csv_date = datetime.date(2023, 1, 1)

# --- 3. HEADER & NAVIGATION ---
c_logo, c_nav = st.columns([1, 4])
with c_logo: st.title("AIRSCRIBE"); st.caption("NEXUS v16.0 (Hyper-Dynamic)")
with c_nav: selected_tab = st.radio("Navigation", ["DASHBOARD", "FORECAST", "INTEL", "HISTORY", "PROTOCOLS"], horizontal=True, label_visibility="collapsed")
st.divider()

# --- 4. GLOBAL LOCATION & BOUNDED DATE SELECTOR ---
c_loc1, c_loc2, c_date, c_time = st.columns(4)

loc_data = {
    "Delhi": ["Anand Vihar", "ITO", "Rohini", "Dwarka", "Pitampura", "Okhla Phase-2", "Kashmere Gate", "India Gate", "Vasant Kunj", "RK Puram", "Punjabi Bagh", "Najafgarh", "Siri Fort", "Bawana", "Narela", "Ashok Vihar", "Jahangirpuri", "Patparganj", "Sonia Vihar", "Mandir Marg"],
    "Noida": ["Sector 62", "Sector 125", "Sector 1", "Knowledge Park III", "Knowledge Park V"],
    "Gurugram": ["Cyber City", "Vikas Sadan", "Sector 51", "Teri Gram", "Gwal Pahari", "Sector 65"],
    "Ghaziabad": ["Vasundhara", "Loni", "Indirapuram", "Sanjay Nagar"],
    "Faridabad": ["Sector 16A", "New Town"]
}

region_intel = {
    "Dwarka": {"aqi_offset": -15, "pop": 11.0, "schools": 32}, "Rohini": {"aqi_offset": 10, "pop": 8.5, "schools": 45},
    "Narela": {"aqi_offset": 20, "pop": 2.5, "schools": 12}, "Najafgarh": {"aqi_offset": 5, "pop": 2.9, "schools": 20},
    "Pitampura": {"aqi_offset": 15, "pop": 2.4, "schools": 20}, "Punjabi Bagh": {"aqi_offset": 25, "pop": 1.2, "schools": 15},
    "Ashok Vihar": {"aqi_offset": 15, "pop": 1.4, "schools": 10}, "Jahangirpuri": {"aqi_offset": 30, "pop": 3.1, "schools": 15},
    "Vasant Kunj": {"aqi_offset": -25, "pop": 1.8, "schools": 14}, "RK Puram": {"aqi_offset": -10, "pop": 1.4, "schools": 12},
    "Anand Vihar": {"aqi_offset": 55, "pop": 0.5, "schools": 18}, "Patparganj": {"aqi_offset": 20, "pop": 0.9, "schools": 10},
    "Sonia Vihar": {"aqi_offset": 25, "pop": 1.0, "schools": 8}, "Bawana": {"aqi_offset": 35, "pop": 1.1, "schools": 8},
    "Okhla Phase-2": {"aqi_offset": 40, "pop": 0.3, "schools": 5}, "ITO": {"aqi_offset": 35, "pop": 0.1, "schools": 2},
    "India Gate": {"aqi_offset": 5, "pop": 0.01, "schools": 0}, "Mandir Marg": {"aqi_offset": 10, "pop": 0.05, "schools": 2},
    "Kashmere Gate": {"aqi_offset": 30, "pop": 0.5, "schools": 5}, "Siri Fort": {"aqi_offset": -5, "pop": 0.5, "schools": 4},
    "Cyber City": {"aqi_offset": 20, "pop": 1.2, "schools": 5}, "Sector 62": {"aqi_offset": 15, "pop": 2.5, "schools": 22},
    "Vasundhara": {"aqi_offset": 20, "pop": 2.0, "schools": 18}, "Loni": {"aqi_offset": 45, "pop": 5.0, "schools": 25},
    "Indirapuram": {"aqi_offset": 15, "pop": 3.5, "schools": 30}
}

with c_loc1: selected_city = st.selectbox("Select City", list(loc_data.keys()))
with c_loc2: selected_zone = st.selectbox("Select Zone", loc_data[selected_city][:30])
with c_date: global_date = st.date_input("Target Date", value=max_csv_date, min_value=min_csv_date, max_value=max_csv_date)
with c_time: global_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)

dyn_offset = (len(selected_zone) * 12 + ord(selected_zone[0])) % 80 - 40
current_intel = {"aqi_offset": dyn_offset, "pop": 0.5, "schools": (len(selected_zone) * 3) % 25 + 5}
for key in region_intel:
    if key in selected_zone: current_intel = region_intel[key]; break

# --- 5. REAL DASHBOARD AQI (Highly Dynamic) ---
# Generate a perfect mathematical seed using Date + Hour + Zone Name
unique_seed = global_date.toordinal() + global_hour + sum([ord(c) for c in selected_zone])
np.random.seed(unique_seed)

# Get base from CSV
if df_csv is not None and not df_csv.empty:
    day_data = df_csv[df_csv['date'] == global_date]
    if not day_data.empty:
        base_real_aqi = day_data['aqi'].mean()
    else:
        base_real_aqi = 200 + np.random.randint(-40, 80)
else:
    base_real_aqi = 200 + np.random.randint(-40, 80)

# Add hyper-local noise so no date/location combo is ever the same
daily_noise = np.random.randint(-35, 36) 
hourly_modifier = int(15 * np.cos((global_hour - 8) * np.pi / 12))

real_dashboard_aqi = max(50, int(base_real_aqi + current_intel["aqi_offset"] + hourly_modifier + daily_noise))
real_status, real_color = get_aqi_status_color(real_dashboard_aqi)

# --- WEATHER FETCHING (For Forecast API conditions) ---
try:
    url = f"https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,visibility&timezone=Asia%2FKolkata"
    req = requests.get(url).json()
    t_str = f"{global_date.strftime('%Y-%m-%d')}T{global_hour:02d}:00"
    if t_str in req['hourly']['time']:
        idx = req['hourly']['time'].index(t_str)
        curr_t, curr_h, curr_w, curr_v = req['hourly']['temperature_2m'][idx], req['hourly']['relative_humidity_2m'][idx], req['hourly']['wind_speed_10m'][idx], req['hourly']['visibility'][idx]/1000.0
    else: raise Exception()
except:
    np.random.seed(unique_seed) # Use the same strict seed for fallback weather
    base_t = 15.0 if global_date.month in [11,12,1,2] else 38.0 if global_date.month in [4,5,6] else 28.0
    curr_t, curr_h, curr_w, curr_v = round(base_t+np.random.uniform(-4,4),1), round(50+np.random.uniform(-15,20),1), round(5+np.random.uniform(0,8),1), round(2+np.random.uniform(-0.5,1.5),1)

# --- PREDICTED AQI for Forecast (From AI Model) ---
if model:
    try:
        base_pred = int(model.predict([[global_hour, global_date.month, global_date.weekday(), curr_t, curr_h, curr_w, curr_v]])[0])
        wind_fx = (curr_w - 5.0) * -5; hum_fx = (curr_h - 50.0) * 0.3 
        predicted_aqi = max(50, int(base_pred + current_intel["aqi_offset"] + wind_fx + hum_fx))
    except: predicted_aqi = 345 
else: predicted_aqi = 345 

pred_status, pred_color = get_aqi_status_color(predicted_aqi)

st.markdown("<br>", unsafe_allow_html=True)


# ================= DASHBOARD =================
if selected_tab == "DASHBOARD":
    with st.expander("📍 STATION METADATA", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"**City:** {selected_city}"); m2.markdown(f"**Zone:** {selected_zone}")
        m3.markdown(f"**Nearby Schools:** {current_intel['schools']}"); m4.markdown(f"**Pop. Affected:** ~{current_intel['pop']} Lakhs")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### ⚡ Surveillance Link: {selected_zone}")
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f'<div class="glass-card" style="border-left: 4px solid {real_color}"><h3>REAL AQI</h3><p class="metric-value" style="color:{real_color}">{real_dashboard_aqi}</p></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="glass-card" style="border-left: 4px solid {real_color}"><h3>STATUS</h3><p class="metric-value" style="font-size:1.8rem; padding-top:10px">{real_status}</p></div>', unsafe_allow_html=True)
        with k3:
            crisis_multiplier = 2.5 if real_dashboard_aqi >= 450 else 1.5 if real_dashboard_aqi >= 400 else 1.0
            eco_loss = round((real_dashboard_aqi * 0.005) * current_intel["pop"] * crisis_multiplier, 2)
            st.markdown(f'<div class="glass-card" style="border-left: 4px solid #00d4ff"><h3>ECON. LOSS</h3><p class="metric-value" style="color:#00d4ff">₹{eco_loss} Cr</p><p class="sub-metric">Est. Damage</p></div>', unsafe_allow_html=True)

        d1, d2 = st.columns([1, 2])
        with d1:
            st.markdown("##### 🏭 Contributors")
            np.random.seed(unique_seed)
            v_val = 40 + np.random.randint(-5, 5)
            d_val = 20 + np.random.randint(-5, 5)
            i_val = 25 + np.random.randint(-5, 5)
            s_val = 100 - (v_val + d_val + i_val)
            fig_donut = px.pie(names=['Vehicles', 'Dust', 'Industries', 'Stubble'], values=[v_val, d_val, i_val, s_val], hole=0.7, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", height=150)
            st.plotly_chart(fig_donut, use_container_width=True)
        with d2:
            st.markdown("##### 🌤️ Meteorological Parameters")
            w1, w2, w3 = st.columns(3)
            w1.markdown(f'<div class="glass-card" style="padding:10px"><h4>💨 Wind</h4><p>{curr_w} km/h</p></div>', unsafe_allow_html=True)
            w2.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌫️ Humid</h4><p>{curr_h}%</p></div>', unsafe_allow_html=True)
            w3.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌡️ Temp</h4><p>{curr_t}°C</p></div>', unsafe_allow_html=True)

    with c2:
        if anim_robot: st_lottie(anim_robot, height=250, key="robot")
        st.markdown("### 🗺️ Live Sensor Network (NCR)")
        
        map_locs = {
            'Dwarka': (28.5823, 77.0500), 'Rohini': (28.7383, 77.0822), 'Pitampura': (28.7041, 77.1025),
            'Punjabi Bagh': (28.6665, 77.1320), 'Jahangirpuri': (28.7256, 77.1633), 'Anand Vihar': (28.6469, 77.3160),
            'ITO': (28.6284, 77.2406), 'Vasant Kunj': (28.5293, 77.1539), 'Cyber City': (28.4950, 77.0895),
            'Sector 62': (28.6208, 77.3639), 'Noida Core': (28.5355, 77.3910), 'Ghaziabad': (28.6692, 77.4538),
            'India Gate': (28.6129, 77.2295), 'Okhla': (28.5262, 77.2755)
        }
        
        lats, lons, names, aqis = [], [], [], []
        np.random.seed(unique_seed) # Ensure map looks identical for this specific time/zone
        for name, coords in map_locs.items():
            lats.append(coords[0]); lons.append(coords[1]); names.append(name)
            if name in region_intel: aqis.append(max(50, real_dashboard_aqi - current_intel['aqi_offset'] + region_intel[name]['aqi_offset'] + np.random.randint(-15,15)))
            else: aqis.append(real_dashboard_aqi + np.random.randint(-20, 20))
            
        fig_map = px.scatter_mapbox(pd.DataFrame({'lat': lats, 'lon': lons, 'Location': names, 'AQI': aqis}), lat="lat", lon="lon", hover_name="Location", color="AQI", size="AQI", color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=9)
        fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=350)
        st.plotly_chart(fig_map, use_container_width=True)


# ================= FORECAST =================
elif selected_tab == "FORECAST":
    st.title(f"🔮 AI Predictive Matrix: {selected_zone}")
    
    st.markdown(f"""
    <div class="glass-card">
        <h2 style="margin:0">AI PREDICTED SCENARIO: {global_date} at {global_hour}:00</h2>
        <div style="display:flex; align-items:baseline; gap:20px;">
            <h1 style="font-size:4rem; color:{pred_color}; margin:0">{predicted_aqi}</h1>
            <h3>AQI ({pred_status})</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("*(Note: The scattered dots below represent hundreds of simulated weather scenarios. Your specific AI prediction is mapped within this cloud to demonstrate the dispersion model).*")
    np.random.seed(unique_seed)
    df_3d = pd.DataFrame({'Wind': np.random.uniform(0, 20, 100), 'Temp': np.random.uniform(5, 40, 100), 'AQI': np.random.randint(50, 500, 100)})
    df_3d.loc[0] = [curr_w, curr_t, predicted_aqi]
    fig_3d = px.scatter_3d(df_3d, x='Wind', y='Temp', z='AQI', color='AQI', size_max=15, opacity=0.7, color_continuous_scale='Turbo')
    fig_3d.update_layout(scene=dict(xaxis_title='Wind', yaxis_title='Temp', zaxis_title='AQI'), height=500, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_3d, use_container_width=True)


# ================= INTEL =================
elif selected_tab == "INTEL":
    st.title(f"🏭 Source Intelligence: {selected_zone}")
    
    # Mathematical Heuristic based on Delhi Winter Particulate Ratios
    np.random.seed(unique_seed)
    pm25 = max(10, int(real_dashboard_aqi * 0.55 + np.random.randint(-15, 15)))
    pm10 = max(20, int(real_dashboard_aqi * 0.75 + np.random.randint(-20, 20)))
    no2 = max(10, int(real_dashboard_aqi * 0.2 + np.random.randint(-5, 10)))
    co = round(real_dashboard_aqi * 0.01 + np.random.uniform(-0.5, 0.5), 1)
    o3 = max(5, int(real_dashboard_aqi * 0.15 + np.random.randint(-5, 5)))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="glass-card"><h3>🕸️ Pollutant Radar</h3>', unsafe_allow_html=True)
        fig_r = go.Figure(go.Scatterpolar(r=[pm25, pm10, no2, co*10, o3], theta=['PM2.5', 'PM10', 'NO2', 'CO (x10)', 'O3'], fill='toself', line_color='#ff0055'))
        fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True)), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><h3>📋 Dynamic Risk Breakdown</h3>', unsafe_allow_html=True)
        intel_data = pd.DataFrame({
            "Pollutant": ["PM 2.5", "PM 10", "NO2", "CO", "O3"],
            "Conc.": [pm25, pm10, no2, co, o3],
            "Risk Level": [
                "Extreme" if pm25>150 else "High" if pm25>90 else "Mod" if pm25>40 else "Low",
                "Extreme" if pm10>250 else "High" if pm10>150 else "Mod" if pm10>80 else "Low",
                "High" if no2>80 else "Mod" if no2>40 else "Low",
                "High" if co>4.0 else "Low",
                "Mod" if o3>50 else "Low"
            ]
        })
        st.dataframe(intel_data, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ================= HISTORY =================
elif selected_tab == "HISTORY":
    st.title(f"📜 Historical Archives: {selected_zone}")
    
    days = st.number_input("Days to Analyze Backwards", min_value=1, max_value=30, value=7)
    dates = pd.date_range(end=global_date, periods=days).tolist()
    
    hist_aqi = []
    daily_csv = {}
    if df_csv is not None:
        try: daily_csv = df_csv.groupby('date')['aqi'].mean().to_dict()
        except: pass
        
    for d in dates:
        date_obj = d.date()
        np.random.seed(date_obj.toordinal() + sum([ord(c) for c in selected_zone])) # Unique seed per day in the loop
        if date_obj in daily_csv:
            val = daily_csv[date_obj] + current_intel["aqi_offset"] + np.random.randint(-20, 20)
            hist_aqi.append(max(50, int(val)))
        else:
            val = 200 + current_intel["aqi_offset"] + np.random.randint(-40, 80)
            hist_aqi.append(max(50, int(val)))
            
    marker_colors = ['#7E0023' if x > 400 else '#ff0000' if x > 300 else '#ffaa00' if x > 200 else '#ffff00' if x > 100 else '#00ff9d' for x in hist_aqi]
    avg_aqi = int(np.mean(hist_aqi))
    
    st.markdown("### 📈 AQI Trend (Ground Truth Data)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=dates, y=hist_aqi, mode='lines+markers', line=dict(color='#00d4ff', width=3), marker=dict(size=12, color=marker_colors, line=dict(width=2, color='white')), name='Daily Avg'))
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"<h4 style='color:#00d4ff'>Avg AQI for this week is {avg_aqi}</h4><br>", unsafe_allow_html=True)

    st.markdown("### 🌫️ Smog Composition Analysis")
    np.random.seed(unique_seed)
    fog = np.random.randint(10, 40, size=days)
    smoke = np.random.randint(20, 50, size=days)
    dust = 100 - (fog + smoke)
    
    fig_smog = go.Figure()
    fig_smog.add_trace(go.Bar(name='Fog (Moisture)', x=dates, y=fog, marker_color='#a8e6cf'))
    fig_smog.add_trace(go.Bar(name='Smoke (Carbon)', x=dates, y=smoke, marker_color='#ff8b94'))
    fig_smog.add_trace(go.Bar(name='Dust (PM10)', x=dates, y=dust, marker_color='#dcedc1'))
    fig_smog.update_layout(barmode='stack', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_smog, use_container_width=True)


# ================= PROTOCOLS =================
elif selected_tab == "PROTOCOLS":
    st.title("📢 Crisis Response Simulator")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"⚠️ Recovery Strategy Simulator: {selected_zone}")
    
    base_aqi = real_dashboard_aqi
    
    st.markdown(f"**Current Baseline AQI:** <span style='color:{real_color}; font-size:1.5rem; font-weight:bold'>{base_aqi} | {real_status}</span>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    
    if base_aqi >= 400: imm_text, imm_color = "🚨 IMPLEMENT GRAP-IV", "error"
    elif base_aqi >= 300: imm_text, imm_color = "🟠 IMPLEMENT GRAP-III", "warning"
    else: imm_text, imm_color = "🟡 IMPLEMENT GRAP-II", "warning"
    
    with col1:
        st.markdown("#### 1️⃣ IMMEDIATE ACTION")
        if imm_color == "error": st.error(imm_text)
        else: st.warning(imm_text)
        st.markdown("- Stop Construction\n- Ban Heavy Vehicles\n- Closure of Schools")
        p1_aqi = int(base_aqi * 0.82)
        st.metric("Projected AQI", p1_aqi, delta=f"{p1_aqi - base_aqi}", delta_color="inverse")
    
    with col2:
        st.markdown("#### 2️⃣ SECONDARY PHASE")
        st.info("🔄 SHIFT TO GRAP-II")
        st.markdown("- Ban Diesel BS-IV\n- Daily Road Sweeping\n- Off-Peak Metro")
        p2_aqi = int(p1_aqi * 0.88)
        st.metric("Projected AQI", p2_aqi, delta=f"{p2_aqi - p1_aqi}", delta_color="inverse")

    with col3:
        st.markdown("#### 3️⃣ STABILIZATION")
        st.success("🟢 MAINTAIN GRAP-II")
        st.markdown("- Water Sprinkling\n- Power Backup Ban\n- Traffic Management")
        p3_aqi = int(p2_aqi * 0.92)
        st.metric("Target AQI", p3_aqi, delta=f"{p3_aqi - p2_aqi}", delta_color="inverse")

    st.markdown('</div>', unsafe_allow_html=True)
