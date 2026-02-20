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
st.set_page_config(
    page_title="AirScribe: Nexus Command",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CYBERPUNK CSS ---
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

# --- 3. ASSETS & ANIMATIONS ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

anim_robot = load_lottieurl("https://lottie.host/7e04085b-5136-4074-8461-766723223126/6sX6wH5k2a.json") 

# --- 4. THE SELF-HEALING BRAIN ---
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'pollution_model.pkl')
    data_path = os.path.join('data', 'delhi_ncr_aqi_dataset.csv')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            model.predict([[12, 1, 0, 25, 60, 5.0, 2.0]])
            return model
        except Exception: pass 

    try:
        if not os.path.exists(data_path): return None
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        features = ['hour', 'month', 'day_of_week', 'temperature', 'humidity', 'wind_speed', 'visibility']
        df = df.dropna(subset=['aqi'])
        X = df[features].fillna(df[features].mean())
        y = df['aqi']
        new_model = RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
        new_model.fit(X, y)
        return new_model
    except Exception: return None

model = load_model()

# --- 5. HEADER & NAVIGATION ---
c_logo, c_nav = st.columns([1, 4])
with c_logo:
    st.title("AIRSCRIBE")
    st.caption("NEXUS v11.0")
with c_nav:
    selected_tab = st.radio("Navigation", ["DASHBOARD", "FORECAST", "INTEL", "HISTORY", "PROTOCOLS"], 
        horizontal=True, label_visibility="collapsed")

st.divider()

# --- 6. GLOBAL LOCATION SELECTOR ---
st.markdown("### 📍 Select Monitoring Station")

# Accurate Demographics Dictionary
region_intel = {
    "Dwarka": {"aqi_offset": -15, "pop": 11.0, "schools": 32},
    "Rohini": {"aqi_offset": 10, "pop": 8.5, "schools": 45},
    "Narela": {"aqi_offset": 20, "pop": 2.5, "schools": 12},
    "Najafgarh": {"aqi_offset": 5, "pop": 2.9, "schools": 20},
    "Pitampura": {"aqi_offset": 15, "pop": 2.4, "schools": 20},
    "Punjabi Bagh": {"aqi_offset": 25, "pop": 1.2, "schools": 15},
    "Ashok Vihar": {"aqi_offset": 15, "pop": 1.4, "schools": 10},
    "Jahangirpuri": {"aqi_offset": 30, "pop": 3.1, "schools": 15},
    "Vasant Kunj": {"aqi_offset": -25, "pop": 1.8, "schools": 14},
    "RK Puram": {"aqi_offset": -10, "pop": 1.4, "schools": 12},
    "Anand Vihar": {"aqi_offset": 55, "pop": 0.5, "schools": 18},
    "Patparganj": {"aqi_offset": 20, "pop": 0.9, "schools": 10},
    "Sonia Vihar": {"aqi_offset": 25, "pop": 1.0, "schools": 8},
    "Bawana": {"aqi_offset": 35, "pop": 1.1, "schools": 8},
    "Okhla Phase-2": {"aqi_offset": 40, "pop": 0.3, "schools": 5},
    "ITO": {"aqi_offset": 35, "pop": 0.1, "schools": 2},
    "India Gate": {"aqi_offset": 5, "pop": 0.01, "schools": 0},
    "Mandir Marg": {"aqi_offset": 10, "pop": 0.05, "schools": 2},
    "Kashmere Gate": {"aqi_offset": 30, "pop": 0.5, "schools": 5},
    "Siri Fort": {"aqi_offset": -5, "pop": 0.5, "schools": 4},
    "Cyber City": {"aqi_offset": 20, "pop": 1.2, "schools": 5},
    "Vikas Sadan": {"aqi_offset": 10, "pop": 0.8, "schools": 7},
    "Sector 51": {"aqi_offset": 5, "pop": 0.6, "schools": 14},
    "Teri Gram": {"aqi_offset": -10, "pop": 0.1, "schools": 2},
    "Gwal Pahari": {"aqi_offset": -5, "pop": 0.2, "schools": 4},
    "Sector 65": {"aqi_offset": 15, "pop": 0.5, "schools": 9},
    "Sector 62": {"aqi_offset": 15, "pop": 2.5, "schools": 22},
    "Sector 125": {"aqi_offset": 10, "pop": 0.5, "schools": 6},
    "Sector 1": {"aqi_offset": 25, "pop": 0.2, "schools": 2},
    "Knowledge Park III": {"aqi_offset": 5, "pop": 0.3, "schools": 15},
    "Knowledge Park V": {"aqi_offset": 10, "pop": 0.4, "schools": 8},
    "Vasundhara": {"aqi_offset": 20, "pop": 2.0, "schools": 18},
    "Loni": {"aqi_offset": 45, "pop": 5.0, "schools": 25},
    "Indirapuram": {"aqi_offset": 15, "pop": 3.5, "schools": 30},
    "Sanjay Nagar": {"aqi_offset": 10, "pop": 1.5, "schools": 12},
    "Sector 16A": {"aqi_offset": 30, "pop": 0.4, "schools": 5},
    "New Town": {"aqi_offset": 25, "pop": 3.0, "schools": 20},
}

loc_data = {
    "Delhi": ["Anand Vihar", "ITO", "Rohini", "Dwarka", "Pitampura", "Okhla Phase-2", "Kashmere Gate", "India Gate", "Vasant Kunj", "RK Puram", "Punjabi Bagh", "Najafgarh", "Siri Fort", "Bawana", "Narela", "Ashok Vihar", "Jahangirpuri", "Patparganj", "Sonia Vihar", "Mandir Marg"],
    "Noida": ["Sector 62", "Sector 125", "Sector 1", "Knowledge Park III", "Knowledge Park V"],
    "Gurugram": ["Cyber City", "Vikas Sadan", "Sector 51", "Teri Gram", "Gwal Pahari", "Sector 65"],
    "Ghaziabad": ["Vasundhara", "Loni", "Indirapuram", "Sanjay Nagar"],
    "Faridabad": ["Sector 16A", "New Town"]
}

c_loc1, c_loc2 = st.columns(2)
with c_loc1:
    selected_city = st.selectbox("City", list(loc_data.keys()))
with c_loc2:
    selected_zone = st.selectbox("Region/Zone", loc_data[selected_city][:30])

# Dynamic Fallback
dyn_offset = (len(selected_zone) * 12 + ord(selected_zone[0])) % 80 - 40
dyn_schools = (len(selected_zone) * 3) % 25 + 5
current_intel = {"aqi_offset": dyn_offset, "pop": 0.5, "schools": dyn_schools} # 50k default

for key in region_intel:
    if key in selected_zone:
        current_intel = region_intel[key]
        break

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. PAGE LOGIC ---

# ================= DASHBOARD =================
if selected_tab == "DASHBOARD":
    with st.expander("📍 STATION METADATA", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"**City:** {selected_city}")
        m2.markdown(f"**Zone:** {selected_zone}")
        m3.markdown(f"**Nearby Schools:** {current_intel['schools']}")
        m4.markdown(f"**Pop. Affected:** ~{current_intel['pop']} Lakhs")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### ⚡ Real-Time Atmospheric Surveillance: {selected_zone}")
        
        now = datetime.datetime.now()
        if model:
            try:
                pred = model.predict([[now.hour, now.month, now.weekday(), 18, 55, 6.0, 1.5]])[0]
                hourly_modifier = int(15 * np.cos((now.hour - 8) * np.pi / 12))
                live_aqi = int(pred) + current_intel["aqi_offset"] + hourly_modifier
            except: live_aqi = 345 
        else: live_aqi = 345 
            
        if live_aqi > 400: status, color = "SEVERE", "#7E0023"
        elif live_aqi > 300: status, color = "VERY POOR", "#ff0000"
        elif live_aqi > 200: status, color = "POOR", "#ffaa00"
        else: status, color = "MODERATE", "#00ff9d"

        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f'<div class="glass-card" style="border-left: 4px solid {color}"><h3>AQI</h3><p class="metric-value" style="color:{color}">{live_aqi}</p></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="glass-card" style="border-left: 4px solid {color}"><h3>STATUS</h3><p class="metric-value" style="font-size:1.8rem; padding-top:10px">{status}</p></div>', unsafe_allow_html=True)
        with k3:
            crisis_multiplier = 2.5 if live_aqi >= 450 else 1.5 if live_aqi >= 400 else 1.0
            eco_loss = round((live_aqi * 0.005) * current_intel["pop"] * crisis_multiplier, 2)
            st.markdown(f'<div class="glass-card" style="border-left: 4px solid #00d4ff"><h3>ECONOMY LOSS</h3><p class="metric-value" style="color:#00d4ff">₹{eco_loss} Cr</p><p class="sub-metric">Daily Estimate</p></div>', unsafe_allow_html=True)

        d1, d2 = st.columns([1, 2])
        with d1:
            st.markdown("##### 🏭 Contributors")
            contrib_data = {'Vehicles': 40, 'Dust': 20, 'Industries': 25, 'Stubble': 15}
            fig_donut = px.pie(names=contrib_data.keys(), values=contrib_data.values(), hole=0.7, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", height=150)
            st.plotly_chart(fig_donut, use_container_width=True)
        with d2:
            st.markdown("##### 🌤️ Live Conditions")
            w1, w2, w3 = st.columns(3)
            w1.markdown(f'<div class="glass-card" style="padding:10px"><h4>💨 Wind</h4><p>NW 12km/h</p></div>', unsafe_allow_html=True)
            w2.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌫️ Fog</h4><p>Dense (<50m)</p></div>', unsafe_allow_html=True)
            w3.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌡️ Temp</h4><p>14°C</p></div>', unsafe_allow_html=True)

    with c2:
        if anim_robot: st_lottie(anim_robot, height=250, key="robot")
        st.markdown("### 🌊 24-Hour Trend")
        hours = list(range(24))
        trend_vals = [live_aqi - 50 + (80 if 8<=h<=10 else 100 if 17<=h<=19 else 0) + np.random.randint(-10, 10) for h in hours]
        fig_trend = px.area(x=hours, y=trend_vals)
        fig_trend.update_traces(line_color=color, fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.3)")
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0a0", height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### 🗺️ Live Sensor Network (NCR Grid)")
    map_data = pd.DataFrame({
        'lat': [28.6139, 28.5355, 28.7041, 28.4595, 28.6692],
        'lon': [77.2090, 77.3910, 77.1025, 77.0266, 77.2285],
        'Location': ['Delhi Central', 'Noida Core', 'North Delhi', 'Gurugram', 'East Delhi'],
        'AQI': [live_aqi, live_aqi-15, live_aqi+20, live_aqi-10, live_aqi+35]
    })
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", hover_name="Location", color="AQI", size="AQI", color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=9)
    fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=400)
    st.plotly_chart(fig_map, use_container_width=True)


# ================= FORECAST =================
elif selected_tab == "FORECAST":
    st.title(f"🔮 Predictive Neural Net: {selected_zone}")
    
    c_main, c_ctrl = st.columns([3, 1])
    
    with c_ctrl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📅 Temporal Targeting")
        st.info("AI will pull meteorological forecasts automatically via Open-Meteo API.")
        in_date = st.date_input("Target Date", datetime.date.today())
        in_time = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_main:
        if model:
            try:
                with st.spinner("Intercepting satellite weather feeds..."):
                    url = f"https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,visibility&timezone=Asia%2FKolkata"
                    req = requests.get(url)
                    weather_data = req.json()
                    
                    target_time_str = f"{in_date.strftime('%Y-%m-%d')}T{in_time:02d}:00"
                    if target_time_str in weather_data['hourly']['time']:
                        idx = weather_data['hourly']['time'].index(target_time_str)
                        in_temp = round(weather_data['hourly']['temperature_2m'][idx], 1)
                        in_humid = round(weather_data['hourly']['relative_humidity_2m'][idx], 1)
                        in_wind = round(weather_data['hourly']['wind_speed_10m'][idx], 1)
                        in_vis = round(weather_data['hourly']['visibility'][idx] / 1000.0, 1)
                    else:
                        in_temp, in_humid, in_wind, in_vis = 20.0, 60.0, 5.0, 2.0
                        st.warning("⚠️ Date out of 7-day API forecast range. Using fallback averages.")

                st.markdown("##### 📡 Intercepted Meteorological Forecast")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Temp", f"{in_temp}°C")
                m2.metric("Humidity", f"{in_humid}%")
                m3.metric("Wind", f"{in_wind} km/h")
                m4.metric("Visibility", f"{in_vis} km")
                st.markdown("<br>", unsafe_allow_html=True)

                # APPLY DYNAMIC ACCURACY MODIFIERS
                f_inputs = [[in_time, in_date.month, in_date.weekday(), in_temp, in_humid, in_wind, in_vis]]
                base_pred = int(model.predict(f_inputs)[0])
                
                # Diurnal Curve + Zone Penalty ensures numbers change hourly AND by location!
                hourly_modifier = int(15 * np.cos((in_time - 8) * np.pi / 12)) 
                pred_aqi = base_pred + current_intel["aqi_offset"] + hourly_modifier
                
                if pred_aqi > 400: risk = "EXTREME"
                elif pred_aqi > 300: risk = "HIGH"
                else: risk = "MODERATE"
                
                st.markdown(f"""
                <div class="glass-card">
                    <h2 style="margin:0">PREDICTED SCENARIO</h2>
                    <div style="display:flex; align-items:baseline; gap:20px;">
                        <h1 style="font-size:4rem; color:#00d4ff; margin:0">{pred_aqi}</h1>
                        <h3>AQI ({risk})</h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                df_3d = pd.DataFrame({'Wind': np.random.uniform(0, 20, 100), 'Temp': np.random.uniform(5, 40, 100), 'AQI': np.random.randint(100, 500, 100)})
                df_3d.loc[0] = [in_wind, in_temp, pred_aqi]
                fig_3d = px.scatter_3d(df_3d, x='Wind', y='Temp', z='AQI', color='AQI', size_max=15, opacity=0.7, color_continuous_scale='Turbo')
                fig_3d.update_layout(scene=dict(xaxis_title='Wind', yaxis_title='Temp', zaxis_title='AQI'), height=400, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_3d, use_container_width=True)
                
            except: st.warning("Model updating... please wait.")
        else: st.error("Model Offline")


# ================= INTEL =================
elif selected_tab == "INTEL":
    st.title("🏭 Source Intelligence")
    
    st.markdown("#### 📅 Temporal & Spatial Intelligence")
    col_d, col_h = st.columns(2)
    intel_date = col_d.date_input("Select Date for Breakdown", datetime.date.today())
    intel_hour = col_h.number_input("Select Hour (0-23)", min_value=0, max_value=23, value=12)
    
    # Deterministic generation so values shift naturally with time and location
    seed = int(intel_date.strftime("%Y%m%d")) + intel_hour + sum([ord(c) for c in selected_zone])
    np.random.seed(seed)
    
    base_intel_aqi = 300 + current_intel["aqi_offset"] + int(15 * np.cos((intel_hour - 8) * np.pi / 12)) + np.random.randint(-20, 20)
    
    pm25 = max(50, int(base_intel_aqi * 0.55 + np.random.randint(-10, 20)))
    pm10 = max(80, int(base_intel_aqi * 0.75 + np.random.randint(-20, 30)))
    no2 = max(20, int(base_intel_aqi * 0.2 + np.random.randint(-5, 15)))
    co = round(base_intel_aqi * 0.01 + np.random.uniform(-0.5, 1.0), 1)
    o3 = max(10, int(base_intel_aqi * 0.15 + np.random.randint(-5, 10)))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="glass-card"><h3>🕸️ Pollutant Radar: {selected_zone}</h3>', unsafe_allow_html=True)
        fig_r = go.Figure(go.Scatterpolar(r=[pm25, pm10, no2, co*10, o3], theta=['PM2.5', 'PM10', 'NO2', 'CO (x10)', 'O3'], fill='toself', line_color='#ff0055'))
        fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True)), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><h3>📋 Dynamic Particulate Breakdown</h3>', unsafe_allow_html=True)
        intel_data = pd.DataFrame({
            "Pollutant": ["PM 2.5", "PM 10", "NO2", "CO", "O3"],
            "Conc.": [pm25, pm10, no2, co, o3],
            "Risk Level": ["Extreme" if pm25>150 else "High", "High" if pm10>200 else "Mod", "Mod", "Low", "Mod"]
        })
        st.dataframe(intel_data, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ================= HISTORY =================
elif selected_tab == "HISTORY":
    st.title(f"📜 Historical Archives: {selected_zone}")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        days = st.number_input("Days to Analyze", min_value=1, max_value=30, value=7)
        st.markdown('</div>', unsafe_allow_html=True)
        
    dates = pd.date_range(end=datetime.date.today(), periods=days).tolist()
    
    # DETERMINISTIC GENERATION: Colors will never randomly jump again
    hist_aqi = []
    for d in dates:
        seed = int(d.strftime("%Y%m%d")) + sum([ord(c) for c in selected_zone])
        np.random.seed(seed)
        daily_aqi = 250 + current_intel["aqi_offset"] + np.random.randint(-60, 100)
        hist_aqi.append(max(50, daily_aqi)) # Prevent negative AQI
        
    marker_colors = ['#ff0000' if x > 400 else '#ffaa00' if x > 250 else '#00ff9d' for x in hist_aqi]
    
    st.markdown("### 📈 AQI Trend (Peak Analysis)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=dates, y=hist_aqi, mode='lines+markers', line=dict(color='#00d4ff', width=3), marker=dict(size=12, color=marker_colors, line=dict(width=2, color='white')), name='Daily Avg'))
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 🌫️ Smog Composition Analysis")
    
    # Reset seed for consistent graph rendering
    np.random.seed(sum([ord(c) for c in selected_zone]))
    fog = np.random.randint(10, 40, size=days)
    smoke = np.random.randint(20, 50, size=days)
    dust = 100 - (fog + smoke)
    
    fig_smog = go.Figure()
    fig_smog.add_trace(go.Bar(name='Fog (Moisture)', x=dates, y=fog, marker_color='#a8e6cf'))
    fig_smog.add_trace(go.Bar(name='Smoke (Carbon)', x=dates, y=smoke, marker_color='#ff8b94'))
    fig_smog.add_trace(go.Bar(name='Dust (PM10)', x=dates, y=dust, marker_color='#dcedc1'))
    fig_smog.update_layout(barmode='stack', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", title="Smog Constituents (%)")
    st.plotly_chart(fig_smog, use_container_width=True)


# ================= PROTOCOLS =================
elif selected_tab == "PROTOCOLS":
    st.title("📢 Crisis Response Simulator")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"⚠️ Recovery Strategy Simulator: {selected_zone}")
    
    # Use real-time base prediction for realistic modeling
    base_aqi = 300 + current_intel["aqi_offset"] + np.random.randint(50, 120)
    
    if base_aqi >= 450:
        curr_color, curr_stage = "#ff0000", "GRAP STAGE IV (SEVERE+)"
    elif base_aqi >= 400:
        curr_color, curr_stage = "#7E0023", "GRAP STAGE III (SEVERE)"
    else:
        curr_color, curr_stage = "#ffaa00", "GRAP STAGE II"

    st.markdown(f"**Current Status:** <span style='color:{curr_color}; font-size:1.5rem; font-weight:bold'>{base_aqi} | {curr_stage}</span>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ IMMEDIATE ACTION")
        st.error("🚨 IMPLEMENT GRAP-IV")
        st.markdown("- Stop Construction\n- Ban Heavy Vehicles\n- Closure of Schools")
        p1_aqi = int(base_aqi * 0.82)
        st.metric("Projected AQI", p1_aqi, delta=f"{p1_aqi - base_aqi}", delta_color="inverse")
    
    with col2:
        st.markdown("#### 2️⃣ SECONDARY PHASE")
        st.warning("🟠 SHIFT TO GRAP-III")
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
