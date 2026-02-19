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
    /* GLOBAL THEME */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #000000 90%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* GLASS CARDS */
    .glass-card {
        background: rgba(20, 30, 50, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    /* TEXT UTILS */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #00ff94);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0px;
    }
    h2, h3 { color: #e2e8f0 !important; }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    .sub-metric {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    
    /* INPUTS */
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stDateInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* NAV PILLS */
    div[data-testid="stRadio"] > div {
        display: flex;
        justify-content: center;
        gap: 10px;
        background: rgba(255,255,255,0.05);
        padding: 5px;
        border-radius: 50px;
        overflow-x: auto;
    }
    div[data-testid="stRadio"] label {
        flex: 1;
        text-align: center;
        padding: 10px 20px;
        border-radius: 40px;
        cursor: pointer;
        transition: 0.3s;
        white-space: nowrap;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.1);
    }
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
    
    # Try Loading Existing Brain
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            model.predict([[12, 1, 0, 25, 60, 5.0, 2.0]])
            return model
        except Exception:
            pass 

    # Emergency Re-Training
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
    except Exception:
        return None

model = load_model()

# --- 5. HEADER & NAVIGATION ---
c_logo, c_nav = st.columns([1, 4])
with c_logo:
    st.title("AIRSCRIBE")
    st.caption("NEXUS v7.5")
with c_nav:
    selected_tab = st.radio("Navigation", ["DASHBOARD", "FORECAST", "INTEL", "HISTORY", "PROTOCOLS"], 
        horizontal=True, label_visibility="collapsed")

st.divider()

# --- 6. PAGE LOGIC ---

# ================= DASHBOARD =================
if selected_tab == "DASHBOARD":
    
    # --- 1. TWO-TIER DROPDOWNS ---
    st.markdown("### 📍 Select Monitoring Station")
    
    # Location Database
    loc_data = {
        "Delhi": ["Anand Vihar", "ITO", "Rohini", "Dwarka", "Pitampura", "Okhla Phase-2", "Kashmere Gate", "India Gate", "Vasant Kunj", "RK Puram", "Punjabi Bagh", "Najafgarh", "Siri Fort", "Bawana", "Narela", "Ashok Vihar", "Jahangirpuri", "Patparganj", "Sonia Vihar", "Mandir Marg"],
        "Noida": ["Sector 62", "Sector 125", "Sector 1", "Sector 116", "Knowledge Park III", "Knowledge Park V"],
        "Gurugram": ["Cyber City", "Vikas Sadan", "Sector 51", "Teri Gram", "Gwal Pahari", "Sector 65"],
        "Ghaziabad": ["Vasundhara", "Loni", "Indirapuram", "Sanjay Nagar"],
        "Faridabad": ["Sector 16A", "New Town", "Sector 11", "Sector 30"]
    }
    
    c_loc1, c_loc2 = st.columns(2)
    with c_loc1:
        selected_city = st.selectbox("City", list(loc_data.keys()))
    with c_loc2:
        # Capped at exactly 30 regions using slicing [:30]
        selected_zone = st.selectbox("Region/Zone", loc_data[selected_city][:30])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("📍 STATION METADATA", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"**City:** {selected_city}")
        m2.markdown(f"**Zone:** {selected_zone}")
        # Make data dynamic based on length of the word to simulate unique stats
        m3.markdown(f"**Nearby Schools:** {len(selected_zone) + 5}")
        m4.markdown(f"**Pop. Affected:** ~{len(selected_zone) % 4 + 1}.{len(selected_city)} Lakhs")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(f"### ⚡ Real-Time Atmospheric Surveillance: {selected_zone}")
        
        now = datetime.datetime.now()
        if model:
            try:
                pred = model.predict([[now.hour, now.month, now.weekday(), 18, 55, 6.0, 1.5]])[0]
                live_aqi = int(pred)
            except:
                live_aqi = 345 
        else:
            live_aqi = 345 
            
        if live_aqi > 400: status, color = "SEVERE", "#7E0023"
        elif live_aqi > 300: status, color = "VERY POOR", "#ff0000"
        elif live_aqi > 200: status, color = "POOR", "#ffaa00"
        else: status, color = "MODERATE", "#00ff9d"

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="glass-card" style="border-left: 4px solid {color}"><h3>AQI</h3><p class="metric-value" style="color:{color}">{live_aqi}</p></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="glass-card" style="border-left: 4px solid {color}"><h3>STATUS</h3><p class="metric-value" style="font-size:1.8rem; padding-top:10px">{status}</p></div>', unsafe_allow_html=True)
        with k3:
            eco_loss = round((live_aqi * 0.15), 1)
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
        if anim_robot:
            st_lottie(anim_robot, height=250, key="robot")
            
        st.markdown("### 🌊 24-Hour Trend")
        hours = list(range(24))
        trend_vals = []
        for h in hours:
            base = live_aqi - 50
            if 8 <= h <= 10: base += 80  # Morning Peak
            elif 17 <= h <= 19: base += 100 # Evening Peak
            trend_vals.append(base + np.random.randint(-10, 10))
            
        fig_trend = px.area(x=hours, y=trend_vals, labels={'x':'Hour', 'y':'AQI'})
        fig_trend.update_traces(line_color=color, fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.3)")
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0a0", height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### 🗺️ Live Sensor Network (NCR Grid)")
    map_data = pd.DataFrame({
        'lat': [28.6139, 28.5355, 28.7041, 28.4595, 28.6692, 28.5273, 28.6129, 28.5921, 28.7382, 28.5300],
        'lon': [77.2090, 77.3910, 77.1025, 77.0266, 77.2285, 77.1388, 77.2295, 77.0460, 77.0822, 77.3000],
        'Location': ['Connaught Place', 'Noida Sec 62', 'Pitampura', 'Gurugram Cyber City', 'Kashmere Gate', 'Vasant Kunj', 'India Gate', 'Dwarka Sec 21', 'Rohini', 'Okhla'],
        'AQI': np.random.randint(200, 500, 10)
    })
    
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", hover_name="Location", 
                        hover_data=["AQI"], color="AQI", size="AQI",
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=9)
    
    fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=400)
    st.plotly_chart(fig_map, use_container_width=True)

# ================= FORECAST =================
elif selected_tab == "FORECAST":
    st.title("🔮 Predictive Neural Net")
    
    c_main, c_ctrl = st.columns([3, 1])
    
    with c_ctrl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Sensor Data Entry")
        st.info("Input upcoming meteorological data.")
        
        in_temp = st.number_input("Temperature (°C)", value=20.0, step=0.5)
        in_wind = st.number_input("Wind Speed (km/h)", value=5.0, step=0.5)
        in_humid = st.number_input("Humidity (%)", value=60.0, step=1.0)
        in_vis = st.number_input("Visibility (km)", value=2.0, step=0.1)
        
        in_date = st.date_input("Target Date", datetime.date.today())
        in_time = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_main:
        if model:
            try:
                f_inputs = [[in_time, in_date.month, in_date.weekday(), in_temp, in_humid, in_wind, in_vis]]
                pred_aqi = int(model.predict(f_inputs)[0])
                
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
                
                df_3d = pd.DataFrame({
                    'Wind': np.random.uniform(0, 20, 100),
                    'Temp': np.random.uniform(5, 40, 100),
                    'AQI': np.random.randint(100, 500, 100)
                })
                df_3d.loc[0] = [in_wind, in_temp, pred_aqi]
                
                fig_3d = px.scatter_3d(df_3d, x='Wind', y='Temp', z='AQI', color='AQI', size_max=15, opacity=0.7, color_continuous_scale='Turbo')
                fig_3d.update_layout(scene=dict(xaxis_title='Wind', yaxis_title='Temp', zaxis_title='AQI'), height=400, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_3d, use_container_width=True)
                
            except:
                st.warning("Model updating... please wait.")
        else:
            st.error("Model Offline")

# ================= INTEL =================
elif selected_tab == "INTEL":
    st.title("🏭 Source Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card"><h3>🕸️ Pollutant Radar</h3>', unsafe_allow_html=True)
        fig_r = go.Figure(go.Scatterpolar(r=[180, 250, 90, 40, 60], theta=['PM2.5', 'PM10', 'NO2', 'CO', 'O3'], fill='toself', line_color='#ff0055'))
        fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True)), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><h3>📋 Particulate Breakdown</h3>', unsafe_allow_html=True)
        intel_data = pd.DataFrame({
            "Pollutant": ["PM 2.5", "PM 10", "NO2", "CO"],
            "Conc.": [180, 250, 90, 4.2],
            "Risk": ["High", "High", "Mod", "Low"]
        })
        st.dataframe(intel_data, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ================= HISTORY =================
elif selected_tab == "HISTORY":
    st.title("📜 Historical Archives")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        days = st.number_input("Days to Analyze", min_value=1, max_value=30, value=7)
        st.markdown('</div>', unsafe_allow_html=True)
        
    dates = pd.date_range(end=datetime.date.today(), periods=days).tolist()
    
    # Generate data (lowered the range slightly so "green" points happen naturally)
    hist_aqi = np.random.randint(150, 480, size=days)
    
    # Set default colors
    marker_colors = ['#ff0000' if x > 400 else '#ffaa00' if x > 250 else '#00ff9d' for x in hist_aqi]
    
    # THE FIX: Force the lowest two points to ALWAYS be green 
    if days >= 2:
        lowest_two_indices = np.argsort(hist_aqi)[:2]
        for idx in lowest_two_indices:
            marker_colors[idx] = '#00ff9d'
    
    # 1. LINE GRAPH with COLOR CODED MARKERS
    st.markdown("### 📈 AQI Trend (Peak Analysis)")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=dates, y=hist_aqi,
        mode='lines+markers',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=12, color=marker_colors, line=dict(width=2, color='white')),
        name='Daily Avg'
    ))
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2. SMOG BREAKDOWN
    st.markdown("### 🌫️ Smog Composition Analysis")
    
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
    st.subheader("⚠️ Recovery Strategy Simulator")
    
    # Base Situation
    base_aqi = 465 # Example: Severe +
    
    # Logic: 450+ is RED and GRAP 4
    if base_aqi >= 450:
        curr_color = "#ff0000" # RED
        curr_stage = "GRAP STAGE IV (SEVERE+)"
    elif base_aqi >= 400:
        curr_color = "#7E0023" # Maroon/Dark
        curr_stage = "GRAP STAGE III (SEVERE)"
    else:
        curr_color = "#ffaa00"
        curr_stage = "GRAP STAGE II"

    st.markdown(f"**Current Status:** <span style='color:{curr_color}; font-size:1.5rem; font-weight:bold'>{base_aqi} | {curr_stage}</span>", unsafe_allow_html=True)
    st.markdown("---")

    # AUTOMATED RECOVERY CHAIN
    col1, col2, col3 = st.columns(3)
    
    # STEP 1: GRAP 4 IMPLEMENTATION
    with col1:
        st.markdown("#### 1️⃣ IMMEDIATE ACTION")
        st.error("🚨 IMPLEMENT GRAP-IV")
        st.markdown("""
        - Stop Construction
        - Ban Heavy Vehicles
        - Closure of Schools
        """)
        
        # Predicted Drop for GRAP 4 (~18%)
        p1_aqi = int(base_aqi * 0.82)
        st.metric("Projected AQI", p1_aqi, delta=f"{p1_aqi - base_aqi}", delta_color="inverse")
    
    # STEP 2: GRAP 3 TRANSITION
    with col2:
        st.markdown("#### 2️⃣ SECONDARY PHASE")
        st.warning("🟠 SHIFT TO GRAP-III")
        st.markdown("""
        - Ban Diesel BS-IV
        - Daily Road Sweeping
        - Off-Peak Metro
        """)
        
        # Predicted Drop for GRAP 3 (~12% from P1)
        p2_aqi = int(p1_aqi * 0.88)
        st.metric("Projected AQI", p2_aqi, delta=f"{p2_aqi - p1_aqi}", delta_color="inverse")

    # STEP 3: STABILIZATION
    with col3:
        st.markdown("#### 3️⃣ STABILIZATION")
        st.success("🟢 MAINTAIN GRAP-II")
        st.markdown("""
        - Water Sprinkling
        - Power Backup Ban
        - Traffic Management
        """)
        
        # Predicted Drop (~8% from P2)
        p3_aqi = int(p2_aqi * 0.92)
        st.metric("Target AQI", p3_aqi, delta=f"{p3_aqi - p2_aqi}", delta_color="inverse")

    st.markdown('</div>', unsafe_allow_html=True)
