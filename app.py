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

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="AirScribe: Nexus Command",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. THE "CYBERPUNK GLASS" CSS ---
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
    
    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 4px; }
    
    /* INPUTS */
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stDateInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* NAV PILLS (FIXED: NOWRAP) */
    div[data-testid="stRadio"] > div {
        display: flex;
        justify-content: center;
        gap: 10px;
        background: rgba(255,255,255,0.05);
        padding: 5px;
        border-radius: 50px;
        overflow-x: auto; /* Allow scroll on very small screens */
    }
    div[data-testid="stRadio"] label {
        flex: 1;
        text-align: center;
        padding: 10px 20px;
        border-radius: 40px;
        cursor: pointer;
        transition: 0.3s;
        white-space: nowrap; /* <--- THE FIX */
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
anim_wind = load_lottieurl("https://lottie.host/0a701967-7359-4670-87a3-5c79893962b9/1x8k7l0i5D.json")

# --- 4. LOAD BRAIN ---
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'pollution_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- 5. HEADER & NAVIGATION ---
c_logo, c_nav = st.columns([1, 4])
with c_logo:
    st.title("AIRSCRIBE")
    st.caption("NEXUS COMMAND v4.0")
with c_nav:
    selected_tab = st.radio("Navigation", ["DASHBOARD", "FORECAST", "INTEL", "HISTORY", "PROTOCOLS"], 
        horizontal=True, label_visibility="collapsed")

st.divider()

# --- 6. PAGE LOGIC ---

# ================= DASHBOARD =================
if selected_tab == "DASHBOARD":
    
    # --- METADATA HEADER ---
    with st.expander("📍 STATION METADATA", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown("**Ward:** Anand Vihar (Zone 4)")
        m2.markdown("**Station ID:** DPCC-AV-042")
        m3.markdown("**Nearby Schools:** 14")
        m4.markdown("**Pop. Affected:** ~2.4 Lakhs")

    # --- MAIN KPI ROW ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### ⚡ Real-Time Atmospheric Surveillance")
        
        # Live Data Simulation
        now = datetime.datetime.now()
        if model:
            pred = model.predict([[now.hour, now.month, now.weekday()]])[0]
            live_aqi = int(pred)
        else:
            live_aqi = 345 # Demo default
            
        # Color Logic
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

        # --- CONTRIBUTORS ---
        d1, d2 = st.columns([1, 2])
        with d1:
            # Donut Chart
            st.markdown("##### 🏭 Contributors")
            contrib_data = {'Vehicles': 40, 'Dust': 20, 'Industries': 25, 'Stubble': 15}
            fig_donut = px.pie(names=contrib_data.keys(), values=contrib_data.values(), hole=0.7, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", height=150)
            st.plotly_chart(fig_donut, use_container_width=True)
        with d2:
            # Weather/Fog
            st.markdown("##### 🌤️ Live Conditions")
            w1, w2, w3 = st.columns(3)
            w1.markdown(f'<div class="glass-card" style="padding:10px"><h4>💨 Wind</h4><p>NW 12km/h</p></div>', unsafe_allow_html=True)
            w2.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌫️ Fog</h4><p>Dense (<50m)</p></div>', unsafe_allow_html=True)
            w3.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌡️ Temp</h4><p>14°C</p></div>', unsafe_allow_html=True)

    with c2:
        if anim_robot:
            st_lottie(anim_robot, height=250, key="robot")
            
        # --- 24H TREND ---
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

    # --- MAP SECTION (NEW) ---
    st.markdown("### 🗺️ Live Sensor Network (NCR Grid)")
    
    # Generate Mock Map Data
    map_data = pd.DataFrame({
        'lat': [28.6139, 28.5355, 28.7041, 28.4595, 28.6692, 28.5273, 28.6129, 28.5921, 28.7382, 28.5300],
        'lon': [77.2090, 77.3910, 77.1025, 77.0266, 77.2285, 77.1388, 77.2295, 77.0460, 77.0822, 77.3000],
        'Location': ['Connaught Place', 'Noida Sec 62', 'Pitampura', 'Gurugram Cyber City', 'Kashmere Gate', 'Vasant Kunj', 'India Gate', 'Dwarka Sec 21', 'Rohini', 'Okhla'],
        'AQI': np.random.randint(200, 500, 10)
    })
    
    # Create Mapbox
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", hover_name="Location", 
                        hover_data=["AQI"], color="AQI", size="AQI",
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=9)
    
    fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=400)
    st.plotly_chart(fig_map, use_container_width=True)


# ================= FORECAST =================
elif selected_tab == "FORECAST":
    st.title("🔮 Predictive Neural Net")
    
    # --- SUMMARY CARD ---
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="margin:0">PREDICTED PEAK AQI</h3>
                <h1 style="font-size:3.5rem; color:#ff0055">378</h1>
                <p style="color:#ff0055">Risk Level: SEVERE</p>
            </div>
            <div style="text-align:right">
                <p><strong>Confidence:</strong> 82%</p>
                <p><strong>Primary Driver:</strong> Vehicular Emissions (42%)</p>
                <p><strong>Wind Factor:</strong> Low Dispersion (18%)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 3D PLOT ---
    st.markdown("### 🧊 Multi-Dimensional Risk Model")
    df_3d = pd.DataFrame({
        'Hour': np.random.randint(0, 24, 300),
        'Traffic Load': np.random.randint(10, 100, 300),
        'AQI': np.random.randint(100, 500, 300)
    })
    fig_3d = px.scatter_3d(df_3d, x='Hour', y='Traffic Load', z='AQI', color='AQI', color_continuous_scale='Turbo', size_max=15, opacity=0.8)
    fig_3d.update_layout(scene=dict(xaxis=dict(backgroundcolor="rgba(0,0,0,0)"), yaxis=dict(backgroundcolor="rgba(0,0,0,0)"), zaxis=dict(backgroundcolor="rgba(0,0,0,0)")), paper_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_3d, use_container_width=True)


# ================= INTEL =================
elif selected_tab == "INTEL":
    st.title("🏭 Deep Analytics")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><h3>🕸️ Pollutant Radar</h3>', unsafe_allow_html=True)
        cats = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3']
        vals = [180, 250, 90, 40, 60]
        fig_r = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', line_color='#00d4ff'))
        fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True)), paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20))
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card"><h3>📋 Particulate Breakdown</h3>', unsafe_allow_html=True)
        intel_data = pd.DataFrame({
            "Pollutant": ["PM 2.5", "PM 10", "NO2", "CO", "O3"],
            "Concentration (µg/m³)": [180, 250, 90, 4.2, 60],
            "Source": ["Combustion", "Dust", "Traffic", "Fuel", "Reaction"],
            "Risk": ["High", "High", "Mod", "Low", "Mod"]
        })
        st.dataframe(intel_data, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ================= HISTORY =================
elif selected_tab == "HISTORY":
    st.title("📜 Historical Archives")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c_hist1, c_hist2 = st.columns([1, 3])
        with c_hist1:
            days = st.number_input("Days to Analyze", min_value=1, max_value=30, value=7)
        with c_hist2:
            st.info(f"Displaying AI-reconstructed data for the last {days} days.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    dates = pd.date_range(end=datetime.date.today(), periods=days).tolist()
    hist_aqi = np.random.randint(200, 450, size=days)
    
    colors = ['#ff0000' if x > 400 else '#00ff9d' if x < 250 else '#ffaa00' for x in hist_aqi]
    
    fig_hist = go.Figure(data=[go.Bar(x=dates, y=hist_aqi, marker_color=colors)])
    fig_hist.update_layout(title="Daily Average AQI", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)


# ================= PROTOCOLS =================
elif selected_tab == "PROTOCOLS":
    st.title("📢 GRAP Response Simulator")
    
    c_main, c_side = st.columns([2, 1])
    
    with c_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Policy Impact Simulator")
        
        base_aqi = 450
        st.markdown(f"**Current Projected AQI:** <span style='color:red; font-size:1.5rem'>{base_aqi}</span>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: p1 = st.checkbox("Ban Construction")
        with c2: p2 = st.checkbox("Odd-Even Traffic")
        with c3: p3 = st.checkbox("Close Schools")
        
        reduction = 0
        if p1: reduction += 40
        if p2: reduction += 60
        if p3: reduction += 15
        
        final_aqi = base_aqi - reduction
        
        st.markdown("---")
        st.markdown(f"### 📉 New Predicted AQI: <span style='color:#00ff9d; font-size:2.5rem'>{final_aqi}</span>", unsafe_allow_html=True)
        st.progress(max(0, min(1.0, final_aqi/500)))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📜 GRAP Status")
        
        if final_aqi > 450:
            st.error("🚨 GRAP STAGE IV (Severe+)")
            st.markdown("- Stop Truck Entry\n- Public Projects Halted")
        elif final_aqi > 400:
            st.error("🔴 GRAP STAGE III (Severe)")
            st.markdown("- BS-III/IV Car Ban\n- No Demolition")
        elif final_aqi > 300:
            st.warning("🟠 GRAP STAGE II (Very Poor)")
            st.markdown("- Diesel Gen Ban\n- Bus Frequency Up")
        else:
            st.success("🟢 GRAP STAGE I (Poor)")
            st.markdown("- Dust Control\n- Mechanized Sweeping")
        st.markdown('</div>', unsafe_allow_html=True)
