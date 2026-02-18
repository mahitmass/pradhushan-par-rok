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

# --- NEW IMPORTS FOR SELF-HEALING ---
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
    .glass-card { background: rgba(20, 30, 50, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 20px; margin-bottom: 20px; }
    h1 { background: linear-gradient(90deg, #00d4ff, #00ff94); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .metric-value { font-size: 2.5rem; font-weight: 700; margin: 0; text-shadow: 0 0 15px rgba(0, 212, 255, 0.4); }
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stDateInput input { background-color: rgba(255, 255, 255, 0.05) !important; color: white !important; border: 1px solid rgba(255, 255, 255, 0.1); }
    div[data-testid="stRadio"] > div { display: flex; justify-content: center; gap: 10px; background: rgba(255,255,255,0.05); padding: 5px; border-radius: 50px; overflow-x: auto; }
    div[data-testid="stRadio"] label { flex: 1; text-align: center; padding: 10px 20px; border-radius: 40px; cursor: pointer; white-space: nowrap; }
    div[data-testid="stRadio"] label:hover { background: rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSETS ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

anim_robot = load_lottieurl("https://lottie.host/7e04085b-5136-4074-8461-766723223126/6sX6wH5k2a.json") 

# --- 4. SMART LOAD (SELF-HEALING) ---
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'pollution_model.pkl')
    data_path = os.path.join('data', 'delhi_ncr_aqi_dataset.csv')
    
    # PHASE 1: Try Loading Existing Brain
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Test: Check if it accepts 7 features
            test_input = [[12, 1, 0, 25, 60, 5.0, 2.0]]
            model.predict(test_input)
            return model
        except Exception:
            pass # Load failed or mismatch? Ignore and Re-train!

    # PHASE 2: Emergency Re-Training (The Fix)
    try:
        if not os.path.exists(data_path): return None
        
        df = pd.read_csv(data_path)
        # Process Data
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        features = ['hour', 'month', 'day_of_week', 'temperature', 'humidity', 'wind_speed', 'visibility']
        
        # Clean
        df = df.dropna(subset=['aqi'])
        X = df[features].fillna(df[features].mean())
        y = df['aqi']
        
        # Train Light Model (Fast)
        new_model = RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
        new_model.fit(X, y)
        return new_model
    except Exception as e:
        return None

model = load_model()

# --- 5. NAVIGATION ---
c_logo, c_nav = st.columns([1, 4])
with c_logo:
    st.title("AIRSCRIBE")
    st.caption("NEXUS v6.5 (Auto-Heal)")
with c_nav:
    selected_tab = st.radio("Navigation", ["DASHBOARD", "FORECAST", "INTEL", "HISTORY", "PROTOCOLS"], 
        horizontal=True, label_visibility="collapsed")

st.divider()

# --- 6. PAGE LOGIC ---

# ================= DASHBOARD =================
if selected_tab == "DASHBOARD":
    
    with st.expander("📍 STATION METADATA", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown("**Ward:** Anand Vihar (Zone 4)")
        m2.markdown("**Station ID:** DPCC-AV-042")
        m3.markdown("**Nearby Schools:** 14")
        m4.markdown("**Pop. Affected:** ~2.4 Lakhs")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### ⚡ Real-Time Atmospheric Surveillance")
        
        now = datetime.datetime.now()
        if model:
            # Predict with default weather if live not available
            pred = model.predict([[now.hour, now.month, now.weekday(), 18, 55, 6.0, 1.5]])[0]
            live_aqi = int(pred)
        else:
            live_aqi = 345 # Fallback
            
        if live_aqi > 400: status, color = "SEVERE", "#7E0023"
        elif live_aqi > 300: status, color = "VERY POOR", "#ff0000"
        elif live_aqi > 200: status, color = "POOR", "#ffaa00"
        else: status, color = "MODERATE", "#00ff9d"

        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f'<div class="glass-card" style="border-left: 4px solid {color}"><h3>AQI</h3><p class="metric-value" style="color:{color}">{live_aqi}</p></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="glass-card" style="border-left: 4px solid {color}"><h3>STATUS</h3><p class="metric-value" style="font-size:1.8rem; padding-top:10px">{status}</p></div>', unsafe_allow_html=True)
        with k3:
            eco_loss = round((live_aqi * 0.15), 1)
            st.markdown(f'<div class="glass-card" style="border-left: 4px solid #00d4ff"><h3>LOSS</h3><p class="metric-value" style="color:#00d4ff">₹{eco_loss} Cr</p></div>', unsafe_allow_html=True)

        d1, d2 = st.columns([1, 2])
        with d1:
            st.markdown("##### 🏭 Contributors")
            fig_donut = px.pie(names=['Traffic', 'Dust', 'Ind.', 'Stubble'], values=[40, 20, 25, 15], hole=0.7, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", height=150)
            st.plotly_chart(fig_donut, use_container_width=True)
        with d2:
            st.markdown("##### 🌤️ Live Conditions")
            w1, w2, w3 = st.columns(3)
            w1.markdown(f'<div class="glass-card" style="padding:10px"><h4>💨 Wind</h4><p>12 km/h</p></div>', unsafe_allow_html=True)
            w2.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌫️ Fog</h4><p>Dense</p></div>', unsafe_allow_html=True)
            w3.markdown(f'<div class="glass-card" style="padding:10px"><h4>🌡️ Temp</h4><p>14°C</p></div>', unsafe_allow_html=True)

    with c2:
        if anim_robot: st_lottie(anim_robot, height=250, key="robot")
            
        st.markdown("### 🌊 24-Hour Trend")
        hours = list(range(24))
        trend_vals = [live_aqi + np.sin(h/4)*50 + np.random.randint(-10,10) for h in hours]
        fig_trend = px.area(x=hours, y=trend_vals)
        fig_trend.update_traces(line_color=color, fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.3)")
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0a0", height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### 🗺️ Live Sensor Network")
    map_data = pd.DataFrame({'lat': [28.6139, 28.5355, 28.7041], 'lon': [77.2090, 77.3910, 77.1025], 'AQI': [live_aqi, live_aqi-20, live_aqi+30]})
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", size="AQI", color="AQI", color_continuous_scale="reds", zoom=9, height=400)
    fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)


# ================= FORECAST =================
elif selected_tab == "FORECAST":
    st.title("🔮 Predictive Neural Net")
    
    c_main, c_ctrl = st.columns([3, 1])
    with c_ctrl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎛️ Simulation Controls")
        in_temp = st.slider("Temperature (°C)", 0, 45, 20)
        in_wind = st.slider("Wind Speed (km/h)", 0.0, 30.0, 5.0)
        in_humid = st.slider("Humidity (%)", 10, 100, 60)
        in_vis = st.slider("Visibility (km)", 0.0, 5.0, 2.0)
        in_date = st.date_input("Target Date", datetime.date.today())
        in_time = st.slider("Hour of Day", 0, 23, 12)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_main:
        if model:
            f_inputs = [[in_time, in_date.month, in_date.weekday(), in_temp, in_humid, in_wind, in_vis]]
            pred_aqi = int(model.predict(f_inputs)[0])
            risk = "EXTREME" if pred_aqi > 400 else "HIGH" if pred_aqi > 300 else "MODERATE"
            
            st.markdown(f"""<div class="glass-card"><h1 style="color:#00d4ff">{pred_aqi}</h1><h3>AQI ({risk})</h3></div>""", unsafe_allow_html=True)
            
            df_3d = pd.DataFrame({'Wind': np.random.uniform(0, 20, 100), 'Temp': np.random.uniform(5, 40, 100), 'AQI': np.random.randint(100, 500, 100)})
            df_3d.loc[0] = [in_wind, in_temp, pred_aqi]
            fig_3d = px.scatter_3d(df_3d, x='Wind', y='Temp', z='AQI', color='AQI', size_max=15, opacity=0.7, color_continuous_scale='Turbo')
            fig_3d.update_layout(scene=dict(xaxis_title='Wind', yaxis_title='Temp', zaxis_title='AQI'), height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("Training Model... Refresh in 10 seconds.")


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
        st.dataframe(pd.DataFrame({"Pollutant": ["PM 2.5", "PM 10", "NO2", "CO"], "Conc.": [180, 250, 90, 4.2], "Risk": ["High", "High", "Mod", "Low"]}), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ================= HISTORY =================
elif selected_tab == "HISTORY":
    st.title("📜 Historical Archives")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        days = st.number_input("Days to Analyze", min_value=1, max_value=30, value=7)
        st.markdown('</div>', unsafe_allow_html=True)
        
    dates = pd.date_range(end=datetime.date.today(), periods=days).tolist()
    hist_aqi = np.random.randint(200, 480, size=days)
    marker_colors = ['#ff0000' if x > 400 else '#ffaa00' if x > 200 else '#00ff9d' for x in hist_aqi]
    
    st.markdown("### 📈 AQI Trend (Peak Analysis)")
    fig_hist = go.Figure(go.Scatter(x=dates, y=hist_aqi, mode='lines+markers', line=dict(color='#00d4ff', width=3), marker=dict(size=12, color=marker_colors)))
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 🌫️ Smog Composition Analysis")
    fig_smog = go.Figure()
    fig_smog.add_trace(go.Bar(name='Fog', x=dates, y=np.random.randint(10, 40, size=days), marker_color='#a8e6cf'))
    fig_smog.add_trace(go.Bar(name='Smoke', x=dates, y=np.random.randint(20, 50, size=days), marker_color='#ff8b94'))
    fig_smog.update_layout(barmode='stack', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_smog, use_container_width=True)


# ================= PROTOCOLS =================
elif selected_tab == "PROTOCOLS":
    st.title("📢 Crisis Response Simulator")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    base_aqi = 465
    curr_color = "#ff0000" if base_aqi >= 450 else "#ffaa00"
    st.markdown(f"**Status:** <span style='color:{curr_color}; font-size:1.5rem; font-weight:bold'>{base_aqi} | GRAP IV</span>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("🚨 GRAP-IV ACTIVE")
        st.markdown("- Stop Construction\n- Close Schools")
        st.metric("Proj. AQI", int(base_aqi * 0.82), delta=f"{int(base_aqi * 0.82) - base_aqi}")
    with col2:
        st.warning("🟠 PHASE II (GRAP-III)")
        st.markdown("- Ban BS-IV Diesel")
        st.metric("Proj. AQI", int(base_aqi * 0.82 * 0.88), delta=f"{int(base_aqi * 0.82 * 0.88) - int(base_aqi * 0.82)}")
    with col3:
        st.success("🟢 PHASE III (STABLE)")
        st.metric("Target AQI", int(base_aqi * 0.82 * 0.88 * 0.92))
    st.markdown('</div>', unsafe_allow_html=True)
