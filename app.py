import streamlit as st
import pandas as pd
import joblib

# --- Page configuration ---
st.set_page_config(
    page_title="F1 Top-Finish Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Animated modern CSS ---
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(160deg, #0a0e27, #1a1f3a, #0f172a);
    font-family: 'Segoe UI', sans-serif;
    color: #ffffff;
}

/* Main title with glow */
.main-title {
    text-align: center;
    font-size: 4rem;
    font-weight: 900;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 15px #60a5fa, 0 0 30px #3b82f6;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    0% { text-shadow: 0 0 10px #60a5fa, 0 0 20px #3b82f6; }
    100% { text-shadow: 0 0 25px #60a5fa, 0 0 50px #3b82f6; }
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 1.4rem;
    color: #cbd5e1;
    margin-bottom: 2rem;
}

/* Input cards */
.input-card {
    background: rgba(15,23,42,0.85);
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 2px solid #3b82f6;
    box-shadow: 0 0 15px rgba(59,130,246,0.4);
    transition: all 0.3s ease;
    animation: pulse 3s infinite alternate;
}
.input-card:hover {
    box-shadow: 0 0 25px #60a5fa, 0 0 50px rgba(59,130,246,0.6);
}

@keyframes pulse {
    0% { box-shadow: 0 0 15px rgba(59,130,246,0.4); }
    100% { box-shadow: 0 0 25px #60a5fa, 0 0 50px rgba(59,130,246,0.6); }
}

/* Number Inputs */
.stNumberInput > div > div > input {
    border-radius: 10px;
    background: rgba(15,23,42,0.8);
    border: 2px solid #3b82f6;
    padding: 0.6rem;
    color: #ffffff;
    font-weight: 500;
    box-shadow: 0 0 10px rgba(59,130,246,0.6);
    transition: all 0.2s ease-in-out;
}
.stNumberInput > div > div > input:focus {
    box-shadow: 0 0 15px #60a5fa;
    outline: none;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 700;
    padding: 0.7rem 3rem;
    border-radius: 40px;
    border: none;
    box-shadow: 0 6px 20px rgba(96,165,250,0.6);
    animation: glow-button 2s infinite alternate;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(96,165,250,0.8);
}
@keyframes glow-button {
    0% { box-shadow: 0 6px 20px rgba(96,165,250,0.6); }
    100% { box-shadow: 0 6px 30px rgba(96,165,250,1); }
}

/* Prediction cards */
.prediction-card {
    border-radius: 25px;
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
    font-weight: 700;
    color: #ffffff;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    animation: glow-card 3s infinite alternate;
}
.prediction-success {
    background: #10b981;
}
.prediction-warning {
    background: #ef4444;
}
@keyframes glow-card {
    0% { box-shadow: 0 0 20px rgba(255,255,255,0.2); }
    100% { box-shadow: 0 0 35px rgba(255,255,255,0.4); }
}

/* Info box */
.info-box {
    background: rgba(15,23,42,0.7);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 5px solid #60a5fa;
    margin: 1rem 0;
    font-weight: 500;
    color: #e2e8f0;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 12px;
    padding: 1rem;
    background: rgba(15,23,42,0.7);
    color: #ffffff;
}

/* Section headers */
h3 {
    color: #60a5fa !important;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('model/best_pipeline.joblib')
pipeline = load_model()

# --- Header ---
st.markdown('<h1 class="main-title">🏎️ F1 TOP-FINISH PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict if a driver will finish in top positions using AI</p>', unsafe_allow_html=True)

# --- Input Layout with proper column mapping ---
display_to_model = {
    "Points": "points",
    "Laps Completed": "laps",
    "Grid Position": "grid",
    "Driver Average Points": "driver_avg_points",
    "Driver Median Grid Position": "driver_median_grid",
    "Constructor Average Points": "constructor_avg_points",
    "Constructor Median Grid Position": "constructor_median_grid",
    "Constructor ID (encoded)": "constructorRef_enc",
    "Circuit ID (encoded)": "circuitRef_enc"
}

sections = {
    "🏁 Performance Metrics": ["Points","Laps Completed","Grid Position"],
    "👤 Driver Statistics": ["Driver Average Points","Driver Median Grid Position"],
    "🏭 Constructor Data": ["Constructor Average Points","Constructor Median Grid Position","Constructor ID (encoded)"],
    "🏟️ Circuit Information": ["Circuit ID (encoded)"]
}

input_data = {}

col1, col2 = st.columns(2, gap="large")
for i, (section, fields) in enumerate(sections.items()):
    target_col = col1 if i%2==0 else col2
    with target_col:
        st.markdown(f'<div class="input-card"><h3>{section}</h3>', unsafe_allow_html=True)
        for field in fields:
            val = st.number_input(field, value=0.0)
            input_data[display_to_model[field]] = val  # store using **model column names**
        st.markdown('</div>', unsafe_allow_html=True)

# --- Display input summary ---
st.markdown("### 📋 Current Input Summary")
input_df = pd.DataFrame(input_data, index=[0])
st.dataframe(input_df, use_container_width=True)
st.markdown('<div class="info-box">ℹ️ The model evaluates driver, constructor, and circuit data to predict top finishes.</div>', unsafe_allow_html=True)

# --- Predict Button ---
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_button = st.button("🚀 PREDICT RESULT", use_container_width=True)

# --- Prediction ---
if predict_button:
    with st.spinner('🏎️ Calculating...'):
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1] if hasattr(pipeline, "predict_proba") else None
    
    st.markdown("---")
    st.markdown("### 🏆 Prediction Results")
    result_col1, result_col2 = st.columns([1,1])
    
    with result_col1:
        if prediction==1:
            st.markdown('<div class="prediction-card prediction-success">✅ TOP FINISH PREDICTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-card prediction-warning">❌ NOT A TOP FINISH</div>', unsafe_allow_html=True)
    
    with result_col2:
        if prob is not None:
            st.markdown("#### 📊 Confidence Level")
            st.progress(int(prob*100))
            st.markdown(f"<p style='text-align:center; color:#60a5fa; font-size:2rem;'>{prob*100:.1f}%</p>", unsafe_allow_html=True)
            if prob>=0.7:
                st.info("🔥 High confidence - Strong indicators for a top finish!")
            elif prob>=0.5:
                st.info("⚡ Moderate confidence - Good chance of top finish.")
            else:
                st.info("📉 Lower confidence - Challenging conditions for top finish.")

st.markdown("---")

