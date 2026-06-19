import streamlit as st
import pickle
import numpy as np

# Load our saved ML Brain
with open('delivery_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page Configuration
st.set_page_config(page_title="SmartBites AI", page_icon="🍔", layout="centered")

st.title("🍔 SmartBites AI")
st.subheader("Predictive Food Delivery Delay Engine")
st.write("Input current metrics to compute precise delivery delays instantly.")

st.markdown("---")

st.write("### 📊 Step 1: Input Delivery Parameters")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance to Restaurant (KM)", min_value=1.0, max_value=20.0, value=3.5, step=0.1)
    num_items = st.number_input("Number of Items Ordered", min_value=1, max_value=15, value=2)
    hour = st.slider("Hour of Day (24hr format)", min_value=11, max_value=23, value=19)

with col2:
    weather = st.slider("Weather (1: Clear ☀️ -> 5: Heavy Rain 🌧️)", min_value=1, max_value=5, value=1)
    traffic = st.slider("Traffic (1: Empty Roads 🟢 -> 5: Jam 🔴)", min_value=1, max_value=5, value=2)

st.markdown("---")

# Predict Button
if st.button("🚀 Calculate Delivery Time", use_container_width=True):
    # Format inputs for model prediction
    input_data = np.array([[distance, num_items, hour, weather, traffic]])
    
    # Make prediction
    predicted_time = model.predict(input_data)[0]
    
    # Display Result
    st.write("### 🔮 AI Prediction Output")
    st.metric(label="Estimated Delivery Time", value=f"{round(predicted_time)} Minutes")
    
    # Simple logic to show risk levels
    if predicted_time < 30:
        st.success("🟢 LOW RISK: Your food will arrive rapidly and fresh!")
    elif 30 <= predicted_time <= 45:
        st.warning("🟡 MEDIUM RISK: Slight delays. Food might arrive moderately warm.")
    else:
        st.error("🔴 HIGH RISK: Major delays predicted. Expect severe backlogs.")