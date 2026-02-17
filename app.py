import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("eye_health_model.pkl")

# -------------------------------
# Recommendation Function
# -------------------------------
def get_recommendation(risk):
    if "HIGH" in risk:
        return [
            "Reduce screen time",
            "Follow 20-20-20 rule",
            "Increase outdoor activities",
            "Consult an eye specialist"
        ]
    elif "MEDIUM" in risk:
        return [
            "Take regular screen breaks",
            "Reduce screen brightness",
            "Practice eye exercises daily"
        ]
    else:
        return [
            "Maintain healthy screen habits",
            "Continue eye exercises",
            "Regular eye checkups"
        ]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Eye Health Prediction", layout="centered")

st.title("👁️ AI-Based Eye Health Prediction System")
st.write("Enter your lifestyle details to monitor eye health risk")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔍 Prediction", "📊 Explainable AI", "ℹ️ About Project"]
)

# ===============================
# TAB 1: Prediction
# ===============================
with tab1:

    st.sidebar.header("🧍 Lifestyle Inputs")

    age = st.sidebar.number_input("Age", 1, 100, 22)
    screen_time = st.sidebar.number_input("Daily Screen Time (hours)", 0.0, 24.0, 6.0)
    exercise = st.sidebar.number_input("Exercise Hours per Day", 0.0, 5.0, 1.0)
    mental_health = st.sidebar.slider("Mental Health Score (1-10)", 1, 10, 5)
    screen_brightness = st.sidebar.slider("Screen Brightness (%)", 0, 100, 50)
    outdoor_light = st.sidebar.number_input("Outdoor Light Exposure (hours)", 0.0, 10.0, 1.0)
    night_mode = st.sidebar.selectbox("Night Mode Usage", ["No", "Yes"])
    night_mode_val = 1 if night_mode == "Yes" else 0
    screen_distance = st.sidebar.number_input("Screen Distance (cm)", 10.0, 100.0, 40.0)
    glasses_number = st.sidebar.number_input("Glasses Power", 0.0, 10.0, 0.0)
    height_cm = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 165.0)

    # Future Simulation
    st.sidebar.markdown("### 🔮 Future Simulation")
    future_screen_time = st.sidebar.slider(
        "Simulate Future Screen Time (hours)", 0.0, 15.0, screen_time
    )

    # -------------------------------
    # Current Prediction
    # -------------------------------
    input_data = np.array([[ 
        exercise,
        mental_health,
        screen_time,
        screen_brightness,
        age,
        height_cm,
        outdoor_light,
        night_mode_val,
        screen_distance,
        glasses_number
    ]])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    risk_map = {
        0: "HIGH RISK ⚠️",
        1: "LOW RISK ✅",
        2: "MEDIUM RISK ⚡"
    }

    st.subheader("🧠 Current Prediction")
    st.success(f"Eye Health Risk: **{risk_map[prediction]}**")

    # Risk Score
    risk_score = round(proba[prediction] * 100, 2)
    st.metric("Risk Score (0-100)", risk_score)

    # Alert
    if risk_score > 70:
        st.error("🚨 High Risk Alert! Reduce screen time immediately.")
    elif risk_score > 40:
        st.warning("⚠️ Moderate Risk. Improve lifestyle habits.")
    else:
        st.success("✅ Low Risk. Maintain healthy habits.")

    # Confidence
    st.subheader("📊 Prediction Confidence")
    labels = ["High Risk", "Low Risk", "Medium Risk"]
    for label, p in zip(labels, proba):
        st.write(f"{label}: {round(p*100, 2)}%")
        st.progress(float(p))

    # Pie Chart
    st.subheader("📊 Risk Distribution")
    fig, ax = plt.subplots()
    ax.pie(proba, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # Recommendations
    st.subheader("👀 Recommendations")
    for tip in get_recommendation(risk_map[prediction]):
        st.write("✔️", tip)

    # -------------------------------
    # SAVE USER HISTORY
    # -------------------------------
    file_path = "user_history.csv"

    new_record = {
        "Date": datetime.now(),
        "Age": age,
        "Screen_Time": screen_time,
        "Outdoor_Light": outdoor_light,
        "Risk": risk_map[prediction],
        "Risk_Score": risk_score
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df = pd.DataFrame([new_record])

    df.to_csv(file_path, index=False)

    # -------------------------------
    # SHOW TREND GRAPH
    # -------------------------------
    st.subheader("📈 Your Risk Trend")
    history = pd.read_csv(file_path)

    if len(history) > 1:
        st.line_chart(history["Risk_Score"])

    # -------------------------------
    # PDF REPORT DOWNLOAD
    # -------------------------------
    def create_pdf():
        doc = SimpleDocTemplate("Eye_Health_Report.pdf")
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Eye Health Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))
        elements.append(Paragraph(f"Risk Level: {risk_map[prediction]}", styles["Normal"]))
        elements.append(Paragraph(f"Risk Score: {risk_score}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        for tip in get_recommendation(risk_map[prediction]):
            elements.append(Paragraph(f"- {tip}", styles["Normal"]))

        doc.build(elements)

    if st.button("📄 Download Report"):
        create_pdf()
        with open("Eye_Health_Report.pdf", "rb") as f:
            st.download_button("Click to Download", f, "Eye_Health_Report.pdf")

    # -------------------------------
    # Future Prediction
    # -------------------------------
    simulated_input = np.array([[ 
        exercise,
        mental_health,
        future_screen_time,
        screen_brightness,
        age,
        height_cm,
        outdoor_light,
        night_mode_val,
        screen_distance,
        glasses_number
    ]])

    future_prediction = model.predict(simulated_input)[0]

    st.subheader("🔮 Future Risk Prediction")
    st.info(
        f"If screen time changes to {future_screen_time} hrs → "
        f"Predicted Risk: {risk_map[future_prediction]}"
    )

# ===============================
# TAB 2: Explainable AI
# ===============================
with tab2:

    st.subheader("📊 Feature Importance")

    features = [
        "Exercise Hours",
        "Mental Health",
        "Screen Time",
        "Screen Brightness",
        "Age",
        "Height",
        "Outdoor Light",
        "Night Mode",
        "Screen Distance",
        "Glasses Power"
    ]

    importance = model.feature_importances_

    fig2, ax2 = plt.subplots()
    ax2.barh(features, importance)
    ax2.set_xlabel("Importance Score")
    st.pyplot(fig2)

# ===============================
# TAB 3: About Project
# ===============================
with tab3:
    st.markdown("""
    ### 🎓 Project Overview
    This system predicts eye health risk using Machine Learning.

    ### 🚀 Advanced Features
    - Real-Time Monitoring
    - Risk Score (0–100)
    - Alert System
    - Future Simulation
    - Explainable AI
    - User History Tracking
    - Risk Trend Analytics
    - PDF Report Generation

    ### 🛠 Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - Random Forest
    - ReportLab
    """)
