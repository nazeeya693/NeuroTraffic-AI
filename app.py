import matplotlib.pyplot as plt
import time
import glob
import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from vehicle_counter import count_vehicles
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(
    page_title="NeuroTraffic AI",
    page_icon="🚦",
    layout="wide"
)
st.sidebar.title("🚦 Control Panel")
st.sidebar.success("🟢 System Online")
st.sidebar.info("🤖 AI Engine Active")
st.sidebar.warning("📡 Traffic Sensors Connected")


theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Neon", "Classic"])

city = st.sidebar.text_input("City Name", "chennai")

auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh Simulation")

# Custom Futuristic UI
st.markdown("""
<style>

    .main {
        background-color: #0f1117;
    }

    .title {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        color: #00FFF7;
        text-shadow: 0px 0px 20px #00FFF7;
        margin-top: 10px;
    }

    .subtitle {
        font-size: 22px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }

    .metric-box {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 0px 15px #00FFF7;
        color: white;
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #00FFF7;
        color: black;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
        box-shadow: 0px 0px 10px #00FFF7;
    }

    .stButton>button:hover {
        background-color: #00cfc7;
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("## 🌍 Smart City Traffic Control System")
st.caption("Real-time AI-based congestion monitoring dashboard")
st.markdown(
    '<div class="title">🚦 NeuroTraffic AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Traffic Forecasting, Route Intelligence and Signal Optimization</div>',
    unsafe_allow_html=True
)

st.info(
    "🌍 AI-powered Smart Traffic Management System using YOLOv8 and Machine Learning"
)


def animated_dashboard(total, prediction, green_time):

    st.markdown("#### ⚡ LIVE TRAFFIC CONTROL DASHBOARD")

    status_box = st.empty()
    progress_bar = st.progress(0)
    metric_box = st.empty()

    for i in range(1, 101, 5):

        time.sleep(0.05)
        progress_bar.progress(i)

        if prediction == "High":
            status = "🚨 CONGESTION BUILDING..."
            color = "🔴"
        elif prediction == "Medium":
            status = "⚠ TRAFFIC FLOW MODERATE"
            color = "🟡"
        else:
            status = "✅ TRAFFIC SMOOTH"
            color = "🟢"

        status_box.markdown(f"##### {color} {status}")
        metric_box.markdown(f"**🚦 Signal Countdown:** {green_time} sec")

    st.success("📡 Live Dashboard Updated Successfully")

def traffic_map(prediction):
    # Center map (you can change to your city)
    m = folium.Map(
       location=[12.97, 77.59],
       zoom_start=13,
       tiles="CartoDB dark_matter"   
)

    # Traffic color logic
    if prediction == "High":
        color = "red"
    elif prediction == "Medium":
        color = "orange"
    else:
        color = "green"

    # Fake traffic points (you can replace with real locations later)
    locations = [
        [12.9716, 77.5946],
        [12.9352, 77.6245],
        [12.9916, 77.5700],
        [12.9600, 77.6100]
    ]

    for loc in locations:
        folium.CircleMarker(
            location=loc,
            radius=12,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Traffic: {prediction}"
        ).add_to(m)

    return m

# Load Dataset
df = pd.read_csv("traffic_dataset.csv")

forecast_df = pd.read_csv("traffic_forecast_v2.csv")

forecast_df = forecast_df.dropna()

forecast_df.columns = forecast_df.columns.str.strip()

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

route_times = {

    ("Tambaram", "Chrompet"): 10,
    ("Tambaram", "Guindy"): 30,
    ("Tambaram", "Airport"): 25,
    ("Tambaram", "T Nagar"): 45,
    ("Tambaram", "Velachery"): 35,

    ("Chrompet", "Tambaram"): 10,
    ("Chrompet", "Guindy"): 20,
    ("Chrompet", "Airport"): 15,
    ("Chrompet", "T Nagar"): 35,
    ("Chrompet", "Velachery"): 25,

    ("Guindy", "Airport"): 10,
    ("Guindy", "T Nagar"): 15,
    ("Guindy", "Velachery"): 20,

    ("Airport", "Guindy"): 10,
    ("Airport", "Velachery"): 15,

    ("Velachery", "T Nagar"): 20
}

forecast_df = pd.get_dummies(
    forecast_df,
    columns=["weather", "event"]
)

forecast_df["day"] = forecast_df["day"].map(day_mapping)

X_forecast = forecast_df.drop("traffic level", axis=1)

y_forecast = forecast_df["traffic level"]

X_train, X_test, y_train, y_test = train_test_split(
    X_forecast,
    y_forecast,
    test_size=0.2,
    random_state=42
)

forecast_model = RandomForestClassifier()

forecast_model.fit(X_train, y_train)

preds = forecast_model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

# Random Forest Model

#rf_model = RandomForestClassifier()

#rf_model.fit(X_train, y_train)

#rf_predictions = rf_model.predict(X_test)

#rf_accuracy = accuracy_score(y_test, rf_predictions)

# Logistic Regression Model

#lr_model = LogisticRegression(max_iter=1000)

#lr_model.fit(X_train, y_train)

#lr_predictions = lr_model.predict(X_test)

#lr_accuracy = accuracy_score(y_test, lr_predictions)

# Final model used for prediction

 #model = rf_model

# Futuristic Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-box">
            <h2>🚗 AI Detection</h2>
            <h1>ACTIVE</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-box">
            <h2>📊 ML Model</h2>
            <h1>READY</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-box">
            <h2>🚦 Traffic Status</h2>
            <h1>LIVE</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("") 

st.subheader("📊 Live Traffic Summary")

col1, col2, col3 = st.columns(3)

col1.metric("🛣 Roads Monitored", "12")
col2.metric("🚦 Active Signals", "8")
col3.metric("📡 AI Status", "ONLINE")

# Input Section
st.subheader("🚘 Enter Traffic Details")

# cars = st.number_input("Enter Number of Cars", min_value=0)

# buses = st.number_input("Enter Number of Buses", min_value=0)

# trucks = st.number_input("Enter Number of Trucks", min_value=0)

hour = st.selectbox(
    "Select Hour",
    [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
)

day = st.selectbox(
    "Select Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

weather = st.selectbox(
    "Select Weather",
    ["Sunny", "Rainy", "Cloudy"]
)

event = st.selectbox(
    "Select Event",
    [
        "Normal",
        "School",
        "Office Rush",
        "Shopping",
        "Festival"
    ]
)

st.subheader("📷 Upload Traffic Image")

uploaded_image = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    with open("uploaded_traffic.jpg", "wb") as f:
        f.write(uploaded_image.getbuffer())

    cars, buses, trucks, total = count_vehicles(
        "uploaded_traffic.jpg"
    )

    st.success("✅ Vehicle Detection Completed")

    detected_files = glob.glob("runs/detect/predict*/*.jpg")

    if detected_files:

        latest_file = max(
            detected_files,
            key=os.path.getctime
        )

        st.image(
            latest_file,
            caption="🎯 YOLO Detection Output",
            use_container_width=True
        )
    
    

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🚗 Cars", cars)
    col2.metric("🚌 Buses", buses)
    col3.metric("🚚 Trucks", trucks)
    col4.metric("🚦 Total", total)

    traffic_density = (
        "High" if total > 25
        else "Medium" if total > 10
        else "Low"
    )

    st.metric(
        "🚦 Live Density",
        traffic_density
    )

emergency = st.checkbox("🚑 Emergency Vehicle Detected")

source = st.selectbox(
    "Source",
    ["Kilambhakkam", "Tambaram", "Chrompet", "Guindy"]
)

destination = st.selectbox(
    "Destination",
    ["Chrompet", "Tambaram", "T Nagar", "Velachery"]
)

# Prediction

if st.button("Predict Traffic", key="predict_btn_main"):

    st.subheader("🚦 Live Traffic Control Panel")

    day_number = day_mapping[day]

    input_data = pd.DataFrame({
        "hour": [hour],
        "day": [day_number],

        "weather_Cloudy": [1 if weather == "Cloudy" else 0],
        "weather_Rainy": [1 if weather == "Rainy" else 0],
        "weather_Sunny": [1 if weather == "Sunny" else 0],

        "event_Festival": [1 if event == "Festival" else 0],
        "event_Normal": [1 if event == "Normal" else 0],
        "event_Office Rush": [1 if event == "Office Rush" else 0],
        "event_School": [1 if event == "School" else 0],
        "event_Shopping": [1 if event == "Shopping" else 0]    
})
    
    prediction = forecast_model.predict(input_data)[0]

    base_time = route_times.get(
    (source, destination),
    20
)

    total = 0

    with st.spinner("🧠 AI analyzing traffic patterns..."):
        time.sleep(1.5)

    if prediction == "High":
        travel_time = base_time + 15

    elif prediction == "Medium":
        travel_time = base_time + 8

    else:
        travel_time = base_time

    if prediction == "High":
        green_time = 60
        delay = "15-20 Minutes"

    elif prediction == "Medium":
        green_time = 40
        delay = "5-10 Minutes"

    else:
        green_time = 20
        delay = "0-3 Minutes"

    if emergency:
        green_time = 90
        st.success("🚑 Ambulance Priority Activated")

    # Smart City Score
    if prediction == "High":
        city_score = 60

    elif prediction == "Medium":
        city_score = 80

    else:
        city_score = 95

    # Congestion Score

    if prediction == "High":
        congestion_score = 90

    elif prediction == "Medium":
        congestion_score = 60

    else:
        congestion_score = 25
  
    if prediction == "High":
        health_score = 30

    elif prediction == "Medium":
        health_score = 65

    else:
        health_score = 95

    # Carbon Emission

    if prediction == "High":
        carbon = "25 Kg CO₂/hr"

    elif prediction == "Medium":
        carbon = "15 Kg CO₂/hr"

    else:
        carbon = "5 Kg CO₂/hr"


    # Peak Hour Detector

    if hour in [8, 9, 18, 19]:
        st.warning("⏰ Peak Hour Detected")
   
    if emergency:
        green_time = 90
        st.error("🚑 Emergency Vehicle Priority Activated")

    if prediction == "High" and weather == "Rainy":
        risk = "High"

    elif prediction == "Medium":
        risk = "Medium"

    else:
        risk = "Low"
    
    if prediction == "High":
        travel_time = 35

    elif prediction == "Medium":
        travel_time = 20

    else:
        travel_time = 10

    st.session_state["result"] = {
        "prediction": prediction,
        "total": total,
        "green_time": green_time,
    }

    st.success(f"🚦 Predicted Traffic Level: {prediction}")

    # ANIMATION CALL (NOW SAFE)
    animated_dashboard(total, prediction, green_time)

    # SMART SIGNAL DISPLAY
    if prediction == "High":
        st.error("🚨 HIGH TRAFFIC DETECTED")
        st.warning(f"🚦 Green Signal: {green_time} sec")
        st.info("📈 Congestion Control Activated")

    elif prediction == "Medium":
        st.warning("⚠ MEDIUM TRAFFIC DETECTED")
        st.info(f"🚦 Green Signal: {green_time} sec")
        st.info("🚗 Monitor Traffic Flow")

    else:
        st.success("✅ LOW TRAFFIC DETECTED")
        st.success(f"🚦 Green Signal: {green_time} sec")
        st.info("🌿 Smooth Traffic")

    # AI INSIGHTS
    st.metric("Selected Hour", hour)
    st.metric("Predicted Traffic", prediction)
    st.metric("Green Signal Timer", f"{green_time} sec")
    st.metric("Expected Delay", delay)
    st.metric("🏙 Smart City Efficiency", f"{city_score}%")
    st.metric("🎯 Model Accuracy", f"{accuracy*100:.2f}%")
    st.metric("🚦 Congestion Score", f"{congestion_score}/100")
    st.progress(congestion_score)
    st.metric("Estimated Travel Time", f"{travel_time} mins")
    st.metric("Route", f"{source} ➜ {destination}")
    st.metric("Estimated Travel Time",f"{travel_time} mins")
    st.metric("❤️ Traffic Health Score", f"{health_score}/100")
    st.metric("Traffic Severity Score",
          100 if prediction=="High"
          else 60 if prediction=="Medium"
          else 20)
    st.metric("Accident Risk", risk)
    st.metric("CO₂ Emission", carbon)
    risk_score = min(total * 3, 100)

    st.metric(
        "⚠ Traffic Risk Score",
        f"{risk_score}%"
    )

    if risk_score > 80:
        st.error("🚨 High Accident Risk Zone")

    elif risk_score > 50:
        st.warning("⚠ Moderate Congestion Risk")

    else:
        st.success("✅ Safe Traffic Conditions")

    # Congestion Status
    if prediction == "High":
        st.error("🔴 Congestion Level : Critical")

    elif prediction == "Medium":
        st.warning("🟡 Congestion Level : Moderate")

    else:
        st.success("🟢 Congestion Level : Smooth")

    st.subheader("🤖 AI Recommendation Engine")

        
    if prediction == "High":
        st.error("🚫 Heavy congestion detected. Use alternate routes.")

    elif prediction == "Medium":
        st.warning("⚠ Moderate traffic. Start 10 minutes earlier.")

    else:
        st.success("✅ Roads are clear. Safe and smooth travel.")

    route_df = pd.DataFrame({
        "Route": [
            "Tambaram → Guindy",
            "Tambaram → Velachery",
            "Tambaram → T Nagar",
            "Chrompet → Airport"
        ],
        "ETA (mins)": [15, 20, 25, 10],
        "Traffic Status": [
            "Low",
            "Medium",
            "High",
            "Low"
        ]
    })

    st.subheader("🗺 AI Alternate Route")

    if prediction == "High":

        if source == "Tambaram" and destination == "Guindy":
            st.info("Suggested Route: Tambaram ➜ Velachery ➜ Guindy")

        elif source == "Chrompet" and destination == "T Nagar":
            st.info("Suggested Route: Chrompet ➜ Guindy ➜ T Nagar")

        else:
            st.info("AI suggests using alternative city roads.")

    else:
        st.success("No alternate route required.")

        st.subheader("🗺 Route Information")

        route_df = pd.DataFrame({
        "Route": [
            "Tambaram → Guindy",
            "Tambaram → Velachery",
            "Tambaram → T Nagar"
        ],
        "ETA (mins)": [15, 20, 25],
        "Traffic": ["Low", "Medium", "High"]
    })
    st.dataframe(route_df)

    st.subheader("📊 Traffic Forecast")

    if hour in [8, 9]:
        st.warning("Morning Rush Hour Expected")

    elif hour in [18, 19]:
        st.warning("Evening Rush Hour Expected")

    else:
        st.success("Normal Traffic Expected")
    
    # =========================
    # SMART SIGNAL TIMING AI
    # =========================

    if prediction == "High":

        green_time = 60
        st.error("🚨 HIGH TRAFFIC DETECTED")
        st.warning(f"🚦 AI Suggested Green Signal Time: {green_time} seconds")
        st.info("📈 Congestion Control Activated")

    elif prediction == "Medium":

        green_time = 40
        st.warning("⚠ MEDIUM TRAFFIC DETECTED")
        st.info(f"🚦 AI Suggested Green Signal Time: {green_time} seconds")
        st.info("🚗 Monitor Traffic Flow Carefully")

    else:

        green_time = 20
        st.success("✅ LOW TRAFFIC DETECTED")
        st.success(f"🚦 AI Suggested Green Signal Time: {green_time} seconds")
        st.info("🌿 Traffic Flow is Smooth")

    # =========================
    # AI INSIGHTS PANEL
    # =========================
    if "result" in st.session_state:

       result = st.session_state["result"]

       st.success(f"🚦 Predicted Traffic Level: {result['prediction']}")

       st.subheader("🧠 AI Traffic Insights")
       st.metric("Total Vehicles", result["total"])
       st.metric("Predicted Traffic", result["prediction"])
       st.metric("Green Signal Timer", f"{result['green_time']} sec")
       

   # Dataset Preview
st.subheader("📂 Live Traffic Dataset")

st.dataframe(df.head())

# Graph Section
st.subheader("📈 Traffic Analytics")

#st.line_chart(df["Total Vehicles"])

# Footer
st.markdown("""
---
<center>
<h4 style='color:white;'>
🚀 Developed with AI, Machine Learning & Computer Vision
</h4>
</center>
""", unsafe_allow_html=True)


# Advanced Analytics

st.subheader("🚀 Key Features")

st.write("""
✅ Traffic Prediction using Machine Learning

✅ Dynamic Signal Timing

✅ Emergency Vehicle Priority

✅ Route ETA Prediction

✅ Accident Risk Analysis

✅ Carbon Emission Monitoring

✅ AI Alternate Route Suggestion

✅ Smart City Efficiency Score
""")

st.subheader("📊 Advanced Traffic Analytics")

# Bar Chart
#st.bar_chart(df[["Cars", "Buses", "Trucks"]])

# Pie Chart Data
#vehicle_totals = [
 #  df["Cars"].sum(),
   #  df["Buses"].sum(),
    #df["Trucks"].sum()
#]

'''pie_data = pd.DataFrame({
    "Vehicles": ["Cars", "Buses", "Trucks"],
    "Count": vehicle_totals
})

st.subheader("🥧 Vehicle Distribution")

st.dataframe(pie_data)'''

# Smart Insights
st.subheader("🧠 AI Traffic Insights")

#total_vehicles = df["Total Vehicles"].mean()

# Footer
st.markdown("""
---
<center>
<h3 style='color:#00FFF7;'>
🚀 NeuroTraffic AI | Smart City Traffic Intelligence System
</h3>
</center>
""", unsafe_allow_html=True)


# =========================
# ADVANCED AI ANALYTICS
# =========================

st.markdown("---")

st.subheader("📊 Advanced AI Traffic Analytics")

# Sample analytics data
traffic_data = {
    "Low": 15,
    "Medium": 30,
    "High": 10
}

# pie chart

fig1, ax1 = plt.subplots(figsize=(3,3))

ax1.pie(
    traffic_data.values(),
    labels=traffic_data.keys(),
    autopct='%1.1f%%'
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.pyplot(fig1)

# Bar Chart
fig2, ax2 = plt.subplots(figsize=(3,3))

ax2.bar(
    traffic_data.keys(),
    traffic_data.values()
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.pyplot(fig2)


# AI Insights
st.subheader("🧠 AI Insights Engine")

st.success("✅ Peak traffic usually occurs during office hours.")

st.warning("⚠ Medium traffic density detected frequently.")

st.info("🚦 AI recommends adaptive signal timing for congestion reduction.")

# Smart City Recommendation
st.subheader("🌆 Smart City Recommendation")

st.write("""
- Deploy adaptive AI traffic signals
- Increase emergency lane priority
- Optimize green signal timing dynamically
- Use AI surveillance for congestion monitoring
""")

st.subheader("⚙ System Architecture")

st.write("""
1. User selects route, weather and event
2. Dataset is processed using Pandas
3. Random Forest ML model predicts traffic
4. AI calculates signal timing
5. ETA, congestion and risk are generated
6. Smart city recommendations are displayed
""")

st.subheader("🏆 System Performance")

st.success("Prediction Engine : Active")
st.success("Traffic Intelligence : Active")
st.success("Emergency Response : Active")
st.success("Route Optimization : Active")


with st.expander("📖 About NeuroTraffic AI"):
    st.write("""
    NeuroTraffic AI is a smart city traffic intelligence system
    that predicts traffic congestion using Machine Learning.

    Features:
    - Traffic Prediction
    - Route ETA Estimation
    - Smart Signal Timing
    - Carbon Emission Monitoring
    - Emergency Vehicle Priority
    - Alternate Route Recommendation
    - Smart City Efficiency Analysis
    """)
