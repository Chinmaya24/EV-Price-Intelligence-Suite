# ============================================
# EV Price Prediction using Random Forest
# ============================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import joblib


# ============================================
# Step 2: Load Dataset
# ============================================
df = pd.read_excel("FEV-data-Excel.xlsx")  # ✅ Replace with your actual file path

print("\n📄 First 5 Rows of the Dataset:")
print(df.head())
print("\n📊 Dataset Info:")
print(df.info())
print("\n🔍 Missing Values Count:")
print(df.isnull().sum())

# ============================================
# Step 3: Handle Missing Values
# ============================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\n✅ Missing values cleaned successfully!")

# ============================================
# Step 4: Encode Categorical Columns
# ============================================
for col in df.select_dtypes(include="object"):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("✅ Text columns converted to numbers!")

# ============================================
# Step 5: Prepare Features (X) and Target (y)
# ============================================
df = df.drop(columns=["Car full name", "Model"], errors='ignore')
X = df.drop(columns=["Minimal price (gross) [PLN]"])
y = df["Minimal price (gross) [PLN]"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split into training and testing sets!")

# ============================================
# Step 6: Train the Random Forest Model
# ============================================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ Model trained successfully!")
print(f"📈 R² Score: {r2:.3f}")
print(f"📉 RMSE: {rmse:.2f}")

# ============================================
# Step 7: Feature Importance
# ============================================
importances = model.feature_importances_
features = np.array(X.columns)
indices = np.argsort(importances)[::-1]

print("\n🔍 Top Features Influencing EV Price:")
for i in range(min(10, len(features))):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.3f}")

feature_importance_fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features[indices][:10][::-1], importances[indices][:10][::-1], color="#1f77b4")
ax.set_xlabel("Importance")
ax.set_title("Top 10 Features Affecting EV Price")
feature_importance_fig.tight_layout()

# ============================================
# Step 8: Save the Model & Feature Columns
# ============================================
joblib.dump(model, "ev_price_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n✅ Model and feature names saved successfully as 'ev_price_model.pkl' and 'feature_names.pkl'!")

# ============================================
# Step 9: Test the Model with New Input
# ============================================
print("\n📋 Expected feature order:")
print(list(X.columns))

# Example input (22 features)
new_data = [[
    1,      # Make (encoded)
    150,    # Engine power [KM]
    300,    # Maximum torque [Nm]
    1,      # Type of brakes
    0,      # Drive type
    60,     # Battery capacity [kWh]
    420,    # Range (WLTP) [km]
    270,    # Wheelbase [cm]
    450,    # Length [cm]
    180,    # Width [cm]
    160,    # Height [cm]
    1650,   # Minimal empty weight [kg]
    2100,   # Permissible gross weight [kg]
    450,    # Maximum load capacity [kg]
    5,      # Number of seats
    4,      # Number of doors
    18,     # Tire size [in]
    160,    # Maximum speed [kph]
    500,    # Boot capacity (VDA) [l]
    7.2,    # Acceleration 0–100 kph [s]
    120,    # Maximum DC charging power [kW]
    15.5    # Energy consumption [kWh/100 km]
]]

# Convert list to DataFrame to avoid warning
new_df = pd.DataFrame(new_data, columns=X.columns)

# Predict and display result
predicted_price = model.predict(new_df)
print(f"\n💰 Predicted EV Price: {predicted_price[0]:,.2f} PLN")

# ============================================
# 🚘 Electric Vehicle Price Predictor (₹ INR)
# ============================================

import streamlit as st
import joblib

# Load model and feature names
model = joblib.load("ev_price_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Conversion: PLN → INR
PLN_TO_INR = 21.0  # Update when needed


def format_range(column: str, precision: int = 1, suffix: str = "") -> str:
    """Return formatted min–max range for a column if available."""
    if column in df.columns:
        col = df[column]
        if np.issubdtype(col.dtype, np.number):
            fmt = f"{{:.{precision}f}}"
            return f"{fmt.format(col.min())} – {fmt.format(col.max())}{suffix}"
        return f"{col.unique()[0]} – {col.unique()[-1]}{suffix}"
    return "Data unavailable"

# Lightweight knowledge base for chatbot responses
FEATURE_TIPS = {
    "engine power": "Higher engine power generally improves performance and pushes the price up because it requires stronger drivetrains.",
    "battery": "Larger battery capacity boosts driving range but adds weight and cost. Balance it with your daily commute needs.",
    "range": "WLTP range is the most realistic lab cycle. Real-world range varies with temperature, driving style, and payload.",
    "charging": "Fast DC charging power tells you how quickly the pack can be recharged from public fast chargers.",
    "weight": "Higher curb weight can reduce efficiency but may signal a larger battery or premium materials.",
    "seats": "Seat and door count mostly influence practicality; they have a minor impact on pricing compared to powertrain specs.",
    "efficiency": "Lower energy consumption (kWh/100 km) means cheaper running costs and can offset a smaller battery.",
}

FAQ_RESPONSES = {
    "predict": "To get a price prediction, use the sliders above and press 'Predict EV Price'. The chatbot focuses on explanations.",
    "model": "The app uses a Random Forest Regressor trained on curated EV specifications and prices from the provided dataset.",
    "currency": "Predictions are made in PLN by the model and converted to INR using the PLN_TO_INR rate configured in the app.",
    "reset": "Need a fresh start? Use the 'Rerun' button in Streamlit to reset sliders and the chat history.",
}


def generate_chatbot_reply(prompt: str) -> str:
    """Simple rule-based assistant to answer EV and app questions."""
    text = prompt.lower().strip()
    responses = []

    # Direct FAQ matches
    for keyword, answer in FAQ_RESPONSES.items():
        if keyword in text:
            responses.append(answer)

    # Feature-specific tips
    for keyword, tip in FEATURE_TIPS.items():
        if keyword in text:
            responses.append(tip)

    # Guidance when user shares numbers
    if any(char.isdigit() for char in text) and "predict" not in text:
        responses.append(
            "If you want an exact price estimate for those numbers, plug them into the sliders above and run the predictor."
        )

    if "hello" in text or "hi" in text:
        responses.append("Hi there! Ask me anything about EV specs, pricing logic, or how to use the app.")

    if "thank" in text:
        responses.append("Happy to help! Let me know if you need anything else.")

    if not responses:
        responses.append(
            "I try to explain how specs affect EV pricing. Ask about power, batteries, range, charging, or how to use the predictor."
        )

    return " ".join(responses)

# ============================================
# Streamlit Page Setup
# ============================================
st.set_page_config(
    page_title="EV Price Intelligence Suite",
    page_icon="⚡",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 0 2rem;
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 40%, #020617 100%);
    }
    .hero {
        background: linear-gradient(120deg, rgba(59,130,246,0.9), rgba(14,165,233,0.9));
        color: white;
        padding: 2.5rem;
        border-radius: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 25px 65px rgba(15, 23, 42, 0.35);
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(14,165,233,0.95), rgba(59,130,246,0.95));
        border-radius: 1rem;
        padding: 1.25rem;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255,255,255,0.2);
        color: #f8fafc;
    }
    .metric-card h3 {
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        color: rgba(248,250,252,0.9);
        text-transform: uppercase;
    }
    .metric-card h2 {
        color: white;
    }
    .section-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 1.25rem;
        padding: 1.5rem;
        box-shadow: 0 25px 90px rgba(2,6,23,0.65);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148,163,184,0.25);
        backdrop-filter: blur(14px);
        color: #e2e8f0;
    }
    .section-card h2,
    .section-card h3,
    .section-card label,
    .section-card p,
    .section-card span {
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.5rem 1.5rem;
        background-color: rgba(255,255,255,0.15);
        color: #cbd5f5;
        border: 1px solid rgba(148,163,184,0.3);
    }
    .stTabs [aria-selected="true"] {
        background-color: #fbbf24 !important;
        color: #0f172a !important;
        border-color: transparent !important;
    }
    .stButton>button, .st-form button {
        border-radius: 999px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        background: linear-gradient(120deg, #fbbf24, #f97316);
        color: #0b1324;
        border: none;
        box-shadow: 0 12px 30px rgba(249,115,22,0.35);
    }
    .stButton>button:hover, .st-form button:hover {
        box-shadow: 0 18px 35px rgba(249,115,22,0.45);
    }
    .stDataFrame {
        border-radius: 1rem;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar Insights
# ============================================
st.sidebar.header("Dataset & Model")
st.sidebar.metric("Records Analysed", f"{len(df):,}")
st.sidebar.metric("R² Score", f"{r2:.2f}")
st.sidebar.metric("RMSE (PLN)", f"{rmse:,.0f}")
st.sidebar.caption("Random Forest with 200 estimators trained on curated EV specs.")
st.sidebar.divider()
st.sidebar.subheader("Quick Tips")
st.sidebar.write("- Keep drivetrain, weight, and battery balanced for efficiency.")
st.sidebar.write("- High torque plus fast charging commands premium pricing.")
st.sidebar.write("- Use the chatbot tab for detailed guidance.")

# ============================================
# Hero Section
# ============================================
with st.container():
    st.markdown(
        """
        <div class="hero">
            <h1>⚡ EV Price Intelligence Suite</h1>
            <p style="font-size:1.2rem; opacity:0.9;">
                A production-ready console to explore EV specs, benchmark feature impact,
                and forecast prices in INR with data-backed confidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# KPI Row
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.markdown(
    f"<div class='metric-card'><h3>Battery Window</h3><h2>{format_range('Battery capacity [kWh]', precision=0, suffix=' kWh')}</h2></div>",
    unsafe_allow_html=True,
)
col_kpi2.markdown(
    f"<div class='metric-card'><h3>Acceleration Range</h3><h2>{format_range('Acceleration 0–100 kph [s]', precision=1, suffix=' s')}</h2></div>",
    unsafe_allow_html=True,
)
col_kpi3.markdown(
    f"<div class='metric-card'><h3>Top Speed Span</h3><h2>{format_range('Maximum speed [kph]', precision=0, suffix=' km/h')}</h2></div>",
    unsafe_allow_html=True,
)

# ============================================
# Tabbed Experience
# ============================================
overview_tab, predictor_tab, insights_tab, assistant_tab = st.tabs(
    ["Overview", "Predictor Studio", "Insights & Benchmarks", "Assistant Chatbot"]
)

with overview_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Why this platform?")
    st.write(
        "- Unified workspace to configure EV specs and instantly view price forecasts.\n"
        "- Feature importance visualizations to justify decisions to stakeholders.\n"
        "- Conversational assistant to interpret specs and model behavior in plain English."
    )
    st.markdown("#### Recently Processed Sample")
    st.dataframe(df.head(5))
    st.markdown("</div>", unsafe_allow_html=True)

with predictor_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Configure Your EV Blueprint")
    with st.form("predict_form"):
        st.markdown("###### Performance Stack")
        col1, col2, col3 = st.columns(3)
        with col1:
            engine_power = st.slider("Engine power [KM]", 50, 500, 150)
            torque = st.slider("Max torque [Nm]", 100, 1000, 300)
        with col2:
            acceleration = st.slider("Acceleration 0–100 kph [s]", 3.0, 15.0, 7.5)
            max_speed = st.slider("Max speed [kph]", 100, 300, 160)
        with col3:
            charging_power = st.slider("DC charging power [kW]", 20, 350, 120)

        st.markdown("###### Battery & Efficiency")
        col4, col5, col6 = st.columns(3)
        with col4:
            battery_capacity = st.slider("Battery capacity [kWh]", 20.0, 200.0, 60.0)
            range_km = st.slider("Range (WLTP) [km]", 100, 800, 420)
        with col5:
            energy_consumption = st.slider("Energy consumption [kWh/100 km]", 10.0, 30.0, 15.0)
            make = st.number_input("Make (encoded)", 0, 10, 1)
        with col6:
            brakes = st.number_input("Type of brakes (encoded)", 0, 5, 1)
            drive_type = st.number_input("Drive type (encoded)", 0, 5, 0)

        st.markdown("###### Dimensions & Structure")
        col7, col8, col9 = st.columns(3)
        with col7:
            wheelbase = st.slider("Wheelbase [cm]", 200, 400, 270)
            length = st.slider("Length [cm]", 350, 600, 450)
            width = st.slider("Width [cm]", 150, 250, 180)
        with col8:
            height = st.slider("Height [cm]", 130, 200, 160)
            min_weight = st.slider("Empty weight [kg]", 800, 3000, 1650)
            gross_weight = st.slider("Gross weight [kg]", 1200, 3500, 2100)
        with col9:
            load_capacity = st.slider("Load capacity [kg]", 200, 1000, 450)
            boot_capacity = st.slider("Boot capacity [l]", 100, 1000, 500)
            tire_size = st.slider("Tire size [in]", 13, 22, 18)

        st.markdown("###### Cabin Layout")
        seats = st.slider("Seats", 2, 9, 5)
        doors = st.slider("Doors", 2, 6, 4)

        input_data = [
            make, engine_power, torque, brakes, drive_type, battery_capacity, range_km,
            wheelbase, length, width, height, min_weight, gross_weight, load_capacity,
            seats, doors, tire_size, max_speed, boot_capacity, acceleration,
            charging_power, energy_consumption
        ]

        submitted = st.form_submit_button("Run Price Forecast")

    if submitted:
        predicted_pln = model.predict([input_data])[0]
        predicted_inr = predicted_pln * PLN_TO_INR
        st.success(f"Estimated Price: ₹ {predicted_inr:,.2f}")
        st.caption(f"({predicted_pln:,.2f} PLN @ ₹{PLN_TO_INR}/PLN)")
        st.balloons()
    st.markdown("</div>", unsafe_allow_html=True)

with insights_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Feature Importance")
    st.pyplot(feature_importance_fig, clear_figure=False)
    st.markdown("#### Key Signals")
    st.write(
        "- Higher battery capacity, range, and torque dominate price movements.\n"
        "- Dimensional attributes influence price indirectly via weight and premium cabin design.\n"
        "- Charging power is a differentiator for performance EVs, reflecting infrastructure readiness."
    )

    st.markdown("#### Design Principles")
    st.metric("Optimal Battery Window", "55 – 95 kWh")
    st.metric("Comfort Sweet Spot", "5 seats • 4 doors • 500L boot")
    st.markdown("</div>", unsafe_allow_html=True)

with assistant_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("💬 EV Assistant Chatbot")
    st.caption("Ask questions about EV specs, the model, or how to interpret the predictions.")
    st.info("Hint: describe the spec or topic you're curious about (e.g., “How does battery size affect price?”). The assistant serves curated tips from a built-in knowledge base.", icon="💡")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hey! I'm your EV assistant. Ask about batteries, range, charging, or how pricing works.",
            }
        ]

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input("Type your EV question here")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)
        response = generate_chatbot_reply(user_prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.write(response)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown(
    "<center>🚗 Developed with ❤️ using Streamlit | EV Price Predictor (INR)</center>",
    unsafe_allow_html=True
)
