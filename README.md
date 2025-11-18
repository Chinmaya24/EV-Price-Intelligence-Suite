# ⚡ EV Price Intelligence Suite

An immersive **Streamlit experience** that forecasts **Electric Vehicle (EV)** prices and explains how specs impact valuation. Backed by a **Random Forest Regressor** trained on curated EV profiles.

---

## 🚀 Highlights

- 🎛️ **Predictor Studio** with multi-section form + KPI insights
- 📊 **Analytics tab** (feature-importance chart, design guidelines)
- 💬 **In-app chatbot** offering contextual EV + model tips
- 🗂️ Automated preprocessing, encoding, and model persistence
- 🪄 Professional glassmorphism UI with responsive layout

---

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| ML | RandomForestRegressor (scikit-learn) |
| Data | pandas, numpy, openpyxl |
| Visualization | matplotlib |
| Serving | Streamlit |
| Packaging | joblib |

---

## 🧩 Dataset Snapshot

`FEV-data-Excel.xlsx` contains:
- Drivetrain specs: engine power, torque, drivetrain type
- Battery & range: capacity, WLTP range, charging power
- Body metrics: wheelbase, dimensions, seating, load/boot capacity
- Performance: acceleration, top speed, efficiency

---

## 🧮 Workflow

1. **Load & Clean** – missing value handling, categorical encoding
2. **Split & Train** – Random Forest (200 estimators, fixed seed)
3. **Evaluate** – R² + RMSE surfaced in sidebar metrics
4. **Persist** – saves `ev_price_model.pkl` + `feature_names.pkl`
5. **Serve** – Streamlit UI for configuration, insights, and chat

---

## 🌐 Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Launch Streamlit
```bash
streamlit run ev_price_app.py
```

### 3️⃣ Explore
- Configure specs in **Predictor Studio** → click “Run Price Forecast”
- Review explanations in **Insights & Benchmarks**
- Ask the **EV Assistant** for help (e.g., “How does battery size affect price?”)

---

## 📦 Project Structure

```
Week1/
├── ev_price_app.py           # Training + Streamlit experience
├── ev_price_model.pkl        # Saved Random Forest model
├── feature_names.pkl         # Ordered feature list
├── FEV-data-Excel.xlsx       # Source dataset
├── requirements.txt          # Dependency list
└── README.md                 # This doc
```

---

## 🤝 Contributions & Notes

- Update `PLN_TO_INR` in `ev_price_app.py` to match current conversion rates.
- Extend the chatbot knowledge base inside `FEATURE_TIPS` / `FAQ_RESPONSES`.
- Pull requests for new visualizations or model improvements are welcome! 👋