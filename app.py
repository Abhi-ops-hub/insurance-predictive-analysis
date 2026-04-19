import streamlit as st
import numpy as np
import pickle

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="🏥",
    layout="centered"
)

# ── Load model & scaler ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ── UI ─────────────────────────────────────────────────────
st.title("🏥 Insurance Charge Predictor")
st.markdown("Fill in the details below to estimate your insurance charges.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex      = st.selectbox("Sex", ["Male", "Female"])
    smoker   = st.selectbox("Smoker?", ["No", "Yes"])
    region   = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.divider()

# ── Preprocessing (mirrors your training pipeline) ─────────
def preprocess(age, bmi, children, sex, smoker, region):
    # Scale numeric
    scaled = scaler.transform([[age, bmi, children]])[0]
    age_s, bmi_s, children_s = scaled

    # Encode categoricals
    is_female        = 1 if sex == "Female" else 0
    is_smoker        = 1 if smoker == "Yes" else 0
    region_southeast = 1 if region == "Southeast" else 0

    # BMI category (obese = BMI > 29.9)
    bmi_category_obese = 1 if bmi > 29.9 else 0

    # Feature order must match training:
    # ['age','is_female','bmi','children','is_smoker','region_southeast','bmi_category_obese']
    return np.array([[age_s, is_female, bmi_s, children_s,
                      is_smoker, region_southeast, bmi_category_obese]])

# ── Predict button ─────────────────────────────────────────
if st.button("💰 Predict My Insurance Charges", use_container_width=True):
    features = preprocess(age, bmi, children, sex, smoker, region)
    prediction = model.predict(features)[0]

    st.success(f"### Estimated Annual Insurance Charge: **${prediction:,.2f}**")

    # Quick insight
    st.markdown("---")
    st.subheader("📊 Key Factors Affecting Your Estimate")

    factors = []
    if smoker == "Yes":
        factors.append("🚬 **Smoking** significantly increases your premium.")
    if bmi > 29.9:
        factors.append("⚖️ **Obese BMI** is linked to higher charges.")
    if age > 50:
        factors.append("👴 **Age above 50** contributes to higher estimates.")
    if not factors:
        factors.append("✅ Your inputs suggest a relatively lower risk profile.")

    for f in factors:
        st.markdown(f)

st.markdown("---")
st.caption("Model: Linear Regression | Accuracy: ~77% R² | Built with scikit-learn & Streamlit")
