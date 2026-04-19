import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="🏥", layout="centered")

# ── Train model from CSV (cached so it only runs once) ─────
@st.cache_resource
def train_model():
    df = pd.read_csv("insurance.csv")
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)

    df_cleaned['sex']    = df_cleaned['sex'].map({"male": 0, "female": 1})
    df_cleaned['smoker'] = df_cleaned['smoker'].map({"no": 0, "yes": 1})
    df_cleaned.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)

    df_cleaned = pd.get_dummies(df_cleaned, columns=["region"], drop_first=True)
    df_cleaned = df_cleaned.astype(int)

    df_cleaned["bmi_category"] = pd.cut(
        df_cleaned["bmi"],
        bins=[0, 18.5, 24.9, 29.9, float('inf')],
        labels=["underweight", "normal", "overweight", "obese"]
    )
    df_cleaned = pd.get_dummies(df_cleaned, columns=["bmi_category"], drop_first=True)
    df_cleaned = df_cleaned.astype(int)

    scaler = StandardScaler()
    cols = ['age', 'bmi', 'children']
    df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

    final_df = df_cleaned[[
        'age', 'is_female', 'bmi', 'children',
        'is_smoker', 'charges', 'region_southeast', 'bmi_category_obese'
    ]]

    X = final_df.drop('charges', axis=1)
    y = final_df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler

model, scaler = train_model()

# ── UI ─────────────────────────────────────────────────────
st.title("🏥 Insurance Charge Predictor")
st.markdown("Fill in your details below to estimate your annual insurance charges.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex    = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker?", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest", "North", "West","South","East"])

st.divider()

def preprocess(age, bmi, children, sex, smoker, region):
    scaled = scaler.transform([[age, bmi, children]])[0]
    age_s, bmi_s, children_s = scaled

    is_female        = 1 if sex == "Female" else 0
    is_smoker        = 1 if smoker == "Yes" else 0
    region_southeast = 1 if region == "Southeast" else 0
    bmi_obese        = 1 if bmi > 29.9 else 0

    return np.array([[age_s, is_female, bmi_s, children_s,
                      is_smoker, region_southeast, bmi_obese]])

if st.button("💰 Predict My Insurance Charges", use_container_width=True):
    features   = preprocess(age, bmi, children, sex, smoker, region)
    prediction = model.predict(features)[0]

    st.success(f"### Estimated Annual Insurance Charge: **${prediction:,.2f}**")

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
st.caption("Model: Linear Regression | ~77% R² Accuracy | Built with scikit-learn & Streamlit")
