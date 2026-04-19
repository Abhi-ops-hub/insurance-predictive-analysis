"""
Run this script once to train and save your model + scaler.
Make sure insurance.csv is in the same folder.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("insurance.csv")

# ── Clean & encode ─────────────────────────────────────────
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)

df_cleaned['sex']    = df_cleaned['sex'].map({"male": 0, "female": 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no": 0, "yes": 1})
df_cleaned.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)

df_cleaned = pd.get_dummies(df_cleaned, columns=["region"], drop_first=True)
df_cleaned = df_cleaned.astype(int)

# ── BMI category ───────────────────────────────────────────
df_cleaned["bmi_category"] = pd.cut(
    df_cleaned["bmi"],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=["underweight", "normal", "overweight", "obese"]
)
df_cleaned = pd.get_dummies(df_cleaned, columns=["bmi_category"], drop_first=True)
df_cleaned = df_cleaned.astype(int)

# ── Scale numeric features ─────────────────────────────────
scaler = StandardScaler()
cols = ['age', 'bmi', 'children']
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

# ── Select final features ──────────────────────────────────
final_df = df_cleaned[[
    'age', 'is_female', 'bmi', 'children',
    'is_smoker', 'charges',
    'region_southeast', 'bmi_category_obese'
]]

X = final_df.drop('charges', axis=1)
y = final_df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# ── Train model ────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Save model & scaler ────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl saved successfully!")
