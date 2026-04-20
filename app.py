import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = Path("artifacts/model.joblib")

if not MODEL_PATH.exists():
    st.error("❌ Model file not found. Train and save model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Try to get expected feature names
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    st.warning("⚠ Model has no feature names. Using fallback manual list.")
    expected_features = None


# ----------------------------
# PREPROCESSING FUNCTION
# ----------------------------
def preprocess_input(df):
    mapping_yes_no = {"Yes": 1, "No": 0}
    mapping_gender = {"Male": 1, "Female": 0}
    mapping_location = {"Urban": 1, "Rural": 0}

    if "cellphone_access" in df:
        df["cellphone_access"] = df["cellphone_access"].map(mapping_yes_no)

    if "gender_of_respondent" in df:
        df["gender_of_respondent"] = df["gender_of_respondent"].map(mapping_gender)

    if "location_type" in df:
        df["location_type"] = df["location_type"].map(mapping_location)

    return df


# ----------------------------
# UI
# ----------------------------
st.title("🌍 Financial Inclusion Prediction App")

st.write("Enter user/business details to predict financial inclusion risk.")

country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
age = st.number_input("Age", 18, 100, 30)
cellphone = st.selectbox("Has Cellphone", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
location = st.selectbox("Location Type", ["Urban", "Rural"])
household_size = st.number_input("Household Size", 1, 20, 3)


# ----------------------------
# BUILD INPUT DATAFRAME
# ----------------------------
input_dict = {
    "country": country,
    "age_of_respondent": age,
    "cellphone_access": cellphone,
    "gender_of_respondent": gender,
    "location_type": location,
    "household_size": household_size
}

input_df = pd.DataFrame([input_dict])


# ----------------------------
# PREPROCESS INPUT (🔥 FIX ADDED HERE)
# ----------------------------
input_df = preprocess_input(input_df)


# ----------------------------
# ALIGN FEATURES
# ----------------------------
if expected_features is not None:
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_features]


# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]

        st.success("Prediction Complete 🎯")

        if prediction == 1:
            st.write("🟢 Likely FINANCIALLY INCLUDED")
        else:
            st.write("🔴 Likely FINANCIALLY EXCLUDED")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")