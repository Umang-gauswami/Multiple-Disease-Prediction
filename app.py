import streamlit as st
import pickle
import pandas as pd
import joblib
import numpy as np

# Load saved models
heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")
cancer_model = joblib.load("models/cancer_model.pkl")
cancer_label_map = joblib.load("models/cancer_label_map.pkl")  # if you need labels

st.set_page_config(page_title="Health Disease Prediction", layout="wide")

st.title("🩺 Disease Prediction System")
st.sidebar.header("Select Model")
choice = st.sidebar.radio("Choose a Disease Model:",
                          ["Heart Disease", "Diabetes", "Cancer"])


# ---------------- HEART DISEASE ----------------
if choice == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age",min_value= 1,max_value= 120,step= 1)

    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ])
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp_val = cp_map[cp]

    # Resting Blood Pressure
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250)

    # Serum Cholesterol
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)

    # Fasting Blood Sugar
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    fbs_val = 1 if fbs == "True" else 0

    # Resting ECG
    restecg = st.selectbox("Resting ECG Results", [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ])
    restecg_map = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }
    restecg_val = restecg_map[restecg]

    # Max Heart Rate
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250)

    # Exercise Induced Angina
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    exang_val = 1 if exang == "Yes" else 0

    # Oldpeak
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)

    # Slope
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [
        "Upsloping",
        "Flat",
        "Downsloping"
    ])
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_val = slope_map[slope]

    # Number of Major Vessels
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3,4])

    # Thalassemia Test
    thal = st.selectbox("Thalassemia Test Result", [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ])
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal_val = thal_map[thal]

    # Collect Features
    features = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                thalach, exang_val, oldpeak, slope_val, ca, thal_val]

    # if st.button("Predict Heart Disease"):
    #     prediction = heart_model.predict(features)[0]
    #     st.success("⚠️ Disease Detected" if prediction == 1 else "✅ No Disease")

    if st.button("Predict Heart Disease"):
        prediction = heart_model.predict([features])
        if prediction[0] == 1:
            st.error("⚠️ Disease Detected")
        else:
            st.success("✅ No Disease")


# ---------------- DIABETES ----------------
elif choice == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")

    # Categorical inputs as readable dropdowns (strings, like in training)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "current", "former", "No Info", "ever", "not current"]
    )

    # Numeric / binary inputs
    age = st.number_input("Age", 1, 120, 30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", 50, 300, 120, step=1)

    if st.button("Predict Diabetes"):
        # Build a ONE-ROW DATAFRAME with the EXACT column names used in training
        X_df = pd.DataFrame([{
            "gender": gender,                                   # string
            "age": int(age),
            "hypertension": 1 if hypertension == "Yes" else 0,  # 0/1
            "heart_disease": 1 if heart_disease == "Yes" else 0,# 0/1
            "smoking_history": smoking_history,                  # string
            "bmi": float(bmi),
            "HbA1c_level": float(HbA1c_level),
            "blood_glucose_level": int(blood_glucose_level),
        }])

        # Predict directly with the pipeline (it will OHE + scale internally)
        pred = diabetes_model.predict(X_df)[0]
        label = {0: "✅ No Diabetes", 1: "⚠️ Diabetes Detected"}[int(pred)]
        (st.error if pred == 1 else st.success)(label)


# ---------------- CANCER ----------------
elif choice == "Cancer":
    st.subheader("🧬 Breast Cancer Prediction")

    # Collect all 30 numeric features from the user
    radius_mean = st.number_input("Radius Mean", 0.0, 50.0, 14.0, step=0.1)
    texture_mean = st.number_input("Texture Mean", 0.0, 50.0, 19.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0, step=0.1)
    area_mean = st.number_input("Area Mean", 0.0, 3000.0, 600.0, step=1.0)
    smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.1, step=0.01)
    compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.1, step=0.01)
    concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.1, step=0.01)
    concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.05, step=0.01)
    symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.2, step=0.01)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.05, step=0.01)

    radius_se = st.number_input("Radius SE", 0.0, 5.0, 0.5, step=0.01)
    texture_se = st.number_input("Texture SE", 0.0, 5.0, 0.5, step=0.01)
    perimeter_se = st.number_input("Perimeter SE", 0.0, 10.0, 1.0, step=0.1)
    area_se = st.number_input("Area SE", 0.0, 1000.0, 40.0, step=1.0)
    smoothness_se = st.number_input("Smoothness SE", 0.0, 1.0, 0.01, step=0.001)
    compactness_se = st.number_input("Compactness SE", 0.0, 1.0, 0.02, step=0.001)
    concavity_se = st.number_input("Concavity SE", 0.0, 1.0, 0.03, step=0.001)
    concave_points_se = st.number_input("Concave Points SE", 0.0, 1.0, 0.01, step=0.001)
    symmetry_se = st.number_input("Symmetry SE", 0.0, 1.0, 0.02, step=0.001)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, 1.0, 0.01, step=0.001)

    radius_worst = st.number_input("Radius Worst", 0.0, 50.0, 16.0, step=0.1)
    texture_worst = st.number_input("Texture Worst", 0.0, 50.0, 25.0, step=0.1)
    perimeter_worst = st.number_input("Perimeter Worst", 0.0, 200.0, 105.0, step=0.1)
    area_worst = st.number_input("Area Worst", 0.0, 3000.0, 700.0, step=1.0)
    smoothness_worst = st.number_input("Smoothness Worst", 0.0, 1.0, 0.15, step=0.01)
    compactness_worst = st.number_input("Compactness Worst", 0.0, 1.0, 0.25, step=0.01)
    concavity_worst = st.number_input("Concavity Worst", 0.0, 1.0, 0.3, step=0.01)
    concave_points_worst = st.number_input("Concave Points Worst", 0.0, 1.0, 0.15, step=0.01)
    symmetry_worst = st.number_input("Symmetry Worst", 0.0, 1.0, 0.3, step=0.01)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.0, 1.0, 0.1, step=0.01)

    if st.button("Predict Cancer"):
        # Build DataFrame with ALL 30 features (same order as training)
        X_df = pd.DataFrame([{
            "radius_mean": radius_mean,
            "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean,
            "area_mean": area_mean,
            "smoothness_mean": smoothness_mean,
            "compactness_mean": compactness_mean,
            "concavity_mean": concavity_mean,
            "concave_points_mean": concave_points_mean,
            "symmetry_mean": symmetry_mean,
            "fractal_dimension_mean": fractal_dimension_mean,
            "radius_se": radius_se,
            "texture_se": texture_se,
            "perimeter_se": perimeter_se,
            "area_se": area_se,
            "smoothness_se": smoothness_se,
            "compactness_se": compactness_se,
            "concavity_se": concavity_se,
            "concave_points_se": concave_points_se,
            "symmetry_se": symmetry_se,
            "fractal_dimension_se": fractal_dimension_se,
            "radius_worst": radius_worst,
            "texture_worst": texture_worst,
            "perimeter_worst": perimeter_worst,
            "area_worst": area_worst,
            "smoothness_worst": smoothness_worst,
            "compactness_worst": compactness_worst,
            "concavity_worst": concavity_worst,
            "concave_points_worst": concave_points_worst,
            "symmetry_worst": symmetry_worst,
            "fractal_dimension_worst": fractal_dimension_worst,
        }])

        pred = cancer_model.predict(X_df)[0]
        label = {0: "✅ Benign (Non-Cancerous)", 1: "⚠️ Malignant (Cancerous)"}[int(pred)]
        (st.error if pred == 1 else st.success)(label)