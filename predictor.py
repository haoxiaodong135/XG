import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

model = joblib.load('XGBoost.pkl')
feature_names = ["histology", "Gleason", "Abiraterone", "prostate_RT", "Metastatic_burden", 
                 "diabetes", "Metformin", "PTEN", "BRCA", "age", "PSA"]  # 确保顺序正确

st.title("Parp inhibitors effectively reduce PSA predictors")

# 使用正确的选项并确保变量名匹配
histology = st.selectbox("Histology:", options=[0, 1], format_func=lambda x: "Other" if x == 1 else "Adenocarcinoma")
Gleason = st.selectbox("Gleason:", options=[0, 1], format_func=lambda x: "9-10" if x == 1 else "6-8")
Abiraterone = st.selectbox("Abiraterone:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
prostate_RT = st.selectbox("Radical prostatectomy:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Metastatic_burden = st.selectbox("Distant metastasis:", options=[0, 1], format_func=lambda x: "Bone/Organ" if x == 1 else "Other")
diabetes = st.selectbox("Diabetes:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Metformin = st.selectbox("Metformin:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PTEN = st.selectbox("PTEN alterations:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
BRCA = st.selectbox("BRCA alterations:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
age = st.selectbox("Age >70:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PSA = st.selectbox("PSA >68ng/mL:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

feature_values = [histology, Gleason, Abiraterone, prostate_RT, Metastatic_burden, 
                  diabetes, Metformin, PTEN, BRCA, age, PSA]

if st.button("Predict"):
    features = pd.DataFrame([feature_values], columns=feature_names)
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    st.write(f"**Predicted Class:** {predicted_class} (1: Effective, 0: Ineffective)")
    st.write(f"**Probabilities:** [Class 0: {predicted_proba[0]:.2f}, Class 1: {predicted_proba[1]:.2f}]")
    
    # 生成正确的建议
    probability = predicted_proba[1] * 100  # 显示有效概率
    advice = f"Parp inhibitors are {'effective' if predicted_class ==1 else 'ineffective'} with {probability:.1f}% probability."
    st.write(advice)
    
    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    plt.figure()
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())
    plt.clf()