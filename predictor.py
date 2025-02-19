import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
model = joblib.load('XGBoost.pkl')
X_test = pd.read_csv('X_test.csv')
feature_names = ["histology", "Gleason", "Abiraterone", "prostate_RT", "Metastatic_burden", "diabetes", "Metformin","PTEN", "BRCA", "age", "PSA"]
st.title("Parp inhibitors effectively reduce PSA predictors")
histology = st.selectbox("Histology:", options=[0, 1], format_func=lambda x: "Others" if x == 1 else "Adenocarcinoma")
Gleason = st.selectbox("Gleason:", options=[0, 1], format_func=lambda x: "9-10" if x == 1 else "6-8")
Abiraterone = st.selectbox("Abiraterone:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
prostate_RT = st.selectbox("Radical prostatectomy:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Metastatic_burden = st.selectbox("The situation of distant metastasis:", options=[0, 1], format_func=lambda x: "Bone or organ metastasis" if x == 1 else "No or lymph node metastasis")
diabetes = st.selectbox("Diabetes:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Metformin = st.selectbox("Metformin:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PTEN = st.selectbox("PTEN alterations:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
BRCA = st.selectbox("BRCA alterations:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
age = st.selectbox("Age:", options=[0, 1], format_func=lambda x: "Age>70" if x == 1 else "Age≤70")
PSA = st.selectbox("PSA:", options=[0, 1], format_func=lambda x: "PSA>68ng/mL" if x == 1 else "PSA≤68ng/mL")
feature_values = [histology, Gleason, Abiraterone, prostate_RT, Metastatic_burden, diabetes, Metformin,PTEN,BRCA, age, PSA]
features = np.array([feature_values])
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Parp inhibitors are effective, 0: Parp inhibitors are ineffective)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, Parp inhibitors are effective. "
            f"The probability of PARP inhibitors being effective is  {probability:.1f}%. "
             )
    else:
        advice = (
            f"According to our model, Parp inhibitors are ineffective. "
            f"The probability of PARP inhibitors being ineffective is {probability:.1f}%. "
            )
    st.write(advice)
    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    # Display the SHAP force plot for the predicted class
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)        
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Ineffective', 'Effective'],
        mode='classification'
    )
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )
    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)