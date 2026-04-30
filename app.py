import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Iris Classifier", layout="wide")

st.title("🌸 Iris Flower Classification")
st.markdown("Production-level ML App ")

# ------------------------------
# LOAD MODEL FILES
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# ------------------------------
# SIDEBAR INPUT
# ------------------------------
st.sidebar.header("Input Features")

sl = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.5)
sw = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
pl = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
pw = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.2)

# ------------------------------
# PREDICTION
# ------------------------------
input_data = pd.DataFrame([[sl, sw, pl, pw]],
                          columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

result = encoder.inverse_transform(prediction)[0]

# ------------------------------
# DISPLAY
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    st.success(f"🌸 {result}")

with col2:
    st.subheader("Probabilities")
    prob_df = pd.DataFrame(prediction_proba, columns=encoder.classes_)
    st.bar_chart(prob_df.T)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Model loaded using joblib | No retraining | Production Ready")