import streamlit as st
import numpy as np
import pandas as pd
from src.models.load_model import load_model

st.set_page_config(page_title="Used Bike Price Predictor", page_icon="🏍️")

st.title("🏍️ Used Bike Price Prediction")
st.markdown("#### Predict the selling price of a used motorcycle using ML models built from scratch.")

# ============================================
# Load name encoding file
# ============================================
df_enc = pd.read_csv("data/name_encoding.csv")

bike_names = df_enc["name"].tolist()

# ============================================
# Model Options
# ============================================
models = {
    "KNN Regression": "knn",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting"
}

st.subheader("🔧 Select Model")
model_choice = st.selectbox("Choose ML Model", list(models.keys()))
model_path = f"models/{models[model_choice]}.pkl"

model = load_model(model_path)   # FIXED → scaler removed


# ============================================
# User Input Fields
# ============================================
st.subheader("🏍️ Enter Bike Details")

col1, col2 = st.columns(2)

with col1:
    bike_name = st.selectbox("Bike Model", bike_names)
    year = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, value=2019)
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])

with col2:
    owner = st.selectbox("Owner Type", ["1st", "2nd", "3rd", "4th"])
    km = st.number_input("Kilometers Driven", min_value=0, value=350)
    showroom = st.number_input("Ex-Showroom Price (₹)", min_value=0, value=70000)


# ============================================
# Convert categorical values (ENCODING)
# ============================================
encoded_name = df_enc[df_enc["name"] == bike_name]["encoded_name"].values[0]
seller_type_num = 0 if seller_type == "Individual" else 1
owner_map = {"1st": 0, "2nd": 1, "3rd": 2, "4th": 3}
owner_num = owner_map[owner]

# 6 features (NO scaler, NO log_km)
X = np.array([[encoded_name, year, seller_type_num, owner_num, km, showroom]])


# ============================================
# Prediction
# ============================================
st.subheader("📊 Prediction Result")

if st.button("Predict Price"):

    pred = model.predict(X)[0]    # FIXED → no scaling needed

    st.success(f"💰 **Predicted Selling Price:** ₹ {round(pred, 2)}")

    st.markdown("---")
    st.markdown("### Model Info")
    st.write(f"**Selected Model:** {model_choice}")
    st.write("**Final Features Used (6):**")
    st.write("- encoded_name\n- year\n- seller_type\n- owner\n- km_driven\n- ex_showroom_price")
