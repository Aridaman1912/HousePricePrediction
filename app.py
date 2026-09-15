import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")
st.subheader("Enter House Details")
st.write("Enter the details of the house to predict its price.")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7
    )

with col2:
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=6000,
        value=1710,
        step=50
    )

col1, col2 = st.columns(2)

with col1:
    garage_cars = st.number_input(
        "Garage Cars",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

with col2:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=856,
        step=50
    )

col1, col2 = st.columns(2)

with col1:
    first_flr_sf = st.number_input(
        "First Floor Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=856,
        step=50
    )

with col2:
    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

col1, col2 = st.columns(2)

with col1:
    tot_rms_abv_grd = st.number_input(
        "Total Rooms Above Ground",
        min_value=1,
        max_value=15,
        value=7,
        step=1
    )

with col2:
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2003,
        step=1
    )

col1, col2 = st.columns(2)

with col1:
    year_remod_add = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2003,
        step=1
    )

if year_remod_add < year_built:
    st.warning("Year Remodeled cannot be earlier than Year Built.")
    
with col2:
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1500,
        value=548,
        step=50
    )

col1, col2 = st.columns(2)

with col1:
    overall_cond = st.slider(
        "Overall Condition",
        min_value=1,
        max_value=10,
        value=5
    )

with col2:
   defaults = joblib.load("house_defaults.pkl")

neighborhood = st.selectbox(
    "Neighborhood",
    options=[
        "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
        "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
        "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill",
        "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer",
        "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"
    ]
)

house_input = pd.DataFrame([defaults])

house_input["OverallQual"] = overall_qual
house_input["GrLivArea"] = gr_liv_area
house_input["GarageCars"] = garage_cars
house_input["TotalBsmtSF"] = total_bsmt_sf
house_input["1stFlrSF"] = first_flr_sf
house_input["FullBath"] = full_bath
house_input["TotRmsAbvGrd"] = tot_rms_abv_grd
house_input["YearBuilt"] = year_built
house_input["YearRemodAdd"] = year_remod_add
house_input["GarageArea"] = garage_area
house_input["OverallCond"] = overall_cond
house_input["Neighborhood"] = neighborhood

st.subheader("Prediction")

if st.button("Predict House Price"):
    if year_remod_add < year_built:
        st.error("Please enter a valid remodeling year.")
    else:
        prediction_log = model.predict(house_input)
        prediction = np.expm1(prediction_log[0])

        st.metric(
            "Estimated House Price",
            f"${prediction:,.2f}"
        )

st.divider()

st.subheader("About the Project")

st.write(
    "This application uses a Gradient Boosting Regression model "
    "to predict house prices based on important property features."
)

st.write(
    "The model was trained on the Kaggle House Prices dataset "
    "and uses log-transformed house prices for better prediction performance."
)