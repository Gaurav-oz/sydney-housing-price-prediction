import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('sydney_rf_model.joblib')

bundle = load_model()
pipeline = bundle['pipeline']

st.title("🏠 Sydney Housing Price Predictor")
st.write(
    "Predicts a sale price for a property in **Paddington**, **Marrickville**, or **Blacktown**, "
    "based on a Random Forest model trained on 114 manually collected sold-property records "
    "from realestate.com.au and domain.com.au."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.selectbox("Suburb", bundle['suburbs'])
        property_type = st.selectbox("Property type", bundle['property_types'])
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
        parking_spaces = st.number_input("Parking spaces", min_value=0, max_value=10, value=1, step=1)
        land_size_known = st.checkbox("I know the land size", value=True)
        if land_size_known:
            land_size_sqm = st.number_input("Land size (sqm)", min_value=0.0, max_value=2000.0, value=300.0, step=10.0)
        else:
            land_size_sqm = None
            st.caption("Land size will be estimated from similar properties of this type.")

    submitted = st.form_submit_button("Predict sale price", use_container_width=True)

if submitted:
    land_missing = 1 if land_size_sqm is None else 0
    land_value = land_size_sqm
    if land_missing:
        land_value = bundle['land_by_type'].get(property_type, bundle['land_overall'])
        if pd.isna(land_value):
            land_value = bundle['land_overall']

    row = pd.DataFrame([{
        'suburb': suburb,
        'property_type': property_type,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'parking_spaces': parking_spaces,
        'land_size_sqm': land_value,
        'land_size_missing': land_missing,
        'parking_missing': 0,
        'bathrooms_missing': 0,
    }])

    prediction = pipeline.predict(row)[0]

    st.success(f"### Predicted sale price: ${prediction:,.0f}")

    st.caption(
        f"This model's typical error on unseen properties is around "
        f"\\${bundle['test_mae']:,.0f} (mean) / \\${bundle['test_median_abs_err']:,.0f} (median), "
        f"based on a held-out test set of 23 properties. Treat this figure as a rough estimate, "
        f"not a valuation, especially for Paddington or for properties with unusual characteristics "
        f"for their suburb."
    )

    if land_missing:
        st.info(
            f"Land size was not provided, so it was estimated as the median for "
            f"**{property_type}** properties (~{land_value:,.0f} sqm). Predictions with an estimated "
            f"land size are generally less reliable."
        )

st.divider()
with st.expander("About this model"):
    st.write(f"""
    - **Model:** Random Forest Regressor (200 trees), chosen after comparing against Linear Regression
      and a Decision Tree via 5-fold cross-validation.
    - **Training data:** 114 sold properties (50 Paddington, 32 Marrickville, 32 Blacktown), manually
      collected from realestate.com.au and domain.com.au sold listings.
    - **Held-out test performance:** R² = {bundle['test_r2']:.3f}, MAE = \${bundle['test_mae']:,.0f},
      RMSE = \${bundle['test_rmse']:,.0f} (23 properties never seen during training or tuning).
    - **Known limitations:** does not capture renovation quality, heritage character, exact
      street-level location, or floorplan,  see the project report for a full discussion of where
      this model struggles most (expensive/heterogeneous Paddington properties especially).
    """)
