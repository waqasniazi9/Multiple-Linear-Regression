import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from train_model import train, FEATURES

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")
@st.cache_resource
def get_model():
    return train()
model, report, comparison, data, training = get_model()
st.title("🏡 House Price Predictor")
st.caption("Capstone-02 • Multiple Linear Regression • Historical Washington, USA housing data")
st.warning("Educational historical-price estimator, not a current-market appraisal. Currency is not explicitly documented in the CSV; prices are shown in dataset units.")
predict_tab, explore_tab, performance_tab = st.tabs(["Predict a price", "Explore data", "Model performance"])
with predict_tab:
    st.subheader("Enter property details")
    with st.form("property"):
        left, middle, right = st.columns(3)
        with left:
            city = st.selectbox("City", sorted(training.city.unique()))
            bedrooms = st.number_input("Bedrooms", 0, 20, 3)
            bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, 0.25)
            sqft_living = st.number_input("Living area (sq ft)", 100, 30000, 1800, 50)
        with middle:
            sqft_lot = st.number_input("Lot area (sq ft)", 100, 5000000, 6000, 100)
            floors = st.selectbox("Floors", sorted(data.floors.unique()), index=list(sorted(data.floors.unique())).index(1.0))
            sqft_basement = st.number_input("Basement area (sq ft)", 0, 30000, 0, 50)
            waterfront = int(st.checkbox("Waterfront property"))
        with right:
            condition = st.selectbox("Condition code", sorted(data.condition.unique()), index=2)
            view = st.selectbox("View code", sorted(data.view.unique()))
            yr_built = st.number_input("Year built", 1800, 2014, 1990)
            yr_renovated = st.number_input("Year renovated (0 = none)", 0, 2014, 0)
        submitted = st.form_submit_button("Estimate house price", type="primary")
    if submitted:
        if sqft_basement > sqft_living:
            st.error("Basement area cannot exceed total living area.")
        elif yr_renovated and yr_renovated < yr_built:
            st.error("Renovation year cannot precede the construction year.")
        else:
            record = dict(city=city, bedrooms=bedrooms, bathrooms=bathrooms, sqft_living=sqft_living, sqft_lot=sqft_lot, floors=floors, sqft_basement=sqft_basement, waterfront=waterfront, condition=condition, view=view, yr_built=yr_built, yr_renovated=yr_renovated)
            row = pd.DataFrame([record])[FEATURES]
            estimate = float(model.predict(row)[0])
            if estimate <= 0:
                st.error("The model produced a nonpositive price. This input is not reliably supported; no valid valuation is available.")
            else:
                st.metric("Estimated historical price (dataset units)", f"{estimate:,.0f}")
                st.caption("This is a point estimate. The test MAE is not a prediction interval.")
            outside = [col for col in FEATURES if col != "city" and not training[col].min() <= row.iloc[0][col] <= training[col].max()]
            if outside:
                st.warning("Outside the training range: " + ", ".join(outside))
            st.dataframe(row, hide_index=True, use_container_width=True)
            st.download_button("Download this estimate", row.assign(predicted_price=estimate).to_csv(index=False), "house_estimate.csv", "text/csv")
with explore_tab:
    a,b,c = st.columns(3)
    a.metric("Original records", report["raw_rows"])
    b.metric("Usable records", report["usable_rows"])
    c.metric("Cities", data.city.nunique())
    st.write("Removed 49 nonpositive-price records. No positive-price outliers were trimmed.")
    st.dataframe(data.head(100), use_container_width=True)
    st.plotly_chart(px.histogram(data, x="price", nbins=60, title="Historical price distribution (includes extreme values)"), use_container_width=True)
    st.plotly_chart(px.scatter(data, x="sqft_living", y="price", color="city", title="Living area and price", opacity=0.5), use_container_width=True)
    st.caption("sqft-prefixed area columns are interpreted as square feet. View and condition are dataset codes, not independently verified ratings.")
with performance_tab:
    st.subheader("Untouched 20% test-set results")
    a,b,c = st.columns(3)
    a.metric("MAE (price units)", f'{report["model"]["MAE"]:,.0f}')
    b.metric("RMSE (price units)", f'{report["model"]["RMSE"]:,.0f}')
    c.metric("R²", f'{report["model"]["R2"]:.3f}')
    st.dataframe(pd.DataFrame({"Linear regression": report["model"], "Training-mean baseline": report["baseline"]}).T, use_container_width=True)
    fig = px.scatter(comparison, x="Actual", y="Predicted", hover_data=["City"], title="Actual vs. predicted prices")
    low = min(comparison.Actual.min(), comparison.Predicted.min())
    high = max(comparison.Actual.max(), comparison.Predicted.max())
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.scatter(comparison, x="Predicted", y="Residual", title="Residuals vs. predicted prices"), use_container_width=True)
    st.write(f'Training rows: {report["train_rows"]}; test rows: {report["test_rows"]}; random seed: 42.')
    st.info("This random split evaluates held-out historical records, not future-market forecasting. Extreme prices remain in the data and can substantially affect squared-error metrics. Rare cities absent from training are encoded as the reference city in test evaluation; no separate location effect is learned for them.")
    st.download_button("Download test predictions", comparison.to_csv(index=False), "test_predictions.csv", "text/csv")
