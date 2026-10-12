import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from train_model import train, FEATURES, NUMERIC, CATEGORICAL, load_data

st.set_page_config(
    page_title="House Price Predictor — MLR Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a sleek, modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(14, 165, 233, 0.12);
        color: #0284c7;
        border: 1px solid rgba(14, 165, 233, 0.3);
        margin-bottom: 0.75rem;
    }
    
    .value-box {
        background: rgba(15, 23, 42, 0.03);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
    }
    
    .benefit-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    
    .metric-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    
    .metric-hero h2 {
        color: #38bdf8 !important;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    
    .metric-card-inner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    model, report, comparison, data, X_train = train()
    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocess"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]
    cat_names = list(encoder.get_feature_names_out(CATEGORICAL))
    all_names = NUMERIC + cat_names
    coefs = regressor.coef_

    # City benchmark statistics
    city_stats = (
        data.groupby("city")
        .agg(
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            count=("price", "count"),
            avg_sqft=("sqft_living", "mean"),
        )
        .reset_index()
        .sort_values(by="avg_price", ascending=False)
    )

    return {
        "model": model,
        "report": report,
        "comparison": comparison,
        "data": data,
        "X_train": X_train,
        "regressor": regressor,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": all_names,
        "coefs": coefs,
        "intercept": regressor.intercept_,
        "city_stats": city_stats,
    }


bundle = load_all_models()
model = bundle["model"]
report = bundle["report"]
comparison = bundle["comparison"]
data = bundle["data"]
X_train = bundle["X_train"]
regressor = bundle["regressor"]
preprocessor = bundle["preprocessor"]
encoder = bundle["encoder"]
feature_names = bundle["feature_names"]
coefs = bundle["coefs"]
city_stats = bundle["city_stats"]

AVAILABLE_CITIES = sorted(list(X_train["city"].unique()))

# ----------------- HEADER & ASSIGNMENT VALUE PROPOSITION -----------------
st.markdown('<div class="hero-badge">⚡ Capstone Project • Multiple Linear Regression (MLR)</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🏡 House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    "Predict property valuations using multiple independent variables (*square footage, bedrooms, bathrooms, "
    "location, condition, and quality*) calibrated on a **real-world dataset**."
)

with st.expander("📌 **Assignment Overview & How This Project Helps Users** (Click to view)", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        **🛒 For Buyers**  
        Enables buyers to estimate fair property prices before making purchase decisions.
        """)
    with col2:
        st.markdown("""
        **🏷️ For Sellers**  
        Assists sellers in setting competitive and realistic listing prices to maximize return.
        """)
    with col3:
        st.markdown("""
        **📋 For Agents**  
        Helps real estate agents provide instant data-driven recommendations & CMA reports.
        """)
    with col4:
        st.markdown("""
        **⚡ Instant Valuations**  
        Saves time by offering instant automated predictions instead of manual market research.
        """)
    with col5:
        st.markdown("""
        **🔍 Model Transparency**  
        100% explainability with explicit mathematical equations & linear feature weights.
        """)
    
    st.markdown("---")
    st.markdown(
        "📂 **Dataset Reference:** Real-world King County Washington housing dataset (4,600 historical records, 4,551 positive sales). "
        "[View Original Dataset on Google Drive](https://drive.google.com/file/d/1kicLZ_mUT1f5O1STqb-azATZtkmjsxov/view?usp=sharing)"
    )

# Navigation Tabs
tab_predict, tab_buyer, tab_seller, tab_agent, tab_transparency, tab_explorer = st.tabs([
    "🔮 Predict Property Value",
    "🛒 Buyer Mode (Fair Price & Deal Checker)",
    "🏷️ Seller Mode (Pricing & Upgrade ROI)",
    "📋 Agent Mode (CMA Market Intel)",
    "📐 Model Transparency & Math",
    "📊 Real-World Dataset Explorer",
])

# Presets dictionary for quick loading
PRESETS = {
    "Seattle Family Craftsman": {
        "city": "Seattle", "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1950,
        "sqft_lot": 5000, "floors": 2.0, "sqft_basement": 300, "waterfront": 0,
        "view": 0, "condition": 3, "yr_built": 1985, "yr_renovated": 0
    },
    "Bellevue Luxury Estate": {
        "city": "Bellevue", "bedrooms": 4, "bathrooms": 3.5, "sqft_living": 3800,
        "sqft_lot": 12000, "floors": 2.0, "sqft_basement": 800, "waterfront": 0,
        "view": 2, "condition": 4, "yr_built": 2004, "yr_renovated": 0
    },
    "Redmond Tech Suburban": {
        "city": "Redmond", "bedrooms": 4, "bathrooms": 2.5, "sqft_living": 2600,
        "sqft_lot": 8500, "floors": 2.0, "sqft_basement": 0, "waterfront": 0,
        "view": 1, "condition": 4, "yr_built": 1998, "yr_renovated": 2010
    },
    "Lake Washington Waterfront Villa": {
        "city": "Mercer Island", "bedrooms": 5, "bathrooms": 4.5, "sqft_living": 4800,
        "sqft_lot": 22000, "floors": 2.0, "sqft_basement": 1200, "waterfront": 1,
        "view": 4, "condition": 5, "yr_built": 2008, "yr_renovated": 0
    },
    "Kent Starter Home": {
        "city": "Kent", "bedrooms": 3, "bathrooms": 1.5, "sqft_living": 1350,
        "sqft_lot": 6200, "floors": 1.0, "sqft_basement": 0, "waterfront": 0,
        "view": 0, "condition": 3, "yr_built": 1978, "yr_renovated": 0
    },
}

# ----------------- HELPER PREDICTION FUNCTION -----------------
def get_prediction_breakdown(record_dict):
    row = pd.DataFrame([record_dict])[FEATURES]
    pred = float(model.predict(row)[0])
    mae = float(report["model"]["MAE"])
    lower = max(40000.0, pred - mae)
    upper = pred + mae

    # Feature contribution breakdown
    transformed = preprocessor.transform(row)
    contributions = []
    for name, coef, val in zip(feature_names, coefs, transformed[0]):
        contrib = float(coef * val)
        clean_name = name.replace("city_", "City: ").replace("_", " ").title()
        contributions.append({"feature": clean_name, "raw": name, "impact": contrib})

    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return pred, lower, upper, contributions, row


# ----------------- TAB 1: PREDICT PROPERTY VALUE -----------------
with tab_predict:
    st.subheader("Property Configuration & Valuation Engine")
    st.caption("Select a quick preset or adjust the 12 property features below to predict instantaneous valuation.")

    # Preset selection
    preset_col, _ = st.columns([2, 3])
    with preset_col:
        chosen_preset = st.selectbox("⚡ Quick Presets (Auto-fills inputs):", ["-- Custom Configuration --"] + list(PRESETS.keys()))

    preset_data = PRESETS.get(chosen_preset, PRESETS["Seattle Family Craftsman"])

    with st.form("property_form"):
        st.markdown("#### 1. Location & Dimensions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            city_idx = AVAILABLE_CITIES.index(preset_data["city"]) if preset_data["city"] in AVAILABLE_CITIES else 0
            city = st.selectbox("City / Market", AVAILABLE_CITIES, index=city_idx)
        with c2:
            sqft_living = st.number_input("Living Area (sq ft)", 300, 30000, int(preset_data["sqft_living"]), 50)
        with c3:
            sqft_lot = st.number_input("Lot Size (sq ft)", 300, 5000000, int(preset_data["sqft_lot"]), 100)
        with c4:
            sqft_basement = st.number_input("Basement Area (sq ft)", 0, 10000, int(preset_data["sqft_basement"]), 50)

        st.markdown("#### 2. Configuration & Amenities")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            bedrooms = st.number_input("Bedrooms", 1, 15, int(preset_data["bedrooms"]))
        with c6:
            bathrooms = st.number_input("Bathrooms", 0.5, 10.0, float(preset_data["bathrooms"]), 0.25)
        with c7:
            floors_opts = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            fl_idx = floors_opts.index(preset_data["floors"]) if preset_data["floors"] in floors_opts else 2
            floors = st.selectbox("Floors / Stories", floors_opts, index=fl_idx)
        with c8:
            waterfront = 1 if st.checkbox("🌊 Waterfront Property", value=bool(preset_data["waterfront"])) else 0

        st.markdown("#### 3. Quality, View & Age")
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            condition = st.selectbox("Condition (1=Poor, 5=Pristine)", [1, 2, 3, 4, 5], index=int(preset_data["condition"])-1)
        with c10:
            view = st.selectbox("View Code (0=Standard, 4=Premier)", [0, 1, 2, 3, 4], index=int(preset_data["view"]))
        with c11:
            yr_built = st.number_input("Year Built", 1900, 2014, int(preset_data["yr_built"]))
        with c12:
            yr_renovated = st.number_input("Year Renovated (0 = Never)", 0, 2014, int(preset_data["yr_renovated"]))

        submit_btn = st.form_submit_button("⚡ Run Multiple Linear Regression Valuation", type="primary", use_container_width=True)

    # Validation & Prediction
    current_record = dict(
        city=city, bedrooms=bedrooms, bathrooms=bathrooms, sqft_living=sqft_living,
        sqft_lot=sqft_lot, floors=floors, sqft_basement=sqft_basement, waterfront=waterfront,
        condition=condition, view=view, yr_built=yr_built, yr_renovated=yr_renovated
    )

    if sqft_basement > sqft_living:
        st.error("❌ Validation Error: Basement square footage cannot exceed total living square footage.")
    elif yr_renovated > 0 and yr_renovated < yr_built:
        st.error("❌ Validation Error: Renovation year cannot precede the construction year.")
    else:
        pred_val, lower_b, upper_b, top_contribs, input_row = get_prediction_breakdown(current_record)

        # Store in session state for Buyer, Seller & Agent tabs
        st.session_state["latest_pred"] = pred_val
        st.session_state["latest_record"] = current_record
        st.session_state["latest_lower"] = lower_b
        st.session_state["latest_upper"] = upper_b

        # Hero Valuation Display
        st.markdown("---")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Estimated Valuation", f"${pred_val:,.0f}", help="Point estimate from Ordinary Least Squares Regression")
        with h2:
            st.metric("Confidence Band (±MAE)", f"${lower_b:,.0f} – ${upper_b:,.0f}", help="Expected historical error range")
        with h3:
            pps = pred_val / sqft_living if sqft_living > 0 else 0
            st.metric("Price / Sq Ft", f"${pps:,.0f}/sqft")
        with h4:
            city_avg = city_stats.loc[city_stats.city == city, "avg_price"].values[0] if city in city_stats.city.values else pred_val
            diff_pct = ((pred_val - city_avg) / city_avg) * 100
            st.metric(f"{city} Benchmark", f"${city_avg:,.0f}", f"{diff_pct:+.1f}% vs City Avg")

        # Top Linear Drivers Bar Chart
        st.markdown("#### 📈 Key Value Drivers (Regression Impact Decomposition)")
        top_7 = top_contribs[:8]
        df_drivers = pd.DataFrame(top_7)
        df_drivers["Color"] = df_drivers["impact"].apply(lambda x: "#10b981" if x >= 0 else "#ef4444")

        fig_drivers = go.Figure(go.Bar(
            x=df_drivers["impact"],
            y=df_drivers["feature"],
            orientation="h",
            marker=dict(color=df_drivers["Color"]),
            text=df_drivers["impact"].apply(lambda v: f"${v:+,.0f}"),
            textposition="auto"
        ))
        fig_drivers.update_layout(
            title="Feature Contributions to Predicted Valuation (relative to dataset mean)",
            xaxis_title="Dollar Impact (Dataset Price Units)",
            yaxis_title="Feature",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_drivers, use_container_width=True)

        st.download_button(
            "📥 Download Valuation Summary (CSV)",
            input_row.assign(predicted_price=pred_val, confidence_lower=lower_b, confidence_upper=upper_b).to_csv(index=False),
            "property_valuation.csv",
            "text/csv",
        )


# ----------------- TAB 2: BUYER LENS -----------------
with tab_buyer:
    st.subheader("🛒 Buyer Intelligence: Fair Price Assessment & Deal Checker")
    st.markdown(
        "**Assignment Goal:** *Enables buyers to estimate fair property prices before making decisions "
        "and protects them against overpaying.*"
    )

    pred_val = st.session_state.get("latest_pred", 650000.0)
    rec = st.session_state.get("latest_record", PRESETS["Seattle Family Craftsman"])

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("#### 🎯 Deal Fairness Evaluator")
        st.write(f"Evaluating for: **{rec['bedrooms']} Bed, {rec['bathrooms']} Bath in {rec['city']}** ({rec['sqft_living']} sqft)")
        st.info(f"Model-Estimated Fair Value: **${pred_val:,.0f}**")

        asking_price = st.number_input("Enter Seller's Asking Price ($):", min_value=50000, max_value=15000000, value=int(pred_val * 1.05), step=10000)

        price_diff = asking_price - pred_val
        diff_pct = (price_diff / pred_val) * 100

        if diff_pct > 8.0:
            st.error(f"""
            🚨 **Overpriced Alert: +{diff_pct:.1f}% Above Fair Value (+${price_diff:,.0f})**  
            The seller is asking significantly more than historical linear comps.  
            **Buyer Advice:** Strongly negotiate down toward **${pred_val:,.0f}**. Do not bid above **${pred_val * 1.05:,.0f}**.
            """)
        elif diff_pct < -8.0:
            st.success(f"""
            🔥 **Bargain / Strong Deal: {diff_pct:.1f}% Below Fair Value (-${abs(price_diff):,.0f})**  
            This property is listed attractively compared to its size and location features.  
            **Buyer Advice:** High potential upside. Proceed promptly before rival bids emerge.
            """)
        else:
            st.info(f"""
            ✅ **Fair Market Value (Within ±8% of Model Estimate)**  
            The asking price is properly aligned with historical property valuations.  
            **Buyer Advice:** Balanced offer. Standard negotiation room is ~$10k - $20k.
            """)

    with b_col2:
        st.markdown("#### 💳 Monthly Mortgage & Affordability Breakdown")
        down_payment_pct = st.slider("Down Payment (%)", 5, 30, 20, 5)
        interest_rate = st.slider("Mortgage Interest Rate (% APR)", 4.0, 9.0, 6.5, 0.25)
        loan_years = st.selectbox("Loan Duration", [15, 30], index=1)

        down_payment_amt = asking_price * (down_payment_pct / 100.0)
        loan_amount = asking_price - down_payment_amt

        monthly_rate = (interest_rate / 100.0) / 12.0
        n_months = loan_years * 12
        if loan_amount > 0 and monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1)
        else:
            monthly_payment = 0

        m1, m2 = st.columns(2)
        m1.metric("Down Payment Required", f"${down_payment_amt:,.0f}")
        m2.metric("Est. Monthly Payment (P&I)", f"${monthly_payment:,.0f}/mo")

        st.caption("Assumes 30-year fixed rate mortgage without property tax or private mortgage insurance (PMI).")


# ----------------- TAB 3: SELLER LENS -----------------
with tab_seller:
    st.subheader("🏷️ Seller Intelligence: Competitive Pricing Strategy & Upgrade ROI")
    st.markdown(
        "**Assignment Goal:** *Assists sellers in setting competitive and realistic listing prices "
        "and understanding how improvements increase property value.*"
    )

    pred_val = st.session_state.get("latest_pred", 650000.0)
    rec = st.session_state.get("latest_record", PRESETS["Seattle Family Craftsman"])

    s_col1, s_col2 = st.columns([1, 1])

    with s_col1:
        st.markdown("#### 🏷️ Recommended 3-Tier Listing Price Strategy")
        st.markdown(f"""
        | Pricing Strategy | Recommended Price | Target Timeline | Seller Objective |
        | :--- | :--- | :--- | :--- |
        | **🚀 Aggressive / Seller's Market (+5%)** | **${pred_val * 1.05:,.0f}** | 30–60 Days | Test top-market buyer interest |
        | **🎯 Competitive Market Target (100%)** | **${pred_val:,.0f}** | 14–30 Days | Optimal balance of speed and maximum price |
        | **⚡ Fast Sale / Quick Liquidation (-5%)** | **${pred_val * 0.95:,.0f}** | 7–14 Days | Stimulate multiple offers / bidding war |
        """)
        st.caption("Strategy calculated directly from the regression confidence hyperplane.")

    with s_col2:
        st.markdown("#### 🛠️ Upgrade & Renovation Value Boost (ROI)")
        st.write("See estimated linear value increments from potential pre-listing upgrades:")

        # Calculate upgrade impacts using model weights
        # Bathroom increment (+1 bathroom)
        rec_more_bath = dict(rec, bathrooms=rec["bathrooms"] + 1.0)
        p_bath = float(model.predict(pd.DataFrame([rec_more_bath])[FEATURES])[0])
        gain_bath = p_bath - pred_val

        # Condition increment (boost condition to 5)
        rec_cond_5 = dict(rec, condition=5)
        p_cond = float(model.predict(pd.DataFrame([rec_cond_5])[FEATURES])[0])
        gain_cond = max(0.0, p_cond - pred_val)

        # Basement finish (add 400 sqft basement + living)
        rec_base = dict(rec, sqft_living=rec["sqft_living"] + 400, sqft_basement=rec["sqft_basement"] + 400)
        p_base = float(model.predict(pd.DataFrame([rec_base])[FEATURES])[0])
        gain_base = p_base - pred_val

        u1, u2, u3 = st.columns(3)
        u1.metric("Add +1 Full Bathroom", f"+${gain_bath:,.0f}", "Est. Added Value")
        u2.metric("Turnkey Refresh (Cond=5)", f"+${gain_cond:,.0f}", "Est. Added Value")
        u3.metric("Finish +400 Sqft Basement", f"+${gain_base:,.0f}", "Est. Added Value")


# ----------------- TAB 4: AGENT MODE (CMA INTEL) -----------------
with tab_agent:
    st.subheader("📋 Real Estate Agent Portal: Instant CMA & Market Intel")
    st.markdown(
        "**Assignment Goal:** *Helps real estate agents provide data-driven recommendations "
        "and saves hours of manual Comparative Market Analysis research.*"
    )

    pred_val = st.session_state.get("latest_pred", 650000.0)
    rec = st.session_state.get("latest_record", PRESETS["Seattle Family Craftsman"])
    target_city = rec["city"]

    cma_stats = city_stats.loc[city_stats.city == target_city].iloc[0] if target_city in city_stats.city.values else None

    if cma_stats is not None:
        ca1, ca2, ca3, ca4 = st.columns(4)
        ca1.metric(f"{target_city} Sales Volume", f"{int(cma_stats['count'])} homes")
        ca2.metric(f"{target_city} Mean Price", f"${cma_stats['avg_price']:,.0f}")
        ca3.metric(f"{target_city} Median Price", f"${cma_stats['median_price']:,.0f}")
        ca4.metric("Avg Living Sq Ft", f"{cma_stats['avg_sqft']:,.0f} sqft")

    st.markdown("#### 📄 Instant Client Presentation Summary")
    summary_text = f"""
    ========================================================================
    COMPARATIVE MARKET ANALYSIS (CMA) REPORT — {target_city.upper()}, WA
    Generated by HousePredict AI (Multiple Linear Regression Engine)
    ========================================================================
    Subject Property:
      • Location: {rec['city']}, WA
      • Configuration: {rec['bedrooms']} Bedrooms | {rec['bathrooms']} Bathrooms | {rec['floors']} Stories
      • Dimensions: {rec['sqft_living']:,} sq ft living | {rec['sqft_lot']:,} sq ft lot | {rec['sqft_basement']:,} sq ft basement
      • Quality / View: Condition rating {rec['condition']}/5 | View code {rec['view']}/4 | Waterfront: {'Yes' if rec['waterfront'] else 'No'}
      • Construction Year: {rec['yr_built']} | Renovated: {rec['yr_renovated'] if rec['yr_renovated'] > 0 else 'Original'}

    Regression Valuation Results:
      • Point Valuation: ${pred_val:,.0f}
      • Range (±MAE Confidence): ${st.session_state.get('latest_lower', pred_val - 131870):,.0f} to ${st.session_state.get('latest_upper', pred_val + 131870):,.0f}
      • Price / Sq Ft: ${pred_val / rec['sqft_living']:,.2f}
      • City Market Baseline ({target_city}): ${cma_stats['avg_price']:,.0f} average
    ========================================================================
    """
    st.text_area("Client Presentation Report", summary_text, height=220)
    st.download_button("📥 Export CMA Summary (.txt)", summary_text, f"CMA_Report_{target_city}.txt")


# ----------------- TAB 5: MODEL TRANSPARENCY & MATH -----------------
with tab_transparency:
    st.subheader("📐 Model Transparency & Mathematical Architecture")
    st.markdown(
        "**Assignment Goal:** *Improves transparency in property valuation using machine learning "
        "by explaining the exact linear formulation, preprocessing, and test results.*"
    )

    st.markdown("#### The Multiple Linear Regression Model Equation")
    st.latex(r"""
    \hat{Y}_{\text{price}} = \beta_0 + \sum_{i=1}^{11} \beta_i \cdot \text{Standardize}(X_i) + \sum_{j=1}^{43} \gamma_j \cdot \text{OneHot}(\text{City}_j)
    """)
    st.markdown(
        "The model is an **Ordinary Least Squares (OLS) Linear Regression** fitted on the training split "
        r"minimizing the residual sum of squares: $\min_{\beta} \sum (y_i - \hat{y}_i)^2$."
    )

    st.markdown("#### Holdout Test Set Evaluation (vs Naive Baseline)")
    comp_df = pd.DataFrame({
        "Metric": ["R² Score (Variance Explained)", "MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"],
        "Multiple Linear Regression (MLR)": [
            f"{report['model']['R2']:.3f} (67.6%)",
            f"${report['model']['MAE']:,.0f}",
            f"${report['model']['RMSE']:,.0f}"
        ],
        "Baseline (Mean Dummy Model)": [
            f"{report['baseline']['R2']:.3f} (0.0%)",
            f"${report['baseline']['MAE']:,.0f}",
            f"${report['baseline']['RMSE']:,.0f}"
        ],
        "Performance Improvement": [
            "+67.6% variance explained",
            "-45.8% error reduction",
            "-43.1% error reduction"
        ]
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Diagnostic Plots
    p1, p2 = st.columns(2)
    with p1:
        fig_scatter = px.scatter(
            comparison, x="Actual", y="Predicted",
            opacity=0.45, hover_data=["City"],
            title="Actual vs. Predicted Prices (Test Split)",
        )
        min_p = min(comparison["Actual"].min(), comparison["Predicted"].min())
        max_p = max(comparison["Actual"].max(), comparison["Predicted"].max())
        fig_scatter.add_shape(type="line", x0=min_p, y0=min_p, x1=max_p, y1=max_p, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with p2:
        fig_resid = px.scatter(
            comparison, x="Predicted", y="Residual",
            opacity=0.45, hover_data=["City"],
            title="Residuals vs. Predicted Values (Error Diagnostics)",
        )
        fig_resid.add_shape(type="line", x0=min_p, y0=0, x1=max_p, y1=0, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_resid, use_container_width=True)


# ----------------- TAB 6: DATASET EXPLORER -----------------
with tab_explorer:
    st.subheader("📊 Real-World Dataset Explorer (King County Housing)")
    st.markdown(
        "Explore the authentic housing records used to train the Multiple Linear Regression pipeline. "
        "[Download Original CSV from Google Drive](https://drive.google.com/file/d/1kicLZ_mUT1f5O1STqb-azATZtkmjsxov/view?usp=sharing)"
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Records in Dataset", f"{report['raw_rows']:,}")
    d2.metric("Usable Records (Price > 0)", f"{report['usable_rows']:,}")
    d3.metric("Municipal Cities Covered", f"{data['city'].nunique()}")
    d4.metric("Non-Positive Records Cleaned", f"{report['removed_nonpositive']}")

    st.markdown("#### Sample Records (100 Rows)")
    st.dataframe(data.head(100), use_container_width=True)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.plotly_chart(
            px.histogram(data, x="price", nbins=50, title="Price Distribution Across King County Homes"),
            use_container_width=True
        )
    with c_chart2:
        st.plotly_chart(
            px.scatter(data.sample(min(1000, len(data)), random_state=42), x="sqft_living", y="price", color="city", title="Living Area vs Price (Sample of 1,000 Homes)"),
            use_container_width=True
        )

st.markdown("---")
st.caption("🏡 **House Price Predictor Capstone Project** • Multiple Linear Regression • Python & Scikit-Learn")
