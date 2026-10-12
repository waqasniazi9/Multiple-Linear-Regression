# 🏡 House Price Predictor — Multiple Linear Regression (MLR)

An end-to-end Machine Learning web application predicting residential property prices using Multiple Linear Regression (MLR) calibrated on a real-world housing dataset.

- 🌐 **Live Streamlit App:** [https://multiple-linear-regression-c3bpnkriislfpaqvtzqddv.streamlit.app/](https://multiple-linear-regression-c3bpnkriislfpaqvtzqddv.streamlit.app/)
- 💻 **GitHub Repository:** [https://github.com/waqasniazi9/Multiple-Linear-Regression](https://github.com/waqasniazi9/Multiple-Linear-Regression)
- 📂 **Real-World Dataset:** [Google Drive Dataset Link](https://drive.google.com/file/d/1kicLZ_mUT1f5O1STqb-azATZtkmjsxov/view?usp=sharing)

---

## 🎯 How This Project Helps Users
1. **Enables buyers to estimate fair property prices before making decisions** — Provides data-driven valuations and a Deal Fairness Checker (evaluating whether a seller's asking price is a bargain, fair market, or overpriced).
2. **Assists sellers in setting competitive and realistic listing prices** — Offers a 3-tier listing strategy (Aggressive, Competitive, Fast Sale) and an Upgrade ROI Estimator to quantify value gains from renovations.
3. **Helps real estate agents provide data-driven recommendations** — Generates instant Comparative Market Analysis (CMA) summary reports and neighborhood benchmarks saving hours of manual research.
4. **Saves time by offering instant predictions instead of manual market research** — Features 1-click quick presets and rapid calculation across 12 independent property variables.
5. **Improves transparency in property valuation using machine learning** — Fully explainable Ordinary Least Squares (OLS) regression formula with standardized coefficients and feature decomposition.

---

## 📊 Model & Performance Metrics

The model is trained on King County, Washington historical housing records (4,600 rows raw; 4,551 usable rows after cleaning non-positive values) using an 80/20 train/test split (`random_state=42`).

### 12 Independent Variables Used:
- **Numerical (11):** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_basement`, `yr_built`, `yr_renovated`
- **Categorical (1):** `city` (one-hot encoded with reference category drop)

### Evaluation on Untouched 20% Holdout Test Set:
| Evaluation Metric | Multiple Linear Regression (MLR) | Training-Mean Dummy Baseline | Improvement |
| :--- | :--- | :--- | :--- |
| **R² Score** | **0.676** (67.6% variance explained) | -0.000 (0.0%) | **+67.6% Explained Variance** |
| **MAE** | **$131,870** | $243,548 | **-45.8% Error Reduction** |
| **RMSE** | **$219,399** | $385,712 | **-43.1% Error Reduction** |

---

## 🚀 Run Locally

```bash
# 1. Clone repository
git clone https://github.com/waqasniazi9/Multiple-Linear-Regression.git
cd Multiple-Linear-Regression

# 2. Setup virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Run model training & evaluation
python train_model.py

# 5. Launch Streamlit application
python -m streamlit run app.py
```

---

## 📱 Application Modules
- **🔮 Predict Property Value:** Interactive parameter controls, quick presets, point estimates, $\pm\text{MAE}$ confidence bands, and feature contribution charts.
- **🛒 Buyer Mode:** Asking price deal fairness checker, overpricing alerts, and 30-year mortgage affordability calculator.
- **🏷️ Seller Mode:** 3-tier listing price recommendations and renovation ROI estimator.
- **📋 Agent Mode:** Instant Comparative Market Analysis (CMA) generator with exportable reports.
- **📐 Model Transparency & Math:** Explicit mathematical formulation, OLS loss minimization, baseline comparison, and residual diagnostic plots.
- **📊 Dataset Explorer:** Searchable dataset browser with interactive Plotly distribution charts.

---

## 🏆 Project Deliverables
- [x] Machine Learning Pipeline (`train_model.py`) with Scikit-Learn ColumnTransformer and OLS Regression.
- [x] Interactive Streamlit Web Application (`app.py`).
- [x] Alternative Full-Stack Web Suite (`server.py` + `static/index.html`).
- [x] Streamlit Cloud Live Deployment: [https://multiple-linear-regression-c3bpnkriislfpaqvtzqddv.streamlit.app/](https://multiple-linear-regression-c3bpnkriislfpaqvtzqddv.streamlit.app/)
- [x] GitHub Repository: [https://github.com/waqasniazi9/Multiple-Linear-Regression](https://github.com/waqasniazi9/Multiple-Linear-Regression)
- [x] LinkedIn Post Draft & Assets: `LINKEDIN_POST.txt` and `linkedin_assets/`.
