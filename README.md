# House Price Predictor — Capstone-02

A Multiple Linear Regression Streamlit app using the supplied data.csv.

## Run locally (Python 3.11 or 3.12 recommended)

```sh
python -m venv .venv
```
Activate on Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Then:
```sh
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
The app trains and caches the same deterministic pipeline at startup, avoiding a dependency on a version-specific saved pickle. train_model.py optionally saves a fitted pipeline for experimentation.

## Dataset and cleaning
Uploaded assignment data.csv: 4600 rows and 18 columns; dates May 2–July 10, 2014; USA/Washington addresses. No missing values or exact duplicate rows in this file. Remove 49 records with price <= 0, leaving 4551. Do not remove positive price outliers. The CSV contains no explicit currency metadata; the app uses dataset price units, not PKR or a confirmed USD label. The sqft area fields are interpreted as square feet.

## Model
Use bedrooms, bathrooms, living area, lot area, floors, waterfront, view, condition, basement area, construction year, renovation year, and city. Numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding; ordinary least-squares LinearRegression. Fit preprocessing only on training rows. Test split 20%, random_state=42. The app uses the evaluated training-only model, not a separately refitted full-data model.

Excluded: price from inputs; street (high-cardinality address); country (constant); date (scope is historical, not temporal forecasting); statezip (city chosen as location representation); sqft_above (redundant with living and basement area). Raw renovation year retained; 0 is the dataset's no-renovation marker.

## Evaluation
See artifacts/metrics.json and artifacts/test_predictions.csv for actual results. MAE and RMSE are price-unit errors. R² is not an accuracy percentage. Mean-price DummyRegressor provides a baseline. No cross-validation or external validation was performed. High-value anomalies remain unverified and can strongly influence the model. Some test cities are absent from training and use the reference-category encoding; app city choices are limited to training cities.

## Deployment
Push app.py, train_model.py, requirements.txt, and data/data.csv to a GitHub repository (along with documentation). On Streamlit Community Cloud choose that repository, branch main, and app.py. Use Python 3.11 or 3.12. Verify all tabs and make a prediction after deployment. Confirm dataset redistribution permission before publishing it in a public repo; original licensing was not verified.

## Submission
Record a 30–60-second demonstration: enter features, predict, show performance. Publish a LinkedIn post with the screenshot/video, deployed app link, and GitHub link. Submit the LinkedIn post URL. Do not claim guaranteed fair prices or current-market validity.

## Execution status
- [x] Model training pipeline verified (Scikit-Learn OLS, R² = 0.676, MAE = 131,870).
- [x] Streamlit web application (`app.py`) tested and verified locally across all tabs and prediction flows.
- [x] Alternative full-stack web UI suite (`server.py` + `static/`) tested and verified.
- [x] LinkedIn showcase assets generated (`linkedin_assets/`) and draft post written (`LINKEDIN_POST.txt`).
- [ ] Push to GitHub and deploy to Streamlit Community Cloud.
- [ ] Publish LinkedIn post and submit final project URL.
