# Sydney Housing Price Prediction and Decision Support System

Mini project for SIT307. Predicts sale prices for properties in three Sydney suburbs (Paddington, Marrickville, Blacktown) using a manually collected dataset of 114 sold properties, and deploys the final model in a small Streamlit app.

## Repo contents

| File | Description |
|---|---|
| `sydney_housing_final.csv` | The collected dataset, 114 sold properties across Paddington, Marrickville, and Blacktown, manually gathered from realestate.com.au and domain.com.au sold listings. Columns: suburb, address, sale_price, sale_date, property_type, bedrooms, bathrooms, parking_spaces, land_size_sqm, agent_description_snippet, listing_url. |
| `sydney_housing_part2.ipynb` | Part 2: data understanding and feature engineering. Price distributions, suburb comparisons, outlier detection, missing value handling, correlation analysis. |
| `sydney_housing_part3.ipynb` | Part 3: model development and evaluation. Trains and compares Linear Regression, Decision Tree, and Random Forest with 5 fold cross validation, diagnoses under/overfitting, picks a final model. |
| `sydney_housing_part4.ipynb` | Part 4: investigating prediction failures. Out of fold error analysis, the five worst individual predictions, discussion of model limitations. |
| `build_final_model.py` | Trains the final Random Forest pipeline, evaluates it once on the held out test set, then retrains on the full dataset and saves it as `sydney_rf_model.joblib` for the app to use. |
| `app.py` | The Streamlit web app. Loads `sydney_rf_model.joblib` and predicts a sale price from user entered property details. |
| `sydney_rf_model.joblib` | The saved, trained model pipeline used by `app.py`. |
| `sydney_housing_final_report.pdf` | The full project report covering Parts 1 to 5. |

## How to run the notebooks

1. Install the requirements:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. Make sure `sydney_housing_final.csv` is in the same folder as the notebooks.
3. Run the notebooks in order: `sydney_housing_part2.ipynb`, then `sydney_housing_part3.ipynb`, then `sydney_housing_part4.ipynb`. Each one rebuilds the cleaned dataset from the raw CSV itself, so they do not depend on each others saved output, but the order matches how the analysis builds up in the report.

## How to run the app

1. Install the requirements:
   ```
   pip install streamlit pandas scikit-learn joblib
   ```
2. Make sure `sydney_rf_model.joblib` is in the same folder as `app.py`. If you want to retrain it yourself instead of using the saved copy, run `python build_final_model.py` first (needs `sydney_housing_final.csv` in the same folder).
3. Run:
   ```
   streamlit run app.py
   ```
4. It opens automatically in your browser, normally at `http://localhost:8501`. Fill in the suburb, property type, bedrooms, bathrooms, parking, and land size, then click Predict sale price.

## Model summary

Random Forest was the best performing model, with a 5 fold cross validated MAE of $335,630 and R2 of 0.775, ahead of a Decision Tree ($343,194 MAE, R2 0.762) and well ahead of Linear Regression, whose cross validated R2 collapsed to -8.01 due to instability on one high leverage property rather than uniformly poor performance. On a genuinely held out test set (used once, for the final evaluation), Random Forest scored R2 = 0.767 and MAE around $342,000. Full detail, including where the model struggles most (expensive, heterogeneous properties in Paddington), is in the PDF report and in Parts 3 and 4 of the notebooks.
