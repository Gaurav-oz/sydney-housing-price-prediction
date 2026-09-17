"""
Part 5: Final model for deployment.

Step 1: Evaluate the chosen model (Random Forest) on the genuinely untouched test set
for the first time in this project - this is the real, final check on how well Part 3's
CV-based model selection generalizes to data the model has never influenced in any way
(not even through imputation medians).

Step 2: Retrain on all 114 rows (train + test combined) for the actual deployed model,
since there's no reason to withhold 20% of the data from the production model once
evaluation is done. This is a standard, defensible practice and is called out explicitly
in the project reflection as a deliberate trade-off.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('sydney_housing_final.csv')
df = df.drop(columns=['agent_description_snippet'])
df['land_size_missing'] = df['land_size_sqm'].isna().astype(int)
df['parking_missing'] = df['parking_spaces'].isna().astype(int)
df['bathrooms_missing'] = df['bathrooms'].isna().astype(int)

feature_cols = ['suburb', 'property_type', 'bedrooms', 'bathrooms', 'parking_spaces',
                'land_size_sqm', 'land_size_missing', 'parking_missing', 'bathrooms_missing']
X = df[feature_cols].copy()
y = df['sale_price']

categorical_features = ['suburb', 'property_type']
numeric_features = ['bedrooms', 'bathrooms', 'parking_spaces', 'land_size_sqm',
                     'land_size_missing', 'parking_missing', 'bathrooms_missing']


def make_imputer(X_fit):
    land_by_type = X_fit.groupby('property_type')['land_size_sqm'].median()
    land_overall = X_fit['land_size_sqm'].median()
    parking_by_type = X_fit.groupby('property_type')['parking_spaces'].median()
    bath_by_type = X_fit.groupby('property_type')['bathrooms'].median()

    def apply_imputation(X):
        X = X.copy()
        X['land_size_sqm'] = X['land_size_sqm'].fillna(X['property_type'].map(land_by_type))
        X['land_size_sqm'] = X['land_size_sqm'].fillna(land_overall)
        mask_studio = X['property_type'] == 'Studio'
        X.loc[mask_studio, 'parking_spaces'] = X.loc[mask_studio, 'parking_spaces'].fillna(0)
        X['parking_spaces'] = X['parking_spaces'].fillna(X['property_type'].map(parking_by_type))
        X['bathrooms'] = X['bathrooms'].fillna(X['property_type'].map(bath_by_type))
        return X
    return apply_imputation, dict(land_by_type=land_by_type, land_overall=land_overall,
                                   parking_by_type=parking_by_type, bath_by_type=bath_by_type)


def make_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features),
    ])
    return Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=200))])


#  Step 1: genuine held-out test evaluation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

impute_train, _ = make_imputer(X_train)
X_train_imp = impute_train(X_train)
X_test_imp = impute_train(X_test)  # medians fit on train only, applied to test - no leakage

pipe = make_pipeline()
pipe.fit(X_train_imp, y_train)
test_preds = pipe.predict(X_test_imp)

test_r2 = r2_score(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_median_abs_err = np.median(np.abs(y_test.values - test_preds))

print("=== Genuine held-out test set evaluation (23 properties, touched for the first time) ===")
print(f"Test R^2: {test_r2:.3f}")
print(f"Test MAE: {test_mae:,.0f}")
print(f"Test RMSE: {test_rmse:,.0f}")
print(f"Test median abs error: {test_median_abs_err:,.0f}")
print()
print("For comparison, Part 3's 5-fold CV on the training set gave:")
print("  CV R^2: 0.775, CV MAE: $335,630")
print()

#  Step 2: retrain on all 114 rows for deployment 
impute_full, _ = make_imputer(X)
X_full_imp = impute_full(X)

deploy_pipe = make_pipeline()
deploy_pipe.fit(X_full_imp, y)

# Save everything the app needs: the fitted pipeline, and the imputation medians fit on
# the full dataset (so the app can impute a missing land_size_sqm on a new property using
# the same logic the model was trained with).
land_by_type = X.groupby('property_type')['land_size_sqm'].median()
land_overall = X['land_size_sqm'].median()
parking_by_type = X.groupby('property_type')['parking_spaces'].median()
bath_by_type = X.groupby('property_type')['bathrooms'].median()

joblib.dump({
    'pipeline': deploy_pipe,
    'land_by_type': land_by_type,
    'land_overall': land_overall,
    'parking_by_type': parking_by_type,
    'bath_by_type': bath_by_type,
    'suburbs': sorted(df['suburb'].unique().tolist()),
    'property_types': sorted(df['property_type'].unique().tolist()),
    'test_r2': test_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'test_median_abs_err': test_median_abs_err,
}, 'sydney_rf_model.joblib')

print("Saved deployment model to sydney_rf_model.joblib")
print(f"Trained on all {len(df)} properties.")
