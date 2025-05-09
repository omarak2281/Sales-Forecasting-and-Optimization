# Data Engineering & Preprocessing Report

## 1. Data Cleaning
- **Handling missing values:**
  - Missing values in date columns (Order Date, Ship Date) are parsed as `NaT` and handled during feature engineering.
  - For numerical features with missing values after transformation, median imputation is applied using `SimpleImputer` (see `retrain.py`).
- **Removing duplicates:**
  - Although duplicates are not explicitly mentioned as a major issue, it is recommended to implement duplicate removal as a precautionary measure to ensure data integrity.
- **Outlier treatment:**
  - Outliers in the `Sales` column are removed by dropping any row where `Sales > 10,500` (see `retrain.py`). Consider implementing a more robust outlier detection method, such as the IQR method, for future improvements.

## 2. Feature Engineering
- **Date features:**
  - Extracted from `Order Date` and `Ship Date`:
    - `Order Day`, `Order Month`, `Order Weekday`, `Ship Day`.
- **Encoding categorical variables:**
  - Although the final model uses only numerical features for prediction, consider exploring encoding methods for categorical variables to potentially improve model performance.
- **Scaling numerical features:**
  - Features are scaled using `StandardScaler` after transformation.
- **Feature selection:**
  - Only features available at prediction time are retained: `Profit`, `Postal Code`, `Discount`, `Order Day`, `Order Month`, `Quantity`, `Ship Day`, `Order Weekday`. Consider implementing recursive feature elimination (RFE) to further optimize feature selection.

## 3. Transformation Pipeline
- Overview of the pipeline steps (see `models/pipeline.pkl`):
  - **Transformer:**
    - Applies log transformation to `Quantity` and `Profit`.
    - Applies square root transformation to `Discount`.
    - Uses `ColumnTransformer` to combine these with passthrough for other features.
  - **Scaler:**
    - `StandardScaler` is applied to all features after transformation.
  - **Model:**
    - The main model is `ExtraTreesRegressor`, with hyperparameter tuning using `RandomizedSearchCV`. Consider exploring other models, such as `GradientBoostingRegressor` or `LGBMRegressor`, to potentially improve performance.

## 4. Data Splitting
- **Train/test split:**
  - Data is split into train and test sets with a 75/25 ratio (`test_size=0.25`, `random_state=75`).
- **Validation strategy:**
  - Cross-validation is performed with 3 folds during hyperparameter search and 5 folds for final model validation (`cross_val_score`). Consider increasing the number of folds for more robust validation results.

## Recommendations for Future Improvements

- Implement duplicate removal to ensure data integrity.
- Explore more robust outlier detection methods, such as the IQR method.
- Investigate encoding methods for categorical variables to potentially improve model performance.
- Implement recursive feature elimination (RFE) to further optimize feature selection.
- Explore other models, such as `GradientBoostingRegressor` or `LGBMRegressor`, to potentially improve performance.
- Increase the number of folds for cross-validation to obtain more robust validation results.

---

*For code details, see the notebook and `src/` code.*
