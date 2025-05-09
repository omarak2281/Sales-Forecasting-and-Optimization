# Modeling Report

## 1. Model Selection
- **Candidate models considered:**
  - ExtraTreesRegressor (primary)
  - RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor, LinearRegression, Ridge, Lasso, KNeighborsRegressor, SVR, DecisionTreeRegressor, and others (explored in notebook)
- **Criteria for selection:**
  - Model performance (R² score, MAE, RMSE)
  - Robustness to outliers and feature scaling
  - Interpretability and feature importance
  - Computational efficiency for retraining and deployment

## 2. Training & Evaluation
- **Training process:**
  - Data split into train/test sets (75/25)
  - Feature engineering and scaling performed in a pipeline
  - Baseline model: ExtraTreesRegressor trained on all features
- **Hyperparameter tuning:**
  - RandomizedSearchCV with a reduced parameter grid (n_iter=4, cv=3) for efficiency
  - Parameters tuned: n_estimators, criterion, max_features, bootstrap, oob_score, max_samples
- **Metrics used:**
  - R² score (primary)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Mean Absolute Percentage Error (MAPE)
  - Root Mean Squared Error (RMSE)
- **Best model:**
  - Optimized ExtraTreesRegressor (selected if it outperformed baseline)

## 3. MLflow Tracking
- **Experiment name:** SuperstoreSalesPrediction
- **Tracked parameters:**
  - Model type, hyperparameters, input features
- **Tracked metrics:**
  - Baseline R², optimized R², mean CV score, MAE, MSE, RMSE, MAPE
- **Model artifacts:**
  - Full pipeline (transformer, scaler, model) saved as `pipeline.pkl` and logged to MLflow

## 4. Results
- **Final performance on test set:**
  - R² score: ~0.89 (cross-validation mean)
  - MAE: ~85.0
  - RMSE: ~170.0
  - (Refer to MLflow logs for exact metrics)
- **Feature importance:**
  - Top features: Profit, Quantity, Discount, Order Day, Postal Code

## 5. Model Limitations and Potential Improvements
- **Limitations:**
  - Model assumes linear relationships between features and target variable
  - May not capture complex interactions between features
  - Limited by quality and quantity of training data
- **Potential improvements:**
  - Incorporating non-linear relationships using polynomial or interaction terms
  - Exploring ensemble methods to combine multiple models
  - Collecting and incorporating additional relevant features
  - Using techniques to handle class imbalance or outliers

---

*For full experiment details, see MLflow UI and `src/app.py`.*
