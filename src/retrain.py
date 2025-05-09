print("RUNNING: src/retrain.py, DATA PATH TEST")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
import logging, joblib, warnings
import mlflow
import mlflow.sklearn
warnings.filterwarnings('ignore')
mlflow.set_experiment('Superstore-Sales-Prediction')
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer

# Ensure logs directory exists
import os
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'model_retraining.log')
# Configure the logger
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger instance
logger = logging.getLogger("model_retraining")
import os
assert os.path.exists("data/Superstore.xlsx"), "File not found: data/Superstore.xlsx. Please ensure the data file exists in the 'data' directory."
df = pd.read_excel("data/Superstore.xlsx",usecols=['Profit','Postal Code','Discount','Quantity','Order Date','Ship Date','Sales'])

for idx in list(df[df['Sales'] > 10500].index):
    df.drop(idx,axis=0,inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'],errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'],errors='coerce')
df['Order Day'] = df['Order Date'].dt.day
df['Order Month'] = df['Order Date'].dt.month
df['Order Weekday'] = df['Order Date'].dt.weekday
df['Ship Day'] = df['Ship Date'].dt.day
# Remove Discount Percentage and Operating Expenses
# Only keep features available at prediction time
features_to_keep = ['Profit', 'Postal Code', 'Discount', 'Order Day', 'Order Month', 'Quantity', 'Ship Day', 'Order Weekday']
df = df[features_to_keep + ['Sales']]
# Drop date columns only if they exist
for col in ['Order Date', 'Ship Date']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

transformer = ColumnTransformer(transformers=[
    ('log_transform', FunctionTransformer(np.log1p), ['Quantity', 'Profit']),
    ('sqrt_transform', FunctionTransformer(np.sqrt), ['Discount'])
], remainder='passthrough')

X = df.drop('Sales', axis=1)
y = df['Sales']
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True,random_state=75)

X_train = transformer.fit_transform(X_train)
X_train = pd.DataFrame(X_train,columns=features)
X_test = transformer.transform(X_test)
X_test = pd.DataFrame(X_test,columns=features)

columns_with_missing_values = X_train.columns[X_train.isnull().any()].tolist()

for col in columns_with_missing_values:
    imputer = SimpleImputer(strategy='median')
    X_train[col] = imputer.fit_transform(X_train[[col]])

columns_with_missing_values = X_test.columns[X_test.isnull().any()].tolist()

for col in columns_with_missing_values:
    imputer = SimpleImputer(strategy='median')
    X_test[col] = imputer.fit_transform(X_test[[col]])


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_model(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    logger.info("Evaluation metrics - R2 Score: %.2f, Mean Absolute Error: %.2f, Mean Squared Error: %.2f, Mean Absolute Percentage Error: %.2f, Root Mean Squared Error: %.2f", r2, mae, mse, mape, rmse)
    return model, r2

with mlflow.start_run():
    model, baseline_r2 = train_and_evaluate_model(ExtraTreesRegressor())
    mlflow.log_param('model_type', 'ExtraTreesRegressor')
    mlflow.log_metric('baseline_r2', baseline_r2)

# SPEED-UP: Reduce parameter grid, folds, and set n_iter, n_jobs
param_grid = {
    'n_estimators': [200, 500],  # fewer options
    'criterion': ['squared_error', 'absolute_error'],  # fewer options
    'max_features': ['auto', 'sqrt'],  # fewer options
    'bootstrap': [True],  # single option
    'oob_score': [True, False],
    'max_samples': [0.75, 1]  # fewer options
}

# Reduce cv folds and n_iter, enable parallel jobs
cv_folds = 3
n_iter_search = 4

# Optionally subsample data for quick test (comment out for full run)
# X_train_sample = X_train[:500]
# y_train_sample = y_train[:500]
# grid_et = RandomizedSearchCV(estimator=ExtraTreesRegressor(), param_distributions=param_grid, cv=cv_folds, n_iter=n_iter_search, verbose=2, n_jobs=-1)
# optimized_model, optimized_r2 = train_and_evaluate_model(grid_et.fit(X_train_sample, y_train_sample).best_estimator_)
grid_et = RandomizedSearchCV(estimator=ExtraTreesRegressor(), param_distributions=param_grid, cv=cv_folds, n_iter=n_iter_search, verbose=2, n_jobs=-1)
optimized_model, optimized_r2 = train_and_evaluate_model(grid_et)

mlflow.log_metric('optimized_r2', optimized_r2)

if baseline_r2 < optimized_r2:
    model = optimized_model

avg_cv_scores = cross_val_score(model,X_test,y_test,scoring='r2',cv=5,verbose=2)
mean_score = round(np.mean(avg_cv_scores),2) * 100
logger.info(f"Mean Cross Validation Performance of Extra Trees Regressor: {mean_score}%")
mlflow.log_metric('mean_cv_score', mean_score)

pipeline = Pipeline(steps=[
    ('transformer',transformer),
    ('scaler',scaler),
    ('model',model)
])

mlflow.sklearn.log_model(pipeline, 'model_pipeline')
mlflow.log_artifact('models/pipeline.pkl')
logging.shutdown()
joblib.dump(pipeline, 'models/pipeline.pkl')
joblib.dump(pipeline, 'pipeline.pkl')  # Save also in root for MLflow log_artifact compatibility