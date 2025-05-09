# MLOps & Deployment Report

## 1. MLflow Integration
- MLflow is used for experiment tracking, logging parameters, metrics, and model artifacts.
- Each training and prediction run is logged under the experiment name `SuperstoreSalesPrediction`.
- Artifacts such as the trained pipeline (`pipeline.pkl`) are stored and versioned for reproducibility.

## 2. Dashboard
- The Flask-based dashboard (see `src/app.py`) provides a user interface for making predictions and monitoring key metrics.
- Real-time metrics, visualizations, and logs are displayed, including model performance and recent predictions.
- MLflow history can be accessed from the dashboard for transparency.

## 3. Retraining Pipeline
- **Automated retraining:**
  - The retraining process is scripted in `src/retrain.py`, which can be scheduled or triggered manually to update the model using the latest data.
  - The pipeline includes data cleaning, feature engineering, model training, hyperparameter tuning, and saving of the updated model.
- **Logging and monitoring:**
  - All retraining runs are logged in MLflow, including metrics and parameters.
  - Logs are also written to `logs/model_retraining.log` for auditing and debugging.

## 4. Deployment
- **Local deployment:**
  - The Flask app serves predictions locally and can be run on any machine with Python and required dependencies.
- **Model serving:**
  - The trained model pipeline is loaded at runtime and used for inference via web forms or API endpoints.
  - The `/predict` endpoint handles user input, applies preprocessing, and returns predictions.
- **Cloud & container deployment:**
  - The app can be containerized using Docker for consistent deployment across environments.
  - Optionally, the dashboard can be deployed to cloud platforms (e.g., Azure, AWS, GCP) or static frontends (e.g., Netlify) with appropriate backend support.

---

*See `src/app.py`, `logs/`, and MLflow UI for details.*
