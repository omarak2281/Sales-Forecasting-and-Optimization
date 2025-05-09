# Superstore Sales Forecasting & Optimization

[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/omarak2281/Sales-Forecasting-and-Optimization)

## Project Overview

The **Superstore Sales Forecasting & Optimization** project is a comprehensive, production-ready solution for predicting and optimizing sales in a retail environment. It covers the entire data science lifecycle:

- **End-to-end workflow:** Data analysis, feature engineering, model training, evaluation, deployment, and monitoring
- **MLOps integration:** MLflow for experiment tracking, model management, and automated retraining
- **Interactive dashboard:** Real-time KPIs, visualizations, and user-driven predictions via a Flask web app
- **Professional documentation:** Detailed markdown reports and stakeholder-ready presentations

This repository is designed for both technical and business users, providing actionable insights, robust code, and clear documentation.

## Directory Structure

```
SUPERSTORE-SALES-PREDICTION/
├── data/                # Raw data (Superstore.xlsx)
├── models/              # Trained model pipeline (pipeline.pkl)
├── notebooks/           # EDA & development notebooks
├── reports/             # Professional markdown reports & presentation
├── src/                 # Flask app, retraining scripts
├── templates/           # Dashboard & MLOps HTML templates
├── logs/                # Training and retraining logs
├── mlruns/              # MLflow experiment data
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization
└── README.md            # Project overview (this file)
```

## 📊 Reports & Presentation

All professional documentation is in the `reports/` directory:
- `data_analysis_report.md`: Exploratory Data Analysis (EDA)
- `data_engineering_report.md`: Data cleaning & feature engineering
- `modeling_report.md`: Model selection, training, and MLflow tracking
- `mlops_deployment_report.md`: MLOps, dashboard, retraining, deployment
- `dashboard_visualization_report.md`: UI & visualization features
- `final_comprehensive_report.md`: Executive summary and synthesis
- `presentation.md`: Stakeholder presentation (Markdown, ready for slides)

## 🚀 How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the dashboard:**
   ```bash
   python src/app.py
   ```
   Visit [http://localhost:8000](http://localhost:8000) in your browser.
3. **Review reports:**
   - Read all markdown reports in `reports/` for technical and business insights.
   - Use `presentation.md` for stakeholder meetings (convert to slides as needed).
4. **MLOps & retraining:**
   - Track experiments and models with MLflow (see `mlruns/` and dashboard MLOps page).
   - Retrain models using `src/retrain.py`.

## 🔑 Key Features
- **Full data science workflow:** From raw data to deployed model and dashboard
- **Interactive web app:** Real-time predictions and analytics
- **MLOps best practices:** Experiment tracking, versioning, and retraining
- **Professional documentation:** For both technical and non-technical audiences

## 🤝 Contributing
Contributions and suggestions are welcome! Please open an issue or pull request.

---

*For full technical and business context, see the markdown reports and presentation in the `reports/` directory. This project is structured for professional, stakeholder-facing delivery.*

| Model                  | R2 Score | RMSE      | MAPE     | Training Time (s) |
|------------------------|----------|-----------|----------|-------------------|
| ExtraTreesRegressor    | 0.976    | 79.75     | 0.014    | 1.89              |
| GradientBoosting       | 0.969    | 91.02     | 0.202    | 262.95            |
| RandomForest           | 0.964    | 97.96     | 0.020    | 339.90            |

## User Interface

The web interface allows users to input features and receive sales predictions. Below is a screenshot of the UI:

![UI Screenshot](images/ui_screenshot.jpeg)

## Prediction Example

A sample prediction using the following inputs:
- Profit: $3846.91
- Discount Percentage: 79.42
- Postal Code: 5937
- Discount: 0.5
- Order Day: 8
- Order Month: 5
- Quantity: 9
- Operating Expenses: $6293.46
- Ship Day: 29
- Order Weekday: 3

**Predicted Sales**: $6367.70

Below is a screenshot of the prediction output:

![Prediction Screenshot](images/prediction_screenshot.png)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/omarak2281/Sales-Forecasting-and-Optimization.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sales-Forecasting-and-Optimization
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python src/app.py
   ```
5. Access the web interface at `http://localhost:5000` in your browser.

## Usage

1. Open the web interface in your browser.
2. Enter the required features (e.g., Profit, Discount, Quantity).
3. Click "Predict Sales" to view the predicted sales value.


## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. Ensure your code follows the project's style and includes appropriate tests.

## Acknowledgments

- The dataset is sourced from the Superstore dataset, commonly used for sales prediction tasks.
- Built with the support of open-source libraries like scikit-learn, pandas, and Flask.