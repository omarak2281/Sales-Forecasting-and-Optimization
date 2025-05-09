from flask import Flask, render_template, request, send_file, redirect, url_for
import joblib, warnings, os
import mlflow
from datetime import datetime
mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))
mlflow.set_experiment("SuperstoreSalesPrediction")
warnings.filterwarnings('ignore')
import pandas as pd
import tempfile

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load pipeline and components globally for prediction
pipeline = joblib.load(os.path.join('models', 'pipeline.pkl'))
transformer = pipeline.named_steps['transformer']
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']

# Flask filter for readable datetime
@app.template_filter('datetimeformat')
def datetimeformat(value):
    try:
        return datetime.fromtimestamp(int(value) / 1000).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return value

# Helper to get logs
def get_logs():
    log_path = os.path.join('logs', 'model_performance_info.log')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return f.read()
    return None

@app.route("/", methods=['GET'])
def home():
    logs = get_logs()
    # Clear prediction and error on fresh load
    return render_template('index.html', logs=logs, prediction=None, error=None)

@app.route("/predict", methods=['POST'])
def predict():
    logs = get_logs()
    try:
        from datetime import datetime
        # Parse dates and compute derived features
        order_date = request.form['order_date']
        ship_date = request.form['ship_date']
        order_date_dt = datetime.strptime(order_date, '%Y-%m-%d')
        ship_date_dt = datetime.strptime(ship_date, '%Y-%m-%d')
        order_day = order_date_dt.day
        order_month = order_date_dt.month
        order_weekday = order_date_dt.weekday()
        ship_day = ship_date_dt.day
        # Get other fields
        postal_code = float(request.form['postal_code'])
        discount = float(request.form['discount'])
        quantity = float(request.form['quantity'])
        profit = float(request.form['profit'])
        # Prepare input dict as in Streamlit logic
        try:
            required_cols = list(transformer.feature_names_in_)
        except Exception:
            required_cols = ['Profit', 'Postal Code', 'Discount', 'Order Day', 'Order Month', 'Quantity', 'Ship Day', 'Order Weekday']
        input_dict = {
            'Profit': profit,
            'Postal Code': postal_code,
            'Discount': discount,
            'Order Day': order_day,
            'Order Month': order_month,
            'Quantity': quantity,
            'Ship Day': ship_day,
            'Order Weekday': order_weekday
        }
        input_df = pd.DataFrame([input_dict], columns=required_cols)
        transformed = transformer.transform(input_df)
        import numpy as np
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(transformed, columns=input_df.columns)
        scaled = scaler.transform(transformed)
        error_msg = None
        prediction = None
        with mlflow.start_run(run_name="prediction"):
            try:
                prediction = model.predict(scaled)[0]
                prediction_fmt = f"{prediction:.2f}"
                mlflow.log_params(input_dict)
                mlflow.log_metric("prediction", prediction)
                mlflow.set_tag("status", "success")
                return render_template('index.html', prediction=prediction_fmt, logs=logs, error=None)
            except Exception as e:
                error_msg = str(e)
                mlflow.log_params(input_dict)
                mlflow.set_tag("status", "error")
                mlflow.set_tag("error_msg", error_msg)
                return render_template('index.html', prediction=None, logs=logs, error=error_msg)
    except Exception as e:
        return render_template('index.html', error=str(e), logs=logs, prediction=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def get_mlflow_history(n=10):
    from mlflow.tracking import MlflowClient
    import os
    client = MlflowClient(tracking_uri="file:///%s" % os.path.abspath("mlruns"))
    experiment = client.get_experiment_by_name("SuperstoreSalesPrediction")
    print("Experiment:", experiment)
    if not experiment:
        print("No experiment found!")
        return []
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=n)
    print(f"Fetched {len(runs)} runs from MLflow.")
    run_summaries = []
    for run in runs:
        print("Run:", run)
        run_summaries.append({
            "run_id": run.info.run_id,
            "start_time": run.info.start_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags
        })
    print("MLflow runs found:", run_summaries)
    return run_summaries

@app.route('/mlops')
def mlops():
    logs = get_logs()
    mlflow_history = get_mlflow_history(10)
    return render_template('mlops.html', logs=logs, mlflow_history=mlflow_history)



from flask import jsonify

@app.route('/api/dashboard_data')
def api_dashboard_data():
    import numpy as np
    segment = request.args.get('segment', 'Consumer')
    # Cache DataFrame for performance
    global superstore_df
    if 'superstore_df' not in globals():
        superstore_df = pd.read_excel("data/Superstore.xlsx")
        # Parse dates
        superstore_df['Order Date'] = pd.to_datetime(superstore_df['Order Date'], errors='coerce')
        superstore_df['Ship Date'] = pd.to_datetime(superstore_df['Ship Date'], errors='coerce')
        superstore_df['Order Year'] = superstore_df['Order Date'].dt.year
    df = superstore_df.copy()
    # Filter by segment
    df = df[df['Segment'] == segment]
    # KPIs (YTD = current year)
    current_year = df['Order Year'].max()
    prev_year = current_year - 1
    ytd = df[df['Order Year'] == current_year]
    pytd = df[df['Order Year'] == prev_year]
    kpis = {
        'sales': {
            'value': round(ytd['Sales'].sum() / 1e6, 2), 'unit': 'M',
            'trend': round((ytd['Sales'].sum() - pytd['Sales'].sum()) / (pytd['Sales'].sum() + 1e-6) * 100, 2) if pytd['Sales'].sum() else 0
        },
        'profit': {
            'value': round(ytd['Profit'].sum() / 1e6, 2), 'unit': 'M',
            'trend': round((ytd['Profit'].sum() - pytd['Profit'].sum()) / (pytd['Profit'].sum() + 1e-6) * 100, 2) if pytd['Profit'].sum() else 0
        },
        'quantity': {
            'value': round(ytd['Quantity'].sum() / 1e3, 2), 'unit': 'K',
            'trend': round((ytd['Quantity'].sum() - pytd['Quantity'].sum()) / (pytd['Quantity'].sum() + 1e-6) * 100, 2) if pytd['Quantity'].sum() else 0
        },
        'profit_margin': {
            'value': round(100 * ytd['Profit'].sum() / (ytd['Sales'].sum() + 1e-6), 2), 'unit': '%',
            'trend': round(100 * ((ytd['Profit'].sum() / (ytd['Sales'].sum() + 1e-6)) - (pytd['Profit'].sum() / (pytd['Sales'].sum() + 1e-6))), 2) if pytd['Sales'].sum() else 0
        }
    }
    # Category Table
    cats = []
    for cat in df['Category'].unique():
        ytd_cat = ytd[ytd['Category'] == cat]
        pytd_cat = pytd[pytd['Category'] == cat]
        yoy = (ytd_cat['Sales'].sum() - pytd_cat['Sales'].sum()) / (pytd_cat['Sales'].sum() + 1e-6) * 100 if pytd_cat['Sales'].sum() else 0
        trend = 'up' if yoy >= 0 else 'down'
        cats.append({
            'category': cat,
            'ytd': round(ytd_cat['Sales'].sum(), 2),
            'pytd': round(pytd_cat['Sales'].sum(), 2),
            'yoy': round(yoy, 2),
            'trend': trend
        })
    # Top/Bottom Products
    prod_sales = ytd.groupby('Product Name')['Sales'].sum().sort_values(ascending=False)
    top_products = [{'name': n, 'sales': round(s, 2)} for n, s in prod_sales.head(5).items()]
    bottom_products = [{'name': n, 'sales': round(s, 2)} for n, s in prod_sales.tail(5).items()]
    # Region Sales
    region = ytd.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    region_list = [{'name': n, 'sales': round(s / 1e6, 2)} for n, s in region.items()]
    # Shipping Sales
    shipping = ytd.groupby('Ship Mode')['Sales'].sum().sort_values(ascending=False)
    shipping_list = [{'name': n, 'sales': round(s / 1e6, 2)} for n, s in shipping.items()]
    # States (for map)
    states = ytd.groupby(['State', 'Region'])['Sales'].sum().reset_index()
    states_list = [{'state': row['State'], 'sales': round(row['Sales'], 2), 'region': row['Region']} for _, row in states.iterrows()]
    data = {
        'kpis': kpis,
        'category_table': cats,
        'top_products': top_products,
        'bottom_products': bottom_products,
        'region': region_list,
        'shipping': shipping_list,
        'states': states_list
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8000, debug=True)