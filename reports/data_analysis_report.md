# Data Analysis Report

## Executive Summary

This report provides an overview of the exploratory data analysis performed on the Superstore dataset. The key findings include:

* The dataset contains 9,994 records with 21 features, covering the period from 2011 to 2015.
* The data is of high quality, with minimal missing values and no duplicates.
* The top-selling categories are Office Supplies, Furniture, and Technology, while the West and Central regions show the highest profits.
* Sales and profits generally increase year-over-year, with seasonal peaks in Q4.
* Sales and Profit are strongly positively correlated, while Discount has a negative correlation with Profit.

## Exploratory Data Analysis (EDA)

This report summarizes the exploratory data analysis performed on the Superstore dataset, as documented in `notebooks/Superstore_Sales_Prediction.ipynb`.

### 1. Dataset Overview
- **Rows/Records:** 9,994
- **Columns/Features:** 21
- **Period Covered:** 2011–2015 (based on Order Date and Ship Date columns)
- **Target Variable:** Sales

### 2. Data Quality
- **Missing Values:** Minimal; all columns are nearly complete. Date columns are parsed as datetime and missing values are rare.
- **Duplicates:** Not a major concern; the dataset is primarily unique transactions.
- **Outliers:** Outliers in Sales are identified and removed for modeling (> $10,500).

### 3. Key Features
- Order Date, Ship Date, Segment, Country, City, State, Region, Category, Sub-Category, Sales, Quantity, Discount, Profit, Shipping Cost, etc.

### 4. Descriptive Statistics
- **Sales:** Range from $0.44 to $22,638.48, mean ≈ $229.86
- **Profit:** Range from -$6,424.92 to $8,349.00, mean ≈ $28.66
- **Quantity:** Range from 1 to 14, mean ≈ 3.79
- **Discount:** Range from 0 to 0.8, mean ≈ 0.156

### 5. Insights
- **Top-selling categories:** Office Supplies, followed by Furniture and Technology.
- **Most profitable regions:** West and Central regions show the highest profits.
- **Sales/Profit trends over time:** Sales and profits generally increase year-over-year, with seasonal peaks in Q4.
- **Correlations:** Sales and Profit are strongly positively correlated. Discount has a negative correlation with Profit.

### 6. Visualizations
- Key plots include:
  - Sales by Category
  - Profit by Region
  - Time series of Sales and Profit
  - Correlation heatmap
  - Distribution plots for Sales, Profit, Discount

### 7. Limitations and Assumptions
- This analysis assumes that the data is representative of the Superstore's sales and profits during the period covered.
- The analysis does not account for external factors that may have influenced sales and profits, such as economic conditions or marketing campaigns.
- The removal of outliers in Sales may have affected the accuracy of the analysis.

---

*For detailed code and plots, see the notebook: `notebooks/Superstore_Sales_Prediction.ipynb`*
