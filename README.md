# 🛒 Big Mart Sales Analysis & Outlet Sales Prediction

An end-to-end **retail sales analytics and predictive modeling project** using the Big Mart Sales dataset. The project explores sales performance across outlets, product categories, regions, and outlet types, followed by machine learning models to predict **Item Outlet Sales**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Project Workflow](#project-workflow)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Findings](#key-findings)
- [Predictive Modeling](#predictive-modeling)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)
- [Author](#author)

---

## Overview

This project analyzes retail sales data to understand the factors associated with **Item Outlet Sales** and differences in performance across outlets and regions.

The analysis covers:

- Sales performance by outlet
- Sales by outlet type and location
- Item/category-level sales performance
- Top products by total sales
- Regional performance comparison
- Sales trends across outlet types
- Item-type performance across outlets
- Predictive modeling of outlet sales

The final stage uses **Random Forest Regression** and **XGBRFRegressor** to predict item-level outlet sales and identify the most important predictive features.

---

## Project Objectives

The main objectives of this project are:

1. Analyze sales performance across different outlets.
2. Compare sales across outlet types and location tiers.
3. Identify the best-performing product categories.
4. Identify the top products based on total revenue.
5. Compare retail performance across Tier 1, Tier 2, and Tier 3 locations.
6. Study how product categories perform across individual outlets.
7. Build machine learning models to predict `Item_Outlet_Sales`.
8. Identify the features that contribute most to sales prediction.

---

## Dataset

The project uses the **Big Mart Sales dataset**, containing **8,523 observations and 12 variables**.

### Dataset Features

| Feature | Description |
|---|---|
| `Item_Identifier` | Unique identifier of the product |
| `Item_Weight` | Weight of the product |
| `Item_Fat_Content` | Fat-content category of the product |
| `Item_Visibility` | Visibility of the product in the outlet |
| `Item_Type` | Product category/type |
| `Item_MRP` | Maximum Retail Price of the product |
| `Outlet_Identifier` | Unique identifier of the outlet |
| `Outlet_Establishment_Year` | Year in which the outlet was established |
| `Outlet_Size` | Size of the outlet |
| `Outlet_Location_Type` | Location tier of the outlet |
| `Outlet_Type` | Type of outlet |
| `Item_Outlet_Sales` | Target variable representing item sales at the outlet |

### Dataset Summary

- **Rows:** 8,523
- **Columns:** 12
- **Target variable:** `Item_Outlet_Sales`
- **Categorical variables:** Product, fat content, item type, outlet, outlet size, location type, and outlet type
- **Numerical variables:** Item weight, item visibility, item MRP, establishment year, and item outlet sales

> The notebook loads the dataset from Google Drive. The original dataset file is not embedded in this repository unless added separately.

---

## Tools & Technologies

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyMC
- ArviZ
- XGBoost

### Machine Learning
- Random Forest Regression
- XGBRFRegressor
- Train/Test Split
- 5-Fold Cross-Validation
- Feature Importance

### Statistical / Analytical Techniques
- Descriptive statistics
- Group-by aggregation
- Regional comparison
- Correlation/relationship exploration
- Bayesian imputation
- Categorical encoding
- Exploratory Data Analysis

---

## Project Workflow

```text
Raw Dataset
     │
     ▼
Data Loading & Inspection
     │
     ▼
Missing Value Treatment
     │
     ├── Item Weight → Bayesian Imputation using PyMC
     │
     └── Outlet Size → Random Forest Classification
     │
     ▼
Data Cleaning
     │
     ├── Standardize Item Fat Content
     └── Create Clean Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ├── Outlet Performance
     ├── Outlet Type Analysis
     ├── Product Category Analysis
     ├── Regional Comparison
     └── Product/Outlet Trends
     │
     ▼
Predictive Modeling
     │
     ├── Random Forest Regression
     └── XGBRFRegressor
     │
     ▼
Model Evaluation & Feature Importance
```

---

## Data Cleaning & Preprocessing

### 1. Missing `Item_Weight`

The dataset contained missing values in `Item_Weight`.

A Bayesian approach using **PyMC** was used to estimate the missing weights:

- Normal prior was specified for the mean.
- Half-Normal prior was specified for the standard deviation.
- Observed item weights were used as the likelihood.
- Posterior samples were generated using MCMC.
- Posterior mean estimates were used to fill the missing values.

### 2. Missing `Outlet_Size`

`Outlet_Size` contained **2,410 missing observations**.

A Random Forest Classifier was trained using:

- `Outlet_Type`
- `Outlet_Location_Type`

The trained classifier was then used to predict and fill the missing outlet sizes.

After preprocessing, the dataset contained **no missing values**.

### 3. Category Standardization

Inconsistent values in `Item_Fat_Content` were standardized:

- `LF` → `Low Fat`
- `low fat` → `Low Fat`
- `reg` → `Regular`

---

## Exploratory Data Analysis

### 1. Sales Performance by Outlet

Total sales were compared across all outlets.

The highest total sales were observed for:

- **OUT027:** approximately ₹34.54 lakh
- **OUT035:** approximately ₹22.68 lakh
- **OUT049:** approximately ₹21.84 lakh
- **OUT017:** approximately ₹21.67 lakh
- **OUT013:** approximately ₹21.43 lakh

The lowest sales were observed for:

- **OUT010:** approximately ₹1.88 lakh
- **OUT019:** approximately ₹1.80 lakh

---

### 2. Sales by Outlet Type

The analysis compares both average and total sales across outlet types and location tiers.

The major outlet categories analyzed were:

- Grocery Store
- Supermarket Type1
- Supermarket Type2
- Supermarket Type3

**Supermarket Type3** recorded the highest average sales per item among the outlet types analyzed, at approximately **₹3,694**.

---

### 3. Item Type Performance

The highest-selling product categories by total sales were:

| Rank | Item Type | Total Sales |
|---:|---|---:|
| 1 | Fruits and Vegetables | ₹28.20 lakh |
| 2 | Snack Foods | ₹27.33 lakh |
| 3 | Household | ₹20.55 lakh |
| 4 | Frozen Foods | ₹18.26 lakh |
| 5 | Dairy | ₹15.23 lakh |

**Fruits and Vegetables** and **Snack Foods** were the strongest categories by total sales.

---

### 4. Top Products by Sales

The top 10 individual products were identified based on their aggregated `Item_Outlet_Sales`.

The highest-selling product in the analysis was:

**FDY55 – Fruits and Vegetables**, with total sales of approximately **₹42,661.80**.

Other high-performing products included products from Dairy, Frozen Foods, Household, Health and Hygiene, and Meat categories.

---

### 5. Regional Performance

Sales were compared across three outlet location tiers.

| Location Type | Total Sales | Average Sales |
|---|---:|---:|
| Tier 3 | ₹76.37 lakh | ₹2,279.63 |
| Tier 2 | ₹64.72 lakh | ₹2,323.99 |
| Tier 1 | ₹44.82 lakh | ₹1,876.91 |

Key observation:

- **Tier 3** generated the highest total sales.
- **Tier 2** had the highest average sales per item.
- Tier 3 contained 4 unique outlets, while Tier 1 and Tier 2 each contained 3 outlets.

---

### 6. Product Performance Across Outlets

A heatmap was created to examine sales of different item categories across individual outlets.

The top-selling category varied across outlets, but **Fruits and Vegetables** and **Snack Foods** appeared frequently among the strongest-performing categories.

Examples:

- OUT027 → Fruits and Vegetables
- OUT013 → Fruits and Vegetables
- OUT017 → Fruits and Vegetables
- OUT018 → Snack Foods
- OUT035 → Snack Foods
- OUT049 → Snack Foods

---

## Key Findings

1. **OUT027 was the strongest outlet** by total sales, generating approximately ₹34.54 lakh.
2. **Fruits and Vegetables** was the highest-selling product category, followed closely by **Snack Foods**.
3. **Tier 3 locations** generated the highest total sales.
4. **Tier 2 locations** recorded the highest average sales per item.
5. **Supermarket Type3** had the highest average sales among the outlet types analyzed.
6. `Outlet_Type` was the most important feature in the XGBRF model.
7. `Store_age`, `Item_MRP`, and `Outlet_Identifier` were also important predictors.
8. The final XGBRF model achieved a test-set **R² of 0.6174**.

---

## Predictive Modeling

The target variable for prediction was:

```text
Item_Outlet_Sales
```

### Feature Engineering

A new feature called `Store_age` was created:

```python
Store_age = 2013 - Outlet_Establishment_Year
```

The original establishment year was then removed.

`Item_Identifier` was also transformed from the full product code to its first two characters.

Categorical variables were converted into numerical representations using `OrdinalEncoder`.

---

### 🌲 Model 1 — Random Forest Regression

A Random Forest Regressor with 100 trees was evaluated using 5-fold cross-validation.

**Cross-Validated R² = 0.5578**

---

### 🚀 Model 2 — XGBRFRegressor

An `XGBRFRegressor` with 100 estimators was evaluated using 5-fold cross-validation.

**Cross-Validated R² = 0.5963**

The XGBRF-based model performed better than the Random Forest model in cross-validation.

---

## Feature Importance

Feature importance from the XGBRF model showed that the strongest predictors were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Outlet_Type` | 0.4408 |
| 2 | `Store_age` | 0.1735 |
| 3 | `Item_MRP` | 0.1643 |
| 4 | `Outlet_Identifier` | 0.1521 |
| 5 | `Outlet_Location_Type` | 0.0331 |
| 6 | `Outlet_Size` | 0.0259 |

The remaining variables had relatively smaller feature importance.

Based on this analysis, several lower-importance features were removed before fitting the final model.

---

## Model Evaluation

The final XGBRF model was trained using an **80/20 train-test split**.

### Test Set Performance

| Metric | Result |
|---|---:|
| MAE | 714.73 |
| RMSE | 1,019.73 |
| R² | **0.6174** |

The final model achieved an **R² of approximately 0.617**, indicating that the model explains around 61.7% of the variation in the test-set sales values.

### 🔮 Example Prediction

The final model was also used to generate a sample sales prediction for a manually specified feature vector.

Example predicted `Item_Outlet_Sales`:

```text
₹727.99
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pymc arviz xgboost
```

### 3. Open the notebook

The analysis was developed in Google Colab.

**Google Colab Notebook:** [Open the Big Mart Sales Analysis Notebook](https://colab.research.google.com/drive/16eYFSFgL3xP256eaOY0_FaaiqHwLgcN0?usp=sharing)

### 4. Update the dataset path

The notebook currently loads the dataset from:

```python
/content/drive/MyDrive/Data/Time Series/city_day.csv
```

For GitHub/Colab use, update this path to the location of your Big Mart Sales CSV file.

### 5. Run the notebook

Run the notebook cells sequentially to reproduce:

- Data cleaning
- Missing value imputation
- Exploratory analysis
- Sales comparisons
- Regional analysis
- Predictive modeling
- Model evaluation
- Feature importance

---

## 📂 Suggested Repository Structure

```text
big-mart-sales-analysis/
│
├── README.md
├── Big_Mart_Sales_Analysis.ipynb
├── data/
│   └── Train.csv
├── images/
│   ├── sales_by_outlet.png
│   ├── sales_by_item_type.png
│   ├── regional_performance.png
│   └── feature_importance.png
│
└── requirements.txt
```

> The exact repository structure can be adjusted depending on whether you upload the dataset and exported visualizations to GitHub.

---

## Conclusion

This project demonstrates how retail sales data can be transformed into actionable insights through a combination of **data cleaning, exploratory data analysis, visualization, and machine learning**.

The analysis highlights substantial differences in sales performance across outlets, product categories, and location tiers. The predictive modeling stage further shows that **outlet characteristics, store age, product MRP, and outlet identity** play an important role in explaining item-level sales.

The final XGBRF model provides a useful baseline for predicting `Item_Outlet_Sales` and can be further improved through advanced feature engineering, hyperparameter tuning, alternative encoding techniques, and additional validation.

---

## Author

**Sujeet Sehgal**
M.Sc. Statistics | Data Analytics & Statistical Modelling

### Skills Demonstrated

- Python
- Pandas & NumPy
- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Statistical Analysis
- Machine Learning
- Regression
- Feature Engineering
- Model Evaluation

---

## 📜 License

This project is intended for educational, portfolio, and analytical purposes.
