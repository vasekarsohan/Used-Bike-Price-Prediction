````markdown
# 🚀 Used Motorcycle Price Prediction

A Machine Learning project that predicts the fair selling price of used motorcycles using **custom ML models implemented completely from scratch** (no scikit-learn regressors).

The project covers **EDA, Data Cleaning, Feature Engineering, Model Training, Evaluation, and Deployment using Streamlit**.

---

## 📁 Project Structure

```text
📦 Used_Motorcycle_Price_Prediction
│
├── data
│   ├── raw
│   │   └── BIKEDETAILS.csv
│   ├── processed
│   │   └── cleaned_data.csv
│   ├── name_encoding.csv
│   └── model_evaluation.csv
│
├── plots
│   ├── distribution
│   │   ├── before/
│   │   └── after/
│   │
│   ├── outliers
│   │   ├── before_outliers.png
│   │   └── after_outliers.png
│   │
│   ├── heatmaps
│   │   └── full_correlation_heatmap.png
│   │
│   ├── insights
│   │   ├── selling_price_distribution.png
│   │   ├── brand_vs_price.png
│   │   ├── year_vs_price.png
│   │   ├── km_vs_price.png
│   │   ├── owner_vs_price.png
│   │   └── ex_price_vs_resale.png
│   │
│   └── evaluation
│       ├── KNN_actual_vs_pred.png
│       ├── KNN_residuals_vs_pred.png
│       ├── DecisionTree_actual_vs_pred.png
│       ├── DecisionTree_residuals_vs_pred.png
│       ├── RandomForest_actual_vs_pred.png
│       ├── RandomForest_residuals_vs_pred.png
│       ├── GradientBoosting_actual_vs_pred.png
│       └── GradientBoosting_residuals_vs_pred.png
│
├── models
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── report
│   └── Used_Motorcycle_Price_Prediction_Report.pdf
│
├── src
│   ├── models
│   │   ├── KNN.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── gradient_boosting.py
│   │   ├── load_model.py
│   │   └── save_model.py
│   │
│   ├── preprocessing
│   │   ├── EDA.py
│   │   └── generate_name_encoding.py
│   │
│   ├── training
│   │   ├── train.py
│   │   └── generate_evaluation_plots.py
│   │
│   └── utils
│       └── metrics.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
````

---

## 🧠 Project Overview

The goal of this project is to build a machine learning system capable of predicting the resale price of used motorcycles based on historical listing data.

All regression models are implemented **from scratch**, without using scikit-learn regressors:

* KNN Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting Regression

---

## 📌 Final Features Used (6 Features)

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| name (encoded)    | Target mean encoding of bike model |
| year              | Manufacturing year                 |
| seller_type       | 0 = Individual, 1 = Dealer         |
| owner             | Ordinal encoding (0–3)             |
| km_driven         | Total kilometers driven            |
| ex_showroom_price | Original showroom price            |

---

## 📊 Exploratory Data Analysis (EDA)

Implemented in: `src/preprocessing/EDA.py`

### ✔ Tasks Performed

* Basic data inspection
* Missing value handling
* Outlier detection and capping using IQR
* Distribution analysis (before and after preprocessing)
* Boxplots for outlier visualization
* Correlation heatmap
* Feature-wise insight visualizations

### 📈 Insight Plots (`plots/insights/`)

| Insight                           | Plot File                      |
| --------------------------------- | ------------------------------ |
| Selling price distribution        | selling_price_distribution.png |
| Brand vs resale price             | brand_vs_price.png             |
| Manufacturing year vs price       | year_vs_price.png              |
| Kilometers driven vs price        | km_vs_price.png                |
| Owner count vs price              | owner_vs_price.png             |
| Ex-showroom price vs resale price | ex_price_vs_resale.png         |

---

## 🔧 Data Preprocessing

Key preprocessing steps:

✔ Median imputation for missing ex_showroom_price
✔ IQR-based outlier capping
✔ Ordinal encoding for owner
✔ Binary encoding for seller_type
✔ Target mean encoding for motorcycle names
✔ Exported cleaned dataset (`cleaned_data.csv`)
✔ Exported name encoding file for Streamlit

---

## 🤖 Model Training

Training script: `src/training/train.py`

Each model is:

* Trained on the cleaned dataset
* Evaluated on a test set
* Saved as `.pkl` in `/models/`
* Metrics stored in `data/model_evaluation.csv`

---

## 📈 Model Evaluation

Evaluation metrics include:

* R² Score
* RMSE
* MAE

### 📊 Evaluation Plots (`plots/evaluation/`)

For **each model**, the following plots are generated:

* Actual vs Predicted Selling Price
* Residuals vs Predicted Price

These plots help analyze prediction accuracy, bias, and error distribution.

---

## 🌐 Streamlit Web App

Built in: `streamlit_app.py`

### Features

* Select motorcycle model (target encoded)
* Choose regression model (KNN, DT, RF, GBDT)
* Input bike details:

  * Manufacturing year
  * Seller type
  * Owner count
  * Kilometers driven
  * Ex-showroom price
* Instant resale price prediction

### Run the app

```bash
streamlit run streamlit_app.py
```

---

## 📄 Project Report

A detailed project report including EDA plots, model explanations, evaluation analysis, and conclusions is available as a PDF:

```
report/Used_Motorcycle_Price_Prediction_Report.pdf
```

---

## ▶️ Running the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run EDA (optional)

```bash
python src/preprocessing/EDA.py
```

### 3️⃣ Generate bike name encoding

```bash
python src/preprocessing/generate_name_encoding.py
```

### 4️⃣ Train all models

```bash
python -m src.training.train
```

### 5️⃣ Generate evaluation plots

```bash
python -m src.training.generate_evaluation_plots
```

### 6️⃣ Launch Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## ✨ Author

**Sohan Vasekar**<br/>
Machine Learning Project
