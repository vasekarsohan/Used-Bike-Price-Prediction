# 🚀 Used Motorcycle Price Prediction

A Machine Learning project that predicts the fair selling price of used motorcycles using custom ML models implemented completely from scratch (no scikit-learn regressors).

This project includes EDA, Data Cleaning, Custom ML Models, Model Evaluation, and a Streamlit Web App.



# 📁 Project Structure

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
│   └── insights
│       ├── selling_price_distribution.png
│       ├── brand_vs_price.png
│       ├── year_vs_price.png
│       ├── km_vs_price.png
│       ├── owner_vs_price.png
│       └── ex_price_vs_resale.png
│
├── models
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
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
│   │   └── train.py
│   │
│   └── utils
│       └── metrics.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md



# 🧠 Project Overview

The goal is to build a machine learning system capable of predicting the selling price of used motorcycles.

All ML models are implemented manually from scratch, including:

* KNN Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor



# 📌 Final Features Used (6 Features)

Your final cleaned dataset contains:

| Feature           | Description                        |
| -- | - |
| name (encoded)    | Target mean encoding of bike model |
| year              | Manufacturing year                 |
| seller_type       | 0 = Individual, 1 = Dealer         |
| owner             | 0,1,2,3 → ordinal mapping          |
| km_driven         | Total kilometers                   |
| ex_showroom_price | Original showroom price            |

# 📊 Exploratory Data Analysis (EDA)

Performed in: `src/preprocessing/EDA.py`

### ✔ Tasks Completed:

* Basic data inspection
* Handling missing values
* Outlier detection & capping using IQR
* Before/After distributions
* Boxplots
* Full correlation heatmap
* Key Insight-based visualizations

### 📈 Insights Visualizations

Saved in `plots/insights/`:

| Insight                                 | Visualization                  |
|  |  |
| Distribution of selling price           | selling_price_distribution.png |
| Premium brands have higher resale value | brand_vs_price.png             |
| Newer bikes sell for higher prices      | year_vs_price.png              |
| Higher km reduces resale price          | km_vs_price.png                |
| Owner count impact                      | owner_vs_price.png             |
| Ex-showroom price drives resale price   | ex_price_vs_resale.png         |



# 🔧 Data Preprocessing

Key steps:

✔ Missing value handling (median imputation for ex_showroom_price)
✔ Outlier capping (IQR)
✔ Ordinal encoding for owner
✔ Binary encoding for seller_type
✔ Target mean encoding for bike names
✔ Export cleaned dataset: `cleaned_data.csv`
✔ Export name encoding file for Streamlit



# 🤖 Model Training

Located in `src/training/train.py`

Trains the following scratch-built models:

* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

Each model is:

✔ Trained on cleaned dataset
✔ Evaluated on test set
✔ Saved as `.pkl` in `/models/`
✔ Metrics stored in `model_evaluation.csv`



# 📈 Model Evaluation

Saved in:

data/model_evaluation.csv

Metrics stored:

* R² Score
* RMSE
* MAE

No comparison plots are used in final version.
No model_comparison.py.



# 🌐 Streamlit App

Built in: `streamlit_app.py`

### App Features

* Choose bike model (target-encoded)

* Select ML model (KNN, Decision Tree, RF, GBDT)

* Enter:

  * Year
  * Seller type
  * Owner count
  * KM driven
  * Ex-showroom price

* Predict resale price instantly

### Run the app:
streamlit run streamlit_app.py

# ▶️ Running the Project

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Run EDA (optional)

python src/preprocessing/EDA.py

### 3️⃣ Generate bike name encoding

python src/preprocessing/generate_name_encoding.py

### 4️⃣ Train all models

python -m src.training.train

### 5️⃣ Launch Streamlit App

streamlit run streamlit_app.py



# ✨ Author

Sohan Vasekar
Machine Learning Project – Semester 5