# Predictive Analytics for Sales Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline utilizing historical sales data from the **Walmart Recruiting - Store Sales Forecasting** Kaggle challenge. This project cleans raw transactional data, handles temporal seasonality, engineers advanced features (such as holiday adjustments), performs multi-stage hyperparameter tuning, and fits an optimized Random Forest Regressor to predict weekly department-wide sales.

---

## 📂 Repository Structure

The repository is structured as a compact, self-contained data science workspace:

```bash
├── Sales Forecasting Machine Learning.ipynb  # Core Jupyter notebook containing the complete pipeline
├── requirements.txt                          # Python package dependencies
├── Submission.csv                            # Final predictions generated for the Walmart test set
└── README.md                                 # Project documentation (this file)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/nandkumarcoder/Predictive-Analytics-for-Sales-Forecasting.git
cd Predictive-Analytics-for-Sales-Forecasting
```

### 2. Install Dependencies
Set up your virtual environment and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Data Preparation
The notebook is configured to load datasets from a relative parent directory structure. Ensure you place the competition CSV files from Kaggle under a folder named `../input/dataset/` relative to your workspace:
```text
../input/dataset/
├── train.csv
├── test.csv
├── stores.csv
├── features.csv
└── sampleSubmission.csv
```

---

## 🛠️ Machine Learning Pipeline

The project follows a rigorous data science workflow in the core notebook:

### 1. Data Cleaning & Feature Selection
* **Data Integration:** Inner-merges `train.csv` and `test.csv` with `stores.csv` and `features.csv` on the `Store` and `Date` fields.
* **Dimensionality Reduction:** Drops columns with low target correlation, high multicollinearity, or sparse values to prevent noise:
  * `MarkDown1` to `MarkDown5` (Removed due to excessive missing values).
  * `Fuel_Price` (Removed due to a strong correlation with `Year`).
  * `Temperature`, `Unemployment`, and `CPI` (Removed due to weak/negligible correlation with `Weekly_Sales`).

### 2. Feature Engineering
* **Categorical Mapping:** Treats `Type` as an ordinal variable and maps its values (`A -> 3`, `B -> 2`, `C -> 1`) reflecting the sales volume hierarchies.
* **Easter Holiday Adjustments:** In the raw dataset, Easter is not marked as a holiday. Because Easter falls on different calendar weeks each year, the notebook explicitly aligns it:
  * **2010:** Week 13 ➡️ `IsHoliday = True`
  * **2011:** Week 16 ➡️ `IsHoliday = True`
  * **2012:** Week 14 ➡️ `IsHoliday = True`
  * **2013 (Test Set):** Week 13 ➡️ `IsHoliday = True`
* **Temporal Features:** Extracts time attributes (`Week`, `Year`) to capture seasonality and annual growth trends.

### 3. Validation Strategy
* **Grouped K-Fold Cross-Validation:** Utilizes a `5-Split KFold` cross-validation grouped by `[Store, Dept]` to evaluate model robustness and guard against data leakage.

### 4. Custom Evaluation Metric
Model evaluation is measured using **Weighted Mean Absolute Error (WMAE)**, which places a higher priority on holiday sales accuracy:

$$\text{WMAE} = \frac{1}{\sum w_i} \sum_{i=1}^{n} w_i |y_i - \hat{y}_i|$$

* $w_i = 5$ if the week is a holiday.
* $w_i = 1$ otherwise.

### 5. Model Exploration & Selection
Baseline evaluations of several regression algorithms (using default parameters) yielded the following cross-validated WMAE scores:

| Model | Baseline WMAE |
| :--- | :---: |
| Extra Trees Regressor | 4024.48 |
| K-Nearest Neighbors (KNN) | 3789.40 |
| Random Forest Regressor | 3672.00 |
| **Linear Regression** | **3332.86** |

*Note: While Linear Regression had a lower baseline on un-tuned folds, the **Random Forest Regressor** was selected for optimization due to its high non-linear capacity and resilience to outliers.*

### 6. Hyperparameter Tuning (Grid Search)
The Random Forest model was optimized sequentially across three parameter stages to minimize cross-validated WMAE:

* **Stage 1 (Depth & Estimators):** Tuned `n_estimators` (56, 58, 60) and `max_depth` (25, 27, 30).
  * *Result:* `n_estimators = 56`, `max_depth = 30` (WMAE: **1528.71**)
* **Stage 2 (Features):** Tuned `max_features` (range 2 to 7).
  * *Result:* `max_features = 7` (WMAE: **1539.80**)
* **Stage 3 (Split/Leaf Samples):** Tuned `min_samples_split` (2, 3, 4) and `min_samples_leaf` (1, 2, 3).
  * *Result:* `min_samples_split = 3`, `min_samples_leaf = 1` (WMAE: **1541.13**)

### 7. Final Model Configuration
The final fitted model uses the following parameters:
```python
RandomForestRegressor(
    n_estimators=56, 
    max_depth=30, 
    max_features=7, 
    min_samples_split=3, 
    min_samples_leaf=1
)
```

---

## 📈 Performance & Outputs

* **Model Serialization:** The trained model is serialized to disk as `SavedModel.sav` using Python's `pickle` library.
* **Test Performance:** Evaluated on Walmart's hidden test set, the tuned Random Forest model achieved a final WMAE score of **`2776.28705`**.
* **Submission File:** Generates predictions matching the Kaggle schema and outputs them to `Submission.csv`.

---

## 🔮 Future Improvements

* **Christmas Sales Adjustment:** Implement the post-processing adjustments mentioned in the notebook observations. Because Christmas week has varying pre-holiday shopping days depending on the day of the week it falls on (e.g., 0 pre-holiday days in 2022 vs. 3 pre-holiday days in 2023), applying an analytical scale factor to predicted values for Christmas week could further decrease evaluation error.
* **Ensemble Modeling:** Stack the tuned Random Forest Regressor with XGBoost, LightGBM, and ARIMA/SARIMAX models to exploit both tabular features and sequential time-series patterns.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
