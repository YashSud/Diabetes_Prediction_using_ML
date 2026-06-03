# Diabetes Prediction using Machine Learning

A comprehensive machine learning project that predicts the onset of diabetes using the **Pima Indians Diabetes Dataset**. The project covers the full ML pipeline — from exploratory data analysis and preprocessing to training, tuning, and comparing seven classification algorithms.

## Overview

This notebook implements a supervised binary classification workflow to determine whether a patient is diabetic (`1`) or non-diabetic (`0`) based on diagnostic health measurements. Seven ML models are trained and compared using accuracy scores and ROC-AUC curves to identify the best-performing classifier.

## Dataset

- **File:** `diabetes.csv`
- **Source:** Pima Indians Diabetes Dataset
- **Samples:** 768 (reduced after outlier removal)
- **Target:** `Outcome` — `0` (Non-Diabetic) or `1` (Diabetic)
- **Features:**

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |

## Project Structure

```
Diabetes_Prediction_using_ML-main/
└── Diabetes_Prediction_Using_ML.ipynb    # Main Jupyter Notebook
```

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Distribution plots for all 8 features
- Outcome class distribution (pie chart + count plot)
- Group-wise aggregations by `Outcome` (mean, max per feature)
- Correlation matrix heatmap
- Pair plots colored by outcome

### 2. Data Preprocessing
- Replace biologically invalid zero values with `NaN`
- Impute missing values using **outcome-stratified median** (separate median per class)
- Visualize missing data with `missingno`

### 3. Outlier Detection & Removal
- IQR-based detection across all features
- Capping Insulin outliers at the upper IQR bound
- **Local Outlier Factor (LOF)** applied to remove multivariate outliers

### 4. Feature Engineering
- **NewBMI** — categorized into 6 BMI groups (Underweight → Obesity 3)
- **NewInsulinScore** — Normal (16–166) or Abnormal
- **NewGlucose** — 4 glucose level categories (Low, Normal, Overweight, High)
- One-hot encoding of all engineered categorical features

### 5. Feature Scaling
- **RobustScaler** applied to numeric features (resistant to outliers)
- **StandardScaler** applied after train-test split

### 6. Model Training & Evaluation

Seven classifiers are trained and evaluated:

| Model | Tuning Method |
|---|---|
| Logistic Regression | Default |
| K-Nearest Neighbors (KNN) | Default |
| Support Vector Machine (SVM) | GridSearchCV |
| Decision Tree | GridSearchCV (50-fold CV) |
| Random Forest | Manual hyperparameters |
| Gradient Boosting (GBDT) | GridSearchCV |
| XGBoost | Manual hyperparameters |

Each model is assessed using:
- Train & test **accuracy score**
- **Confusion matrix**
- **Classification report** (precision, recall, F1)
- **ROC-AUC curve** (multi-model comparison plot)

### 7. Model Export
- Best model saved as `diabetes.pkl` using `pickle`

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- missingno
- statsmodels

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost missingno statsmodels jupyter
```

## Usage

```bash
git clone https://github.com/your-username/Diabetes_Prediction_using_ML.git
cd Diabetes_Prediction_using_ML
jupyter notebook Diabetes_Prediction_Using_ML.ipynb
```

Ensure `diabetes.csv` is in the same directory as the notebook before running.

## Outputs

- **Distribution plots** for all features
- **Correlation heatmap** of raw features
- **Pair plot** colored by outcome
- **Box plots** before/after outlier treatment
- **Model comparison table** — all 7 models ranked by test accuracy
- **ROC curve** — overlaid for all 7 classifiers
- **Bar chart** — accuracy and AUC side-by-side per model
- **`diabetes.pkl`** — serialized best model for deployment

## Key Concepts

| Concept | Purpose |
|---|---|
| Stratified Median Imputation | Fills missing values separately per class to avoid data leakage |
| Local Outlier Factor (LOF) | Detects outliers based on local density rather than global statistics |
| RobustScaler | Scales features using median and IQR, less sensitive to outliers than StandardScaler |
| GridSearchCV | Exhaustive hyperparameter search with cross-validation |
| ROC-AUC | Measures classifier performance independent of decision threshold |
