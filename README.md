# 🏥 Insurance Predictive Analysis

A supervised machine learning project that predicts healthcare insurance costs based on real-world patient attributes such as age, BMI, smoking status, and region. The project walks through the full ML pipeline — from exploratory data analysis and data cleaning to feature engineering, feature selection, and model training.

---

## 📌 Project Overview

Medical insurance costs vary significantly across individuals. This project aims to uncover the key factors that drive those costs and build a regression model capable of predicting charges with reasonable accuracy. It is designed as a hands-on, end-to-end data science workflow using Python.

---

## 📂 Repository Structure

```
insurance-predictive-analysis/
│
├── analysis.py                    # Main Python script (full ML pipeline)
├── insurance.csv                  # Dataset
│
├── Distribution of age.png        # Histogram: age distribution
├── Distribution of bmi.png        # Histogram: BMI distribution
├── Distribution of charges.png    # Histogram: charges distribution
├── Distribution of children.png   # Histogram: children distribution
│
├── age.png                        # Boxplot: age
├── bmi.png                        # Boxplot / histogram: BMI
├── bmi.png                        # Boxplot: BMI
├── charges.png                    # Boxplot: charges
├── children.png                   # Boxplot: children
└── correlation.png                # Heatmap: feature correlations
```

---

## 📊 Dataset

The dataset (`insurance.csv`) contains **1,338 records** with the following features:

| Feature    | Type        | Description                                      |
|------------|-------------|--------------------------------------------------|
| `age`      | Numeric     | Age of the primary beneficiary                   |
| `sex`      | Categorical | Gender of the beneficiary (`male` / `female`)    |
| `bmi`      | Numeric     | Body mass index                                  |
| `children` | Numeric     | Number of children/dependents covered            |
| `smoker`   | Categorical | Whether the beneficiary smokes (`yes` / `no`)    |
| `region`   | Categorical | US region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges`  | Numeric     | Individual medical costs billed by insurance (**target variable**) |

---

## 🔬 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Inspects shape, data types, and summary statistics
- Checks for null values
- Plots histograms (with KDE) for all numeric columns
- Generates boxplots to identify outliers
- Produces a correlation heatmap for numeric features

### 2. Data Cleaning & Preprocessing
- Removes duplicate rows
- Encodes categorical variables using **label encoding** (`sex`, `smoker`) and **one-hot encoding** (`region`)
- Renames encoded columns for clarity (`sex` → `is_female`, `smoker` → `is_smoker`)

### 3. Feature Engineering
- Creates a new `bmi_category` column by binning BMI into:
  - `underweight` (< 18.5)
  - `normal` (18.5–24.9)
  - `overweight` (25–29.9)
  - `obese` (≥ 30)
- One-hot encodes BMI categories

### 4. Feature Scaling
- Applies **StandardScaler** to continuous features: `age`, `bmi`, `children`

### 5. Feature Selection
- **Pearson Correlation** used to assess linear relationships between each feature and `charges`
- **Chi-Square Test** (`chi2_contingency`) used to evaluate association between categorical features and binned charge groups
- Features with p-value < 0.05 are retained; others are dropped

**Final selected features:**
`age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_obese`

### 6. Model Training & Evaluation
- Train/test split: **67% train / 33% test**
- Model: **Linear Regression** (`sklearn.linear_model.LinearRegression`)
- Evaluation metrics:
  - **R² Score**
  - **Adjusted R² Score**

> 📈 The model achieves approximately **77% R² accuracy**, meaning it explains ~77% of the variance in insurance charges.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhi-ops-hub/insurance-predictive-analysis.git
   cd insurance-predictive-analysis
   ```

2. Run the analysis script:
   ```bash
   python analysis.py
   ```

This will print EDA summaries, feature selection results, and model performance metrics to the console, and display all plots inline.

---

## 📈 Key Findings

- **Smoking status** is the strongest predictor of insurance charges by a significant margin.
- **BMI** and **age** are also positively correlated with higher charges.
- Being classified as **obese** (BMI ≥ 30) is a meaningful categorical signal for elevated costs.
- **Region** (particularly `southeast`) shows some association with charges.
- **Gender** and **number of children** have relatively weaker influence on predicted costs.

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML model & preprocessing |
| SciPy | Statistical feature selection |

---

## 📄 License

This project is open-source and available for educational and personal use.

---

## 🙋 Author

**Abhi-ops-hub**  
[GitHub Profile](https://github.com/Abhi-ops-hub)
