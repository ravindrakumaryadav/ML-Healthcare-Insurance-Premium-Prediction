# 🛡️ Health Insurance Premium Predictor

A production-grade machine learning web application that predicts annual health insurance premiums based on patient demographics, health profile, and financial information — built for **Health Insurance Premium Prediction** as part of an end-to-end ML project.

---

## 📊 Project Overview

Insurance underwriters traditionally estimate premiums manually — a slow, inconsistent, and error-prone process. This project automates that process using a trained **XGBoost regression model** deployed via a **Streamlit web application**, allowing underwriters to get instant, accurate premium predictions from anywhere.

| Item | Detail |
|------|--------|
| **Client** | Shield Insurance |
| **Service Provider** | AtliQ AI |
| **Model** | XGBoost Regressor |
| **Dataset Size** | 50,000 records |
| **Model Accuracy** | R² = 98.19% |
| **Avg Prediction Error** | RMSE = ₹1,130 |
| **Target** | Annual Premium Amount (₹) |

---

## 🎯 Key Results

- ✅ Achieved **R² of 98.19%** — exceeding client target of 97%
- ✅ RMSE of **₹1,130** average prediction error
- ✅ Identified top premium drivers: Insurance Plan (0.83), Age (0.77), Risk Score (0.52)
- ✅ Reduced error by **49.7%** compared to baseline Linear Regression (RMSE ₹2,247 → ₹1,130)

---

## 🖥️ App Demo

> **Input patient details → Get instant premium prediction**

![App Screenshot](assets/app_screenshot.png)

### Input Features
| Category | Features |
|----------|----------|
| **Personal** | Age, Gender, Marital Status, Region, Number of Dependants |
| **Financial** | Annual Income, Employment Status, Insurance Plan |
| **Health** | BMI Category, Smoking Status, Medical History, Genetical Risk Score |

---

## 🗂️ Project Structure

```
Healthcare-Premium-Prediction/
│
├── app/
│   ├── main.py                  # Streamlit web application
│   └── prediction_helper.py     # Preprocessing & prediction logic
│
├── notebooks/
│   └── Dataset_segmentation.ipynb  # EDA, Data Segmentation
│   └── Healthcare-Premium-Prediction.ipynb  # EDA, model training & evaluation
│   └── Healthcare-Premium-Prediction-Old_Age.ipynb  # EDA, model training & evaluation for older age
│   └── Healthcare-Premium-Prediction-Young_Age.ipynb  # EDA, model training & evaluation for younger age
│   └── Healthcare-Premium-Prediction-Young_Age-with_gr.ipynb  # EDA, model training & evaluation for younger age with genetic factor
│
├── app/artifacts/
│   ├── model__rest.joblib       # Trained XGBoost model-for age>25
│   ├── model_young.joblib       # Trained XGBoost modelfor age<2=5
│   ├── scaler_rest.joblib       # Feature scaler for age>25
│   └── scaler_young.joblib      # Feature scaler for age<=25
│
│
├── documents/
│   ├── sow.pdf       # scope of work file
│
│
├── dataset/
│   └── premimum_old_age.xlsx       # Training dataset splitedwith age>25
│   └── premimum_young_age.xlsx     # Training dataset splitedwith age<=25
│   └── premimum.xlsx       # Training dataset (50,000 records)
│   └── premimum_young_with_gr.xlsx       # Training dataset with genetic factor for young age
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed 50,000 records with 13 features
- Identified 3 columns with missing values (`smoking_status`, `employment_status`, `income_level`)
- Correlation heatmap revealed top predictors and multicollinearity
- Dropped `income_level` (r=0.91 with `income_lakhs`) to resolve multicollinearity

### 2. Data Preprocessing
- Handled missing values via mode imputation
- Applied **Label Encoding** for ordinal features (insurance plan, income level)
- Applied **One-Hot Encoding** for nominal features (gender, region, BMI, etc.)
- Engineered `normalized_risk_score` from medical history using min-max scaling

### 3. Model Building & Evaluation

| Model | R² | RMSE |
|-------|----|------|
| Linear Regression (Baseline) | 92.84% | ₹2,247 |
| XGBoost (Default) | 98.07% | ₹1,165 |
| **XGBoost (Tuned)** | **98.19%** | **₹1,130** |

### 4. Hyperparameter Tuning
Used two-step tuning approach:
- **RandomizedSearchCV** — explored 18,000+ combinations, sampled 50 (150 fits)
- **GridSearchCV** — refined search around best region (27 combinations, 81 fits)
- Both used **3-fold cross validation** for reliable evaluation

**Best Parameters (RandomizedSearchCV):**
```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=7,
    random_state=42
)
```

---

## 🚀 How To Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Healthcare-Premium-Prediction.git
cd Healthcare-Premium-Prediction
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
cd app
streamlit run main.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 📦 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Web App** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Model Saving** | Joblib |
| **IDE** | Jupyter Notebook, PyCharm |

---

## 📈 Feature Importance

Top features driving premium predictions (by correlation with target):

```
Insurance Plan        ████████████████████  0.83
Age                   ████████████████░░░░  0.77
Normalized Risk Score ██████████░░░░░░░░░░  0.52
Marital Status        ██████████░░░░░░░░░░ -0.52
Number of Dependants  ████████░░░░░░░░░░░░  0.41
```

---

## 🧠 Key Learnings

- **Multicollinearity detection** using correlation heatmap — removed `income_level` (r=0.91)
- **Feature engineering** — engineered `normalized_risk_score` from raw medical history
- **Learning rate vs trees tradeoff** — lower learning rate (0.05) with more trees (300) = same accuracy as higher rate (0.1) with fewer trees (100)
- **CV R² ≈ Test R²** — confirmed no overfitting, strong generalization

---

## 👤 Author

**Ravindra Yadav**
- 📧 yadavravi.it@gmail.com
- 🌐 [Portfolio](https://ravindraph.blogspot.com/)
- 💼 Senior Data Scientist | ML Engineer

---

## 📄 License

This project is for educational and portfolio purposes.

---

⭐ **If you found this project helpful, please give it a star!**