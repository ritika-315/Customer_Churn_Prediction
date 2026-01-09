# 📊 Customer Churn Prediction for a Telecommunications Company

## 📝 Overview
Customer churn — the loss of customers over time — is a major challenge for telecommunications companies. Accurately predicting churn enables companies to proactively address customer concerns, improve services, and reduce attrition.

This project leverages **Python and Machine Learning** to build models that predict whether a customer is likely to churn based on their demographics, account details, and usage behavior.

---

## 🧠 Project Workflow

### 1. 📚 Importing Libraries
We use essential libraries including:
- `pandas`, `numpy` – for data manipulation
- `matplotlib`, `seaborn` – for data visualization
- `scikit-learn` – for machine learning models
- Other advanced packages like `xgboost` and `keras` (for neural networks)

---

### 2. 📂 Loading the Dataset
The dataset (CSV format) contains:
- Customer demographics
- Account information
- Service usage
- Target variable: `Churn`

---

### 3. 🔍 Exploratory Data Analysis (EDA)
We explore the dataset using:
- **Histograms & bar plots** – to understand distributions
- **Heatmaps & correlation matrices** – to identify relationships between features
- **Class imbalance checks** – to inspect churn proportions

---

### 4. 🚨 Outlier Detection (IQR Method)
Outliers can mislead model training. We use the **Interquartile Range (IQR)** method to detect and treat extreme values:
- Values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR are considered outliers

---

### 5. 🧹 Data Cleaning & Transformation
- Handle **missing values**
- Fix **data types**
- Drop **irrelevant columns**
- Normalize or transform skewed features

---

### 6. 🔄 One-Hot Encoding
Convert categorical variables into binary (0/1) features using **one-hot encoding**, ensuring compatibility with machine learning algorithms.

---

### 7. 🧾 Rearranging Columns
Reorganize the dataset for clarity:
- Target column `Churn` is moved to the end
- Features grouped logically

---

### 8. 📏 Feature Scaling
Normalize numerical features using:
- **StandardScaler** (Z-score)
- **MinMaxScaler** (scales between 0 and 1)

---

### 9. 📉 Feature Selection
Improve model performance by selecting the most important features:
- **Correlation analysis**
- **Recursive Feature Elimination (RFE)**
- **Principal Component Analysis (PCA)**

---

## 🤖 Model Building & Evaluation

We build and evaluate multiple classifiers:

| Model                          | Description                                                   |
|-------------------------------|---------------------------------------------------------------|
| **Logistic Regression**        | Baseline binary classifier using sigmoid function             |
| **Support Vector Classifier**  | Finds optimal separating hyperplane                           |
| **Decision Tree**              | Rule-based classification using tree structures               |
| **K-Nearest Neighbors (KNN)**  | Classifies based on proximity to other data points            |
| **AdaBoost**                   | Boosting technique combining weak learners                    |
| **Gradient Boosting**          | Sequentially builds trees to correct predecessor errors       |
| **Stochastic Gradient Boosting** | Uses data subsampling to reduce overfitting                |
| **XGBoost**                    | Fast, regularized gradient boosting                           |
| **Neural Networks**            | Deep learning model capturing complex patterns                |

Each model is trained, tested, and evaluated using metrics like:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC-AUC Curve**

---

## ✅ Conclusion

The Neural Network model achieved an accuracy of approximately 89.67% on the test dataset.

This project demonstrates a full machine learning pipeline for **customer churn prediction**:
- From raw data cleaning to model deployment
- Multiple models compared to identify the best performer

Telecom companies can use this approach to:
- Understand churn patterns
- Take action to improve customer retention
- Optimize marketing and customer service strategies
