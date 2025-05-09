# 🩺 Diabetes Prediction using Machine Learning

This project uses machine learning techniques to predict whether a patient is likely to have diabetes based on medical attributes. The model is trained on the Pima Indians Diabetes Dataset using Python and Jupyter Notebook.

---


---

## 📊 Dataset Overview

- **Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples**: 768 patients
- **Features**: 8 input variables (e.g., glucose, insulin, BMI)
- **Target**: `Outcome` (0 = No diabetes, 1 = Has diabetes)

---

## 🧠 Machine Learning Pipeline

The notebook implements an end-to-end machine learning pipeline:

1. **Data Loading & Exploration**  
   Using `pandas`, `matplotlib`, and `seaborn` for EDA.

2. **Data Cleaning & Preprocessing**  
   Handling missing values, feature scaling.

3. **Model Building**  
   Algorithms used:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Decision Tree

4. **Model Evaluation**  
   Metrics used:
   - Accuracy
   - Precision, Recall, F1-Score
   - AUC (Area Under the ROC Curve)
   - Confusion Matrix

---

## ✅ Results

| Model                | Accuracy | Precision | Recall | F1-Score | AUC Score |
|---------------------|----------|-----------|--------|----------|-----------|
| **Logistic Regression** | 0.76     | 0.72      | 0.52   | 0.60     | 0.7937    |
| **Random Forest**        | 0.76     | 0.70      | 0.56   | 0.62     | 0.7851    |
| **SVM**                  | 0.77     | 0.76      | 0.52   | 0.62     | 0.7920    |
| **K-Nearest Neighbors** | 0.72     | 0.65      | 0.44   | 0.53     | 0.7556    |
| **Decision Tree**        | 0.71     | 0.61      | 0.46   | 0.53     | 0.6515    |

> Models were evaluated on a test set of 154 samples. AUC and F1-score were emphasized due to medical importance of detecting positive cases (Outcome = 1).

---

## 🛠️ Requirements

To run this project, install the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter


