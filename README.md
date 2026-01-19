# Online Shoppers Purchase Prediction - Classification Models

## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether an online shopping session will result in a purchase (`Revenue = True/False`).  
Six different classification algorithms are implemented and compared using standard evaluation metrics. A Streamlit web application is developed to allow test data upload and model evaluation.

---

## Dataset Description

**Dataset Name:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  

The dataset contains user behavior data collected from an online shopping website. Each row represents a user session, and the target variable indicates whether the session ended with a purchase.

- **Number of instances:** 12,330  
- **Number of features:** 17 (numerical and categorical)  
- **Target variable:** `Revenue` (Binary: 0 = No purchase, 1 = Purchase)

### Key Features
- Administrative, Informational, ProductRelated  
- BounceRates, ExitRates  
- PageValues, SpecialDay  
- Month, VisitorType, Weekend  

This dataset satisfies the assignment requirements:
- Minimum 12 features  
- More than 500 instances  

---

## Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

All models were trained offline and evaluated using uploaded test data through a Streamlit application.

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.88 | 0.86 | 0.76 | 0.37 | 0.50 | 0.48 |
| Decision Tree | 0.85 | 0.72 | 0.52 | 0.55 | 0.54 | 0.45 |
| KNN | 0.87 | 0.79 | 0.62 | 0.37 | 0.47 | 0.41 |
| Naive Bayes | 0.77 | 0.80 | 0.38 | 0.67 | 0.49 | 0.38 |
| Random Forest (Ensemble) | 0.90 | 0.91 | 0.73 | 0.56 | 0.63 | 0.58 |
| XGBoost (Ensemble) | 0.90 | 0.92 | 0.71 | 0.60 | 0.65 | 0.59 |

> *Note: Metrics may vary slightly depending on random state and test split.*

## Model Performance Observations

| ML Model Name | Observation |
|--------------|-------------|
| Logistic Regression | Performs well as a baseline model but struggles with non-linear relationships. |
| Decision Tree | Easy to interpret but prone to overfitting. |
| KNN | Performs reasonably well but is sensitive to feature scaling and dataset size. |
| Naive Bayes | Fast and simple but assumes feature independence, limiting accuracy. |
| Random Forest (Ensemble) | Handles non-linearity well and improves overall performance. |
| XGBoost (Ensemble) | Best-performing model due to gradient boosting and optimized tree learning. |

---

## Deployment on Streamlit Community Cloud

The application is deployed on **Streamlit Community Cloud** with the following features:

- CSV upload option for test data  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  

All models are pre-trained offline and loaded during runtime to ensure fast and efficient deployment.

---

## Project Structure

```text
Project-folder/
│── streamlit_app.py
│── train_models.py
│── requirements.txt
│── evaluate_models.py
└── model/
    └── saved_models/
        ├── logistic.pkl
        ├── decision_tree.pkl
        ├── knn.pkl
        ├── naive_bayes.pkl
        ├── random_forest.pkl
        ├── xgboost.json
        ├── scaler.pkl
        └── encoders.pkl
