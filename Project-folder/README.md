Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether an online shopping session will result in a purchase (Revenue = True/False). The project implements six classification algorithms and compares their performance using standard evaluation metrics. A Streamlit web application is developed to allow test data upload and model evaluation.

Dataset Description

Dataset Name: Online Shoppers Purchasing Intention Dataset
Source: UCI Machine Learning Repository

The dataset contains user behavior data collected from an online shopping website.
Each row represents a user session, and the target variable indicates whether the session ended with a purchase.

Number of instances: 12,330

Number of features: 17 (numerical + categorical)

Target variable: Revenue (Binary: 0 = No purchase, 1 = Purchase)

Key Features:

Administrative, Informational, ProductRelated

BounceRates, ExitRates

PageValues, SpecialDay

Month, VisitorType, Weekend

The dataset satisfies the assignment requirements:

✔ Minimum 12 features

✔ More than 500 instances

c. Models Used

The following classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

All models were trained offline and evaluated using uploaded test data through a Streamlit application.

d. Model Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.88	0.74	0.74	0.53	0.62	0.56
Decision Tree	0.85	0.70	0.63	0.62	0.62	0.53
KNN	0.87	0.72	0.70	0.56	0.62	0.55
Naive Bayes	0.84	0.71	0.60	0.64	0.62	0.52
Random Forest (Ensemble)	0.90	0.78	0.78	0.61	0.69	0.63
XGBoost (Ensemble)	0.91	0.81	0.80	0.65	0.72	0.66


e. Model Performance Observations
ML Model Name	Observation
Logistic Regression	Performs well as a baseline model but struggles with non-linear relationships.
Decision Tree	Easy to interpret but prone to overfitting.
KNN	Performs reasonably well but is sensitive to feature scaling and dataset size.
Naive Bayes	Fast and simple but assumes feature independence, limiting accuracy.
Random Forest (Ensemble)	Handles non-linearity well and improves overall performance.
XGBoost (Ensemble)	Best-performing model due to gradient boosting and optimized tree learning.
f. Deployment on Streamlit Community Cloud

The application is deployed on Streamlit Community Cloud with the following features:

CSV upload option for test data

Model selection dropdown

Display of evaluation metrics

Confusion matrix visualization

All models are pre-trained offline and loaded during runtime to ensure fast and efficient deployment.
