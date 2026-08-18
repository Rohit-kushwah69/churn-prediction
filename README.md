# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to **churn (leave the service)** based on customer information and service usage.

The project covers the complete Machine Learning workflow, starting from Exploratory Data Analysis (EDA) to model training, evaluation, and saving the trained model.

---

## 🎯 Project Objective

The main objective of this project is to predict customer churn.

In this project:

- `0` → Customer does not churn
- `1` → Customer churns

Customer churn means that a customer leaves or cancels the company's service.

The model can help businesses identify customers who are at risk of leaving and take appropriate retention actions.

---

## 🗂️ Project Structure

```text
Customer-Churn-Prediction/
│
├── 01_EDA_Churn_Analysis.ipynb
├── 02_Churn_ML_Model.ipynb
│
├── customer_churn.csv
├── churn_cleaned.csv
│
├── churn_logistic_model.pkl
├── churn_scaler.pkl
│
└── README.md