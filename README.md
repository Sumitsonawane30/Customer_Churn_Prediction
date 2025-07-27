# 📉 Customer Churn Prediction App

A machine learning-based Streamlit web application that predicts whether a customer is likely to **churn (leave the service)** based on their usage and account details.

## 🚀 Demo

[🔗 View Live App ](https://customerchurnprediction-8cmbhcplqb92fvzkk5zrzh.streamlit.app/)

---

## 🧠 About the Project

This project uses a **Random Forest Classifier** trained on a telecom customer dataset. The model predicts customer churn based on 19 key input features such as tenure, contract type, monthly charges, internet service, etc.

The project includes:
- Data preprocessing (label encoding, missing value handling)
- Model training (with SMOTE to balance the dataset)
- Saving the trained model and encoders using `pickle`
- A Streamlit frontend for user interaction

---

## 📂 Folder Structure

Customer_Churn_Prediction/
├── app.py # Streamlit web app

├── model.pkl # Trained RandomForestClassifier model

├── encoders.pkl # Encoders for categorical columns

├── churn_prediction.ipynb # Jupyter notebook used for training

├── requirements.txt # Python dependencies

└── README.md # Project overview

---

## 🏗️ Features

- 📋 Takes 19 inputs from the user (via dropdowns, sliders, and text boxes)
- 🔄 Preprocesses data with same encoders used during training
- 🤖 Loads trained model and predicts churn
- 📊 Displays results directly in the Streamlit interface

---
