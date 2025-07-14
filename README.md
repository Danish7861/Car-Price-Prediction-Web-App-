# 🚘 Car Price Prediction Web App using FastAPI & Machine Learning

![Car Price Prediction UI](car price image.png)

A modern, AI-powered web application that predicts used car prices in PKR using a trained Machine Learning model and a sleek HTML/CSS frontend. The backend is built with FastAPI, and the UI is designed for an engaging user experience with dropdowns and clean layout.

---

## 🔍 Project Overview

This project uses historical car sales data to train a machine learning model that predicts the price of a used car based on features like brand, year, fuel type, and more. The model is deployed using FastAPI with a professional frontend built using HTML, CSS, and Jinja2.

---

## ✨ Features

- 🎯 Predict car prices in PKR with high accuracy
- ✅ Handles missing values using `New_Price_Missing`
- 🔄 Converts model output from `log1p` using `np.expm1`
- 🧠 ML pipeline: preprocessing + model in one `.pkl` file
- 🎨 Colorful, responsive UI with dropdowns and clean design
- ⚡ Real-time prediction via FastAPI

---

## 🧠 Machine Learning Highlights

**Car Sales Analysis Using Machine Learning**

- 📊 Conducted EDA on historical car sales data to identify trends, seasonality, and feature importance.
- 🧹 Preprocessed data with outlier detection, missing value treatment, and categorical encoding.
- 🧪 Built and compared Linear Regression, Random Forest, and XGBoost models.
- ✅ Final model: Random Forest (selected based on RMSE and R² scores).
- 📈 Delivered business insights and visualizations for stakeholders.

---

## 🛠 Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: HTML5, CSS3, Jinja2 Templates
- **Machine Learning**: scikit-learn, pandas, numpy
- **Model**: Random Forest Regressor with ColumnTransformer preprocessing

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the FastAPI server
uvicorn main:app --reload

# 3. Visit the app
http://127.0.0.1:8000
