
# 📈 Predict‑CPI: Consumer Price Index Prediction Web App

A full-stack Flask application that uses a machine learning model (Lasso Regressor) to forecast the **General CPI index** based on CPI component values (e.g., cereals, clothing, rural, urban, etc.).

---

## 🧭 Table of Contents

1. [Overview](#overview)  
2. [Motivation](#motivation)  
3. [Dataset](#dataset)  
4. [Model Training](#model-training)  
5. [Flask App](#flask-app)  
6. [Frontend (HTML)](#frontend-html)  
7. [How to Run Locally](#how-to-run-locally)  
8. [API Endpoints](#api-endpoints)  
9. [Future Work](#future-work)  
10. [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

This project predicts the **General CPI index** using 29 numeric CPI components (e.g., rural, urban, clothing, food). The model is built using Lasso regression and served via a Flask-powered web interface.

---

## 💡 Motivation

Accurate CPI forecasts are essential for:

- 📉 Inflation monitoring  
- 💰 Budget planning  
- 🏛️ Economic policy design  

This app enables quick, intuitive predictions from user-supplied CPI component data.

---

## 📚 Dataset

- **Source**: Kaggle's All India CPI dataset (2013–2020)  
- **Features**: 29 detailed CPI categories (e.g., “Cereals and products”, “Health”, “Rural”, “Urban”, “Combined”)  
- **Target**: `General index` (the label to predict)  
- **Preprocessing**:
  - Pivoted `Sector` rows into separate `Rural`, `Urban`, and `Combined` columns  
  - Interpolated missing values  
  - Chose numeric features only

---

## 🛠️ Model Training

- **Model used**: Lasso Regression  
- **Training script**:  
  - Reads and preprocesses data  
  - Trains Lasso using `feature_columns`  
  - Evaluates using RMSE and R²  
  - Saved using `pickle` as `model.pkl`
    
---

## 🌐 Flask App

- **Backend**: `app.py`
  - Loads `model.pkl`  
  - Serves prediction form (GET `/`)  
  - Handles submissions and returns predictions (POST `/predict`)  
  - Includes validation, error handling, and fallback dummy model  

- **Folder structure**:
  ```
  .
  ├── app.py
  ├── model.pkl
  ├── requirements.txt
  ├── templates/
  │   └── index.html
  ```

---

## 🖥️ Frontend (HTML)

- **File**: `templates/index.html`  
- **Features**:
  - Auto-generated input fields for all 29 CPI features  
  - Submits via `/predict` POST  
  - Displays rounded predictions  

---

## 🚀 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/vinay‑vk‑kumar/Predict‑CPI.git
   cd Predict‑CPI
   ```

2. **Install dependencies** (recommended via venv):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```
   → Visit `http://127.0.0.1:5000` in your browser

4. **Try making a prediction**:
   - Fill values for 29 CPI components  
   - Submit and view the **General CPI** prediction

---

## 🧪 API Endpoints

- `GET /api/model-info` → returns model type and feature count  
- `GET /api/features` → returns list of all feature column names  
- `GET /api/sample-data` → returns example values for testing  
- `POST /predict` → returns JSON with `prediction`, `model_type`, timestamp, or validation errors

---

## 🔮 Future Work

- 🔃 Add support for **time-based features** (Month, Year)  
- 🔧 Include **optional feature selection**  
- 📈 Build **visual tools** to compare predicted vs actual CPI  
- ☁️ Deploy on **Heroku / AWS** for public use

---

## 🤝 Acknowledgments

- Thanks to **Kaggle** for the CPI dataset  
- Inspired by SmartBridge’s ML coursework  

---
