# 🚗 Vehicle Telemetry ML Pipeline

An end-to-end production-style machine learning system that predicts **Diesel Particulate Filter (DPF) soot levels** and **regeneration needs** using simulated vehicle telemetry data.

---

## 🔥 Key Highlights

* 📊 ~3.9M rows of synthetic fleet telemetry data
* 🤖 Dual-model system:

  * Regression → soot load prediction
  * Classification → regeneration recommendation
* ⚡ FastAPI-based inference API (real-time + batch)
* 🧠 Feature engineering with time-series signals
* 🔁 CI/CD pipeline using GitHub Actions
* 📦 Dockerized deployment on Hugging Face Spaces

---

## 🎯 Problem Statement

Diesel vehicles accumulate soot in their DPF systems.
If not regenerated at the right time, it can lead to:

* Reduced engine efficiency
* Increased emissions
* Potential system failure

This project predicts:

* **Soot load (%)**
* **Whether regeneration is needed**

---

## ⚙️ System Overview

```text
Data Generation → Feature Engineering → Model Training → API → Deployment
```

* Synthetic telemetry simulates real-world vehicle behavior
* Time-series features capture driving patterns
* Models are trained and deployed for real-time inference

---

## 🧠 Approach

### 1. Data Simulation

* Multi-sensor telemetry (RPM, speed, temperature, load)
* Realistic soot accumulation + regeneration cycles
* Noise, drift, and edge cases included

---

### 2. Feature Engineering

* Rolling temperature averages
* Load ratios (idle / high load)
* Time since last regeneration
* Trip-based features

---

### 3. Modeling

Two models are trained:

* **Regression (Soot Load)**

  * RandomForest vs XGBoost
* **Classification (Regen Needed)**

  * RandomForest vs XGBoost

### ⚠️ Important:

* Removed **data leakage features** (`diff_pressure`, `minutes_since_regen`)
* Ensured models learn real patterns, not shortcuts

---

## 📊 Results

| Model        | MAE (Regression) | F1 Score (Classification) |
| ------------ | ---------------- | ------------------------- |
| RandomForest | ~0.52            | ~0.67                     |
| XGBoost      | ~0.52            | ~0.67                     |

### ✅ Final Model:

* Regressor → XGBoost / RandomForest (based on validation)
* Classifier → XGBoost / RandomForest

---

## 🔍 Feature Importance Insights

### Regressor:

* Temperature trends
* Engine load behavior
* Driving patterns

### Classifier:

* High load ratio
* Exhaust temperature
* Engine activity

👉 Matches real-world DPF behavior

---

## 🚀 API Endpoints

### 🔹 Health Check

```
GET /health
```

---

### 🔹 Predict Soot Load

```
POST /predict/soot-load
```

#### Example Input:

```json
{
  "engine_load": 0.6,
  "rpm": 2000,
  "speed": 60,
  "exhaust_temp_pre": 500,
  "exhaust_temp_post": 470,
  "flow_rate": 50,
  "ambient_temp": 30,
  "temp_roll_mean_10": 480,
  "temp_roll_mean_60": 470,
  "temp_delta": 30,
  "idle_ratio_30": 0.1,
  "high_load_ratio_30": 0.5
}
```

#### Example Output:

```json
{
  "soot_load_percent": 100.0,
  "confidence_interval": 0.0,
  "regen_recommended": false
}
```

---

## 🧪 CI/CD Pipeline

* GitHub Actions workflow
* Runs on push & pull request
* Includes:

  * Linting (flake8)
  * Automated testing (pytest)

---

## 📦 Deployment

* Dockerized application
* Deployed on Hugging Face Spaces
* Pre-trained models used for fast startup

---

## ⚠️ Key Engineering Decisions

* Used **time-based split** to avoid data leakage
* Removed **leaky features** after detecting unrealistic performance
* Added **output constraints (0–100%)** for safe predictions
* Handled **NumPy → JSON serialization issues** in API
* Implemented fallback for **model-specific behavior (RF vs XGB)**

---

## 💡 What Makes This Project Strong

* Realistic system simulation (not random data)
* Proper ML validation (no leakage)
* Model comparison and selection
* Production-ready API
* CI/CD + deployment
* Debugging real-world issues (memory, serialization, model loading)

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt

python data_generation/generate_data.py
python pipeline/feature_engineering.py
python models/train.py

uvicorn api.app:app --reload
```

---

## 📌 Future Improvements

* SHAP-based explainability
* Real-time streaming pipeline (Kafka)
* Cloud deployment (AWS/GCP)
* Monitoring + drift detection

---

## 👨‍💻 Author

Piyush Jha
Aspiring Data Scientist | IIT Madras BS Data Science

---
