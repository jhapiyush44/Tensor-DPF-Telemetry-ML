# 🚗 Vehicle Telemetry ML Pipeline

An end-to-end production-style machine learning system that predicts **Diesel Particulate Filter (DPF) soot levels** and **regeneration needs** using simulated vehicle telemetry data.

---

## 🔗 Live Demo (Hugging Face)

👉 **Try the API here:**
https://huggingface.co/spaces/jhapiyush44/Tensor-DPF-Telementry-ML

👉 Test endpoints:

* `http://jhapiyush44-tensor-dpf-telementry-ml.hf.space/health` **GET**
* `http://jhapiyush44-tensor-dpf-telementry-ml.hf.space/predict/soot-load` **POST**

---

## 🔥 Key Highlights

* 📊 ~3.9M rows of synthetic telemetry data
* 🤖 Dual-model system (Regression + Classification)
* ⚡ FastAPI inference API (real-time)
* 🔁 CI/CD pipeline (GitHub Actions)
* 📦 Dockerized deployment

---

## 🎯 Problem Statement

Diesel vehicles accumulate soot in their DPF systems. Poor regeneration timing leads to:

* Reduced engine efficiency
* Increased emissions
* Potential system failure

This system predicts:

* **Soot load (%)**
* **Whether regeneration is required**

---

## ⚙️ System Overview

```text
Data → Features → Model → API → Deployment
```

---

## 🧠 Approach

### Data Simulation

* Multi-sensor telemetry (RPM, speed, temp, load)
* Realistic soot accumulation + regeneration cycles
* Noise, drift, and edge cases included

---

### Feature Engineering

* Rolling temperature features
* Load ratios (idle / high load)
* Trip behavior metrics

---

### Modeling

* RandomForest vs XGBoost
* Time-based train/validation/test split
* Removed leakage features:

  * `diff_pressure`
  * `minutes_since_regen`

---

## 📊 Results

| Model        | MAE   | F1 Score |
| ------------ | ----- | -------- |
| RandomForest | ~0.52 | ~0.67    |
| XGBoost      | ~0.52 | ~0.67    |

👉 Final models selected based on validation performance

---

## 🔍 Feature Insights

* Regressor → temperature + load behavior
* Classifier → high load + exhaust temperature

👉 Matches real-world DPF behavior

---

## 🚀 API Example

### Endpoint:

```
POST /predict/soot-load
```

### Input:

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

### Output:

```json
{
  "soot_load_percent": 100.0,
  "confidence_interval": 0.0,
  "regen_recommended": false
}
```

---

## 🧪 CI/CD

* GitHub Actions
* Runs on push & PR
* Includes:

  * Linting (flake8)
  * Testing (pytest)

---

## ⚠️ Deployment Note (Important)

Due to **Hugging Face free tier limitations**:

* Large binary files (`.pkl`) cannot be pushed via Git
* Models were **uploaded manually via HF UI**

👉 This ensures:

* fast startup
* avoids memory and storage issues

👉 Locally, the full pipeline (data → training → inference) works end-to-end.

---

## ▶️ Run Locally

### 1. Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

### 2. Create environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run full pipeline

```bash
python data_generation/generate_data.py
python pipeline/feature_engineering.py
python models/train.py
```

---

### 5. Start API

```bash
uvicorn api.app:app --reload
```

---

### 6. Open docs

```
http://127.0.0.1:8000/docs
```

---

## 💡 Key Engineering Decisions

* Removed feature leakage after debugging unrealistic performance
* Used time-series split to avoid data leakage
* Added output constraints (0–100%)
* Handled NumPy serialization issues in API
* Added fallback for model-specific behavior (RF vs XGB)

---

## 🚀 Future Work

* SHAP explainability
* Real-time streaming (Kafka)
* Monitoring & drift detection
* Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

Piyush Jha

---
