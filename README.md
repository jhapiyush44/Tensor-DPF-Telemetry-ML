# 🚗 Tensor-DPF Telemetry ML Pipeline

> **Production-ready, end-to-end ML system** for predicting Diesel Particulate Filter (DPF) soot accumulation and regeneration needs from vehicle telemetry data.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-FFD21F?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/jhapiyush44/Tensor-DPF-Telementry-ML)

---

## 📌 Overview

Diesel engines accumulate soot in their **Diesel Particulate Filters (DPF)**. Poorly timed regeneration cycles cause reduced engine efficiency, elevated emissions, and costly system failures.

This project simulates the full ML lifecycle — from **raw telemetry ingestion** to a **live inference API** — to predict:

| Target | Type | Description |
|--------|------|-------------|
| `soot_load_percent` | Regression | Current DPF soot fill level (0–100%) |
| `regen_recommended` | Classification | Whether active regeneration should trigger |

**Dataset scale:** ~3.9 million synthetic telemetry rows with realistic noise, drift, and regeneration cycles.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Pipeline                             │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Data         │───▶│ Feature      │───▶│ Model        │  │
│  │ Generation   │    │ Engineering  │    │ Training     │  │
│  │              │    │              │    │              │  │
│  │ 3.9M rows    │    │ 40+ features │    │ RF + XGBoost │  │
│  │ Multi-sensor │    │ Lag, rolling │    │ Time-split   │  │
│  │ Noise + drift│    │ Load ratios  │    │ CV + tuning  │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                 │           │
└─────────────────────────────────────────────────┼───────────┘
                                                  │
                                    ┌─────────────▼────────────┐
                                    │     FastAPI Service      │
                                    │  /predict/soot-load      │
                                    │  /health                 │
                                    │  Batch + single inference│
                                    └─────────────┬────────────┘
                                                  │
                               ┌──────────────────▼──────────────┐
                               │         Docker Container         │
                               │   + GitHub Actions CI/CD         │
                               │   + Hugging Face Deployment      │
                               └─────────────────────────────────┘
```

---

## 🔥 Key Results

| Model | Task | MAE | R² | F1 Score |
|-------|------|-----|----|----------|
| RandomForest | Regression (soot load) | ~0.52 | 0.92 | — |
| RandomForest | Classification (regen) | — | — | 0.89 |
| XGBoost | Regression (soot load) | ~0.52 | — | — |
| XGBoost | Classification (regen) | — | — | ~0.67 |

> ✅ Final models selected based on held-out time-series validation performance.  
> ⚠️ Leakage features (`diff_pressure`, `minutes_since_regen`) were removed after identifying unrealistic early performance.

---

## 🧠 Technical Deep Dive

### 1. Data Simulation (`data_generation/`)

Synthetic multi-sensor telemetry designed to mirror real DPF operating conditions:

- Engine RPM, vehicle speed, engine load
- Exhaust temperature (pre/post DPF)
- Exhaust flow rate, ambient temperature
- Realistic **soot accumulation curves** and **active regeneration cycles**
- Injected sensor noise, temporal drift, and edge cases

### 2. Feature Engineering (`pipeline/`)

40+ engineered features capturing temporal and behavioral patterns:

| Category | Features |
|----------|----------|
| Rolling statistics | `temp_roll_mean_10`, `temp_roll_mean_60` |
| Thermal delta | `temp_delta` (pre vs post DPF) |
| Load behaviour | `idle_ratio_30`, `high_load_ratio_30` |
| Trip context | High-load %, speed variance |

**Time-based train/val/test split** used throughout to prevent data leakage.

### 3. Modeling (`models/`)

- **RandomForestRegressor** and **RandomForestClassifier** as primary models
- **XGBoost** benchmarked for comparison
- Hyperparameter tuning via grid search + cross-validation
- Output constrained to `[0, 100]` for soot load predictions
- NumPy serialization handled explicitly for API compatibility

---

## 🚀 Live Demo

**🤗 Hugging Face Space:** [jhapiyush44/Tensor-DPF-Telementry-ML](https://huggingface.co/spaces/jhapiyush44/Tensor-DPF-Telementry-ML)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Service health check |
| `/predict/soot-load` | `POST` | Single + batch inference |
| `/docs` | `GET` | Interactive Swagger UI |

> **Note on deployment:** Model `.pkl` files were uploaded manually via the Hugging Face UI due to free-tier Git LFS limitations. Locally, the full pipeline runs end-to-end without any manual steps.

---

## 📡 API Reference

### `POST /predict/soot-load`

**Request body:**

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

**Response:**

```json
{
  "soot_load_percent": 73.4,
  "confidence_interval": 4.2,
  "regen_recommended": false
}
```

---

## ▶️ Run Locally

### Prerequisites

- Python 3.10+
- Docker (optional, for containerised run)

### Setup

```bash
# 1. Clone
git clone https://github.com/jhapiyush44/Tensor-DPF-Telemetry-ML.git
cd Tensor-DPF-Telemetry-ML

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Generate synthetic telemetry data (~3.9M rows)
python data_generation/generate_data.py

# Feature engineering — produces Parquet files (80% smaller than CSV)
python pipeline/feature_engineering.py

# Train models (RandomForest + XGBoost, regression + classification)
python models/train.py
```

### Start the API

```bash
uvicorn api.app:app --reload
# Swagger docs → http://127.0.0.1:8000/docs
```

### Run with Docker

```bash
docker build -t tensor-dpf-ml .
docker run -p 8000:8000 tensor-dpf-ml
```

---

## 🧪 Testing & CI/CD

```bash
# Run tests locally
pytest tests/

# Linting
flake8 .
```

The **GitHub Actions** pipeline runs automatically on every push and pull request:

- ✅ Linting with `flake8`
- ✅ Unit + integration tests with `pytest`

---

## 🗂️ Project Structure

```
Tensor-DPF-Telemetry-ML/
├── .github/
│   └── workflows/         # GitHub Actions CI/CD
├── api/
│   └── app.py             # FastAPI inference service
├── data_generation/
│   └── generate_data.py   # Synthetic telemetry simulation
├── models/
│   └── train.py           # Model training + evaluation
├── pipeline/
│   └── feature_engineering.py  # Feature construction
├── tests/                 # Pytest test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔧 Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Apache Parquet for feature storage | 80% size reduction vs CSV; faster I/O |
| Time-series train/val/test split | Prevents temporal data leakage |
| Removed `diff_pressure` + `minutes_since_regen` | Were directly correlated with target — caused leakage |
| Output clipping to `[0, 100]` | Enforces physical validity of soot load predictions |
| RandomForest as primary model | Robust to noise; interpretable feature importances |

---

## 🔮 Roadmap

- [ ] SHAP explainability for feature attribution
- [ ] Real-time streaming with Apache Kafka
- [ ] Model monitoring & drift detection (Evidently AI)
- [ ] Cloud deployment on AWS ECS / GCP Cloud Run
- [ ] Dashboard for live soot load visualisation

---

## 👨‍💻 Author

**Piyush Jha** — ML Engineer  
[GitHub](https://github.com/jhapiyush44) · [LinkedIn](https://www.linkedin.com/in/piyush-jha-3904a81a6/) · jhapiyush44@gmail.com

---

*If you found this project useful, consider leaving a ⭐ — it helps others discover the work!*
