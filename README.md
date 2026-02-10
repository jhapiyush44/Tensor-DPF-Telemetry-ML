# Vehicle Telemetry ML Pipeline

A complete end-to-end Machine Learning system that generates synthetic vehicle telemetry data, performs advanced feature engineering, trains predictive models, and deploys them via a FastAPI inference service. This pipeline predicts soot load levels and regeneration requirements for vehicle exhaust systems.

---

## Architecture Overview

The system follows a clean, modular architecture with four main stages:

```
Data Generation → Feature Engineering → Model Training → API Deployment
     ↓                    ↓                    ↓                ↓
generate_data.py  feature_engineering.py    train.py      uvicorn (FastAPI)
     ↓                    ↓                    ↓                ↓
CSV files         Parquet features       Trained models    HTTP endpoints
```

**Data Flow:**
1. **Data Generation** (`generate_data.py`): Creates synthetic telemetry, maintenance, and trip data
2. **Feature Engineering** (`feature_engineering.py`): Transforms raw data into ML-ready features
3. **Model Training** (`train.py`): Trains RandomForest models for regression and classification
4. **API Deployment** (`api/app.py`): Serves real-time predictions with FastAPI

---

## Data Generation Explanation

The data generation module creates realistic vehicle telemetry data simulating multiple vehicles over an extended period.

**Configuration Parameters:**
- **N_VEHICLES**: 30 vehicles in the fleet
- **DAYS**: 90 days of historical data
- **FREQ**: 1-minute measurement frequency (measurements taken every minute)

**Total Rows Calculation:**
```
Rows per vehicle = DAYS × Hours per day × Minutes per hour
Rows per vehicle = 90 × 24 × 60 = 129,600 rows
Total dataset = 129,600 rows × 30 vehicles = 3,888,000 rows (~3.9M)
```

**Generated Files:**

| File | Purpose | Rows | Columns |
|------|---------|------|---------|
| `telemetry.csv` | Raw sensor measurements (pressure, temperature, emissions) | ~3.9M | 8 |
| `maintenance.csv` | Maintenance events and schedules | Variable | 5 |
| `trips.csv` | Trip-level aggregations with distance and duration | ~2,700 | 6 |

These files form the foundation for feature engineering and model training.

---

## Memory/RAM Optimization

The default configuration (30 vehicles, 90 days, 1-minute frequency) requires moderate computational resources. Use this guide to adjust data generation parameters based on your available RAM.

**Recommended Configurations by Available RAM:**

| RAM | Vehicles | Days | Frequency | Total Rows | Notes |
|-----|----------|------|-----------|-----------|-------|
| **16GB** | 30 | 90 | 1 min | 3.9M | Full dataset, recommended |
| **16GB** | 50 | 90 | 1 min | 6.5M | Large-scale testing |
| **8GB** | 15 | 90 | 1 min | 1.95M | Comfortable processing |
| **8GB** | 10 | 60 | 1 min | 864K | Conservative approach |
| **4GB** | 5 | 30 | 5 min | 216K | Minimal dataset |
| **4GB** | 10 | 7 | 10 min | 100.8K | Very limited resources |

**How to Adjust:**
Edit the configuration in `data_generation/generate_data.py`:

```python
N_VEHICLES = 30  # Reduce to 15 or 10
DAYS = 90        # Reduce to 30 or 60
FREQ = "1min"    # Change to "5min" or "10min"
```

**General Guidelines:**
- Reducing `DAYS` has the largest impact on dataset size
- Increasing `FREQ` (e.g., from 1 min to 5 min) reduces rows by 5x
- Start conservative and increase if you have headroom

---

## Clean Restart Instructions

If you need to reset the pipeline and regenerate all data, follow these steps to clean your workspace completely.

**Why Clean Restart is Necessary:**
- Old data files may contain obsolete schemas or configurations
- Cached model artifacts can cause loading errors with updated feature sets
- Parquet format can cache schema information, causing conflicts
- Starting fresh ensures reproducibility and prevents subtle bugs

**Clean Restart Steps:**

```powershell
# Windows PowerShell

# Delete generated data files
Remove-Item -Path "data\*.csv" -Force
Remove-Item -Path "data\*.parquet" -Force

# Delete trained models
Remove-Item -Path "models\*.pkl" -Force
Remove-Item -Path "models\*.joblib" -Force
Remove-Item -Path "models\metrics.json" -Force

# Verify cleanup
Get-ChildItem -Path data\
Get-ChildItem -Path models\
```

**Linux/Mac Equivalent:**

```bash
# Delete generated data files
rm -f data/*.csv data/*.parquet

# Delete trained models
rm -f models/*.pkl models/*.joblib models/metrics.json

# Verify cleanup
ls -la data/
ls -la models/
```

After cleanup, proceed with fresh data generation (see **Step-by-Step Pipeline Commands** below).

---

## Setup Instructions

Before running the pipeline, set up your Python environment with all required dependencies.

**Step 1: Create Virtual Environment**

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Step 2: Activate Virtual Environment**

Verify your virtual environment is active (you should see `(venv)` in your terminal prompt).

**Step 3: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


---

## Step-by-Step Pipeline Commands

Execute these commands in sequence to complete the ML pipeline. Each step builds on the previous one.

### Step 1: Generate Synthetic Data
**Purpose:** Create raw vehicle telemetry, maintenance, and trip data  
**Input:** None  
**Output:** `data/telemetry.csv`, `data/maintenance.csv`, `data/trips.csv`  
**Approximate Time:** 30-60 seconds

```bash
python data_generation/generate_data.py
```

Expected output:
```
Generated 3,888,000 telemetry records
Generated 450 maintenance records
Generated 2,700 trip records
✓ Data generation complete
```

### Step 2: Feature Engineering
**Purpose:** Transform raw data into ML-ready features with temporal aggregations  
**Input:** `data/telemetry.csv`, `data/maintenance.csv`, `data/trips.csv`  
**Output:** `data/features.parquet`  
**Approximate Time:** 2-5 minutes

```bash
python pipeline/feature_engineering.py
```

Expected output:
```
Loading data files...
Engineering features...
Computing rolling statistics...
Computing lag features...
✓ Features saved to data/features.parquet
Features shape: (3,888,000, 42)
```

### Step 3: Train Models
**Purpose:** Train RandomForest models for soot load prediction and regen classification  
**Input:** `data/features.parquet`  
**Output:** `models/regressor.pkl`, `models/classifier.pkl`, `models/metrics.json`, `models/feature_cols.pkl`
**Approximate Time:** 5-15 minutes

```bash
python models/train.py
```

Expected output:
```
Loading features...
Splitting data (80/20 train/test)...
Training soot load regressor...
Training regen classifier...
Evaluation metrics saved
✓ Model training complete
```

### Step 4: Launch API Server
**Purpose:** Start FastAPI server for real-time inference  
**Input:** `models/regressor.pkl`, `models/classifier.pkl`  
**Output:** HTTP server on `http://localhost:8000`  
**Approximate Time:** Immediate startup

```bash
uvicorn api.app:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
```

Visit the interactive API docs at: **http://localhost:8000/docs**

---

## API Usage

Once the API server is running, you can make HTTP requests to the following endpoints.

### 1. Health Check
Verify that the API and models are ready for inference.

**Endpoint:** `GET /health`

**Example Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

### 2. Single Prediction - Soot Load
Predict soot load percentage for a single vehicle measurement.

**Endpoint:** `POST /predict/soot-load`

**Request Body:**
```json
{
  "engine_load": 0.6,
  "rpm": 2000,
  "speed": 60,
  "exhaust_temp_pre": 500,
  "exhaust_temp_post": 470,
  "flow_rate": 50,
  "ambient_temp": 30,
  "diff_pressure": 10,
  "temp_roll_mean_10": 480,
  "temp_roll_mean_60": 470,
  "temp_delta": 30,
  "minutes_since_regen": 120,
  "idle_ratio_30": 0.1,
  "high_load_ratio_30": 0.5
}

```

**Response:**
```json
{
  "soot_load_percent": 22.17,
  "regen_recommended": false
}
```

### 3. Batch Predictions
Predict soot load and regeneration needs for multiple measurements in one request.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "measurements": [
  {
    "engine_load": 0.60,
    "rpm": 2000,
    "speed": 55,
    "exhaust_temp_pre": 510,
    "exhaust_temp_post": 480,
    "flow_rate": 45,
    "ambient_temp": 30,
    "diff_pressure": 10,
    "temp_roll_mean_10": 500,
    "temp_roll_mean_60": 490,
    "temp_delta": 30,
    "minutes_since_regen": 100,
    "idle_ratio_30": 0.1,
    "high_load_ratio_30": 0.5
  },
  {
    "engine_load": 0.85,
    "rpm": 2800,
    "speed": 85,
    "exhaust_temp_pre": 620,
    "exhaust_temp_post": 580,
    "flow_rate": 65,
    "ambient_temp": 33,
    "diff_pressure": 25,
    "temp_roll_mean_10": 610,
    "temp_roll_mean_60": 600,
    "temp_delta": 40,
    "minutes_since_regen": 300,
    "idle_ratio_30": 0.01,
    "high_load_ratio_30": 0.85
  }
]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "soot_load_percent": 22.17,
      "regen_recommended": false
    },
    {
      "soot_load_percent": 54.74,
      "regen_recommended": false
    }
  ]
}
```

### Interactive API Documentation
FastAPI automatically generates interactive Swagger documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

Use these interfaces to test endpoints directly from your browser with auto-complete and request validation.

---

## Design Decisions

This section explains key architectural and technical choices made in the system design.

### Why Apache Parquet for Features?

**Decision:** Store engineered features in Parquet format instead of CSV

**Rationale:**
- **Compression:** Parquet reduces file size by 70-80% compared to CSV (4GB → 0.8GB)
- **Performance:** 10-50x faster read times for analytical queries
- **Type Safety:** Preserves data types (float32, int64) without re-parsing
- **Columnar Storage:** Enables efficient feature subset loading
- **Streaming:** Supports lazy evaluation for large datasets

**Trade-off:** Requires `pyarrow` dependency; not human-readable. Mitigated by using CSV for raw data (easier inspection).

### Why Time-Based Train/Test Split?

**Decision:** Split data by timestamp (80% earlier data, 20% recent data) instead of random split

**Rationale:**
- **Realistic Evaluation:** Simulates real-world scenario where you train on historical data and predict future
- **Prevents Data Leakage:** Temporal dependencies within vehicle trips stay together
- **Trend Detection:** Tests model's ability to handle temporal patterns
- **Maintenance Cycles:** Captures degradation patterns that emerge over time

**Alternative Considered:** Random stratified split (not suitable for time-series data)

### Why Compute Features Inside API?

**Decision:** API accepts raw inputs and computes features on-the-fly instead of requiring pre-computed features

**Rationale:**
- **Consistency:** Same feature computation logic used during training and inference
- **Simplicity:** Clients send raw sensor values; API handles transformation
- **Flexibility:** Easy to update feature logic without client changes
- **Portability:** Models remain self-contained with their feature requirements

**Trade-off:** Slight latency increase (typically <50ms for single prediction); mitigated by batch endpoint for bulk requests.

---

## Project Structure

```
Tensor Internship/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory
│   ├── telemetry.csv                  # Raw sensor data (~3.9M rows)
│   ├── maintenance.csv                # Maintenance events
│   ├── trips.csv                      # Trip aggregations
│   └── features.parquet               # Engineered features (generated)
│
├── data_generation/
│   └── generate_data.py               # Synthetic data generator
│                                      # Creates telemetry, maintenance, trips
│
├── pipeline/
│   └── feature_engineering.py         # Feature transformation & aggregation
│                                      # Converts CSV → Parquet with ML features
│
├── models/
│   ├── train.py                       # Model training script
│   │                                  # Trains RandomForest regresssor/classifier
│   ├── soot_load_model.pkl            # Trained soot load predictor (generated)
│   ├── regen_classifier.pkl           # Trained regen classifier (generated)
│   └── metrics.json                   # Model performance metrics (generated)
│
├── tests/
│   └── test_features.py               # Feature engineering unit tests
│
└── api/
    ├── app.py                         # FastAPI application
    │                                  # Endpoints: /health, /predict/soot-load, /predict/batch
    └── __pycache__/                   # Python cache (auto-generated)
```

**Directory Descriptions:**
- **data/**: Input/output for data generation and feature engineering
- **data_generation/**: Scripts for synthetic data creation
- **pipeline/**: Feature engineering and data transformation logic
- **models/**: Training scripts and serialized model artifacts
- **api/**: FastAPI application for inference deployment
- **tests/**: Unit tests and validation scripts

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Core implementation |
| **Data Processing** | pandas | Data manipulation & aggregation |
| **ML Framework** | scikit-learn | RandomForest models & metrics |
| **API Framework** | FastAPI | HTTP endpoints & inference |
| **Model Serialization** | joblib | Save/load trained models |
| **Data Format** | Apache Parquet | Efficient feature storage |
| **Data Format** | CSV | Raw data interchange |

**Key Versions:**
- pandas ≥ 1.3.0 (for performance and API)
- scikit-learn ≥ 1.0.0 (RandomForest stability)
- FastAPI ≥ 0.95.0 (async support)
- pyarrow ≥ 8.0.0 (Parquet support)

---

## Summary

This end-to-end ML pipeline demonstrates a complete workflow for building, training, and deploying vehicle telemetry models:

✓ **Data:** Generates 3.9M rows of realistic synthetic telemetry across 30 vehicles over 90 days

✓ **Features:** Engineers 40+ advanced features including rolling statistics, lag features, and temporal aggregations

✓ **Models:** Trains two RandomForest models—regression for soot load prediction and classification for regeneration detection

✓ **Deployment:** Serves predictions via FastAPI with health checks, single, and batch inference endpoints

✓ **Optimization:** Configurable parameters for different computational resources and use cases

**Next Steps:**
1. Follow the **Setup Instructions** to configure your environment
2. Execute the **Step-by-Step Pipeline Commands** in order
3. Test endpoints using the **API Usage** section
4. Customize parameters in config files based on your hardware

For questions or issues, refer to the relevant script documentation or test files.

---

**Last Updated:** February 2026  
**Status:** Production-ready
