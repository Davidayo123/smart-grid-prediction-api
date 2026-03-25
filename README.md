# Smart Grid Hybrid AI — Energy Prediction Service

A real-time energy consumption prediction service using a **GRU + LightGBM hybrid model** with Bayesian uncertainty estimation.

## Architecture

| Component | Role |
|---|---|
| **TE-GRU** (TFLite) | Learns temporal patterns from 24-hour sensor windows |
| **LightGBM** | Corrects GRU residuals for sharper accuracy |
| **MH Sampler** | Estimates prediction confidence bounds via MCMC |
| **FastAPI** | HTTP API for manual testing + serves dashboard |
| **MQTT Bridge** | Production data flow (subscribes to `room/sensors`) |

## Performance

- **R² = 0.97** on 20K-row dataset
- **MAE ≈ 0.20 kW**, RMSE ≈ 0.32 kW
- **90%+ confidence interval coverage**

## Features

- 18 engineered features: sensor readings, time cyclicals, lag windows, rolling stats
- Date/time input — control the temporal context for predictions
- Adaptive uncertainty bounds that tighten as prediction history accumulates

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Manual sensor input → prediction |
| `GET` | `/predict_next` | Step through CSV dataset sequentially |
| `GET` | `/` | Serves the test dashboard UI |

### POST `/predict` — Request Body

```json
{
  "temperature_c": 28.0,
  "humidity": 60.0,
  "lux": 400.0,
  "occupancy": 1,
  "datetime_str": "2024-06-15T14:30"
}
```

## Local Setup

```bash
pip install -r requirements.txt
python test_prediction_api.py
```

The dashboard opens at `http://localhost:5000`.

## Tech Stack

Python 3.10+ · FastAPI · TensorFlow Lite · LightGBM · NumPy · Pandas · Paho MQTT
