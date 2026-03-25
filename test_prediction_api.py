"""
test_prediction_api.py — Smart Grid Hybrid AI Prediction Service
=================================================================

Dual-protocol ML service:
  • MQTT (primary)  — subscribes to room/sensors, runs inference on every
                       sensor message, publishes contract-compliant results
                       to room/ml/predictions for rule_engine, mqtt_logger,
                       and the dashboard.
  • HTTP  (testing) — POST /predict lets test.py and test_dashboard.html
                       send manual sensor inputs and receive predictions.
                       GET  /predict_next advances through the CSV for
                       quick sequential testing.

Port: 8000 (same port the rest of the system expects).
"""

import os
import json
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import paho.mqtt.client as mqtt


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = "."
CSV_PATH = os.path.join(BASE_DIR, "abs_smart_grid_dataset_20k.csv")
WINDOW_SIZE = 24

# MQTT
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_CLIENT_ID = "ml-prediction-service"
TOPIC_SENSORS = "room/sensors"
TOPIC_ML_PREDICTIONS = "room/ml/predictions"

# Energy thresholds (for contract compatibility)
PEAK_DEMAND_KW = float(os.environ.get("PEAK_DEMAND_KW", 2.4))

# Simulation index (CSV row pointer)
current_sim_index = WINDOW_SIZE


# ═══════════════════════════════════════════════════════════════════
# LOAD AI ASSETS ON MODULE IMPORT
# ═══════════════════════════════════════════════════════════════════
print("Loading AI assets...")
df_sim = pd.read_csv(CSV_PATH)
if 'Luminous_Intensity' in df_sim.columns and 'Luminous_Intensity_Lux' not in df_sim.columns:
    df_sim = df_sim.rename(columns={'Luminous_Intensity': 'Luminous_Intensity_Lux'})
df_sim['Timestamp'] = pd.to_datetime(df_sim['Timestamp'])

scaler = joblib.load(os.path.join(BASE_DIR, "v2_scalar.joblib"))
lgb_corrector = joblib.load(os.path.join(BASE_DIR, "v3_residual_lightgbm_model.joblib"))

interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "v2_final_te_gru_model.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[OK] AI assets loaded")


# ═══════════════════════════════════════════════════════════════════
# BAYESIAN UNCERTAINTY ESTIMATOR
# ═══════════════════════════════════════════════════════════════════
class MHUncertaintyEstimator:
    """Metropolis-Hastings sampler for posterior sigma estimation.

    Uses log-normal proposals (guarantees positivity) and a weakly
    informative half-normal prior on sigma.
    """
    def __init__(self, iterations=500, burn_in=100, proposal_scale=0.15):
        self.iterations = iterations
        self.burn_in = burn_in
        self.proposal_scale = proposal_scale

    def estimate_sigma(self, residuals: np.ndarray, initial_sigma: float) -> float:
        if len(residuals) < 2:
            return max(0.01, initial_sigma)
        initial_sigma = max(0.001, initial_sigma)

        n = len(residuals)
        ss = float(np.sum(residuals ** 2))  # precompute sum-of-squares

        def log_posterior(sigma):
            if sigma <= 1e-6:
                return -np.inf
            # Gaussian likelihood: residuals ~ N(0, sigma)
            ll = -n * np.log(sigma) - ss / (2.0 * sigma ** 2)
            # Weakly informative half-normal prior (scale=1.0)
            lp = -0.5 * (sigma ** 2)
            return ll + lp

        current_sigma = initial_sigma
        current_lp = log_posterior(current_sigma)
        chain = []

        rng = np.random.RandomState(int(abs(ss * 1e4)) % (2 ** 31))

        for _ in range(self.iterations):
            # Log-normal proposal — always positive
            proposed = current_sigma * np.exp(
                rng.normal(0, self.proposal_scale)
            )
            proposed_lp = log_posterior(proposed)

            # Hastings ratio includes Jacobian correction for log-normal
            log_alpha = (proposed_lp - current_lp
                         + np.log(proposed) - np.log(current_sigma))

            if np.log(rng.rand()) < log_alpha:
                current_sigma = proposed
                current_lp = proposed_lp

            chain.append(current_sigma)

        post_burn = chain[self.burn_in:]
        if len(post_burn) > 0:
            return float(np.median(post_burn))  # median is more robust
        return current_sigma


class ResidualTracker:
    """Rolling window of (predicted - actual) residuals for uncertainty."""
    def __init__(self, max_size=200):
        self.residuals: list[float] = []
        self.max_size = max_size

    def add(self, predicted: float, actual: float | None):
        if actual is not None and not np.isnan(actual):
            self.residuals.append(predicted - actual)
            if len(self.residuals) > self.max_size:
                self.residuals.pop(0)

    def get(self) -> np.ndarray | None:
        return np.array(self.residuals) if len(self.residuals) >= 3 else None


mh_estimator = MHUncertaintyEstimator()
residual_tracker = ResidualTracker()


def unscale_prediction(scaled_pred: float, y_raw: np.ndarray, y_scaled: np.ndarray) -> float:
    m, c = np.polyfit(y_scaled, y_raw, 1)
    return float((scaled_pred * m) + c)


# ═══════════════════════════════════════════════════════════════════
# CORE PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════
def run_prediction(live_window: pd.DataFrame) -> dict:
    """Run the full GRU + LightGBM hybrid pipeline on a 25-row window.

    Returns a dict with all prediction outputs.
    """
    current_hour_data = live_window.iloc[-1].copy()

    # Engineer Time Features
    live_window['Hour'], live_window['DayOfWeek'], live_window['Month'] = live_window['Timestamp'].dt.hour, live_window['Timestamp'].dt.dayofweek, live_window['Timestamp'].dt.month
    live_window['Hour_Sin'], live_window['Hour_Cos'] = np.sin(2 * np.pi * live_window['Hour'] / 24), np.cos(2 * np.pi * live_window['Hour'] / 24)
    live_window['DayOfWeek_Sin'], live_window['DayOfWeek_Cos'] = np.sin(2 * np.pi * live_window['DayOfWeek'] / 7), np.cos(2 * np.pi * live_window['DayOfWeek'] / 7)
    live_window['Month_Sin'], live_window['Month_Cos'] = np.sin(2 * np.pi * live_window['Month'] / 12), np.cos(2 * np.pi * live_window['Month'] / 12)
    live_window['Is_Weekend'] = (live_window['DayOfWeek'] >= 5).astype(int)

    live_window['Lag_1h'] = live_window['Energy_kW'].shift(1).bfill()
    live_window['Lag_2h'] = live_window['Energy_kW'].shift(2).bfill()
    live_window['Lag_3h'] = live_window['Energy_kW'].shift(3).bfill()
    live_window['Lag_24h'] = live_window['Energy_kW'].shift(24).bfill()
    live_window['Rolling_Mean_3h'] = live_window['Energy_kW'].shift(1).rolling(3, min_periods=1).mean()
    live_window['Rolling_Std_3h'] = live_window['Energy_kW'].shift(1).rolling(3, min_periods=1).std().fillna(0)


    expected_gru = ['Energy_kW', 'Temperature_C', 'Humidity_%', 'Luminous_Intensity_Lux', 'Lag_1h', 'Lag_2h', 'Lag_3h', 'Lag_24h', 'Rolling_Mean_3h', 'Rolling_Std_3h', 'Occupancy', 'Is_Weekend', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos']
    expected_lgb = expected_gru[1:]

    # STRICT COLUMN ALIGNMENT FIX
    scaler_cols = [c.split('__')[-1] for c in scaler.get_feature_names_out()]
    scaled_window = pd.DataFrame(scaler.transform(live_window[scaler_cols].fillna(0.0)), columns=scaler_cols)

    # TE-GRU Inference (Only give it the first 24 rows)
    tensor_input = np.array([scaled_window[expected_gru].iloc[:-1].values], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], tensor_input)
    interpreter.invoke()
    gru_scaled = interpreter.get_tensor(output_details[0]['index'])[0][0]

    gru_raw = unscale_prediction(gru_scaled, live_window['Energy_kW'].iloc[:-1].values, scaled_window['Energy_kW'].iloc[:-1].values)

    # LightGBM Inference (Only give it the 25th live row)
    lgb_input = scaled_window[expected_lgb].iloc[-1:].reset_index(drop=True)
    residual_correction = lgb_corrector.predict(lgb_input)[0]

    hybrid_final_kwh = gru_raw + residual_correction

    # Bayesian Uncertainty Bounds (uses real prediction residuals)
    tracked = residual_tracker.get()
    if tracked is not None:
        init_sigma = float(np.std(tracked))
        dynamic_sigma = mh_estimator.estimate_sigma(tracked, initial_sigma=init_sigma)
    else:
        # Cold start: use tight default (~5% of prediction) until residuals accumulate
        dynamic_sigma = max(0.01, abs(hybrid_final_kwh) * 0.05)

    lower_bound = max(0.0, hybrid_final_kwh - 1.96 * dynamic_sigma)
    upper_bound = hybrid_final_kwh + 1.96 * dynamic_sigma

    actual_val = current_hour_data['Energy_kW']
    actual_kw = round(float(actual_val), 4) if pd.notna(actual_val) else None

    # Track residual for future uncertainty estimation
    residual_tracker.add(hybrid_final_kwh, actual_kw)

    return {
        "timestamp": str(current_hour_data['Timestamp']),
        "live_sensors": {
            "temperature_c": float(current_hour_data['Temperature_C']),
            "humidity": float(current_hour_data['Humidity_%']),
            "lux": float(current_hour_data['Luminous_Intensity_Lux']),
            "occupancy": int(current_hour_data['Occupancy'])
        },
        "predictions": {
            "actual_energy_kw": actual_kw,
            "base_gru_kwh": round(gru_raw, 4),
            "lgbm_correction_kwh": round(residual_correction, 4),
            "hybrid_final_kwh": round(hybrid_final_kwh, 4),
            "safety_lower_bound": round(lower_bound, 4),
            "safety_upper_bound": round(upper_bound, 4)
        }
    }


# ═══════════════════════════════════════════════════════════════════
# MQTT BRIDGE (Primary system workflow)
# ═══════════════════════════════════════════════════════════════════
mqtt_client = mqtt.Client(
    client_id=MQTT_CLIENT_ID,
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
)


def build_mqtt_payload(result: dict) -> dict:
    """Convert internal result into the flat contract payload
    expected by rule_engine.py, mqtt_logger.py, and the dashboard.
    """
    pred = result["predictions"]
    return {
        "predicted_energy_kw": pred["hybrid_final_kwh"],
        "upper_bound_energy_kw": pred["safety_upper_bound"],
        "predicted_energy_range": pred["safety_upper_bound"],
        "peak_demand": PEAK_DEMAND_KW,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "fastapi-local-model",
    }


def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe(TOPIC_SENSORS, qos=1)
        print(f"[OK] MQTT bridge connected - subscribed to '{TOPIC_SENSORS}'")
    else:
        print(f"[ERROR] MQTT connection failed (rc={rc})")


def on_mqtt_message(client, userdata, msg):
    """Sensor message arrives → run prediction → publish result to MQTT."""
    global current_sim_index
    try:
        if current_sim_index >= len(df_sim):
            print("[WARN] Simulation finished - resetting index")
            current_sim_index = WINDOW_SIZE

        live_window = df_sim.iloc[
            current_sim_index - WINDOW_SIZE : current_sim_index + 1
        ].copy()
        current_sim_index += 1

        result = run_prediction(live_window)
        mqtt_payload = build_mqtt_payload(result)

        client.publish(TOPIC_ML_PREDICTIONS, json.dumps(mqtt_payload), qos=1)
        print(
            f"  -> MQTT published to {TOPIC_ML_PREDICTIONS}: "
            f"{mqtt_payload['predicted_energy_kw']:.4f} kW"
        )
    except Exception as exc:
        print(f"[ERROR] MQTT prediction error: {exc}")


def start_mqtt_bridge():
    """Connect MQTT client and start network loop.

    If the broker is not reachable yet (e.g. Mosquitto started after this
    service), a background thread retries every 5 seconds until it connects.
    """
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _try_connect():
        while True:
            try:
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                mqtt_client.loop_start()
                print(f"MQTT bridge started -> {MQTT_BROKER}:{MQTT_PORT}")
                return  # success — stop retrying
            except OSError as exc:
                print(f"[WARN] MQTT broker unreachable ({exc}) - retrying in 5s...")
                import time
                time.sleep(5)

    # Run the connection attempts in a daemon thread so FastAPI starts immediately
    t = threading.Thread(target=_try_connect, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP (HTTP — for testing only)
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Start MQTT bridge on boot, clean up on shutdown."""
    start_mqtt_bridge()
    yield
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("MQTT bridge disconnected.")


app = FastAPI(
    title="Smart Grid Hybrid AI - TEST SIMULATOR",
    description="HTTP endpoints are for manual testing only. "
                "Production data flows through MQTT.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic model for manual input ---
class SensorInput(BaseModel):
    """Sensor values typed by the user for testing."""
    temperature_c: float = 28.0
    humidity: float = 60.0
    lux: float = 400.0
    occupancy: int = 1
    datetime_str: str | None = None  # ISO-format string, e.g. "2024-06-15T14:30"


# --- HTTP endpoint: manual input (for test.py / test_dashboard.html) ---
@app.post("/predict")
def predict_manual(sensor: SensorInput):
    """Accept manually-typed sensor values, inject them into the model's
    current CSV window row, and return the prediction.

    The GRU still uses 24 rows of CSV history for context, but YOUR
    values replace the 25th (current) row so you can see how the model
    responds to your inputs.
    """
    global current_sim_index

    if current_sim_index >= len(df_sim):
        current_sim_index = WINDOW_SIZE

    live_window = df_sim.iloc[
        current_sim_index - WINDOW_SIZE : current_sim_index + 1
    ].copy()

    # Override the last row with manual inputs
    idx = live_window.index[-1]
    live_window.loc[idx, 'Temperature_C'] = sensor.temperature_c
    live_window.loc[idx, 'Humidity_%'] = sensor.humidity
    live_window.loc[idx, 'Luminous_Intensity_Lux'] = sensor.lux
    live_window.loc[idx, 'Occupancy'] = sensor.occupancy
    live_window.loc[idx, 'Energy_kW'] = np.nan

    # Use user-provided datetime (or default to now)
    if sensor.datetime_str:
        try:
            user_ts = pd.Timestamp(sensor.datetime_str)
        except Exception:
            user_ts = pd.Timestamp.now()
    else:
        user_ts = pd.Timestamp.now()
    live_window.loc[idx, 'Timestamp'] = user_ts

    res = run_prediction(live_window)
    current_sim_index += 1
    return res


# --- HTTP endpoint: CSV auto-step (quick sequential test) ---
@app.get("/predict_next")
def predict_next_hour():
    """Advance one row through the CSV dataset and return the prediction."""
    global current_sim_index

    if current_sim_index >= len(df_sim):
        return {"error": "Simulation finished. End of dataset."}

    live_window = df_sim.iloc[
        current_sim_index - WINDOW_SIZE : current_sim_index + 1
    ].copy()
    current_sim_index += 1

    return run_prediction(live_window)


# --- Serve test_dashboard.html at root ---
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = os.path.join(BASE_DIR, "test_dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>test_dashboard.html not found in ML/</h1>"


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)