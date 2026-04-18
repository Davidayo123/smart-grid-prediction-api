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
CSV_PATH = os.path.join(BASE_DIR, "perfectdata3.csv")
WINDOW_SIZE = 168  # Expanded to allow for lag_168 feature

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
rename_map = {
    'timestamp': 'Timestamp',
    'temperature': 'Temperature_C',
    'humidity': 'Humidity_%',
    'lux': 'Luminous_Intensity_Lux',
    'occupancy': 'Occupancy',
    'energy': 'Energy_kW'
}
df_sim = df_sim.rename(columns=rename_map)
if 'Luminous_Intensity' in df_sim.columns and 'Luminous_Intensity_Lux' not in df_sim.columns:
    df_sim = df_sim.rename(columns={'Luminous_Intensity': 'Luminous_Intensity_Lux'})
df_sim['Timestamp'] = pd.to_datetime(df_sim['Timestamp'])

scaler_dict = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
scaler_X = scaler_dict['scaler_X']
scaler_y = scaler_dict['scaler_y']
EXPECTED_FEATURES = scaler_dict['feature_names']

lgb_model = joblib.load(os.path.join(BASE_DIR, "lightgbm_model.pkl"))

interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "te_gru_model_final.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[OK] AI assets loaded")


def _find_csv_index(datetime_str: str) -> int | None:
    """Find the CSV row index to provide the best matching lag history context.

    If an exact match (within 1 hour) exists, it returns it.
    If the date is far in the future or not in the dataset, it performs
    a fallback search for a similar historical profile to construct lag features:
      Step 1: Matches Month, Day-of-Week, and Hour
      Step 2: Matches Day-of-Week and Hour
      Step 3: Matches just Hour
    Always ensures the found index has >= 168 rows before it for the window.
    """
    try:
        target = pd.Timestamp(datetime_str)
    except Exception:
        return None

    # Only consider rows that have enough history
    valid_mask = df_sim.index >= WINDOW_SIZE
    valid_df = df_sim[valid_mask]
    
    if len(valid_df) == 0:
        return WINDOW_SIZE

    # Try 1: Exact timestamp match (within 1 hour)
    diffs = (valid_df['Timestamp'] - target).abs()
    best_idx = int(diffs.idxmin())
    if diffs.loc[best_idx] <= pd.Timedelta(hours=1):
        return best_idx

    # Try 2: Semantic Match - Month, Day of Week, and Hour
    mask_mdh = (
        (valid_df['Timestamp'].dt.month == target.month) &
        (valid_df['Timestamp'].dt.dayofweek == target.dayofweek) &
        (valid_df['Timestamp'].dt.hour == target.hour)
    )
    matches_mdh = valid_df[mask_mdh]
    if len(matches_mdh) > 0:
        return int(matches_mdh.index[-1])  # Use most recent comparable occurrence

    # Try 3: Semantic Match - Day of Week and Hour (ignore month)
    mask_dh = (
        (valid_df['Timestamp'].dt.dayofweek == target.dayofweek) &
        (valid_df['Timestamp'].dt.hour == target.hour)
    )
    matches_dh = valid_df[mask_dh]
    if len(matches_dh) > 0:
        return int(matches_dh.index[-1])

    # Try 4: Semantic Match - Just Hour
    mask_h = (valid_df['Timestamp'].dt.hour == target.hour)
    matches_h = valid_df[mask_h]
    if len(matches_h) > 0:
        return int(matches_h.index[-1])

    # Safest structural fallback
    return WINDOW_SIZE


# ═══════════════════════════════════════════════════════════════════
# BAYESIAN UNCERTAINTY & ADAPTIVE WEIGHT ESTIMATOR
# ═══════════════════════════════════════════════════════════════════
class MHWeightEstimator:
    """Metropolis-Hastings sampler for adaptive component weighting."""
    def __init__(self, n_iterations=1000, proposal_std=0.02, temperature=1e-4):
        self.n_iterations = n_iterations
        self.proposal_std = proposal_std
        self.temperature = temperature

    def estimate_weight(self, y_true: np.ndarray, pred_gru: np.ndarray, pred_lgbm: np.ndarray, w_init=0.5) -> float:
        if len(y_true) < 2:
            return w_init
            
        def hybrid_loss(w):
            blended = w * pred_gru + (1 - w) * pred_lgbm
            return float(np.mean((y_true - blended) ** 2))

        w_current = w_init
        loss_current = hybrid_loss(w_current)
        best_w = w_current
        best_loss = loss_current

        rng = np.random.RandomState(int(abs(loss_current * 1e5)) % (2 ** 31))

        for _ in range(self.n_iterations):
            w_proposed = np.clip(w_current + rng.normal(0, self.proposal_std), 0.0, 1.0)
            loss_proposed = hybrid_loss(w_proposed)

            delta_loss = loss_proposed - loss_current
            # Avoid division by zero
            alpha = min(1.0, np.exp(-delta_loss / max(self.temperature, 1e-8)))

            if rng.uniform(0, 1) < alpha:
                w_current = w_proposed
                loss_current = loss_proposed
                if loss_current < best_loss:
                    best_loss = loss_current
                    best_w = w_current

        return best_w


class HistoryTracker:
    """Rolling window of true values and component predictions for MH weighting and bounds."""
    def __init__(self, max_size=200):
        self.y_true = []
        self.pred_gru = []
        self.pred_lgbm = []
        self.max_size = max_size

    def add(self, actual, gru_val, lgbm_val):
        if actual is not None and not np.isnan(actual):
            self.y_true.append(actual)
            self.pred_gru.append(gru_val)
            self.pred_lgbm.append(lgbm_val)
            if len(self.y_true) > self.max_size:
                self.y_true.pop(0)
                self.pred_gru.pop(0)
                self.pred_lgbm.pop(0)

    def get(self):
        if len(self.y_true) >= 3:
            return np.array(self.y_true), np.array(self.pred_gru), np.array(self.pred_lgbm)
        return None, None, None


mh_estimator = MHWeightEstimator()
history_tracker = HistoryTracker()


# ═══════════════════════════════════════════════════════════════════
# CORE PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════
def run_prediction(live_window: pd.DataFrame) -> dict:
    """Run the full GRU + LightGBM hybrid pipeline on the window context.

    Returns a dict with all prediction outputs.
    """
    current_hour_data = live_window.iloc[-1].copy()

    # Engineer Time Features
    live_window['hour'] = live_window['Timestamp'].dt.hour
    live_window['day_of_week_num'] = live_window['Timestamp'].dt.dayofweek
    live_window['month'] = live_window['Timestamp'].dt.month
    
    live_window['hour_sin'] = np.sin(2 * np.pi * live_window['hour'] / 24)
    live_window['hour_cos'] = np.cos(2 * np.pi * live_window['hour'] / 24)
    live_window['dow_sin'] = np.sin(2 * np.pi * live_window['day_of_week_num'] / 7)
    live_window['dow_cos'] = np.cos(2 * np.pi * live_window['day_of_week_num'] / 7)
    live_window['month_sin'] = np.sin(2 * np.pi * live_window['month'] / 12)
    live_window['month_cos'] = np.cos(2 * np.pi * live_window['month'] / 12)
    live_window['is_weekend'] = (live_window['day_of_week_num'] >= 5).astype(int)

    # Engineer Lag Features
    live_window['lag_1'] = live_window['Energy_kW'].shift(1).bfill()
    live_window['lag_2'] = live_window['Energy_kW'].shift(2).bfill()
    live_window['lag_3'] = live_window['Energy_kW'].shift(3).bfill()
    live_window['lag_24'] = live_window['Energy_kW'].shift(24).bfill()
    live_window['lag_168'] = live_window['Energy_kW'].shift(168).bfill()
    
    live_window['rolling_mean_3'] = live_window['Energy_kW'].shift(1).rolling(3, min_periods=1).mean()
    live_window['rolling_std_3'] = live_window['Energy_kW'].shift(1).rolling(3, min_periods=1).std().fillna(0)
    live_window['rolling_mean_24'] = live_window['Energy_kW'].shift(1).rolling(24, min_periods=1).mean()
    live_window['rolling_std_24'] = live_window['Energy_kW'].shift(1).rolling(24, min_periods=1).std().fillna(0)

    # Note: perfectdata3 natively has 'time_of_day' as well as temperature, humidity, lux, occupancy
    # which we've renamed appropriately. We need to ensure columns match scaler expected names exactly:
    feature_df = pd.DataFrame()
    feature_df['time_of_day'] = live_window.get('time_of_day', live_window['hour']) # fallback to hour
    feature_df['temperature'] = live_window['Temperature_C']
    feature_df['humidity'] = live_window['Humidity_%']
    feature_df['lux'] = live_window['Luminous_Intensity_Lux']
    feature_df['occupancy'] = live_window['Occupancy']
    feature_df['hour'] = live_window['hour']
    feature_df['day_of_week_num'] = live_window['day_of_week_num']
    feature_df['month'] = live_window['month']
    feature_df['is_weekend'] = live_window['is_weekend']
    feature_df['hour_sin'] = live_window['hour_sin']
    feature_df['hour_cos'] = live_window['hour_cos']
    feature_df['dow_sin'] = live_window['dow_sin']
    feature_df['dow_cos'] = live_window['dow_cos']
    feature_df['month_sin'] = live_window['month_sin']
    feature_df['month_cos'] = live_window['month_cos']
    feature_df['lag_1'] = live_window['lag_1']
    feature_df['lag_2'] = live_window['lag_2']
    feature_df['lag_3'] = live_window['lag_3']
    feature_df['lag_24'] = live_window['lag_24']
    feature_df['lag_168'] = live_window['lag_168']
    feature_df['rolling_mean_3'] = live_window['rolling_mean_3']
    feature_df['rolling_std_3'] = live_window['rolling_std_3']
    feature_df['rolling_mean_24'] = live_window['rolling_mean_24']
    feature_df['rolling_std_24'] = live_window['rolling_std_24']

    # Final feature alignment check
    feature_df = feature_df[EXPECTED_FEATURES]

    # Scaling
    scaled_features_arr = scaler_X.transform(feature_df.fillna(0.0))
    
    # ── TE-GRU Inference ──
    # The GRU needs exactly a 24-step sequence. We'll take the last 24 steps which include
    # the very last line where manual input variables exist for inference.
    tensor_input = np.array([scaled_features_arr[-24:]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], tensor_input)
    interpreter.invoke()
    gru_scaled = interpreter.get_tensor(output_details[0]['index'])[0][0]
    gru_raw = scaler_y.inverse_transform([[gru_scaled]])[0][0]

    # ── LightGBM Inference ──
    # LightGBM uses just the current row's features (excluding the true target, which is what we predict)
    lgb_input = scaled_features_arr[-1:]
    lgbm_scaled = lgb_model.predict(lgb_input)[0]
    lgbm_raw = scaler_y.inverse_transform([[lgbm_scaled]])[0][0]

    # ── MH Adaptive Weight Blending & Bounds ──
    y_hist, gru_hist, lgbm_hist = history_tracker.get()
    
    if y_hist is not None:
        best_w = mh_estimator.estimate_weight(y_hist, gru_hist, lgbm_hist)
        blended_hist = best_w * gru_hist + (1 - best_w) * lgbm_hist
        residual_std = float(np.std(y_hist - blended_hist))
    else:
        # Default: favor LightGBM (R²≈0.90) over GRU (R²≈0.62)
        best_w = 0.3
        residual_std = max(0.05, abs(lgbm_raw) * 0.10)

    hybrid_final_kwh = best_w * gru_raw + (1 - best_w) * lgbm_raw
    
    z = 1.5
    lower_bound = max(0.0, hybrid_final_kwh - z * residual_std)
    upper_bound = hybrid_final_kwh + z * residual_std

    actual_val = current_hour_data['Energy_kW']
    actual_kw = round(float(actual_val), 4) if pd.notna(actual_val) else None

    # Track components for future weighting and bounds estimation
    history_tracker.add(actual_kw, gru_raw, lgbm_raw)

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
            "lgbm_kwh": round(lgbm_raw, 4),
            "hybrid_final_kwh": round(hybrid_final_kwh, 4),
            "safety_lower_bound": round(lower_bound, 4),
            "safety_upper_bound": round(upper_bound, 4),
            "hybrid_weight_gru": round(best_w, 4)
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
        "lower_bound_energy_kw": pred["safety_lower_bound"],
        "predicted_energy_range": [pred["safety_lower_bound"], pred["safety_upper_bound"]],
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
            except Exception as exc:
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
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT bridge disconnected.")
    except Exception:
        pass


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
    context window and return the prediction.

    If a datetime_str is provided, the system looks up the matching row
    in the CSV and uses that row's full context window (168 preceding
    rows) so that lag features are historically accurate.  The user's
    sensor values still override the current row.

    Without datetime_str, the current sim-index position is used.
    """
    global current_sim_index

    # ── Determine which context window to use ──
    matched_idx = None
    if sensor.datetime_str:
        matched_idx = _find_csv_index(sensor.datetime_str)

    if matched_idx is not None:
        # Use the matched row's own context — lag features will be correct
        ctx_index = matched_idx
    else:
        # Fall back to current sim position
        if current_sim_index >= len(df_sim):
            current_sim_index = WINDOW_SIZE
        ctx_index = current_sim_index

    live_window = df_sim.iloc[
        ctx_index - WINDOW_SIZE : ctx_index + 1
    ].copy()

    # Override the last row with manual inputs
    idx = live_window.index[-1]
    live_window.loc[idx, 'Temperature_C'] = sensor.temperature_c
    live_window.loc[idx, 'Humidity_%'] = sensor.humidity
    live_window.loc[idx, 'Luminous_Intensity_Lux'] = sensor.lux
    live_window.loc[idx, 'Occupancy'] = sensor.occupancy
    live_window.loc[idx, 'Energy_kW'] = np.nan

    # Sync timestamp and time_of_day
    if sensor.datetime_str:
        try:
            user_ts = pd.Timestamp(sensor.datetime_str)
        except Exception:
            user_ts = live_window.loc[idx, 'Timestamp']
    else:
        user_ts = live_window.loc[idx, 'Timestamp']
    live_window.loc[idx, 'Timestamp'] = user_ts
    live_window.loc[idx, 'time_of_day'] = user_ts.hour

    res = run_prediction(live_window)
    # Manual predictions don't advance the simulation clock.
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