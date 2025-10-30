import os
from flask import Flask, request, jsonify
from typing import Any
import mlflow.pyfunc
import numpy as np
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from mlflow.exceptions import RestException
import sys
import time
import json
import logging


app = Flask(__name__)
# url in http://localhost:8080/apidocs/
swagger = Swagger(app)


# === CONFIGURACIÓN LOGGING ===
LOG_FILE = "/tmp/predict_monitor.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === MONITOR INTERNO ===
monitor_stats = {
    "total_calls": 0,
    "total_errors": 0,
    "recent_calls": [],  # guarda últimos N llamados
}
MAX_RECENT = 50

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # fuerza a usar CPU

# === setting up Rate Limiter ===
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


# === getting API KEY ===
MLFLOWAPI_API_KEY = os.getenv("MLFLOWAPI_API_KEY")
if not MLFLOWAPI_API_KEY:
    print("ERROR: La variable de entorno MLFLOWAPI_API_KEY no está definida. No se puede iniciar el servicio.")
    sys.exit(1)


def require_apikey(f: Any) -> Any:
    """Decorator to require API key in the request header 'X-API-KEY'."""

    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        key = request.headers.get("X-API-KEY")
        if not key or key != MLFLOWAPI_API_KEY:
            return jsonify({"error": "Acceso no autorizado"}), 401
        return f(*args, **kwargs)

    return decorated


model_production_uri = "models:/boston_model/Production"
model = None


def load_model_safe():
    """
    Allows to load the model safely, catching exceptions.
    Sets the global 'model' variable.
    """
    global model
    try:
        print(f"Intentando cargar modelo desde: {model_production_uri}")
        model = mlflow.pyfunc.load_model(model_production_uri)
        print("INFO: Model loaded successfully.")
    except RestException as e:
        print(f"ERROR: Model can not be loaded: {e}")
        model = None
    except Exception as e:
        print(f"ERROR: Unknown error loading the model: {e}")
        model = None


# load model at startup
load_model_safe()


# === ENDPOINTS ===
@app.route("/ping", methods=["GET"])
@limiter.limit("5 per minute")
def ping():
    """Verifies that the service is running
    ---
    responses:
      200:
        description: The service is running correctly
    """
    return "ok", 200


@app.route("/reload", methods=["POST"])
@require_apikey
def reload():
    """Reloads the model from MLflow
    ---
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: reloaded successfully
    """
    load_model_safe()
    if model is None:
        return (
            jsonify({"error": "The model is not available after reload, please check mlflow server"}),
            500,
        )
    return jsonify({"status": "Model reloaded correctly"}), 200


@app.route("/predict", methods=["POST"])
@require_apikey
def predict():
    """Performs predictions using the loaded model
    ---
    security:
      - ApiKeyAuth: []
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            data:
              type: array
              items:
                type: array
                items:
                  type: number
              example: [[0.1, 25.0, 5.13, 0, 0.453, 6.5, 45.0, 5.3, 4, 320, 15.3, 390.0, 12.0]]
    responses:
      200:
        description: Prediction results
        schema:
          type: object
          properties:
            predictions:
              type: array
              items:
                type: number
              example: [24.5]
      400:
        description: Error in the request
      401:
        description: Not authorized
      500:
        description: Internal server error
    x-code-samples:
      - lang: cURL
        label: Predicción ejemplo
        source: |
          curl -X POST http://127.0.0.1:8080/predict \
            -H "Content-Type: application/json" \
            -H "X-API-KEY: myapikey" \
            -d '{"data": [[0.1, 25.0, 5.13, 0, 0.453, 6.5, 45.0, 5.3, 4, 320, 15.3, 390.0, 12.0]]}'
    """
    global model
    if model is None:
        return (
            jsonify({"error": "The model is not available, Inference cannot be performed"}),
            503,
        )

    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "The body of the requests requires 'data' entry"}), 400

        arr = np.array(data["data"], dtype=np.float32)

        # Validar tipo y forma del array
        if arr.ndim != 2:
            return (
                jsonify({"error": f" Waiting for 2D array (list of lists), and we got {arr.shape}"}),
                400,
            )

        if arr.shape[1] != 13:
            return (
                jsonify({"error": f"Invalid dimension: wating 13 columns, we got {arr.shape[1]}"}),
                400,
            )
        start_time = time.time()
        monitor_stats["total_calls"] += 1
        client_ip = get_remote_address()
        preds = model.predict(arr)

        elapsed = round(time.time() - start_time, 4)

        # === Logging ===
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "client_ip": client_ip,
            "num_inputs": len(arr),
            "params": arr.tolist(),
            "predictions": preds.tolist(),
            "elapsed_sec": elapsed,
        }
        logging.info(json.dumps(log_entry))
        monitor_stats["recent_calls"].append(log_entry)
        if len(monitor_stats["recent_calls"]) > MAX_RECENT:
            monitor_stats["recent_calls"].pop(0)

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/monitor", methods=["GET"])
@require_apikey
def monitor():
    """Provides monitoring statistics
    ---
    security:
      - ApiKeyAuth: []
    tags:
      - Monitoring
    summary: Returns runtime monitoring statistics
    description: |
      This endpoint provides basic monitoring information about the API usage,
      including the total number of calls, total errors, and the most recent calls.
    responses:
      200:
        description: Monitoring statistics successfully retrieved
        schema:
          type: object
          properties:
            total_calls:
              type: integer
              description: Total number of API calls received
              example: 124
            total_errors:
              type: integer
              description: Total number of calls that resulted in an error
              example: 5
            recent_calls:
              type: array
              description: List of the last 5 API call logs
              items:
                type: object
                properties:
                  endpoint:
                    type: string
                    description: The API endpoint that was called
                    example: "/predict"
                  timestamp:
                    type: string
                    format: date-time
                    description: Time when the call was made
                    example: "2025-10-29T13:57:00Z"
                  params:
                    type: object
                    description: Parameters or data passed in the call
                    example:
                      data: [[0.1, 25.0, 5.13, 0, 0.453, 6.5, 45.0, 5.3, 4, 320, 15.3, 390.0, 12.0]]
                  prediction:
                    type: array
                    items:
                      type: number
                    description: Model predictions returned in that call
                    example: [24.5]
      401:
        description: Not authorized (invalid or missing API key)
    """
    return jsonify(
        {
            "total_calls": monitor_stats["total_calls"],
            "total_errors": monitor_stats["total_errors"],
            "recent_calls": monitor_stats["recent_calls"][-5:],  # últimos 5
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
