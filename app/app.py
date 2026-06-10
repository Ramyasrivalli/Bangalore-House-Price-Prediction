"""
app.py
------
Flask web application exposing the house price prediction API
and serving the browser-based UI.
"""

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Repo root on sys.path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import get_predictor
from src.utils import get_logger

logger = get_logger(__name__)

app = Flask(__name__)


# ── Predictor – loaded once ───────────────────────────────────────────────────

try:
    predictor = get_predictor()
    logger.info("Predictor loaded. %d locations available.", len(predictor.locations))
except FileNotFoundError as exc:
    logger.error("Model artifacts not found: %s", exc)
    logger.error("Run 'python train.py' first to generate models/trained_model.pkl")
    predictor = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    locations = predictor.locations if predictor else []
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST endpoint that accepts JSON and returns a price prediction.

    Request body
    ------------
    .. code-block:: json

        {
            "location": "Indiranagar",
            "sqft": 1200,
            "bath": 2,
            "bhk": 2
        }

    Response
    --------
    .. code-block:: json

        { "price": 95.43 }

    or on error:

    .. code-block:: json

        { "error": "<message>" }
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded. Run training pipeline first."}), 503

    data = request.get_json(silent=True) or {}

    # ── Input extraction & validation ─────────────────────────────────────────
    try:
        location = str(data.get("location", "")).strip()
        sqft = float(data.get("sqft", 0))
        bath = int(data.get("bath", 0))
        bhk = int(data.get("bhk", 0))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input types: {exc}"}), 400

    if not location:
        return jsonify({"error": "Location is required."}), 400

    # ── Prediction ────────────────────────────────────────────────────────────
    try:
        price = predictor.predict(location=location, sqft=sqft, bath=bath, bhk=bhk)
        logger.info("Prediction: %s | %.0f sqft | %d bath | %d BHK → ₹%.2f L", location, sqft, bath, bhk, price)
        return jsonify({"price": price})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected prediction error")
        return jsonify({"error": "Internal server error."}), 500


@app.route("/locations")
def locations():
    """Return the list of known locations as JSON."""
    if predictor is None:
        return jsonify({"error": "Model not loaded."}), 503
    return jsonify({"locations": predictor.locations})


@app.route("/health")
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "model_loaded": predictor is not None})


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
