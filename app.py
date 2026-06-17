from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from hmm_model import BrownHMMTagger


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_IMAGE_DIR = PROJECT_ROOT / "doc" / "latex-report" / "resultImage"


app = Flask(__name__)


@lru_cache(maxsize=1)
def get_model() -> BrownHMMTagger:
    return BrownHMMTagger.from_project_data(PROJECT_ROOT)


@app.route("/")
def index():
    model = get_model()
    return render_template("index.html", stats=model.stats, tags=model.tags)


@app.route("/api/tag", methods=["POST"])
def tag_text():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter a sentence to tag."}), 400

    model = get_model()
    predictions = model.predict_text(text)
    return jsonify(
        {
            "tokens": predictions,
            "token_count": len(predictions),
            "oov_count": sum(1 for item in predictions if not item["known"]),
        }
    )


@app.route("/report-image/<path:filename>")
def report_image(filename: str):
    return send_from_directory(RESULT_IMAGE_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
