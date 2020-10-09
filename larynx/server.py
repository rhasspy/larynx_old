#!/usr/bin/env python3
"""Web server for synthesis"""
import argparse
import logging
import time
import uuid
from pathlib import Path

from flask import Flask, Response, render_template, request
from flask_cors import CORS

from larynx.synthesize import Synthesizer

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("larynx.server")

# -----------------------------------------------------------------------------

# TODO: Show phonemes table
# TODO: Allow phoneme input

def get_app(synthesizer: Synthesizer):
    """Create Flask app and endpoints"""
    app = Flask("larynx", template_folder=str(_DIR / "templates"))
    app.secret_key = str(uuid.uuid4())
    CORS(app)

    @app.route("/")
    def app_index():
        return render_template("index.html")

    @app.route("/api/tts", methods=["GET"])
    def api_tts():
        text = request.args.get("text")
        _LOGGER.debug("Text: %s", text)

        start_time = time.time()
        wav_bytes = synthesizer.synthesize(text)
        end_time = time.time()

        _LOGGER.debug(
            "Synthesized %s byte(s) in %s second(s)",
            len(wav_bytes),
            end_time - start_time,
        )

        return Response(wav_bytes, mimetype="audio/wav")

    return app


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
