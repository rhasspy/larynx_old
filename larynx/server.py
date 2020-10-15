#!/usr/bin/env python3
"""Web server for synthesis"""
import hashlib
import logging
import time
import typing
import uuid
from pathlib import Path

from flask import Flask, Response, render_template, request, send_from_directory
from flask_cors import CORS

from larynx.synthesize import Synthesizer

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("larynx.server")

# -----------------------------------------------------------------------------

# TODO: Show phonemes table
# TODO: Allow phoneme input


def get_app(
    synthesizer: Synthesizer, cache_dir: typing.Optional[typing.Union[str, Path]] = None
):
    """Create Flask app and endpoints"""
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    def text_to_wav(text: str) -> bytes:
        _LOGGER.debug("Text: %s", text)

        wav_bytes: typing.Optional[bytes] = None
        cached_wav_path: typing.Optional[Path] = None

        if cache_dir:
            # Check cache first
            sentence_hash = hashlib.md5()
            sentence_hash.update(text.encode())
            cached_wav_path = cache_dir / f"{sentence_hash.hexdigest()}.wav"

            if cached_wav_path.is_file():
                _LOGGER.debug("Loading WAV from cache: %s", cached_wav_path)
                wav_bytes = cached_wav_path.read_bytes()

        if not wav_bytes:
            _LOGGER.debug("Synthesizing...")
            start_time = time.time()
            wav_bytes = synthesizer.synthesize(text)
            end_time = time.time()

            _LOGGER.debug(
                "Synthesized %s byte(s) in %s second(s)",
                len(wav_bytes),
                end_time - start_time,
            )

            # Save to cache
            if cached_wav_path:
                cached_wav_path.write_bytes(wav_bytes)

        return wav_bytes

    # -------------------------------------------------------------------------

    app = Flask("larynx", template_folder=str(_DIR / "templates"))
    app.secret_key = str(uuid.uuid4())
    CORS(app)

    @app.route("/")
    def app_index():
        return render_template("index.html")

    @app.route("/css/<path:filename>", methods=["GET"])
    def css(filename) -> Response:
        """CSS static endpoint."""
        return send_from_directory("css", filename)

    @app.route("/img/<path:filename>", methods=["GET"])
    def img(filename) -> Response:
        """Image static endpoint."""
        return send_from_directory("img", filename)

    @app.route("/api/tts", methods=["GET", "POST"])
    def api_tts():
        """Text to speech endpoint"""
        if request.method == "POST":
            text = request.data.encode()
        else:
            text = request.args.get("text")

        wav_bytes = text_to_wav(text)

        return Response(wav_bytes, mimetype="audio/wav")

    # MaryTTS compatibility layer
    @app.route("/process", methods=["GET"])
    def api_process():
        """MaryTTS-compatible /process endpoint"""
        text = request.args.get("INPUT_TEXT", "")
        wav_bytes = text_to_wav(text)

        return Response(wav_bytes, mimetype="audio/wav")

    @app.route("/voices", methods=["GET"])
    def api_voices():
        """MaryTTS-compatible /voices endpoint"""
        return "default\n"

    return app
