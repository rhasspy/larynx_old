#!/usr/bin/env python3
"""Web server for synthesis"""
import hashlib
import io
import logging
import time
import typing
import uuid
import wave
from pathlib import Path

from flask import Flask, Response, render_template, request, send_from_directory
from flask_cors import CORS

import gruut
from larynx.synthesize import Synthesizer

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("larynx.server")

# -----------------------------------------------------------------------------

# TODO: Show phonemes table
# TODO: Allow phoneme input


def get_app(
    synthesizer: Synthesizer,
    gruut_lang: typing.Optional[gruut.Language] = None,
    cache_dir: typing.Optional[typing.Union[str, Path]] = None,
):
    """Create Flask app and endpoints"""
    sample_rate = synthesizer.sample_rate

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    def text_to_wav(text: str, text_is_phonemes: bool = False) -> bytes:
        _LOGGER.debug("Text: %s", text)

        wav_bytes: typing.Optional[bytes] = None
        cached_wav_path: typing.Optional[Path] = None

        if cache_dir:
            # Check cache first
            sentence_hash = hashlib.md5()
            sentence_hash.update(f"phonemes-{text_is_phonemes}_{text}".encode())
            cached_wav_path = cache_dir / f"{sentence_hash.hexdigest()}.wav"

            if cached_wav_path.is_file():
                _LOGGER.debug("Loading WAV from cache: %s", cached_wav_path)
                wav_bytes = cached_wav_path.read_bytes()

        if not wav_bytes:
            _LOGGER.debug("Synthesizing...")
            start_time = time.time()

            # Synthesize each line separately.
            # Accumulate into a single WAV file.
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setframerate(sample_rate)
                    wav_file.setsampwidth(2)
                    wav_file.setnchannels(1)

                    for line_index, line in enumerate(text.strip().splitlines()):
                        _LOGGER.debug(
                            "Synthesizing line %s (%s char(s))",
                            line_index + 1,
                            len(line),
                        )
                        line_wav_bytes = synthesizer.synthesize(
                            line, text_is_phonemes=text_is_phonemes
                        )
                        _LOGGER.debug(
                            "Got %s WAV byte(s) for line %s",
                            len(line_wav_bytes),
                            line_index + 1,
                        )

                        # Open up and add to main WAV
                        with io.BytesIO(line_wav_bytes) as line_wav_io:
                            with wave.open(line_wav_io) as line_wav_file:
                                wav_file.writeframes(
                                    line_wav_file.readframes(line_wav_file.getnframes())
                                )

                wav_bytes = wav_io.getvalue()

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
        return render_template("index.html", lang=gruut_lang)

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
            text = request.data.decode()
        else:
            text = request.args.get("text")

        # True if input text is actually phonemes
        text_is_phonemes = request.args.get("phonemes", "").strip().lower() == "true"

        wav_bytes = text_to_wav(text, text_is_phonemes=text_is_phonemes)

        return Response(wav_bytes, mimetype="audio/wav")

    @app.route("/api/phonemize", methods=["GET", "POST"])
    def api_phonemize():
        """Text to speech endpoint"""
        if request.method == "POST":
            text = request.data.decode()
        else:
            text = request.args.get("text", "")

        text_phonemes = []
        if gruut_lang:
            # Use gruut
            for sentence in gruut_lang.tokenizer.tokenize(text):
                # Use first pronunciation
                word_phonemes = [
                    wp[0]
                    for wp in gruut_lang.phonemizer.phonemize(
                        sentence.clean_words, word_indexes=True, word_breaks=True
                    )
                    if wp
                ]

                text_phonemes.extend(p for ps in word_phonemes for p in ps)

        return "".join(text_phonemes)

    # MaryTTS compatibility layer
    @app.route("/process", methods=["GET", "POST"])
    def api_process():
        """MaryTTS-compatible /process endpoint"""
        if request.method == "POST":
            text = request.data.decode()
        else:
            text = request.args.get("INPUT_TEXT", "")

        wav_bytes = text_to_wav(text)

        return Response(wav_bytes, mimetype="audio/wav")

    @app.route("/voices", methods=["GET"])
    def api_voices():
        """MaryTTS-compatible /voices endpoint"""
        return "default\n"

    return app
