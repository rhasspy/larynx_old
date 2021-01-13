# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
# Contains Kathleen voice (https://github.com/rhasspy/dataset-voice-kathleen)
#
# Application code and voice are at /app
# WAV files are cached at /cache
#
# Requires Docker buildx: https://docs.docker.com/buildx/working-with-buildx/
# See scripts/build-voices.sh
# -----------------------------------------------------------------------------

ARG DOCKER_REGISTRY
ARG LARYNX_TAG=latest
FROM $DOCKER_REGISTRY/rhasspy/larynx:$LARYNX_TAG

COPY voices/en-kathleen/tts/ /app/voice/tts/
COPY voices/en-kathleen/vocoder/ /app/voice/vocoder/

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/.venv/bin/python3", "-m", "larynx", "serve", \
            "--model", "/app/voice/tts/en-kathleen_tts-v1.pth.tar", \
            "--vocoder-model", "/app/voice/vocoder/en-kathleen_vocoder-v1.pth.tar", \
            "--cache-dir", "/cache"]
