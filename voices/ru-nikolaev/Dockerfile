# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
# Contains Russian nikolaev voice (https://github.com/rhasspy/ru_larynx-nikolaev)
#
# Application code and voice are at /app
# WAV files are cached at /cache
#
# Requires Docker buildx: https://docs.docker.com/buildx/working-with-buildx/
# See scripts/build-docker.sh
#
# The IFDEF statements are handled by docker/preprocess.sh. These are just
# comments that are uncommented if the environment variable after the IFDEF is
# not empty.
#
# The build-docker.sh script will optionally add apt/pypi proxies running locally:
# * apt - https://docs.docker.com/engine/examples/apt-cacher-ng/ 
# * pypi - https://github.com/jayfk/docker-pypi-cache
# -----------------------------------------------------------------------------

ARG DOCKER_REGISTRY
FROM $DOCKER_REGISTRY/rhasspy/larynx

# Remove proxy
# IFDEF PROXY
#! RUN rm -f /etc/apt/apt.conf.d/01proxy
# ENDIF

# IFDEF PYPI
#! ENV PIP_INDEX_URL=''
#! ENV PIP_TRUSTED_HOST=''
# ENDIF

COPY voices/ru-nikolaev/tts/ /app/voice/tts/
COPY voices/ru-nikolaev/vocoder/ /app/voice/vocoder/

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/.venv/bin/python3", "-m", "larynx", "serve", \
            "--model", "/app/voice/tts/ru-nikolaev_tts-v1.pth.tar", \
            "--vocoder-model", "/app/voice/vocoder/ru-nikolaev_vocoder-v1.pth.tar", \
            "--cache-dir", "/cache"]
