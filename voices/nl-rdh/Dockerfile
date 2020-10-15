# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
# Contains Dutch rdh voice (https://github.com/rhasspy/nl_larynx-rdh)
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

FROM rhasspy/larynx

# Remove proxy
# IFDEF PROXY
#! RUN rm -f /etc/apt/apt.conf.d/01proxy
# ENDIF

# IFDEF PYPI
#! ENV PIP_INDEX_URL=''
#! ENV PIP_TRUSTED_HOST=''
# ENDIF

COPY voices/nl-rdh/tts/ /app/voice/tts/
COPY voices/nl-rdh/vocoder/ /app/voice/vocoder/

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/.venv/bin/python3", "-m", "larynx", "serve", \
            "--model", "/app/voice/tts/nl-rdh_tts-v1.pth.tar", \
            "--vocoder-model", "/app/voice/vocoder/nl-rdh_vocoder-v1.pth.tar", \
            "--cache-dir", "/tmp/larynx"]
