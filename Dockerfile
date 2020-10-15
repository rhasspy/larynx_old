# -----------------------------------------------------------------------------
# Dockerfile for Larynx (https://github.com/rhasspy/larynx)
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

FROM ubuntu:eoan as base
# IFDEF DOCKER_BUILDX
#! ARG TARGETARCH
#! ARG TARGETVARIANT
# ENDIF

ENV LANG C.UTF-8

# IFDEF PROXY
#! RUN echo 'Acquire::http { Proxy "http://${PROXY}"; };' >> /etc/apt/apt.conf.d/01proxy
# ENDIF

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv python3-dev

# -----------------------------------------------------------------------------

FROM base as build
# IFDEF DOCKER_BUILDX
#! ARG TARGETARCH
#! ARG TARGETVARIANT
# ENDIF

RUN apt-get install --yes --no-install-recommends \
        python3-dev build-essential

# IFDEF PYPI
#! ENV PIP_INDEX_URL=http://${PYPI}/simple/
#! ENV PIP_TRUSTED_HOST=${PYPI_HOST}
# ENDIF

COPY download/ /app/download/

COPY requirements.txt /app/
COPY scripts/create-venv.sh /app/scripts/
COPY TTS/ /app/TTS/

# Install app
RUN cd /app && \
    export PIP_INSTALL='install -f /app/download' && \
    scripts/create-venv.sh

# -----------------------------------------------------------------------------

FROM base as run
# IFDEF DOCKER_BUILDX
#! ARG TARGETARCH
#! ARG TARGETVARIANT
# ENDIF

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 libpython3.7 libsndfile1


# Copy virtual environment
COPY --from=build /app/.venv/ /app/.venv/

# Copy TTS with compiled extension
COPY --from=build /app/TTS/ /app/TTS/

# Copy other files
COPY larynx/ /app/larynx/

# Need this since we installed numba as root but will be running as a regular user
ENV NUMBA_CACHE_DIR=/tmp

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/.venv/bin/python3", "-m", "larynx", "serve"]
