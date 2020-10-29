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

FROM ubuntu:eoan as build-ubuntu

ENV LANG C.UTF-8

# IFDEF PROXY
#! RUN echo 'Acquire::http { Proxy "http://${PROXY}"; };' >> /etc/apt/apt.conf.d/01proxy
# ENDIF

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------

FROM build-ubuntu as build-amd64

FROM build-ubuntu as build-armv7

RUN apt-get install --no-install-recommends --yes \
        llvm-7-dev libatlas-base-dev libopenblas-dev gfortran \
        libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
        libharfbuzz-dev libfribidi-dev libxcb1-dev

ENV LLVM_CONFIG=/usr/bin/llvm-config-7

FROM build-ubuntu as build-arm64

RUN apt-get install --no-install-recommends --yes \
        llvm-7-dev libatlas-base-dev libopenblas-dev gfortran \
        libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
        libharfbuzz-dev libfribidi-dev libxcb1-dev

ENV LLVM_CONFIG=/usr/bin/llvm-config-7

# -----------------------------------------------------------------------------

ARG TARGETARCH
ARG TARGETVARIANT
FROM build-$TARGETARCH$TARGETVARIANT as build

# IFDEF PYPI
#! ENV PIP_INDEX_URL=http://${PYPI}/simple/
#! ENV PIP_TRUSTED_HOST=${PYPI_HOST}
# ENDIF

COPY requirements.txt /app/
COPY scripts/create-venv.sh /app/scripts/
COPY TTS/ /app/TTS/

RUN cd /app && \
    export stage=0 end_stage=0 && \
    scripts/create-venv.sh

COPY download/ /app/download/

# Install app
RUN cd /app && \
    export PIP_INSTALL='install -f /app/download' && \
    export SETUP_DEVELOP='-f /app/download' && \
    scripts/create-venv.sh

# -----------------------------------------------------------------------------

FROM ubuntu:eoan as run

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-distutils python3-llvmlite libpython3.7 \
        libsndfile1 libgomp1 libatlas3-base libgfortran4 libopenblas-base \
        libjpeg8 libopenjp2-7 libtiff5 libxcb1 \
        libnuma1


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
