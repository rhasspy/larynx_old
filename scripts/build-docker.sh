#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

download="${src_dir}/download"
mkdir -p "${download}"

version="$(cat "${src_dir}/VERSION")"

# -----------------------------------------------------------------------------

: "${PLATFORMS=linux/amd64,linux/arm/v7,linux/arm64}"
: "${DOCKER_REGISTRY=docker.io}"
: "${DOCKER_BUILDX=1}"

DOCKERFILE="${src_dir}/Dockerfile"

if [[ -n "${PROXY}" ]]; then
    if [[ -z "${PROXY_IP}" ]]; then
        export PROXY_IP="$(hostname -I | awk '{print $1}')"
    fi

    export PROXY_PORT=3142
    export PROXY="${PROXY_IP}:${PROXY_PORT}"
    export PYPI_PORT=4000
    export PYPI="${PROXY_IP}:${PYPI_PORT}"
    export PYPI_HOST="${PROXY_IP}"

    # Use temporary file instead
    temp_dockerfile="$(mktemp -p "${src_dir}")"
    function cleanup {
        rm -f "${temp_dockerfile}"
    }

    trap cleanup EXIT

    # Run through pre-processor to replace variables
    "${src_dir}/docker/preprocess.sh" < "${DOCKERFILE}" > "${temp_dockerfile}"
    DOCKERFILE="${temp_dockerfile}"
fi

if [[ -n "${DOCKER_BUILDX}" ]]; then
    docker buildx build \
           "${src_dir}" \
           -f "${DOCKERFILE}" \
           "--platform=${PLATFORMS}" \
           --build-arg "DOCKER_REGISTRY=${DOCKER_REGISTRY}" \
           --tag "${DOCKER_REGISTRY}/rhasspy/larynx:${version}" \
           --tag "${DOCKER_REGISTRY}/rhasspy/larynx:latest" \
           --push \
           "$@"
else
    docker build \
           "${src_dir}" \
           -f "${DOCKERFILE}" \
           --build-arg "DOCKER_REGISTRY=${DOCKER_REGISTRY}" \
           --tag "${DOCKER_REGISTRY}/rhasspy/larynx:${version}" \
           --tag "${DOCKER_REGISTRY}/rhasspy/larynx:latest" \
           "$@"
fi
