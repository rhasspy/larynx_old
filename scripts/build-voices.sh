#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

download="${src_dir}/download"
mkdir -p "${download}"

# -----------------------------------------------------------------------------

: "${PLATFORMS=linux/amd64,linux/arm/v7,linux/arm64}"
: "${DOCKER_REGISTRY=docker.io}"
: "${DOCKER_BUILDX=1}"

# -----------------------------------------------------------------------------

temp_files=()
function cleanup {
    rm -f "${temp_files[@]}"
}

trap cleanup EXIT

# -----------------------------------------------------------------------------

voices_dir="${src_dir}/voices"

voice_names="$(mktemp)"
temp_files+=("${voice_names[@]}")

if [[ -z "$1" ]]; then
    # All voices
    find "${voices_dir}" -maxdepth 1 -mindepth 1 -type d > "${voice_names}"
else
    # Specified voices
    while [[ -n "$1" ]]; do
        echo "${voices_dir}/$1" >> "${voice_names}"
        shift
    done
fi

while read -r voice_dir; do
    voice_name="$(basename "${voice_dir}")"
    version="$(cat "${voice_dir}/VERSION")"

    echo "Building voice ${voice_name} version ${version}..."

    DOCKERFILE="${voice_dir}/Dockerfile"

    if [[ -n "${NOBUILDX}" ]]; then
        # Don't use docker buildx (single platform)
        docker build \
               "${src_dir}" \
               -f "${DOCKERFILE}" \
               --build-arg "DOCKER_REGISTRY=${DOCKER_REGISTRY}" \
               --tag "${DOCKER_REGISTRY}/rhasspy/larynx:${voice_name}-${version}" \
               "$@"
    else
        # Use docker buildx (multi-platform)
        docker buildx build \
               "${src_dir}" \
               -f "${DOCKERFILE}" \
               "--platform=${PLATFORMS}" \
               --build-arg "DOCKER_REGISTRY=${DOCKER_REGISTRY}" \
               --tag "${DOCKER_REGISTRY}/rhasspy/larynx:${voice_name}-${version}" \
               --push \
               "$@"
    fi

done < "${voice_names}"
