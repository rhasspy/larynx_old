#!/usr/bin/env bash
set -e

if [[ -z "${PIP_INSTALL}" ]]; then
    PIP_INSTALL='install'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

: "${DOWNLOAD_DIR=${src_dir}/download}"

# -----------------------------------------------------------------------------

venv="${src_dir}/.venv"

function maybe_download {
    if [[ ! -s "$2" ]]; then
        mkdir -p "$(dirname "$2")"
        curl -sSfL -o "$2" "$1" || { echo "Can't download $1"; exit 1; }
        echo "$1 => $2"
    fi
}

gruut_file="${DOWNLOAD_DIR}/gruut-0.5.0.tar.gz"
gruut_url='https://github.com/rhasspy/gruut/archive/v0.5.0.tar.gz'

maybe_download "${gruut_url}" "${gruut_file}"

# -----------------------------------------------------------------------------

: "${PYTHON=python3}"

python_version="$(${PYTHON} --version)"

: "${stage=0}"
: "${end_stage=5}"

# Stage 0: create virtual environment
if [[ "${stage}" -le 0 && "${end_stage}" -ge 0 ]]; then
    # Create virtual environment
    echo "Creating virtual environment at ${venv} (${python_version})"
    rm -rf "${venv}"
    "${PYTHON}" -m venv "${venv}"
    source "${venv}/bin/activate"

    # Install Python dependencies
    echo 'Installing Python dependencies'
    pip3 ${PIP_INSTALL} --upgrade pip
    pip3 ${PIP_INSTALL} --upgrade wheel setuptools

    deactivate
fi

source "${venv}/bin/activate"

# Stage 1: install requirements
if [[ "${stage}" -le 1 && "${end_stage}" -ge 1 ]]; then
    # Preinstallation
    if [[ -n "${PIP_PREINSTALL_PACKAGES}" ]]; then
        pip3 ${PIP_INSTALL} ${PIP_PREINSTALL_PACKAGES}
    fi

    if [[ -f "${gruut_file}" ]]; then
        echo 'Installing gruut'
        pip3 ${PIP_INSTALL} "${gruut_file}"
    fi

    if [[ -f requirements.txt ]]; then
        echo 'Installing requirements'
        pip3 ${PIP_INSTALL} -r requirements.txt
    fi
fi

# Stage 2: install torch
if [[ "${stage}" -le 2 && "${end_stage}" -ge 2 ]]; then
    # Install torch
    CPU_ARCH="$(uname -m)"
    case "${CPU_ARCH}" in
        aarch64|arm64v8)
            PLATFORM=aarch64
            ;;

        *)
            PLATFORM="${CPU_ARCH}"
            ;;
    esac

    torch_wheel="${DOWNLOAD_DIR}/torch-1.6.0-cp37-cp37m-linux_${PLATFORM}.whl"
    if [[ -f "${torch_wheel}" ]]; then
        echo 'Using local torch wheel'
        pip3 ${PIP_INSTALL} "${torch_wheel}"
    else
        echo "No torch wheel found at ${torch_wheel}"
    fi
fi

# Stage 3: install MozillaTTS requirements
if [[ "${stage}" -le 3 && "${end_stage}" -ge 3 ]]; then
    # Install MozillaTTS
    echo 'Installing MozillaTTS'
    pip3 ${PIP_INSTALL} -r TTS/requirements.txt
fi

# Stage 4: do develop setup
if [[ "${stage}" -le 4 && "${end_stage}" -ge 4 ]]; then
    echo 'Setting up MozillaTTS development'
    pushd TTS
    python3 setup.py develop "${SETUP_DEVELOP}"
    popd
fi

# Stage 5: Install dev dependencies (linters)
if [[ "${stage}" -le 5 && "${end_stage}" -ge 5 ]]; then
    # Development dependencies
    if [[ -f requirements_dev.txt ]]; then
        pip3 ${PIP_INSTALL} -r requirements_dev.txt || echo 'Failed to install development dependencies' >&2
    fi
fi

# -----------------------------------------------------------------------------

echo "OK"
