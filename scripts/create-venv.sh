#!/usr/bin/env bash
set -e

if [[ -z "${PIP_INSTALL}" ]]; then
    PIP_INSTALL='install'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

download_dir="${src_dir}/download"

# -----------------------------------------------------------------------------

venv="${src_dir}/.venv"

# -----------------------------------------------------------------------------

: "${PYTHON=python3}"

python_version="$(${PYTHON} --version)"

# Create virtual environment
echo "Creating virtual environment at ${venv} (${python_version})"
rm -rf "${venv}"
"${PYTHON}" -m venv "${venv}"
source "${venv}/bin/activate"

# Install Python dependencies
echo 'Installing Python dependencies'
pip3 ${PIP_INSTALL} --upgrade pip
pip3 ${PIP_INSTALL} --upgrade wheel setuptools

if [[ -f requirements.txt ]]; then
    pip3 ${PIP_INSTALL} -r requirements.txt
fi

# Install torch
CPU_ARCH="$(uname --m)"
case "${CPU_ARCH}" in
    aarch64|arm64v8)
        PLATFORM=aarch64
        ;;

    *)
        PLATFORM="${CPU_ARCH}"
        ;;
esac

torch_wheel="${download_dir}/torch-1.6.0-cp37-cp37m-linux_${PLATFORM}.whl"
if [[ -f "${torch_wheel}" ]]; then
    echo 'Using local torch wheel'
    pip3 ${PIP_INSTALL} "${torch_wheel}"
fi

# Install MozillaTTS
echo 'Installing MozillaTTS'
pip3 ${PIP_INSTALL} -r TTS/requirements.txt
pushd TTS
python3 setup.py develop
popd

# Development dependencies
if [[ -f requirements_dev.txt ]]; then
    pip3 ${PIP_INSTALL} -r requirements_dev.txt || echo 'Failed to install development dependencies' >&2
fi

# -----------------------------------------------------------------------------

echo "OK"
