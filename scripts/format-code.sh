#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    source "${venv}/bin/activate"
fi

python_files=("${src_dir}/ipa_tts/")

export PYTHONPATH="${src_dir}:${PYTHONPATH}"

# -----------------------------------------------------------------------------

black "${python_files[@]}"
isort --recursive "${python_files[@]}"

# -----------------------------------------------------------------------------

echo "OK"
