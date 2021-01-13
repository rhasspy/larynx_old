#!/usr/bin/env bash
voice="$1"
version="$2"

if [[ -z "${voice}" ]]; then
    echo 'Usage: run-voice.sh <voice>'
    exit 1
fi

if [[ -z "${version}" ]]; then
    version='1'
fi

docker run -it \
       -p 5002:5002 \
       --device /dev/snd:/dev/snd \
       "rhasspy/larynx:${voice}-${version}"
