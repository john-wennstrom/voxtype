#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
native=${TRANSCRIBE_DIR:-"$root/target/native-cuda"}
binary=${VOXTYPE_NEMOTRON_BINARY:-"$root/target/nemotron-cuda/debug/voxtype"}
config=${VOXTYPE_NEMOTRON_CONFIG:-"$root/config/nemotron.toml"}
filename=nemotron-speech-streaming-en-0.6b-Q8_0.gguf
model=${VOXTYPE_NEMOTRON_MODEL:-"$root/models/$filename"}

if [[ ! -x "$binary" || ! -f "$native/lib/libtranscribe.so" ]]; then
    printf 'Native Nemotron development build is missing. Run just build-nemotron.\n' >&2
    exit 1
fi
if [[ ! -f "$model" ]]; then
    printf 'Model is missing: %s\nPlace the Nemotron Q8 GGUF there, or set VOXTYPE_NEMOTRON_MODEL.\n' "$model" >&2
    exit 1
fi
if [[ ! -f "$config" ]]; then
    printf 'Config is missing: %s\n' "$config" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if (( $# == 0 )); then
    set -- daemon
fi
if [[ "$1" == daemon ]]; then
    if ! command -v eitype >/dev/null; then
        printf 'eitype is required by config/nemotron.toml. Install it before starting the daemon.\n' >&2
        exit 1
    fi
    accessible=false
    for device in /dev/input/event*; do
        if [[ -r "$device" ]]; then accessible=true; break; fi
    done
    if [[ "$accessible" != true ]]; then
        printf 'Input-device access is required for Super+V. Grant access and log in again before starting.\n' >&2
        exit 1
    fi
    if ! eitype ""; then
        printf 'eitype authorization was declined; daemon startup cancelled.\n' >&2
        exit 1
    fi
fi

exec "$binary" --config "$config" --engine nemotron --model "$model" "$@"