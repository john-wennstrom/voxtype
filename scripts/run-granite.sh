#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
native=${TRANSCRIBE_DIR:-"$root/../target/native-cuda"}
binary=${VOXTYPE_GRANITE_BINARY:-"$root/target/granite-cuda/debug/voxtype"}
config=${VOXTYPE_GRANITE_CONFIG:-"$root/config/granite.toml"}
filename=granite-speech-5.0-470m-turboctc-Q8_0.gguf
model=${VOXTYPE_GRANITE_MODEL:-"${XDG_DATA_HOME:-$HOME/.local/share}/granitevox/$filename"}

if [[ ! -f "$model" && -z ${VOXTYPE_GRANITE_MODEL:-} ]]; then
    model="${XDG_DATA_HOME:-$HOME/.local/share}/voxtype/models/$filename"
fi
if [[ ! -x "$binary" || ! -f "$native/lib/libtranscribe.so" ]]; then
    printf 'Native Granite development build is missing. Run just build.\n' >&2
    exit 1
fi
if [[ ! -f "$model" ]]; then
    printf 'Model is missing: %s\nRun: %s setup --download --model %s\n' "$model" "$binary" "$filename" >&2
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
        printf 'eitype is required by config/granite.toml. Install it before starting the daemon.\n' >&2
        exit 1
    fi
    accessible=false
    for device in /dev/input/event*; do
        if [[ -r "$device" ]]; then
            accessible=true
            break
        fi
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

exec "$binary" --config "$config" --engine granite --model "$model" "$@"