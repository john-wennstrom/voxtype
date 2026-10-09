set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
toolchain := env("VOXTYPE_RUST_TOOLCHAIN", "nightly-2026-08-22")
export TRANSCRIBE_DIR := env("TRANSCRIBE_DIR", root / "target/native-cuda")
export CARGO_TARGET_DIR := root / "target/granite-cuda"
export RUSTFLAGS := "-C link-arg=-Wl,-rpath," + TRANSCRIBE_DIR / "lib"
export PATH := root / "target/tools/cmake-3.31.6-linux-x86_64/bin" + ":" + env("PATH")

default: build-nemotron
    bash scripts/run-nemotron.sh

review: build-nemotron
    VOXTYPE_TRANSCRIPT_POPUP=true VOXTYPE_TRANSCRIPT_POPUP_REVIEW=true bash scripts/run-nemotron.sh

granite: build
    bash scripts/run-granite.sh

build-nemotron:
    [[ -f "$TRANSCRIBE_DIR/lib/libtranscribe.so" ]] || { printf 'Set TRANSCRIBE_DIR to a transcribe.cpp 0.3.1 shared runtime.\n' >&2; exit 1; }
    PKG_CONFIG_PATH="{{root}}/target/tools/wayland-dev/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" LIBRARY_PATH="{{root}}/target/tools/wayland-dev/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}" CARGO_TARGET_DIR="{{root}}/target/nemotron-cuda" cargo +{{toolchain}} build --locked --features nemotron-cuda,osd-native --bin voxtype --bin voxtype-osd-native

build:
    [[ -f "$TRANSCRIBE_DIR/lib/libtranscribe.so" ]] || { printf 'Set TRANSCRIBE_DIR to a transcribe.cpp 0.3.1 shared runtime.\n' >&2; exit 1; }
    cargo +{{toolchain}} build --locked --features granite-cuda --bin voxtype

test:
    cargo +{{toolchain}} test --locked --features granite-cuda --quiet

check:
    cargo +{{toolchain}} fmt --check
    cargo +{{toolchain}} clippy --locked --features granite-cuda --all-targets -- -D warnings