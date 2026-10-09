set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
toolchain := env("VOXTYPE_RUST_TOOLCHAIN", "nightly-2026-08-22")
export TRANSCRIBE_DIR := env("TRANSCRIBE_DIR", root / "target/native-cuda")
export CARGO_TARGET_DIR := root / "target/granite-cuda"
export RUSTFLAGS := "-C link-arg=-Wl,-rpath," + TRANSCRIBE_DIR / "lib"
export PATH := root / "target/tools/cmake-3.31.6-linux-x86_64/bin" + ":" + env("PATH")

default: build
    bash scripts/run-granite.sh

build:
    [[ -f "$TRANSCRIBE_DIR/lib/libtranscribe.so" ]] || { printf 'Set TRANSCRIBE_DIR to a transcribe.cpp 0.3.1 shared runtime.\n' >&2; exit 1; }
    cargo +{{toolchain}} build --locked --features granite-cuda --bin voxtype

test:
    cargo +{{toolchain}} test --locked --features granite-cuda --quiet

check:
    cargo +{{toolchain}} fmt --check
    cargo +{{toolchain}} clippy --locked --features granite-cuda --all-targets -- -D warnings