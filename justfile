set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
toolchain := env("VOXTYPE_RUST_TOOLCHAIN", "nightly-2026-08-22")
export TRANSCRIBE_DIR := env("TRANSCRIBE_DIR", root / "target/native-cuda")
export CARGO_TARGET_DIR := root / "target/granite-cuda"
export RUSTFLAGS := "-C link-arg=-Wl,-rpath," + TRANSCRIBE_DIR / "lib"
export PATH := root / "target/tools/cmake-3.31.6-linux-x86_64/bin" + ":" + env("PATH")
export NLTK_DATA := env("VOXTYPE_NLTK_DATA", root / "../.voxtype-nltk-data")

default: build-nemotron
    bash scripts/run-nemotron.sh

review: build-nemotron
    bash scripts/run-nemotron.sh review

granite: build
    bash scripts/run-granite.sh

build-nemotron:
    [[ -f "$TRANSCRIBE_DIR/lib/libtranscribe.so" ]] || { printf 'Set TRANSCRIBE_DIR to a transcribe.cpp 0.3.1 shared runtime.\n' >&2; exit 1; }
    PKG_CONFIG_PATH="{{root}}/target/tools/wayland-dev/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" LIBRARY_PATH="{{root}}/target/tools/wayland-dev/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}" CARGO_TARGET_DIR="{{root}}/target/nemotron-cuda" cargo +{{toolchain}} build --locked --features nemotron-cuda,osd-native --bin voxtype --bin voxtype-osd-native

install-cli: build-nemotron
    #!/usr/bin/env bash
    set -euo pipefail
    binary="{{root}}/target/nemotron-cuda/debug/voxtype"
    launcher="{{root}}/scripts/run-nemotron.sh"
    destination="$HOME/.local/bin/voxtype"
    mkdir -p "$HOME/.local/bin"
    if [[ -e "$destination" || -L "$destination" ]]; then
        [[ -L "$destination" && ( "$(readlink "$destination")" == "$binary" || "$(readlink "$destination")" == "$launcher" ) ]] || { printf 'Refusing to replace %s. Remove or rename it explicitly first.\n' "$destination" >&2; exit 1; }
        ln -sfn "$launcher" "$destination"
    else
        ln -s "$launcher" "$destination"
    fi
    printf 'Installed %s. Keep this checkout and its native runtime in place.\n' "$destination"
    case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) printf 'Add ~/.local/bin to PATH: export PATH="$HOME/.local/bin:$PATH"\n' ;; esac

build:
    [[ -f "$TRANSCRIBE_DIR/lib/libtranscribe.so" ]] || { printf 'Set TRANSCRIBE_DIR to a transcribe.cpp 0.3.1 shared runtime.\n' >&2; exit 1; }
    cargo +{{toolchain}} build --locked --features granite-cuda --bin voxtype

test:
    cargo +{{toolchain}} test --locked --features granite-cuda --quiet

check:
    cargo +{{toolchain}} fmt --check
    cargo +{{toolchain}} clippy --locked --features granite-cuda --all-targets -- -D warnings

lexicon-setup:
    python3 -m venv "{{root}}/.venv-lexicon"
    "{{root}}/.venv-lexicon/bin/python" -m pip install -r "{{root}}/tools/requirements-lexicon.txt"
    just --justfile "{{root}}/justfile" lexicon-nltk

lexicon-nltk:
    #!/usr/bin/env bash
    set -euo pipefail
    python="{{root}}/.venv-lexicon/bin/python"
    [[ -x "$python" ]] || { printf 'Run just lexicon-setup first.\n' >&2; exit 1; }
    mkdir -p -m 700 "$NLTK_DATA"
    "$python" -c 'import nltk; nltk.data.find("taggers/averaged_perceptron_tagger.zip"); nltk.data.find("taggers/averaged_perceptron_tagger_eng/"); nltk.data.find("corpora/cmudict.zip")' >/dev/null 2>&1 || "$python" -c 'import os, nltk; nltk.download(["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"], download_dir=os.environ["NLTK_DATA"], raise_on_error=True)'

lexicon-generate project=root backend="g2p_en": lexicon-nltk
    #!/usr/bin/env bash
    set -euo pipefail
    python="{{root}}/.venv-lexicon/bin/python"
    [[ -x "$python" ]] || { printf 'Run just lexicon-setup first.\n' >&2; exit 1; }
    directory="{{project}}/.voxtype"
    mkdir -p "$directory"
    [[ -e "$directory/terms.tsv" ]] || cp "{{root}}/examples/lexicon/words.txt" "$directory/terms.tsv"
    "$python" "{{root}}/tools/lexicon_g2p.py" "$directory/terms.tsv" "$directory/terms.phon.tsv" --backend "{{backend}}" --export-query-dict "$directory/querydict.phon.tsv"
    printf 'Generated %s/terms.phon.tsv and %s/querydict.phon.tsv\n' "$directory" "$directory"