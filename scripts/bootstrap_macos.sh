#!/bin/sh

set -eu

missing=0

print_install_hint() {
    tool=$1
    hint=$2

    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'MISSING %-12s run manually: %s\n' "$tool" "$hint"
        missing=$((missing + 1))
    fi
}

printf 'OpenShogiAI macOS bootstrap (diagnostic only)\n'
printf 'This script does not install software or request administrator privileges.\n\n'

if ! xcode-select -p >/dev/null 2>&1; then
    printf 'MISSING Xcode tools  run manually: xcode-select --install\n'
    missing=$((missing + 1))
fi

print_install_hint git "install Xcode Command Line Tools"
print_install_hint rustup "install rustup from https://rustup.rs, then run: rustup toolchain install stable"
print_install_hint cargo "rustup toolchain install stable"
print_install_hint make "install Xcode Command Line Tools"

if command -v brew >/dev/null 2>&1; then
    print_install_hint uv "brew install uv"
else
    print_install_hint uv "install from https://docs.astral.sh/uv/getting-started/installation/"
fi

if command -v uv >/dev/null 2>&1 && ! command -v python3.12 >/dev/null 2>&1; then
    printf 'MISSING Python 3.12  run manually: uv python install 3.12\n'
    missing=$((missing + 1))
fi

if command -v rustup >/dev/null 2>&1 \
    && ! rustup target list --installed | grep -qx "wasm32-unknown-unknown"; then
    printf 'MISSING Wasm target  run manually: rustup target add wasm32-unknown-unknown\n'
    missing=$((missing + 1))
fi

wasm_bindgen="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)/local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen"
if [ ! -x "$wasm_bindgen" ] \
    || [ "$("$wasm_bindgen" --version 2>/dev/null || true)" != "wasm-bindgen 0.2.127" ]; then
    printf '%s\n' \
        'MISSING wasm-bindgen  run manually: cargo install wasm-bindgen-cli --version 0.2.127 --locked --root local/tooling/wasm-bindgen-0.2.127'
    missing=$((missing + 1))
fi

printf '\n'
if [ "$missing" -ne 0 ]; then
    printf 'Found %s required setup item(s). Run the printed commands, then rerun this script.\n' "$missing"
    exit 1
fi

./scripts/check_environment.sh
printf '\nBootstrap inspection complete. No changes were made to the machine.\n'
