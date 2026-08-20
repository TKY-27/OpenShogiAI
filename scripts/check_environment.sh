#!/bin/sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
wasm_bindgen="$project_root/local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen"
failures=0

require_command() {
    command_name=$1
    description=$2

    if command -v "$command_name" >/dev/null 2>&1; then
        version=$("$command_name" --version 2>/dev/null | head -n 1)
        printf 'OK   %-12s %s\n' "$description" "$version"
    else
        printf 'FAIL %-12s command not found: %s\n' "$description" "$command_name"
        failures=$((failures + 1))
    fi
}

printf 'OpenShogiAI environment\n'
printf 'OS:   macOS %s (%s)\n' "$(sw_vers -productVersion)" "$(uname -m)"

if command -v system_profiler >/dev/null 2>&1; then
    chip=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F ': ' '/^[[:space:]]+Chip:/{print $2; exit}')
    memory=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F ': ' '/^[[:space:]]+Memory:/{print $2; exit}')
    if [ -n "$chip" ] || [ -n "$memory" ]; then
        printf 'Host: %s, %s memory\n' "${chip:-unknown chip}" "${memory:-unknown}"
    fi
fi

require_command git Git
require_command rustc Rust
require_command cargo Cargo
require_command uv uv
require_command python3.12 Python
require_command make Make

if command -v rustc >/dev/null 2>&1; then
    rust_version=$(rustc --version | awk '{print $2}')
    rust_major=$(printf '%s\n' "$rust_version" | awk -F '.' '{print $1}')
    rust_minor=$(printf '%s\n' "$rust_version" | awk -F '.' '{print $2}')
    if [ "$rust_major" -lt 1 ] || { [ "$rust_major" -eq 1 ] && [ "$rust_minor" -lt 89 ]; }; then
        printf 'FAIL Rust         expected >=1.89, found %s\n' "$rust_version"
        failures=$((failures + 1))
    fi
fi

if command -v python3.12 >/dev/null 2>&1; then
    python_minor=$(python3.12 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$python_minor" != "3.12" ]; then
        printf 'FAIL Python       expected 3.12, found %s\n' "$python_minor"
        failures=$((failures + 1))
    fi
fi

if command -v rustup >/dev/null 2>&1 \
    && rustup target list --installed | grep -qx "wasm32-unknown-unknown"; then
    printf 'OK   %-12s %s\n' "Wasm target" "wasm32-unknown-unknown"
else
    printf 'FAIL %-12s run: rustup target add wasm32-unknown-unknown\n' "Wasm target"
    failures=$((failures + 1))
fi

if [ -x "$wasm_bindgen" ] \
    && [ "$("$wasm_bindgen" --version 2>/dev/null)" = "wasm-bindgen 0.2.127" ]; then
    printf 'OK   %-12s %s\n' "wasm-bindgen" "0.2.127"
else
    printf 'FAIL %-12s expected wasm-bindgen 0.2.127 at %s\n' \
        "wasm-bindgen" "$wasm_bindgen"
    failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
    printf '\nEnvironment check failed with %s required issue(s).\n' "$failures"
    exit 1
fi

printf '\nRequired development tools are available.\n'
