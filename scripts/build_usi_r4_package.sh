#!/bin/sh
# Build the relocatable R4 USI launch package under local/usi-r4/: the freshly built
# native engine, the pinned OSAI R4 weights, an executable launcher, checksums, and a
# short usage file. The model hash is read from configs/models/distribution.json so the
# package never invents a second source of truth.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

selection=r4c4
package_dir="${PACKAGE_DIR:-$project_root/local/usi-r4}"

mkdir -p "$package_dir"

if ! python3 - "$project_root" "$selection" > "$package_dir/.expected.identity" <<'PY'
import json, sys
manifest = json.load(open(f"{sys.argv[1]}/configs/models/distribution.json"))
model = next(m for m in manifest["models"] if m["selection"] == sys.argv[2])
weights = model["weights"]
assert weights["asset"] == "osai-r4.osaval03", weights["asset"]
print(manifest["release_tag"])
print(weights["sha256"])
PY
then
    echo "error: cannot read the R4 weights identity from configs/models/distribution.json" >&2
    rm -f "$package_dir/.expected.identity"
    exit 1
fi
release_tag=$(sed -n '1p' "$package_dir/.expected.identity")
expected_hash=$(sed -n '2p' "$package_dir/.expected.identity")
rm -f "$package_dir/.expected.identity"
download_url="https://github.com/TKY-27/OpenShogiAI/releases/download/${release_tag}/osai-r4.osaval03"
if ! printf '%s' "$expected_hash" | grep -Eq '^[0-9a-f]{64}$'; then
    echo "error: manifest returned an invalid hash" >&2
    exit 1
fi

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

mkdir -p "$package_dir"
model_path="$package_dir/osai-r4.osaval03"
model_ok=false
if [ -f "$model_path" ] && [ "$(checksum "$model_path")" = "$expected_hash" ]; then
    model_ok=true
else
    # Prefer a verified local release copy; otherwise fetch the pinned Release asset
    # with bounded retries and atomic staging. Nothing unverified is ever parked at
    # the final model path.
    candidate="$project_root/local/release/$release_tag/osai-r4.osaval03"
    if [ -f "$candidate" ] && [ "$(checksum "$candidate")" = "$expected_hash" ]; then
        cp "$candidate" "$model_path.tmp"
        model_ok=true
    else
        echo "downloading the pinned R4 weights from $download_url"
        attempt=1
        while [ "$attempt" -le 3 ]; do
            if curl -fL --connect-timeout 15 --max-time 300 "$download_url" -o "$model_path.tmp"; then
                model_ok=true
                break
            fi
            rm -f "$model_path.tmp"
            attempt=$((attempt + 1))
            sleep 2
        done
        [ "$model_ok" = true ] || {
            echo "error: download failed after 3 attempts; place osai-r4.osaval03 manually" >&2
            exit 1
        }
    fi
    [ "$(checksum "$model_path.tmp")" = "$expected_hash" ] || {
        echo "error: staged weights hash mismatch; refusing to install them" >&2
        rm -f "$model_path.tmp"
        exit 1
    }
    mv "$model_path.tmp" "$model_path"
fi
actual_hash=$(checksum "$model_path")
[ "$actual_hash" = "$expected_hash" ] || {
    echo "error: R4 weights hash mismatch; refusing to build the package" >&2
    exit 1
}

# Release build through the native-only entry; target-cpu stays default so the binary
# can run on machines other than the build host.
CARGO_TARGET_DIR="$project_root/target/pure" ./scripts/build_pure_native.sh
cp "$project_root/target/pure/release/open-shogi-cli" "$package_dir/open-shogi-cli"

cat > "$package_dir/osai-r4-usi.sh" <<LAUNCHER
#!/bin/sh
# OpenShogiAI R4 (pure_learned, OSAVAL03) USI launcher. Relocatable and silent:
# the only stdout traffic is the USI protocol itself.
set -eu
# pwd -P resolves symlinked install parents to a physical path, which the engine's
# no-symlink secure model loader requires.
here=\$(CDPATH= cd -- "\$(dirname -- "\$0")" && pwd -P)
exec "\$here/open-shogi-cli" usi \\
    --model "\$here/osai-r4.osaval03" \\
    --model-format OSAVAL03 \\
    --model-sha256 "$expected_hash" \\
    --profile pure_learned
LAUNCHER
chmod +x "$package_dir/osai-r4-usi.sh" "$package_dir/open-shogi-cli"

{
    echo "$expected_hash  osai-r4.osaval03"
    echo "$(checksum "$package_dir/open-shogi-cli")  open-shogi-cli"
    echo "$(checksum "$package_dir/osai-r4-usi.sh")  osai-r4-usi.sh"
} > "$package_dir/SHA256SUMS"

source_revision=$(git -C "$project_root" rev-parse HEAD 2>/dev/null || echo unknown)
if [ -n "$(git -C "$project_root" status --porcelain 2>/dev/null)" ]; then
    source_revision="${source_revision}-dirty"
fi
{
    echo "source_revision=$source_revision"
    echo "model_asset=osai-r4.osaval03 ($release_tag)"
    echo "model_sha256=$expected_hash"
    echo "options=USI_Hash spin 1..1024 default 32; USI_Ponder check default false; Threads spin 1..1 default 1; RuntimeProfile combo pure_learned"
    echo "play_policy=book-free learned play, one search worker, pondering off"
} > "$package_dir/PACKAGE_IDENTITY.txt"

cat > "$package_dir/USAGE.txt" <<USAGE
OpenShogiAI R4 USI package
==========================

Contents
  osai-r4-usi.sh      launcher to register in ShogiHome (keep next to the engine)
  open-shogi-cli      native engine binary built from the recorded source revision
  osai-r4.osaval03    fixed public OSAI R4 weights, SHA-256 verified on load
  SHA256SUMS          checksums of the files above
  PACKAGE_IDENTITY.txt source revision, model hash, supported protocol options

Build from source (Linux or macOS, Rust stable)
  cargo build --locked --release -p open-shogi-cli --no-default-features --features pure-only
  ./scripts/build_usi_r4_package.sh        # assembles this package

Register in ShogiHome (1.29.0 names)
  1. Make the launcher executable once: chmod +x osai-r4-usi.sh
  2. Open 設定 → エンジン設定... (Settings → Engines...).
  3. Click 追加 (Add) and select the file osai-r4-usi.sh (the launcher, not the
     binary).
  4. Keep USI_Ponder false and Threads 1; USI_Hash defaults to 32 MiB.

The engine plays book-free learned shogi with one search worker. TT budget
(USI_Hash) is the transposition-table size, not the process's total memory.
USAGE

echo "package ready: $package_dir"
