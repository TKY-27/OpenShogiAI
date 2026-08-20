# Security Policy

OpenShogiAI is pre-release research software. No production security or compatibility claim is
made.

Report only a minimal non-sensitive description through the repository's available private
security channel. Do not put credentials, personal data, private games, exploit payloads
against third-party services, or unlicensed data in an issue.

CSA, SFEN, USI, downloaded data, teacher output, model files, manifests, registries, Wasm
messages, and generated artifacts are untrusted inputs. Preserve size limits, closed schemas,
path containment, stable-file identity, hashes, legality checks, and resource bounds.

Teacher executables and evaluation files must remain under ignored `local/teacher/` storage.
Bootstrap and setup scripts must not require administrator privileges, silently install
software, or pipe downloaded content into a shell.

Security fixes target the active private candidate branch until a versioned support policy is
published.
