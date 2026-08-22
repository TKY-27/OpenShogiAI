//! Native OSAVAL02 inspection/inference entry point used by the parity gate.

use std::{env, fs, path::PathBuf};

use open_shogi_core::{Osaval02Evaluator, Osaval02History, parse_sfen, to_sfen};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_CORPUS_BYTES: u64 = 1024 * 1024;
const MAX_FIXTURES: usize = 128;

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Corpus {
    schema: String,
    provenance_policy: String,
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Fixture {
    #[serde(rename = "fixtureId")]
    id: String,
    sfen: String,
    categories: Vec<String>,
    position_sha256: String,
    provenance: FixtureProvenance,
    #[serde(default)]
    history: Osaval02History,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct FixtureProvenance {
    kind: String,
    source: String,
    contains_labels: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FixtureResult {
    fixture_id: String,
    inference: open_shogi_core::Osaval02Inference,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ParityOutput<'a> {
    schema: &'static str,
    model_identity: &'a open_shogi_core::Osaval02Identity,
    fixtures: Vec<FixtureResult>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let model_path = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: osaval02_infer MODEL CORPUS".to_owned())?;
    let corpus_path = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: osaval02_infer MODEL CORPUS".to_owned())?;
    if arguments.next().is_some() {
        return Err("usage: osaval02_infer MODEL CORPUS".to_owned());
    }
    let evaluator = Osaval02Evaluator::load_file(&model_path)
        .map_err(|error| format!("model validation failed: {error}"))?;
    let metadata =
        fs::metadata(&corpus_path).map_err(|error| format!("cannot inspect corpus: {error}"))?;
    if !metadata.is_file() || metadata.len() > MAX_CORPUS_BYTES {
        return Err("parity corpus must be a regular file no larger than 1 MiB".to_owned());
    }
    let encoded = fs::read(&corpus_path).map_err(|error| format!("cannot read corpus: {error}"))?;
    let corpus: Corpus = serde_json::from_slice(&encoded)
        .map_err(|error| format!("invalid corpus JSON: {error}"))?;
    if corpus.schema != "open_shogiai_osaval02_parity_corpus/v1" {
        return Err("parity corpus schema is incompatible".to_owned());
    }
    if corpus.provenance_policy
        != "synthetic positions only; no game records, evaluations, labels, or training examples"
    {
        return Err("parity corpus provenance policy is incompatible".to_owned());
    }
    if corpus.fixtures.is_empty() || corpus.fixtures.len() > MAX_FIXTURES {
        return Err("parity corpus fixture count is outside 1..=128".to_owned());
    }
    let mut results = Vec::with_capacity(corpus.fixtures.len());
    for fixture in corpus.fixtures {
        if fixture.id.is_empty() || fixture.id.len() > 64 {
            return Err("fixture id must contain 1..=64 bytes".to_owned());
        }
        if fixture.categories.is_empty()
            || fixture.provenance.kind != "synthetic-rule-state"
            || fixture.provenance.source != "hand-authored-from-public-shogi-rules"
            || fixture.provenance.contains_labels
        {
            return Err(format!(
                "fixture {} provenance is not source-safe",
                fixture.id
            ));
        }
        let position = parse_sfen(&fixture.sfen)
            .map_err(|error| format!("fixture {} SFEN is invalid: {error}", fixture.id))?;
        let first = evaluator
            .infer(&position, fixture.history)
            .map_err(|error| format!("fixture {} inference failed: {error}", fixture.id))?;
        let position_sfen = to_sfen(&position);
        let canonical = position_sfen
            .rsplit_once(' ')
            .map_or(position_sfen.as_str(), |(state, _)| state);
        if format!("{:x}", Sha256::digest(canonical.as_bytes())) != fixture.position_sha256 {
            return Err(format!(
                "fixture {} position hash does not match",
                fixture.id
            ));
        }
        let first_json = serde_json::to_vec(&first).map_err(|error| error.to_string())?;
        let second = evaluator
            .infer(&position, fixture.history)
            .map_err(|error| format!("fixture {} repeat failed: {error}", fixture.id))?;
        if first_json != serde_json::to_vec(&second).map_err(|error| error.to_string())? {
            return Err(format!(
                "fixture {} repeated inference is not deterministic",
                fixture.id
            ));
        }
        results.push(FixtureResult {
            fixture_id: fixture.id,
            inference: first,
        });
    }
    let output = ParityOutput {
        schema: "open_shogiai_osaval02_native_parity/v1",
        model_identity: evaluator.identity(),
        fixtures: results,
    };
    println!(
        "{}",
        serde_json::to_string(&output).map_err(|error| error.to_string())?
    );
    Ok(())
}
