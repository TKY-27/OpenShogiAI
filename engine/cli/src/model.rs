use std::{
    io::{BufRead, BufReader, Cursor},
    path::{Path, PathBuf},
    time::Instant,
};

use open_shogi_core::{
    AnchoredDir, EvaluationConfig, MAX_NEURAL_MODEL_BYTES, NeuralActivation, NeuralEvaluator,
    NeuralQuantization, evaluate, parse_sfen, to_sfen,
};
use serde::Serialize;

use crate::{
    args::{next_value, path_next},
    checksum::read_file_artifact,
};

const INSPECTION_SCHEMA: &str = "phase5_model_inspection/v1";
const INFERENCE_SCHEMA: &str = "phase5_model_inference/v1";
const HANDCRAFTED_INFERENCE_SCHEMA: &str = "phase5_handcrafted_inference/v1";
const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 16 * 1024;
const MAX_POSITIONS: usize = 10_000;
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInspection<'a> {
    schema: &'a str,
    model_path: String,
    artifact_sha256: String,
    artifact_size: u64,
    payload_sha256: String,
    format_version: u32,
    architecture_version: u32,
    feature_schema_version: u32,
    feature_flags: u32,
    input_dimension: usize,
    hidden_layers: usize,
    hidden_dimension: usize,
    activation: &'a str,
    quantization: &'a str,
    layer_count: usize,
    output_scale_cp: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceRecord<'a> {
    schema: &'a str,
    model_artifact_sha256: &'a str,
    model_payload_sha256: &'a str,
    index: usize,
    sfen: String,
    score_cp: i32,
    elapsed_ns: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HandcraftedInferenceRecord<'a> {
    schema: &'a str,
    evaluator_profile: &'a str,
    index: usize,
    sfen: String,
    score_cp: i32,
    elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HandcraftedProfile {
    Baseline,
    Experimental,
}

impl HandcraftedProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "handcrafted-baseline" => Ok(Self::Baseline),
            "handcrafted-experimental" => Ok(Self::Experimental),
            _ => {
                Err("--profile must be handcrafted-baseline or handcrafted-experimental".to_owned())
            }
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "handcrafted-baseline",
            Self::Experimental => "handcrafted-experimental",
        }
    }

    const fn evaluation(self) -> EvaluationConfig {
        match self {
            Self::Baseline => EvaluationConfig::handcrafted_baseline(),
            Self::Experimental => EvaluationConfig::handcrafted_experimental(),
        }
    }
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let Some(command) = arguments.first().map(String::as_str) else {
        return Err("model requires `inspect`, `infer`, or `infer-handcrafted`".to_owned());
    };
    match command {
        "inspect" => inspect(&arguments[1..]),
        "infer" => infer(&arguments[1..]),
        "infer-handcrafted" => infer_handcrafted(&arguments[1..]),
        "help" | "--help" | "-h" if arguments.len() == 1 => {
            print_help();
            Ok(())
        }
        _ => Err("model requires `inspect`, `infer`, or `infer-handcrafted`".to_owned()),
    }
}

fn print_help() {
    println!("OpenShogiAI model tools");
    println!("  open-shogi-cli model inspect --model FILE");
    println!(
        "  open-shogi-cli model infer --model FILE (--sfen SFEN | --input FILE) [--output FILE]"
    );
    println!(
        "  open-shogi-cli model infer-handcrafted --profile handcrafted-baseline|handcrafted-experimental (--sfen SFEN | --input FILE) [--output FILE]"
    );
}

fn inspect(arguments: &[String]) -> Result<(), String> {
    let model_path = parse_single_model_path(arguments)?;
    let (model, artifact_sha256, artifact_size) = load_model_artifact(&model_path)?;
    let identity = model.identity();
    let inspection = ModelInspection {
        schema: INSPECTION_SCHEMA,
        model_path: model_path.display().to_string(),
        artifact_sha256,
        artifact_size,
        payload_sha256: identity.sha256_hex(),
        format_version: identity.format_version,
        architecture_version: identity.architecture_version,
        feature_schema_version: identity.feature_schema_version,
        feature_flags: identity.feature_flags,
        input_dimension: identity.input_dimension,
        hidden_layers: identity.hidden_layers,
        hidden_dimension: identity.hidden_dimension,
        activation: activation_name(identity.activation),
        quantization: quantization_name(identity.quantization),
        layer_count: identity.layer_count,
        output_scale_cp: identity.output_scale_cp,
    };
    println!(
        "{}",
        serde_json::to_string(&inspection)
            .map_err(|error| format!("cannot serialize model inspection: {error}"))?
    );
    Ok(())
}

fn infer(arguments: &[String]) -> Result<(), String> {
    let mut model_path = None;
    let mut sfen = None;
    let mut input = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--model" => {
                let value = path_next(arguments, &mut index, "--model")?;
                set_once(&mut model_path, value, "--model")?;
            }
            "--sfen" => {
                let value = next_value(arguments, &mut index, "--sfen")?.to_owned();
                set_once(&mut sfen, value, "--sfen")?;
            }
            "--input" => {
                let value = path_next(arguments, &mut index, "--input")?;
                set_once(&mut input, value, "--input")?;
            }
            "--output" => {
                let value = path_next(arguments, &mut index, "--output")?;
                set_once(&mut output, value, "--output")?;
            }
            argument => return Err(format!("unknown model infer argument: {argument}")),
        }
        index += 1;
    }
    let model_path = model_path.ok_or_else(|| "model infer requires --model".to_owned())?;
    let positions = match (sfen, input) {
        (Some(value), None) => vec![value],
        (None, Some(path)) => read_sfen_lines(&path)?,
        _ => return Err("model infer requires exactly one of --sfen or --input".to_owned()),
    };
    let (model, model_artifact_sha256, _) = load_model_artifact(&model_path)?;
    let model_payload_sha256 = model.identity().sha256_hex();
    let mut records = Vec::with_capacity(positions.len());
    for (index, raw_sfen) in positions.into_iter().enumerate() {
        let position = parse_inference_sfen(&raw_sfen, index + 1)?;
        let started = Instant::now();
        let score_cp = model.evaluate(&position);
        records.push(InferenceRecord {
            schema: INFERENCE_SCHEMA,
            model_artifact_sha256: &model_artifact_sha256,
            model_payload_sha256: &model_payload_sha256,
            index,
            sfen: to_sfen(&position),
            score_cp,
            elapsed_ns: json_safe_elapsed_ns(started.elapsed())?,
        });
    }
    let mut encoded = String::new();
    for record in records {
        encoded.push_str(
            &serde_json::to_string(&record)
                .map_err(|error| format!("cannot serialize inference result: {error}"))?,
        );
        encoded.push('\n');
    }
    write_output(output.as_deref(), &encoded)
}

fn infer_handcrafted(arguments: &[String]) -> Result<(), String> {
    let mut profile = None;
    let mut sfen = None;
    let mut input = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--profile" => {
                let value =
                    HandcraftedProfile::parse(next_value(arguments, &mut index, "--profile")?)?;
                set_once(&mut profile, value, "--profile")?;
            }
            "--sfen" => {
                let value = next_value(arguments, &mut index, "--sfen")?.to_owned();
                set_once(&mut sfen, value, "--sfen")?;
            }
            "--input" => {
                let value = path_next(arguments, &mut index, "--input")?;
                set_once(&mut input, value, "--input")?;
            }
            "--output" => {
                let value = path_next(arguments, &mut index, "--output")?;
                set_once(&mut output, value, "--output")?;
            }
            argument => {
                return Err(format!(
                    "unknown model infer-handcrafted argument: {argument}"
                ));
            }
        }
        index += 1;
    }
    let profile = profile.ok_or_else(|| "model infer-handcrafted requires --profile".to_owned())?;
    let positions = match (sfen, input) {
        (Some(value), None) => vec![value],
        (None, Some(path)) => read_sfen_lines(&path)?,
        _ => {
            return Err(
                "model infer-handcrafted requires exactly one of --sfen or --input".to_owned(),
            );
        }
    };
    let evaluation = profile.evaluation();
    let mut encoded = String::new();
    for (index, raw_sfen) in positions.into_iter().enumerate() {
        let position = parse_inference_sfen(&raw_sfen, index + 1)?;
        let started = Instant::now();
        let score_cp = evaluate(&position, &evaluation);
        let record = HandcraftedInferenceRecord {
            schema: HANDCRAFTED_INFERENCE_SCHEMA,
            evaluator_profile: profile.name(),
            index,
            sfen: to_sfen(&position),
            score_cp,
            elapsed_ns: json_safe_elapsed_ns(started.elapsed())?,
        };
        encoded.push_str(
            &serde_json::to_string(&record).map_err(|error| {
                format!("cannot serialize handcrafted inference result: {error}")
            })?,
        );
        encoded.push('\n');
    }
    write_output(output.as_deref(), &encoded)
}

fn parse_single_model_path(arguments: &[String]) -> Result<PathBuf, String> {
    let mut model_path = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--model" => {
                let value = path_next(arguments, &mut index, "--model")?;
                set_once(&mut model_path, value, "--model")?;
            }
            argument => return Err(format!("unknown model inspect argument: {argument}")),
        }
        index += 1;
    }
    model_path.ok_or_else(|| "model inspect requires --model".to_owned())
}

fn load_model_artifact(path: &Path) -> Result<(NeuralEvaluator, String, u64), String> {
    let artifact = read_file_artifact(
        path,
        u64::try_from(MAX_NEURAL_MODEL_BYTES).unwrap_or(u64::MAX),
    )?;
    let model = NeuralEvaluator::from_bytes(&artifact.bytes)
        .map_err(|error| format!("cannot load neural model {}: {error}", path.display()))?;
    Ok((model, artifact.sha256, artifact.size))
}

fn read_sfen_lines(path: &Path) -> Result<Vec<String>, String> {
    let artifact = read_file_artifact(path, MAX_INPUT_BYTES)?;
    if artifact.size == 0 {
        return Err(format!(
            "SFEN input size must be 1..={MAX_INPUT_BYTES} bytes"
        ));
    }
    let mut reader = BufReader::new(Cursor::new(artifact.bytes));
    let mut bytes = Vec::new();
    let mut positions = Vec::new();
    loop {
        bytes.clear();
        let read = reader
            .read_until(b'\n', &mut bytes)
            .map_err(|error| format!("cannot read SFEN input: {error}"))?;
        if read == 0 {
            break;
        }
        if bytes.len() > MAX_LINE_BYTES {
            return Err(format!("SFEN input line exceeds {MAX_LINE_BYTES} bytes"));
        }
        while matches!(bytes.last(), Some(b'\n' | b'\r')) {
            bytes.pop();
        }
        if bytes.is_empty() {
            return Err("SFEN input contains an empty line".to_owned());
        }
        let line = String::from_utf8(bytes.clone())
            .map_err(|_| "SFEN input is not valid UTF-8".to_owned())?;
        positions.push(line);
        if positions.len() > MAX_POSITIONS {
            return Err(format!("SFEN input exceeds {MAX_POSITIONS} positions"));
        }
    }
    Ok(positions)
}

fn parse_inference_sfen(value: &str, line: usize) -> Result<open_shogi_core::Position, String> {
    if value.is_empty() || value.len() > MAX_LINE_BYTES {
        return Err(format!(
            "SFEN at input line {line} must be 1..={MAX_LINE_BYTES} bytes"
        ));
    }
    let candidate = if value.split_whitespace().count() == 3 {
        format!("{value} 1")
    } else {
        value.to_owned()
    };
    parse_sfen(&candidate).map_err(|error| format!("invalid SFEN at input line {line}: {error}"))
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("duplicate {option}"));
    }
    Ok(())
}

fn json_safe_elapsed_ns(elapsed: std::time::Duration) -> Result<u64, String> {
    let nanoseconds = elapsed.as_nanos();
    if nanoseconds > u128::from(MAX_JSON_SAFE_INTEGER) {
        return Err("inference duration exceeds the JSON-safe nanosecond range".to_owned());
    }
    u64::try_from(nanoseconds).map_err(|_| "inference duration exceeds u64".to_owned())
}

fn write_output(path: Option<&Path>, contents: &str) -> Result<(), String> {
    if let Some(path) = path {
        let name = path
            .file_name()
            .ok_or_else(|| "inference output path has no file name".to_owned())?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let directory = AnchoredDir::open_or_create_all(parent).map_err(|error| {
            format!("cannot anchor inference output directory without symlinks: {error}")
        })?;
        publish_output_to_anchored(&directory, name, path, contents)?;
    } else {
        print!("{contents}");
    }
    Ok(())
}

fn publish_output_to_anchored(
    directory: &AnchoredDir,
    name: &std::ffi::OsStr,
    display_path: &Path,
    contents: &str,
) -> Result<(), String> {
    directory
        .publish_new_atomic(name, contents.as_bytes())
        .map_err(|error| {
            format!(
                "cannot publish inference output {}: {error}",
                display_path.display()
            )
        })
}

const fn activation_name(activation: NeuralActivation) -> &'static str {
    match activation {
        NeuralActivation::Relu => "relu",
    }
}

const fn quantization_name(quantization: NeuralQuantization) -> &'static str {
    match quantization {
        NeuralQuantization::Float32 => "float32",
        NeuralQuantization::Int8 => "int8",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use sha2::{Digest, Sha256};

    use super::{
        json_safe_elapsed_ns, load_model_artifact, parse_inference_sfen,
        publish_output_to_anchored, run,
    };

    #[test]
    fn model_commands_require_bounded_explicit_inputs() {
        assert!(run(&[]).is_err());
        assert!(run(&["inspect".into()]).is_err());
        assert!(run(&["infer".into(), "--model".into(), "missing".into()]).is_err());
        assert!(run(&["infer-handcrafted".into()]).is_err());
        assert!(run(&["help".into()]).is_ok());
    }

    #[test]
    fn inference_elapsed_time_stays_in_the_cross_runtime_integer_range() {
        assert_eq!(
            json_safe_elapsed_ns(std::time::Duration::from_nanos(
                super::MAX_JSON_SAFE_INTEGER,
            ))
            .unwrap(),
            super::MAX_JSON_SAFE_INTEGER
        );
        assert!(
            json_safe_elapsed_ns(std::time::Duration::from_nanos(
                super::MAX_JSON_SAFE_INTEGER + 1,
            ))
            .is_err()
        );
    }

    #[test]
    fn inference_accepts_state_sfen_without_a_move_number() {
        let position = parse_inference_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -",
            1,
        )
        .unwrap();
        assert_eq!(position.move_number(), 1);
    }

    #[test]
    fn model_identity_and_evaluator_share_one_file_snapshot() {
        let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-model-snapshot-{}-{}.osaval",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes = model_bytes();
        std::fs::write(&path, &bytes).unwrap();
        let (model, artifact_sha256, artifact_size) = load_model_artifact(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(artifact_sha256, format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(artifact_size, bytes.len() as u64);
        assert_eq!(model.evaluate(&open_shogi_core::Position::startpos()), 60);
    }

    #[test]
    fn handcrafted_inference_emits_the_closed_comparison_schema() {
        let output = temporary_path("handcrafted.jsonl");
        run(&[
            "infer-handcrafted".into(),
            "--profile".into(),
            "handcrafted-baseline".into(),
            "--sfen".into(),
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -".into(),
            "--output".into(),
            output.to_string_lossy().into_owned(),
        ])
        .unwrap();

        let contents = std::fs::read_to_string(&output).unwrap();
        assert!(contents.ends_with('\n'));
        assert_eq!(contents.lines().count(), 1);
        let record: serde_json::Value = serde_json::from_str(contents.trim_end()).unwrap();
        let keys = record
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            keys,
            BTreeSet::from([
                "elapsedNs",
                "evaluatorProfile",
                "index",
                "schema",
                "scoreCp",
                "sfen",
            ])
        );
        assert_eq!(record["schema"], "phase5_handcrafted_inference/v1");
        assert_eq!(record["evaluatorProfile"], "handcrafted-baseline");
        assert_eq!(record["index"], 0);
        assert_eq!(record["scoreCp"], 10);
        assert_eq!(
            record["sfen"],
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        );
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn handcrafted_inference_rejects_aliases_and_duplicate_options() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -";
        assert!(
            run(&[
                "infer-handcrafted".into(),
                "--profile".into(),
                "handcrafted".into(),
                "--sfen".into(),
                sfen.into(),
            ])
            .is_err()
        );
        assert!(
            run(&[
                "infer-handcrafted".into(),
                "--profile".into(),
                "handcrafted-baseline".into(),
                "--profile".into(),
                "handcrafted-experimental".into(),
                "--sfen".into(),
                sfen.into(),
            ])
            .is_err()
        );

        let existing = temporary_path("existing-handcrafted.jsonl");
        std::fs::write(&existing, "sentinel").unwrap();
        assert!(
            run(&[
                "infer-handcrafted".into(),
                "--profile".into(),
                "handcrafted-baseline".into(),
                "--sfen".into(),
                sfen.into(),
                "--output".into(),
                existing.to_string_lossy().into_owned(),
            ])
            .is_err()
        );
        assert_eq!(std::fs::read_to_string(&existing).unwrap(), "sentinel");
        std::fs::remove_file(existing).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn inference_output_parent_swap_cannot_write_to_the_replacement_directory() {
        use std::os::unix::fs::symlink;

        let root = temporary_path("parent-swap");
        let external = temporary_path("parent-swap-external");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        let anchored = open_shogi_core::AnchoredDir::open_existing(&root).unwrap();
        let moved = root.with_extension("moved");
        std::fs::rename(&root, &moved).unwrap();
        symlink(&external, &root).unwrap();

        publish_output_to_anchored(
            &anchored,
            std::ffi::OsStr::new("inference.jsonl"),
            &root.join("inference.jsonl"),
            "{}\n",
        )
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(moved.join("inference.jsonl")).unwrap(),
            "{}\n"
        );
        assert!(!external.join("inference.jsonl").exists());
    }

    fn temporary_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-model-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn model_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OSAVAL01");
        for value in [1_u32, 1, 1, 1 << 2, 1, 1, 1, 0, 0, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&10.0_f32.to_le_bytes());
        for weight in [2.0_f32, 3.0] {
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&weight.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }
}
