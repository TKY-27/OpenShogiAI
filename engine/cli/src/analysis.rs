//! Minimal line-oriented reference client for the versioned analysis protocol.

use std::io::{self, BufRead, Write};

use open_shogi_core::{
    ANALYSIS_SCHEMA, AnalysisCacheKey, AnalysisService, AnalysisUpdate, AnalysisUpdateSource,
    Position, SearchConfig, SearchEngine, Side, TimeControl, TimeManager, parse_sfen, to_usi_move,
};
use serde::{Deserialize, Serialize};

use crate::args::{BoundedInputLine, read_bounded_line};

const MAX_COMMAND_BYTES: usize = 16 * 1024;
const MAX_STEP_NODES: u64 = 10_000_000;

#[derive(Clone, Debug)]
struct CurrentRoot {
    position: Position,
    model_hash: String,
    evaluator_config_hash: String,
    feature_schema_hash: String,
    evaluation_semantics_hash: String,
    search_options_hash: String,
    opening_profile_hash: String,
    multi_pv: u8,
}

impl CurrentRoot {
    fn key(&self) -> Result<AnalysisCacheKey, String> {
        AnalysisCacheKey::new(
            &self.position,
            self.model_hash.clone(),
            self.evaluator_config_hash.clone(),
            self.feature_schema_hash.clone(),
            self.evaluation_semantics_hash.clone(),
            self.search_options_hash.clone(),
            self.opening_profile_hash.clone(),
            self.multi_pv,
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "command",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Command {
    Start {
        schema: String,
        position_sfen: String,
        model_hash: String,
        evaluator_config_hash: String,
        feature_schema_hash: String,
        evaluation_semantics_hash: String,
        search_options_hash: String,
        opening_profile_hash: String,
        multi_pv: u8,
    },
    ChangePosition {
        schema: String,
        position_sfen: String,
    },
    ChangeMultiPv {
        schema: String,
        multi_pv: u8,
    },
    Step {
        schema: String,
        nodes: u64,
        max_depth: u8,
        timestamp_ms: u64,
    },
    Stop { schema: String },
    WorkerFailed { schema: String },
    Restart { schema: String },
    Quit { schema: String },
}

impl Command {
    fn schema(&self) -> &str {
        match self {
            Self::Start { schema, .. }
            | Self::ChangePosition { schema, .. }
            | Self::ChangeMultiPv { schema, .. }
            | Self::Step { schema, .. }
            | Self::Stop { schema }
            | Self::WorkerFailed { schema }
            | Self::Restart { schema }
            | Self::Quit { schema } => schema,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Response<'a> {
    schema: &'static str,
    event: &'a str,
    updates: Vec<UpdateRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct UpdateRecord {
    source: &'static str,
    canonical_position: String,
    position_hash: String,
    model_hash: String,
    evaluator_config_hash: String,
    feature_schema_hash: String,
    evaluation_semantics_hash: String,
    search_options_hash: String,
    opening_profile_hash: String,
    multi_pv: u8,
    depth: u8,
    nodes: u64,
    nps: u64,
    score: i32,
    mate_score: Option<i32>,
    lines: Vec<LineRecord>,
    root_move_statistics: Vec<RootMoveRecord>,
    timestamp_ms: u64,
    engine_version: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LineRecord {
    rank: u8,
    score: i32,
    mate_score: Option<i32>,
    depth: u8,
    nodes: u64,
    pv: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RootMoveRecord {
    movement: String,
    score: i32,
    depth: u8,
    nodes: u64,
    pv: Vec<String>,
}

impl From<AnalysisUpdate> for UpdateRecord {
    fn from(update: AnalysisUpdate) -> Self {
        Self {
            source: match update.source {
                AnalysisUpdateSource::Cache => "cache",
                AnalysisUpdateSource::Search => "search",
            },
            canonical_position: update.key.canonical_position,
            position_hash: format!("{:016x}", update.key.position_hash),
            model_hash: update.key.model_hash,
            evaluator_config_hash: update.key.evaluator_config_hash,
            feature_schema_hash: update.key.feature_schema_hash,
            evaluation_semantics_hash: update.key.evaluation_semantics_hash,
            search_options_hash: update.key.search_options_hash,
            opening_profile_hash: update.key.opening_profile_hash,
            multi_pv: update.key.multi_pv,
            depth: update.entry.completed_depth,
            nodes: update.entry.nodes,
            nps: update.entry.nps,
            score: update.entry.score,
            mate_score: update.entry.mate_score,
            lines: update
                .entry
                .lines
                .into_iter()
                .map(|line| LineRecord {
                    rank: line.rank,
                    score: line.score,
                    mate_score: line.mate_score,
                    depth: line.depth,
                    nodes: line.nodes,
                    pv: line.pv.into_iter().map(to_usi_move).collect(),
                })
                .collect(),
            root_move_statistics: update
                .entry
                .root_move_statistics
                .into_iter()
                .map(|root| RootMoveRecord {
                    movement: to_usi_move(root.movement),
                    score: root.score,
                    depth: root.depth,
                    nodes: root.nodes,
                    pv: root.pv.into_iter().map(to_usi_move).collect(),
                })
                .collect(),
            timestamp_ms: update.entry.updated_at_ms,
            engine_version: update.entry.engine_version,
        }
    }
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("analysis accepts newline-delimited protocol commands on standard input".to_owned());
    }
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    run_protocol(&mut reader, &mut writer)
}

#[expect(
    clippy::too_many_lines,
    reason = "the small reference protocol keeps its closed command state machine together"
)]
fn run_protocol<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> Result<(), String> {
    let mut service = AnalysisService::new(
        SearchEngine::new(SearchConfig::default()),
        open_shogi_core::DEFAULT_ANALYSIS_CACHE_ENTRIES,
    )?;
    let mut current: Option<CurrentRoot> = None;
    loop {
        let line = match read_bounded_line(reader, MAX_COMMAND_BYTES)
            .map_err(|error| error.to_string())?
        {
            BoundedInputLine::Line(line) => line,
            BoundedInputLine::TooLong => {
                write_response(writer, "error", None, Some("command exceeds 16384 bytes".to_owned()))?;
                continue;
            }
            BoundedInputLine::Eof => break,
        };
        let command: Command = match serde_json::from_str(&line) {
            Ok(command) => command,
            Err(error) => {
                write_response(writer, "error", None, Some(error.to_string()))?;
                continue;
            }
        };
        if command.schema() != ANALYSIS_SCHEMA {
            write_response(
                writer,
                "error",
                None,
                Some(format!("schema must be {ANALYSIS_SCHEMA}")),
            )?;
            continue;
        }
        match command {
            Command::Start {
                position_sfen,
                model_hash,
                evaluator_config_hash,
                feature_schema_hash,
                evaluation_semantics_hash,
                search_options_hash,
                opening_profile_hash,
                multi_pv,
                ..
            } => {
                let root = CurrentRoot {
                    position: parse_sfen(&position_sfen)
                        .map_err(|error| format!("invalid position: {error}"))?,
                    model_hash,
                    evaluator_config_hash,
                    feature_schema_hash,
                    evaluation_semantics_hash,
                    search_options_hash,
                    opening_profile_hash,
                    multi_pv,
                };
                let cached = service.start(root.position.clone(), root.key()?)?;
                current = Some(root);
                write_response(writer, "started", cached.map(Into::into), None)?;
            }
            Command::ChangePosition { position_sfen, .. } => {
                let root = current
                    .as_mut()
                    .ok_or_else(|| "analysis has not been started".to_owned())?;
                root.position = parse_sfen(&position_sfen)
                    .map_err(|error| format!("invalid position: {error}"))?;
                let cached = service.start(root.position.clone(), root.key()?)?;
                write_response(writer, "position-changed", cached.map(Into::into), None)?;
            }
            Command::ChangeMultiPv { multi_pv, .. } => {
                let root = current
                    .as_mut()
                    .ok_or_else(|| "analysis has not been started".to_owned())?;
                root.multi_pv = multi_pv;
                let cached = service.start(root.position.clone(), root.key()?)?;
                write_response(writer, "multipv-changed", cached.map(Into::into), None)?;
            }
            Command::Step {
                nodes,
                max_depth,
                timestamp_ms,
                ..
            } => {
                if nodes == 0 || nodes > MAX_STEP_NODES {
                    return Err(format!("analysis step nodes must be 1..={MAX_STEP_NODES}"));
                }
                let side = current
                    .as_ref()
                    .ok_or_else(|| "analysis has not been started".to_owned())?
                    .position
                    .side_to_move();
                let plan = analysis_plan(side, nodes, max_depth)?;
                let step = service.step(plan, timestamp_ms)?;
                for update in step.updates {
                    write_response(writer, "update", Some(update.into()), None)?;
                }
                write_response(writer, "step-complete", None, None)?;
            }
            Command::Stop { .. } => {
                service.stop();
                write_response(writer, "stopped", None, None)?;
            }
            Command::WorkerFailed { .. } => {
                let cached = service.record_worker_failure();
                write_response(writer, "worker-failed", cached.map(Into::into), None)?;
            }
            Command::Restart { .. } => {
                let cached = service.restart_after_worker_failure()?;
                write_response(writer, "restarted", cached.map(Into::into), None)?;
            }
            Command::Quit { .. } => break,
        }
    }
    Ok(())
}

fn analysis_plan(side: Side, nodes: u64, max_depth: u8) -> Result<open_shogi_core::TimePlan, String> {
    TimeManager::default().plan(
        side,
        TimeControl {
            nodes: Some(nodes),
            depth: Some(max_depth),
            casual: false,
            ..TimeControl::casual()
        },
        max_depth,
    )
}

fn write_response(
    writer: &mut impl Write,
    event: &str,
    update: Option<UpdateRecord>,
    message: Option<String>,
) -> Result<(), String> {
    serde_json::to_writer(
        &mut *writer,
        &Response {
            schema: ANALYSIS_SCHEMA,
            event,
            updates: update.into_iter().collect(),
            message,
        },
    )
    .map_err(|error| error.to_string())?;
    writeln!(writer).map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn reference_protocol_starts_updates_switches_and_stops() {
        let hash = |digit: char| digit.to_string().repeat(64);
        let start = serde_json::json!({
            "command": "start",
            "schema": ANALYSIS_SCHEMA,
            "positionSfen": open_shogi_core::to_sfen(&Position::startpos()),
            "modelHash": hash('1'),
            "evaluatorConfigHash": hash('2'),
            "featureSchemaHash": hash('3'),
            "evaluationSemanticsHash": hash('4'),
            "searchOptionsHash": hash('5'),
            "openingProfileHash": hash('6'),
            "multiPv": 2,
        });
        let commands = format!(
            "{}\n{}\n{}\n{}\n",
            start,
            serde_json::json!({
                "command": "step", "schema": ANALYSIS_SCHEMA,
                "nodes": 1000, "maxDepth": 2, "timestampMs": 7
            }),
            serde_json::json!({
                "command": "change-multi-pv", "schema": ANALYSIS_SCHEMA, "multiPv": 1
            }),
            serde_json::json!({"command": "stop", "schema": ANALYSIS_SCHEMA}),
        );
        let mut output = Vec::new();
        run_protocol(&mut Cursor::new(commands), &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"event\":\"started\""));
        assert!(text.contains("\"event\":\"update\""));
        assert!(text.contains("\"event\":\"multipv-changed\""));
        assert!(text.contains("\"event\":\"stopped\""));
    }
}
