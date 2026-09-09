//! Strict browser runtime sharing the format-selected pure loader and search.
use open_shogi_core::{
    CancellationToken, PurePlayingEvaluator, SearchConfig, SearchEngine, SearchLimits, parse_sfen,
    to_usi_move,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// A validated, explicitly hash-bound pure learned engine. No default constructor exists.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct PureEngine {
    model: PurePlayingEvaluator,
    engine: SearchEngine,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl PureEngine {
    /// Load the exact a1 artifact before allowing evaluation or search.
    ///
    /// # Errors
    /// Rejects any other profile, missing/corrupt/schema-incompatible model or wrong hash.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(bytes: &[u8], expected_sha256: &str, profile: &str) -> Result<PureEngine, String> {
        Self::new_with_format(bytes, expected_sha256, profile, "OSAT10A1")
    }

    /// Load only the explicitly selected format, never falling back to another loader.
    /// # Errors
    /// Rejects unknown formats, profiles, malformed models and hash mismatches.
    pub fn new_with_format(
        bytes: &[u8],
        expected_sha256: &str,
        profile: &str,
        format: &str,
    ) -> Result<PureEngine, String> {
        if profile != "pure_learned" {
            return Err("profile must be pure_learned".into());
        }
        let model = PurePlayingEvaluator::from_bytes(format, bytes, expected_sha256)?;
        let engine = model.search_engine(SearchConfig::default(), expected_sha256)?;
        Ok(Self { model, engine })
    }

    /// Return learned cp and WDL logits for parity verification.
    ///
    /// # Errors
    /// Rejects an invalid SFEN.
    pub fn evaluate(&self, sfen: &str) -> Result<String, String> {
        self.evaluate_history(sfen, "{}")
    }

    /// Evaluate explicit standalone history facts for cross-runtime semantic verification.
    /// This does not change the authoritative move history used by search.
    ///
    /// # Errors
    /// Rejects invalid SFEN or malformed, oversized or inconsistent history facts.
    pub fn evaluate_history(&self, sfen: &str, history_json: &str) -> Result<String, String> {
        let position = parse_sfen(sfen).map_err(|error| error.to_string())?;
        let history = parse_history(history_json)?;
        let inference = self.model.infer(&position, history)?;
        Ok(serde_json::json!({"schema":"open_shogiai_phase10t_pure_runtime/v1", "compiled_evaluators":open_shogi_core::COMPILED_EVALUATORS, "profile":"pure_learned", "model_sha256":self.model.artifact_sha256(),"history":history,"model_format":self.model.format(),"cp":inference["cp"],"wdl_logits":inference.get("wdl_logits"),"inference":inference}).to_string())
    }

    /// Run a bounded legal search and expose the engine-owned runtime proof.
    ///
    /// # Errors
    /// Rejects invalid SFEN, zero nodes, or depth outside 1..64.
    pub fn search(&mut self, sfen: &str, depth: u8, nodes: u32) -> Result<String, String> {
        self.reset_engine()?;
        let position = parse_sfen(sfen).map_err(|error| error.to_string())?;
        self.search_position(&position, depth, nodes)
    }

    /// Replay exact legal game history before searching the resulting position.
    ///
    /// # Errors
    /// Rejects invalid SFEN, malformed/beyond-500 move history, illegal moves or search limits.
    pub fn search_history(
        &mut self,
        initial_sfen: &str,
        moves_json: &str,
        depth: u8,
        nodes: u32,
    ) -> Result<String, String> {
        if moves_json.len() > 8192 {
            return Err("move history exceeds byte limit".to_owned());
        }
        let tokens: Vec<String> =
            serde_json::from_str(moves_json).map_err(|error| error.to_string())?;
        if tokens.len() > 500 {
            return Err("move history exceeds 500 moves".to_owned());
        }
        let initial = parse_sfen(initial_sfen).map_err(|error| error.to_string())?;
        let mut position = initial.clone();
        let mut moves = Vec::new();
        for token in tokens {
            let movement =
                open_shogi_core::parse_usi_move(&token).map_err(|error| error.to_string())?;
            position
                .make_move(movement)
                .map_err(|error| error.to_string())?;
            moves.push(movement);
        }
        self.reset_engine()?;
        self.engine.set_pure_history(&initial, &moves)?;
        self.search_position(&position, depth, nodes)
    }
}

impl PureEngine {
    fn reset_engine(&mut self) -> Result<(), String> {
        self.engine = self
            .model
            .search_engine(SearchConfig::default(), self.model.artifact_sha256())?;
        Ok(())
    }

    fn search_position(
        &mut self,
        position: &open_shogi_core::Position,
        depth: u8,
        nodes: u32,
    ) -> Result<String, String> {
        if !(1..=64).contains(&depth) || nodes == 0 {
            return Err("depth must be 1..64 and nodes positive".to_owned());
        }
        let result = self.engine.search(
            position,
            SearchLimits {
                max_depth: depth,
                max_nodes: Some(u64::from(nodes)),
                movetime: None,
            },
            &CancellationToken::new(),
        );
        if result.termination == open_shogi_core::SearchTermination::EvaluationError {
            return Err("pure-only inference failed; no result is available".to_owned());
        }
        let proof = self
            .engine
            .runtime_proof(result.stats, self.model.artifact_sha256().to_owned());
        Ok(serde_json::json!({"schema":"open_shogiai_phase10t_pure_runtime/v1", "compiled_evaluators":open_shogi_core::COMPILED_EVALUATORS, "best_move":result.best_move.map(to_usi_move), "score":result.score,"depth":result.depth,"nodes":result.nodes,"proof":proof}).to_string())
    }
}

fn parse_history(json: &str) -> Result<open_shogi_core::Osaval02History, String> {
    if json.len() > 1024 {
        return Err("history JSON exceeds byte limit".to_owned());
    }
    serde_json::from_str(json).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{PureEngine, parse_history};
    #[test]
    fn history_json_rejects_unknown_fields_types_and_oversize() {
        assert!(parse_history(r#"{"available":true,"repetition_count":2}"#).is_err());
        assert!(parse_history(r#"{"repetitionCount":256}"#).is_err());
        assert!(parse_history(&" ".repeat(1025)).is_err());
        assert!(
            parse_history(r#"{"available":true,"repetitionCount":2,"continuousCheckByThem":true}"#)
                .is_ok()
        );
    }

    #[test]
    fn missing_model_and_wrong_profile_fail_closed() {
        assert!(PureEngine::new(&[], &"a".repeat(64), "pure_learned").is_err());
        assert!(PureEngine::new(&[], &"a".repeat(64), "standard").is_err());
    }
}
