//! Explicit format-selected pure-playing boundary. A loader never retries another format.
use crate::{
    Osaval02Evaluator, Osaval02History, Phase10TEvaluator, Position, SearchConfig, SearchEngine,
};
use std::{path::Path, sync::Arc};

/// Immutable validated model selected before play.
#[derive(Clone)]
pub enum PurePlayingEvaluator {
    Osaval02(Arc<Osaval02Evaluator>),
    Osat10a1(Arc<Phase10TEvaluator>),
    Osaval03(Arc<crate::Phase10VEvaluator>),
}
impl PurePlayingEvaluator {
    /// Load exactly the named format and bind the complete artifact hash.
    /// # Errors
    /// Rejects unknown formats, malformed models, schemas and mismatched hashes.
    pub fn from_bytes(format: &str, bytes: &[u8], hash: &str) -> Result<Self, String> {
        let model = match format {
            "OSAVAL02" => Self::Osaval02(Arc::new(
                Osaval02Evaluator::from_bytes(bytes).map_err(|e| e.to_string())?,
            )),
            "OSAT10A1" => Self::Osat10a1(Arc::new(
                Phase10TEvaluator::from_bytes(bytes).map_err(|e| e.to_string())?,
            )),
            "OSAVAL03" => Self::Osaval03(Arc::new(
                crate::Phase10VEvaluator::from_bytes(bytes).map_err(|e| e.to_string())?,
            )),
            _ => return Err("model format must be OSAVAL02, OSAT10A1 or OSAVAL03".into()),
        };
        model.validate_hash(hash)?;
        Ok(model)
    }
    /// Load a bounded file through the selected format's secure loader.
    /// # Errors
    /// Rejects invalid files, formats, models or hashes without fallback.
    pub fn load_file(format: &str, path: impl AsRef<Path>, hash: &str) -> Result<Self, String> {
        let model = match format {
            "OSAVAL02" => Self::Osaval02(Arc::new(
                Osaval02Evaluator::load_file(path).map_err(|e| e.to_string())?,
            )),
            "OSAT10A1" => Self::Osat10a1(Arc::new(
                Phase10TEvaluator::load_file(path).map_err(|e| e.to_string())?,
            )),
            "OSAVAL03" => Self::Osaval03(Arc::new(
                crate::Phase10VEvaluator::load_file(path).map_err(|e| e.to_string())?,
            )),
            _ => return Err("model format must be OSAVAL02, OSAT10A1 or OSAVAL03".into()),
        };
        model.validate_hash(hash)?;
        Ok(model)
    }
    fn validate_hash(&self, hash: &str) -> Result<(), String> {
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || hash != self.artifact_sha256()
        {
            return Err("pure playing model SHA-256 mismatch or invalid expected identity".into());
        }
        Ok(())
    }
    #[must_use]
    pub const fn format(&self) -> &'static str {
        match self {
            Self::Osaval02(_) => "OSAVAL02",
            Self::Osat10a1(_) => "OSAT10A1",
            Self::Osaval03(_) => "OSAVAL03",
        }
    }
    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        match self {
            Self::Osaval02(m) => &m.identity().artifact_sha256,
            Self::Osat10a1(m) => &m.identity().artifact_sha256,
            Self::Osaval03(m) => &m.identity().artifact_sha256,
        }
    }
    /// Create a pure search instance with this immutable model.
    /// # Errors
    /// Rejects invalid expected model identity.
    pub fn search_engine(&self, config: SearchConfig, hash: &str) -> Result<SearchEngine, String> {
        match self {
            Self::Osaval02(m) => SearchEngine::with_pure_learned(config, Arc::clone(m), hash),
            Self::Osat10a1(m) => SearchEngine::with_phase10t(config, Arc::clone(m), hash),
            Self::Osaval03(m) => SearchEngine::with_phase10v(config, Arc::clone(m), hash),
        }
    }
    /// Return format-specific, lossless inference evidence with a common cp field.
    /// # Errors
    /// Rejects invalid history or inference failure.
    pub fn infer(
        &self,
        position: &Position,
        history: Osaval02History,
    ) -> Result<serde_json::Value, String> {
        match self {
            Self::Osaval02(m) => {
                let inference = m.infer(position, history).map_err(|e| e.to_string())?;
                Ok(serde_json::json!({"cp":inference.calibrated_score_cp(),"osaval02":inference}))
            }
            Self::Osaval03(m) => {
                let inference = m.infer(position);
                Ok(serde_json::json!({"cp":inference.cp,"wdl_logits":inference.wdl_logits}))
            }
            Self::Osat10a1(m) => {
                let state = m
                    .accumulator(position, history)
                    .map_err(|e| e.to_string())?;
                let inference = m.infer_accumulator(&state);
                Ok(serde_json::json!({"cp":inference.cp,"wdl_logits":inference.wdl_logits}))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PurePlayingEvaluator;
    #[test]
    fn explicit_format_never_retries_another_loader() {
        let hash = "a".repeat(64);
        let mut a1 = vec![0; 4064];
        a1[..8].copy_from_slice(b"OSAT10A1");
        {
            use sha2::{Digest, Sha256};
            let checksum = Sha256::digest(&a1);
            a1.extend_from_slice(&checksum);
        }
        let mut c0 = vec![0; 256];
        c0[..8].copy_from_slice(b"OSAVAL02");
        assert!(
            PurePlayingEvaluator::from_bytes("OSAVAL02", &a1, &hash)
                .err()
                .unwrap()
                .contains("magic")
        );
        assert!(
            PurePlayingEvaluator::from_bytes("OSAT10A1", &c0, &hash)
                .err()
                .unwrap()
                .contains("magic")
        );
        assert!(PurePlayingEvaluator::from_bytes("auto", &a1, &hash).is_err());
    }
}
