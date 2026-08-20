//! Resource budgets coordinating separate play and analysis engine instances.

/// Versioned resource-budget schema shared by host adapters.
pub const RESOURCE_BUDGET_SCHEMA: &str = "open_shogi_resource_budget/v1";

/// Closed host budget. Search is currently single-threaded, so each logical instance supports
/// one worker thread; a host may set analysis threads to zero to disable analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    pub play_threads: u8,
    pub analysis_threads: u8,
    pub play_hash_megabytes: usize,
    pub analysis_hash_megabytes: usize,
    pub analysis_pause_during_ai_turn: bool,
    pub maximum_aggregate_memory_megabytes: usize,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            play_threads: 1,
            analysis_threads: 1,
            play_hash_megabytes: 32,
            analysis_hash_megabytes: 32,
            analysis_pause_during_ai_turn: true,
            maximum_aggregate_memory_megabytes: 256,
        }
    }
}

impl ResourceBudget {
    /// Validates thread, hash, and aggregate-memory constraints.
    ///
    /// # Errors
    ///
    /// Returns an error when an enabled instance exceeds the closed resource bounds.
    pub fn validate(self) -> Result<(), String> {
        if self.play_threads != 1 {
            return Err("playThreads must be 1 for the current single-threaded search".to_owned());
        }
        if self.analysis_threads > 1 {
            return Err(
                "analysisThreads must be 0 or 1 for the current single-threaded search".to_owned(),
            );
        }
        if self.play_hash_megabytes == 0
            || (self.analysis_threads > 0 && self.analysis_hash_megabytes == 0)
        {
            return Err("enabled engine instances require a positive hash allocation".to_owned());
        }
        let hash_total = self
            .play_hash_megabytes
            .saturating_add(self.analysis_hash_megabytes);
        if self.maximum_aggregate_memory_megabytes == 0
            || hash_total > self.maximum_aggregate_memory_megabytes
        {
            return Err("hash allocations exceed maximum aggregate memory".to_owned());
        }
        Ok(())
    }
}

/// Lightweight host coordinator; play and analysis engines remain separate objects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceCoordinator {
    budget: ResourceBudget,
    play_active: bool,
}

impl ResourceCoordinator {
    /// Creates a coordinator from a validated resource budget.
    ///
    /// # Errors
    ///
    /// Returns an error when the resource budget is invalid.
    pub fn new(budget: ResourceBudget) -> Result<Self, String> {
        budget.validate()?;
        Ok(Self {
            budget,
            play_active: false,
        })
    }

    pub fn begin_play(&mut self) {
        self.play_active = true;
    }

    pub fn finish_play(&mut self) {
        self.play_active = false;
    }

    #[must_use]
    pub const fn analysis_permitted(&self) -> bool {
        self.budget.analysis_threads > 0
            && !(self.play_active && self.budget.analysis_pause_during_ai_turn)
    }

    #[must_use]
    pub const fn budget(&self) -> ResourceBudget {
        self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_can_pause_analysis_without_sharing_an_engine() {
        let mut coordinator = ResourceCoordinator::new(ResourceBudget::default()).unwrap();
        assert!(coordinator.analysis_permitted());
        coordinator.begin_play();
        assert!(!coordinator.analysis_permitted());
        coordinator.finish_play();
        assert!(coordinator.analysis_permitted());
    }

    #[test]
    fn aggregate_memory_and_thread_counts_fail_closed() {
        assert!(
            ResourceCoordinator::new(ResourceBudget {
                play_threads: 2,
                ..ResourceBudget::default()
            })
            .is_err()
        );
        assert!(
            ResourceCoordinator::new(ResourceBudget {
                maximum_aggregate_memory_megabytes: 63,
                ..ResourceBudget::default()
            })
            .is_err()
        );
    }
}
