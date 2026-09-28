//! Bounded direct-mapped transposition table with sharded internal locking.
//!
//! One logical table serves any number of search workers: each direct-mapped slot lives in
//! exactly one shard, so a probe or store locks a small slice instead of a global mutex.
//! Entries still retain a full [`Position`] plus the raw hash, which keeps hash collisions
//! harmless (a slot is only a hit when the stored position is the same state), and keeps
//! stored mate-distance scores validated by the caller's score encoding.

use std::{mem::size_of, sync::Mutex};

use crate::Position;

/// Number of independent lock domains. A power of two; the slot-to-shard mapping is a
/// contiguous range per shard so the direct-mapped index stays `hash % entry_count`.
const SHARD_COUNT: usize = 256;

/// Score namespace/bound of a stored search result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Debug)]
pub(crate) struct TranspositionEntry {
    pub(crate) hash: u64,
    position: Position,
    pub(crate) depth: u8,
    pub(crate) score: i32,
    pub(crate) bound: Bound,
    pub(crate) best_move: Option<crate::Move>,
}

/// Why a probe found nothing: the slot was vacant, or a different position held it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProbeMiss {
    Empty,
    Collision,
}

/// Copy of the fields a probe may use, taken while holding the shard lock. The stored
/// full position never leaves the table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TranspositionHit {
    pub(crate) depth: u8,
    pub(crate) score: i32,
    pub(crate) bound: Bound,
    pub(crate) best_move: Option<crate::Move>,
}

struct Shard {
    entries: Vec<Option<TranspositionEntry>>,
}

/// Reusable direct-mapped table shareable across concurrent search workers.
pub struct TranspositionTable {
    shards: Vec<Mutex<Shard>>,
    /// Contiguous slot count owned by every shard except possibly the last.
    per_shard: usize,
    /// Slot length of every shard, in shard order; captured at construction when no
    /// lock can be contended.
    shard_lengths: Vec<usize>,
}

impl TranspositionTable {
    /// Builds a table with at most `entry_count` direct-mapped slots.
    #[must_use]
    pub fn new(entry_count: usize) -> Self {
        let per_shard = entry_count.div_ceil(SHARD_COUNT).max(1);
        let mut shards = Vec::new();
        let mut shard_lengths = Vec::new();
        let mut remaining = entry_count;
        while remaining > 0 {
            let len = per_shard.min(remaining);
            shards.push(Mutex::new(Shard {
                entries: vec![None; len],
            }));
            shard_lengths.push(len);
            remaining -= len;
        }
        Self {
            shards,
            per_shard,
            shard_lengths,
        }
    }

    /// Returns the actual storage size of one direct-mapped table slot.
    #[must_use]
    pub const fn entry_size_bytes() -> usize {
        size_of::<Option<TranspositionEntry>>()
    }

    /// Converts a byte budget into the largest whole number of table entries that fits.
    #[must_use]
    pub const fn entries_for_bytes(bytes: usize) -> usize {
        bytes / Self::entry_size_bytes()
    }

    /// Converts a mebibyte budget into a bounded whole-entry count.
    #[must_use]
    pub const fn entries_for_megabytes(megabytes: usize) -> usize {
        let bytes = megabytes.saturating_mul(1024 * 1024);
        Self::entries_for_bytes(bytes)
    }

    fn total_entry_count(&self) -> usize {
        self.shard_lengths.iter().sum()
    }

    fn shard_slot(&self, hash: u64) -> (usize, usize) {
        let total = u64::try_from(self.total_entry_count())
            .unwrap_or(u64::MAX)
            .max(1);
        let slot = usize::try_from(hash % total).unwrap_or_default();
        let shard = (self.shards.len().saturating_sub(1)).min(slot / self.per_shard);
        (shard, slot - shard * self.per_shard)
    }

    /// Probes the slot for `position`. A hit requires both the raw hash and the stored
    /// full position to match; anything else reports why it missed.
    pub(crate) fn probe(
        &self,
        position: &Position,
        hash: u64,
    ) -> Result<TranspositionHit, ProbeMiss> {
        let (shard, slot) = self.shard_slot(hash);
        let Ok(shard) = self.shards[shard].lock() else {
            return Err(ProbeMiss::Empty);
        };
        let Some(entry) = shard.entries.get(slot).and_then(|slot| slot.as_ref()) else {
            return Err(ProbeMiss::Empty);
        };
        if entry.hash == hash && entry.position.same_state(position) {
            Ok(TranspositionHit {
                depth: entry.depth,
                score: entry.score,
                bound: entry.bound,
                best_move: entry.best_move,
            })
        } else {
            Err(ProbeMiss::Collision)
        }
    }

    /// Stores one entry under the always-prefer-deeper replacement rule.
    pub(crate) fn store(
        &self,
        position: &Position,
        hash: u64,
        depth: u8,
        score: i32,
        bound: Bound,
        best_move: Option<crate::Move>,
    ) {
        let (shard, slot) = self.shard_slot(hash);
        let Ok(mut shard) = self.shards[shard].lock() else {
            return;
        };
        let Some(slot) = shard.entries.get_mut(slot) else {
            return;
        };
        let replace = slot
            .as_ref()
            .is_none_or(|entry| entry.hash != hash || depth >= entry.depth);
        if replace {
            *slot = Some(TranspositionEntry {
                hash,
                position: position.clone(),
                depth,
                score,
                bound,
                best_move,
            });
        }
    }

    /// Invalidates every entry.
    pub fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut shard) = shard.lock() {
                shard.entries.fill(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_sfen, parse_usi_move};

    #[test]
    fn store_then_probe_round_trips_every_field() {
        let table = TranspositionTable::new(64);
        let position = crate::Position::startpos();
        let hash = position.zobrist_hash();
        table.store(
            &position,
            hash,
            5,
            123,
            Bound::Exact,
            Some(crate::Move::Drop {
                piece: crate::HandPiece::Pawn,
                to: crate::Square::new(5, 5).expect("square"),
            }),
        );
        let hit = table.probe(&position, hash).expect("hit");
        assert_eq!(hit.depth, 5);
        assert_eq!(hit.score, 123);
        assert_eq!(hit.bound, Bound::Exact);
        assert!(hit.best_move.is_some());
    }

    #[test]
    fn different_position_on_same_slot_is_a_collision_not_a_hit() {
        let table = TranspositionTable::new(1);
        let first = crate::Position::startpos();
        let mut second = first.clone();
        second
            .make_move(parse_usi_move("7g7f").expect("legal move"))
            .expect("play");
        table.store(&first, first.zobrist_hash(), 2, 7, Bound::Lower, None);
        // Both positions map to the single slot; the full-position check separates them.
        assert!(table.probe(&first, first.zobrist_hash()).is_ok());
        assert!(matches!(
            table.probe(&second, second.zobrist_hash()),
            Err(ProbeMiss::Collision)
        ));
    }

    #[test]
    fn deeper_entries_replace_shallower_ones_but_not_the_reverse() {
        let table = TranspositionTable::new(4);
        let position = crate::Position::startpos();
        let hash = position.zobrist_hash();
        table.store(&position, hash, 3, 10, Bound::Exact, None);
        table.store(&position, hash, 2, 20, Bound::Upper, None);
        assert_eq!(table.probe(&position, hash).expect("kept").score, 10);
        table.store(&position, hash, 4, 30, Bound::Exact, None);
        assert_eq!(table.probe(&position, hash).expect("kept").score, 30);
    }

    #[test]
    fn clear_invalidates_every_slot() {
        let table = TranspositionTable::new(512);
        let position = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("position");
        table.store(&position, position.zobrist_hash(), 1, 1, Bound::Exact, None);
        table.clear();
        assert!(matches!(
            table.probe(&position, position.zobrist_hash()),
            Err(ProbeMiss::Empty)
        ));
    }

    #[test]
    fn tiny_tables_still_map_every_slot() {
        for entries in [1_usize, 2, 3, 100, 257] {
            let table = TranspositionTable::new(entries);
            let position = crate::Position::startpos();
            table.store(&position, position.zobrist_hash(), 1, 9, Bound::Exact, None);
            assert!(table.probe(&position, position.zobrist_hash()).is_ok());
        }
    }

    #[test]
    fn entry_budget_accounting_matches_slot_size() {
        assert_eq!(
            TranspositionTable::entries_for_megabytes(1),
            TranspositionTable::entries_for_bytes(1024 * 1024)
        );
        assert!(TranspositionTable::entry_size_bytes() >= size_of::<Position>());
        let bytes = 1024;
        let entries = TranspositionTable::entries_for_bytes(bytes);
        assert!(entries.saturating_mul(TranspositionTable::entry_size_bytes()) <= bytes);
    }
}
