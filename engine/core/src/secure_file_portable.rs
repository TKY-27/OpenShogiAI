//! Safe fallback for targets without Unix directory-descriptor APIs.
//!
//! The engine's byte-slice model parser remains portable. Filesystem loading fails closed on
//! targets where this crate does not yet have a no-follow handle identity implementation.

use std::{fs::File, io, path::Path};

/// A regular model file opened by the portable filesystem fallback.
#[derive(Debug)]
pub struct AnchoredFile {
    file: File,
}

impl AnchoredFile {
    /// Rejects filesystem loading on targets without strong handle identity support.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` before opening the supplied path. Callers may instead load already
    /// obtained bytes through the portable model parser.
    pub fn open_existing(_path: &Path) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "strong no-follow filesystem identity is unavailable on this target; load bytes explicitly",
        ))
    }

    /// Returns a mutable reader for the already-open descriptor.
    pub fn reader(&mut self) -> &mut File {
        &mut self.file
    }

    /// Returns a mutable writer for the already-open descriptor.
    pub fn writer(&mut self) -> &mut File {
        &mut self.file
    }

    /// Synchronizes the already-open file descriptor.
    ///
    /// # Errors
    ///
    /// Returns the operating-system synchronization error.
    pub fn sync_all(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    /// Rejects weak descriptor/path identity on this target.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported`; only the Unix implementation currently establishes this identity.
    pub fn stable_identity(&self) -> io::Result<StableFileIdentity> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "strong filesystem identity is unavailable on this target",
        ))
    }

    /// Verifies descriptor/path identity and the exact read count after a bounded read.
    ///
    /// # Errors
    ///
    /// Returns an error if the file changed or the read count differs from its length.
    pub fn verify_stable_read(
        &self,
        before: &StableFileIdentity,
        bytes_read: u64,
    ) -> io::Result<()> {
        let _ = (before, bytes_read);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "strong filesystem identity is unavailable on this target",
        ))
    }
}

/// Portable descriptor/path identity used around bounded reads.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StableFileIdentity {
    length: u64,
}

impl StableFileIdentity {
    /// Returns the descriptor length captured by this identity.
    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn filesystem_loading_fails_before_a_weak_identity_can_be_accepted() {
        let error = super::AnchoredFile::open_existing(std::path::Path::new("model.osnn"))
            .expect_err("portable filesystem loading must fail closed");
        assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    }
}
