//! Small Unix directory-FD boundary for race-resistant local artifact I/O.
//!
//! # Platform and `unsafe` boundary
//!
//! This is the core crate's only `unsafe`-enabled module. Its `libc` calls are a narrow Unix
//! platform boundary for operations that Rust's standard library cannot express with retained
//! directory descriptors. The corresponding non-Unix module rejects filesystem loading with
//! `Unsupported`; it never substitutes weaker pathname checks.
//!
//! The safety and integrity invariants for this boundary are:
//!
//! - every relative component is a validated single name and is resolved under a retained
//!   directory FD with `O_NOFOLLOW`; no syscall receives an unvalidated interior NUL;
//! - each successful raw descriptor is checked and transferred exactly once into `OwnedFd` or
//!   `File`, while borrowed descriptors remain owned and live for the entire syscall;
//! - `MaybeUninit` stat buffers are read only after the initializing syscall succeeds, and DIR
//!   entries are copied before the next `readdir` call or stream close;
//! - directory enumeration, collision retries, quarantine growth, reads, and publication
//!   verification are bounded so an untrusted directory cannot force unbounded work;
//! - callers retain opened files and directories across a transaction. Publication completes
//!   only after final path-to-FD identity plus exact content verification and directory sync;
//! - cleanup removes only an entry whose retained identity proves ownership. A foreign
//!   replacement is preserved and the operation fails closed.
//!
//! Each `unsafe` block below documents the local pointer, lifetime, initialization, and
//! ownership preconditions that discharge these module-level invariants.

use std::{
    ffi::{CStr, CString, OsStr, OsString},
    fs::{File, Metadata},
    io::{self, Read, Seek, Write},
    mem::MaybeUninit,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::ffi::{OsStrExt, OsStringExt},
    },
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);
// Arena recovery deliberately preserves a bounded number of unproven crash remnants. Keep the
// publication retry budget larger than that preservation cap so PID reuse cannot make an
// otherwise valid resume collide with every candidate name.
const TEMPORARY_CREATE_ATTEMPTS: usize = 512;
const CLEANUP_DIRECTORY_NAME: &str = ".open-shogi-cleanup";
const CLEANUP_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
/// Maximum crash remnants retained in one private identity-atomic cleanup quarantine.
///
/// Entries cannot be automatically attributed after a process crash, so they are never
/// consumed or deleted by a later process. New cleanup operations fail closed once this bound
/// is reached and identify the directory for manual inspection.
pub const MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntryKind {
    File,
    Directory,
    Symlink,
    Other,
}

#[derive(Debug)]
pub struct DirectoryEntry {
    pub name: OsString,
    pub kind: EntryKind,
}

#[derive(Debug)]
pub struct AnchoredDir {
    fd: OwnedFd,
    display: PathBuf,
}

impl AnchoredDir {
    /// Opens every existing directory component without following symlinks.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, missing, symlinked, or non-directory components.
    pub fn open_existing(path: &Path) -> io::Result<Self> {
        Self::open_path(path, false)
    }

    /// Opens a directory path, creating missing components under retained descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error when a component is invalid, cannot be created, or is a symlink.
    pub fn open_or_create_all(path: &Path) -> io::Result<Self> {
        Self::open_path(path, true)
    }

    /// Opens a file path's parent and returns its validated final component.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing file name or an unsafe/unavailable parent path.
    pub fn open_parent(path: &Path, create: bool) -> io::Result<(Self, OsString)> {
        let name = path
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?
            .to_os_string();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let directory = if create {
            Self::open_or_create_all(parent)?
        } else {
            Self::open_existing(parent)?
        };
        validate_name(&name)?;
        Ok((directory, name))
    }

    fn open_path(path: &Path, create: bool) -> io::Result<Self> {
        Self::open_path_with_hook(path, create, |_| Ok(()))
    }

    fn open_path_with_hook(
        path: &Path,
        create: bool,
        mut after_component: impl FnMut(&OsStr) -> io::Result<()>,
    ) -> io::Result<Self> {
        let (start, mut display) = if path.is_absolute() {
            (Path::new("/"), PathBuf::from("/"))
        } else {
            (Path::new("."), std::env::current_dir()?)
        };
        // Keeping every ancestor descriptor is essential: resolving `..` by opening it from
        // the child would allow a concurrent directory reparent to redirect the traversal.
        let mut descriptors = vec![open_directory_path(start)?];
        for component in path.components() {
            let name = match component {
                Component::RootDir | Component::CurDir => continue,
                Component::ParentDir => {
                    if descriptors.len() == 1 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "path attempts to escape its opened starting directory",
                        ));
                    }
                    descriptors.pop();
                    display.pop();
                    continue;
                }
                Component::Normal(name) => name,
                Component::Prefix(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "platform path prefix is unsupported",
                    ));
                }
            };
            validate_name(name)?;
            let parent = descriptors
                .last()
                .expect("the starting descriptor is always retained");
            let next = match open_directory_at(parent.as_raw_fd(), name) {
                Ok(next) => next,
                Err(error) if create && error.kind() == io::ErrorKind::NotFound => {
                    mkdir_at(parent.as_raw_fd(), name)?;
                    fsync_fd(parent.as_raw_fd())?;
                    open_directory_at(parent.as_raw_fd(), name)?
                }
                Err(error) => return Err(error),
            };
            descriptors.push(next);
            display.push(name);
            after_component(name)?;
        }
        let fd = descriptors
            .pop()
            .expect("the starting descriptor is always retained");
        Ok(Self { fd, display })
    }

    /// Returns the diagnostic path associated with this retained descriptor.
    #[must_use]
    pub fn display(&self) -> &Path {
        &self.display
    }

    /// Captures the stable device/inode identity of this retained directory.
    ///
    /// # Errors
    ///
    /// Returns an error when descriptor metadata cannot be obtained.
    pub fn stable_identity(&self) -> io::Result<StableDirectoryIdentity> {
        StableDirectoryIdentity::from_fd(self.fd.as_raw_fd())
    }

    /// Verifies that the diagnostic pathname still resolves to this retained directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the path was rebound, symlinked, or became unavailable.
    pub fn verify_display_identity(&self) -> io::Result<()> {
        let reopened = Self::open_existing(&self.display)?;
        if self.stable_identity()? != reopened.stable_identity()? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "retained directory pathname was rebound",
            ));
        }
        Ok(())
    }

    /// Acquires a non-blocking exclusive advisory lock on the retained directory inode.
    ///
    /// The authority cannot be bypassed by unlinking and recreating a child lock file.
    ///
    /// # Errors
    ///
    /// Returns `WouldBlock` when another open description already holds the lock.
    pub fn try_lock_exclusive(&self) -> io::Result<()> {
        // SAFETY: the descriptor is a live directory descriptor retained by `self`.
        if unsafe { libc::flock(self.fd.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Opens one child directory relative to this retained descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is invalid or is not a non-symlink directory.
    pub fn open_dir(&self, name: &OsStr) -> io::Result<Self> {
        validate_name(name)?;
        Ok(Self {
            fd: open_directory_at(self.fd.as_raw_fd(), name)?,
            display: self.display.join(name),
        })
    }

    /// Creates and durably opens one child directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the name exists, is invalid, or cannot be synchronized.
    pub fn create_dir(&self, name: &OsStr) -> io::Result<Self> {
        validate_name(name)?;
        mkdir_at(self.fd.as_raw_fd(), name)?;
        self.sync()?;
        self.open_dir(name)
    }

    /// Opens one regular child and binds it to this directory and name.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid, missing, symlinked, or non-regular entries.
    pub fn open_regular(&self, name: &OsStr) -> io::Result<AnchoredFile> {
        // Opening a FIFO read-only would otherwise wait indefinitely for a writer before the
        // descriptor can be classified. Non-blocking mode is ignored for regular files and
        // lets the metadata check below reject special files without leaving the caller stuck.
        let file = open_file_at(
            self.fd.as_raw_fd(),
            name,
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOCTTY,
            0,
        )?;
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "entry is not a regular file",
            ));
        }
        let anchored = AnchoredFile {
            file,
            parent: self.try_clone()?,
            name: name.to_os_string(),
            display: self.display.join(name),
        };
        anchored.stable_identity()?;
        Ok(anchored)
    }

    /// Opens a normalized relative regular-file path component by component.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/non-normalized path or any unsafe component.
    pub fn open_relative_regular(&self, path: &Path) -> io::Result<AnchoredFile> {
        let mut components = path.components().peekable();
        let mut directory = self.try_clone()?;
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "relative file path is not normalized",
                ));
            };
            if components.peek().is_none() {
                return directory.open_regular(name);
            }
            directory = directory.open_dir(name)?;
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "relative file path is empty",
        ))
    }

    /// Duplicates this retained directory descriptor.
    ///
    /// # Errors
    ///
    /// Returns the operating-system descriptor duplication error.
    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(Self {
            // `OwnedFd::try_clone` uses the platform's close-on-exec duplication path. A plain
            // `dup` would clear `FD_CLOEXEC` and leak this directory authority to child engines.
            fd: self.fd.try_clone()?,
            display: self.display.clone(),
        })
    }

    /// Opens or creates a regular advisory-lock file relative to this directory.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid, symlinked, or non-regular entry.
    pub fn open_lock_file(&self, name: &OsStr) -> io::Result<AnchoredFile> {
        let file = open_file_at(
            self.fd.as_raw_fd(),
            name,
            libc::O_RDWR | libc::O_CREAT | libc::O_NONBLOCK | libc::O_NOCTTY,
            0o600,
        )?;
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "lock entry is not a regular file",
            ));
        }
        let anchored = AnchoredFile {
            file,
            parent: self.try_clone()?,
            name: name.to_os_string(),
            display: self.display.join(name),
        };
        anchored.stable_identity()?;
        Ok(anchored)
    }

    /// Inspects one child entry without following a final symlink.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid name or failed metadata lookup.
    pub fn entry_kind(&self, name: &OsStr) -> io::Result<Option<EntryKind>> {
        validate_name(name)?;
        stat_at(self.fd.as_raw_fd(), name)
    }

    /// Lists bounded entry names and their no-follow kinds from a fresh directory offset.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be listed or an entry cannot be inspected.
    pub fn entries(&self, maximum: usize) -> io::Result<Vec<DirectoryEntry>> {
        list_directory(self.fd.as_raw_fd(), maximum)?
            .into_iter()
            .map(|name| {
                let kind = self.entry_kind(&name)?.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotFound, "directory entry disappeared")
                })?;
                Ok(DirectoryEntry { name, kind })
            })
            .collect()
    }

    /// Counts preserved identity-atomic cleanup remnants without consuming any entry.
    ///
    /// # Errors
    ///
    /// Returns an error when the reserved entry is not a private owner-only directory, an entry
    /// is not a regular app-shaped quarantine remnant, or `maximum` would be exceeded.
    pub fn cleanup_quarantine_entry_count(&self, maximum: usize) -> io::Result<usize> {
        let name = OsStr::new(CLEANUP_DIRECTORY_NAME);
        match self.entry_kind(name)? {
            None => return Ok(0),
            Some(EntryKind::Directory) => {}
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "reserved cleanup quarantine entry is not a directory",
                ));
            }
        }
        let cleanup = self.open_dir(name)?;
        validate_private_directory(cleanup.fd.as_raw_fd())?;
        lock_cleanup_quarantine(&cleanup)?;
        let entries = list_cleanup_quarantine_entries(&cleanup, maximum)?;
        validate_cleanup_quarantine_entries(&entries)?;
        Ok(entries.len())
    }

    /// Creates a new regular child and returns a pathname-bound writable descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is invalid, exists, or cannot be created.
    pub fn create_new_regular(&self, name: &OsStr) -> io::Result<AnchoredFile> {
        validate_publication_name(name)?;
        let file = open_file_at(
            self.fd.as_raw_fd(),
            name,
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
            0o600,
        )?;
        let anchored = AnchoredFile {
            file,
            parent: self.try_clone()?,
            name: name.to_os_string(),
            display: self.display.join(name),
        };
        anchored.stable_identity()?;
        Ok(anchored)
    }

    /// Durably replaces a regular child through a same-directory temporary entry.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe target or any write, rename, or sync failure.
    pub fn replace_atomic(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        self.replace_atomic_with_hook(name, bytes, || Ok(()))
    }

    fn replace_atomic_with_hook(
        &self,
        name: &OsStr,
        bytes: &[u8],
        after_rename: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        validate_publication_name(name)?;
        if matches!(self.entry_kind(name)?, Some(kind) if kind != EntryKind::File) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mutable target is not a regular file",
            ));
        }
        let (temporary, mut file) = open_unique_temporary(self.fd.as_raw_fd(), name)?;
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            let source_identity = StableFileIdentity::from_metadata(&file.metadata()?);
            rename_at(self.fd.as_raw_fd(), &temporary, self.fd.as_raw_fd(), name)?;
            after_rename()?;
            if !stable_identity_at(self.fd.as_raw_fd(), name)?.same_file_contents(&source_identity)
                || !file_matches_bytes(&file, bytes)?
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "mutable publication was relinked during commit",
                ));
            }
            self.sync()?;
            if !stable_identity_at(self.fd.as_raw_fd(), name)?.same_file_contents(&source_identity)
                || !file_matches_bytes(&file, bytes)?
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "mutable publication bytes changed during commit",
                ));
            }
            Ok(())
        })();
        if result.is_err()
            && let Ok(identity) = file
                .metadata()
                .map(|metadata| StableFileIdentity::from_metadata(&metadata))
        {
            let _ = self.unlink_if_identity(&temporary, &identity);
        }
        result
    }

    /// Durably publishes a new immutable child through an exclusive hard link.
    ///
    /// # Errors
    ///
    /// Returns an error if the target exists or any write, link, or sync step fails.
    pub fn publish_new_atomic(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        self.publish_new_atomic_with_hooks(name, bytes, |_| Ok(()), || Ok(()), || Ok(()))
    }

    fn publish_new_atomic_with_hooks(
        &self,
        name: &OsStr,
        bytes: &[u8],
        before_link: impl FnOnce(&OsStr) -> io::Result<()>,
        after_link: impl FnOnce() -> io::Result<()>,
        after_temporary_unlink: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        validate_publication_name(name)?;
        if self.entry_kind(name)?.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "immutable target already exists",
            ));
        }
        let (temporary, mut file) = open_unique_temporary(self.fd.as_raw_fd(), name)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        let source_identity = StableFileIdentity::from_metadata(&file.metadata()?);
        let result = (|| {
            before_link(&temporary)?;
            link_at(self.fd.as_raw_fd(), &temporary, self.fd.as_raw_fd(), name)?;
            after_link()?;
            let source_path_identity = stable_identity_at(self.fd.as_raw_fd(), &temporary).ok();
            let target_identity = stable_identity_at(self.fd.as_raw_fd(), name).ok();
            let valid = source_path_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity))
                && target_identity
                    .as_ref()
                    .is_some_and(|identity| identity.same_file_contents(&source_identity))
                && file_matches_bytes(&file, bytes)?;
            if !valid {
                rollback_untrusted_link(
                    self,
                    name,
                    target_identity.as_ref(),
                    &source_identity,
                    source_path_identity.as_ref(),
                );
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "immutable publication was relinked during commit",
                ));
            }
            if !self.unlink_if_identity(&temporary, &source_identity)? {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "immutable publication temporary changed before cleanup",
                ));
            }
            after_temporary_unlink()?;
            self.sync()?;
            let final_target = stable_identity_at(self.fd.as_raw_fd(), name).ok();
            if !final_target
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity))
                || !file_matches_bytes(&file, bytes)?
            {
                // The temporary evidence no longer exists, so a target mismatch cannot be
                // safely unlinked: it may be a foreign replacement or the final link to bytes
                // changed in place. Preserve it for inspection and fail closed.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "immutable publication target changed during final commit",
                ));
            }
            Ok(())
        })();
        if result.is_err() && file_matches_bytes(&file, bytes).unwrap_or(false) {
            let _ = self.unlink_if_identity(&temporary, &source_identity);
        }
        result
    }

    /// Hard-links one regular child to a new child in another retained directory.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe source/target or a link/synchronization failure.
    pub fn link_new_to(
        &self,
        source: &OsStr,
        target_directory: &Self,
        target: &OsStr,
    ) -> io::Result<()> {
        self.link_new_to_with_hooks(source, target_directory, target, || Ok(()), || Ok(()))
    }

    fn link_new_to_with_hooks(
        &self,
        source: &OsStr,
        target_directory: &Self,
        target: &OsStr,
        before_link: impl FnOnce() -> io::Result<()>,
        after_link: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        validate_publication_name(target)?;
        let source_file = self.open_regular(source)?;
        let source_identity = source_file.stable_identity()?;
        let source_digest = digest_file_exact(&source_file.file, source_identity.length)?;
        if target_directory.entry_kind(target)?.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "hard-link target already exists",
            ));
        }
        before_link()?;
        link_at(
            self.fd.as_raw_fd(),
            source,
            target_directory.fd.as_raw_fd(),
            target,
        )?;
        after_link()?;
        let source_path_identity = stable_identity_at(self.fd.as_raw_fd(), source).ok();
        let target_identity = stable_identity_at(target_directory.fd.as_raw_fd(), target).ok();
        let descriptor_stable = source_file
            .stable_identity()
            .is_ok_and(|identity| identity.same_file_contents(&source_identity));
        let digest_stable = digest_file_exact(&source_file.file, source_identity.length)
            .is_ok_and(|digest| digest == source_digest);
        let valid = descriptor_stable
            && digest_stable
            && source_path_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity))
            && target_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity));
        if !valid {
            rollback_untrusted_link(
                target_directory,
                target,
                target_identity.as_ref(),
                &source_identity,
                source_path_identity.as_ref(),
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "hard-link source was relinked during publication",
            ));
        }
        target_directory.sync()?;
        let source_path_identity = stable_identity_at(self.fd.as_raw_fd(), source).ok();
        let target_identity = stable_identity_at(target_directory.fd.as_raw_fd(), target).ok();
        let valid_after_sync = source_file
            .stable_identity()
            .is_ok_and(|identity| identity.same_file_contents(&source_identity))
            && digest_file_exact(&source_file.file, source_identity.length)
                .is_ok_and(|digest| digest == source_digest)
            && source_path_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity))
            && target_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity));
        if !valid_after_sync {
            rollback_untrusted_link(
                target_directory,
                target,
                target_identity.as_ref(),
                &source_identity,
                source_path_identity.as_ref(),
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "hard-link source bytes changed during publication",
            ));
        }
        Ok(())
    }

    /// Removes a child only when it still names the supplied pathname-bound file.
    ///
    /// # Errors
    ///
    /// Returns an error if the entry is missing/replaced or unlink/synchronization fails.
    pub fn remove_anchored_file(&self, file: &AnchoredFile) -> io::Result<()> {
        self.remove_anchored_file_with_hook(file, || Ok(()))
    }

    fn remove_anchored_file_with_hook(
        &self,
        file: &AnchoredFile,
        before_quarantine: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        let expected = file.stable_identity()?;
        if !self.quarantine_unlink_if_identity(
            &file.name,
            &expected,
            IdentityComparison::SameContents,
            before_quarantine,
            || Ok(()),
        )? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cleanup entry no longer names the retained file",
            ));
        }
        Ok(())
    }

    /// Removes a target only when it still hard-links the supplied source file.
    ///
    /// # Errors
    ///
    /// Returns an error if the target is missing/replaced or unlink/synchronization fails.
    pub fn remove_link_to(&self, target: &OsStr, source: &AnchoredFile) -> io::Result<()> {
        let expected = StableFileIdentity::from_metadata(&source.file.metadata()?);
        if !self.quarantine_unlink_if_identity(
            target,
            &expected,
            IdentityComparison::SameContents,
            || Ok(()),
            || Ok(()),
        )? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cleanup target no longer names the retained file",
            ));
        }
        Ok(())
    }

    /// Verifies that a target still hard-links the supplied retained source.
    ///
    /// # Errors
    ///
    /// Returns an error if either pathname changed or the entries are not the same file.
    pub fn verify_link_to(&self, target: &OsStr, source: &AnchoredFile) -> io::Result<()> {
        let expected = source.stable_identity()?;
        let observed = stable_identity_at(self.fd.as_raw_fd(), target)?;
        if !observed.same_file_contents(&expected) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "publication target no longer names the retained file",
            ));
        }
        Ok(())
    }

    /// Verifies a published path against a retained descriptor after its temporary pathname was
    /// removed, including the exact expected SHA-256 and byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if the path no longer names the retained descriptor, either identity
    /// changes during hashing, or the retained bytes differ from the supplied content identity.
    pub fn verify_retained_link_content(
        &self,
        target: &OsStr,
        source: &AnchoredFile,
        expected_sha256: &[u8; 32],
        expected_size: u64,
    ) -> io::Result<()> {
        let descriptor_before = StableFileIdentity::from_metadata(&source.file.metadata()?);
        let target_before = stable_identity_at(self.fd.as_raw_fd(), target)?;
        if descriptor_before != target_before || descriptor_before.length != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "publication target no longer names the retained descriptor",
            ));
        }
        let observed_sha256 = digest_file_exact(&source.file, expected_size)?;
        let descriptor_after = StableFileIdentity::from_metadata(&source.file.metadata()?);
        let target_after = stable_identity_at(self.fd.as_raw_fd(), target)?;
        if descriptor_after != descriptor_before
            || target_after != descriptor_after
            || &observed_sha256 != expected_sha256
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "publication target or retained content changed during final validation",
            ));
        }
        Ok(())
    }

    /// Synchronizes directory metadata.
    ///
    /// # Errors
    ///
    /// Returns the operating-system synchronization error.
    pub fn sync(&self) -> io::Result<()> {
        fsync_fd(self.fd.as_raw_fd())
    }

    fn cleanup_directory(&self) -> io::Result<Self> {
        let name = OsStr::new(CLEANUP_DIRECTORY_NAME);
        match mkdir_at_mode(self.fd.as_raw_fd(), name, 0o700) {
            Ok(()) => self.sync()?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        let cleanup = self.open_dir(name)?;
        validate_private_directory(cleanup.fd.as_raw_fd())?;
        lock_cleanup_quarantine(&cleanup)?;
        // Leave capacity for the entry about to be moved. Existing crash remnants have no
        // durable attribution proof, so preserving them and failing closed is the only safe
        // automatic recovery policy.
        let maximum_existing = MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES.saturating_sub(1);
        let entries =
            list_cleanup_quarantine_entries(&cleanup, maximum_existing).map_err(|error| {
                if error.kind() == io::ErrorKind::InvalidData {
                    cleanup_capacity_error(&cleanup)
                } else {
                    error
                }
            })?;
        validate_cleanup_quarantine_entries(&entries)?;
        Ok(cleanup)
    }

    fn quarantine_unlink_if_identity(
        &self,
        name: &OsStr,
        expected: &StableFileIdentity,
        comparison: IdentityComparison,
        before_quarantine: impl FnOnce() -> io::Result<()>,
        after_quarantine: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<bool> {
        let observed = match stable_identity_at(self.fd.as_raw_fd(), name) {
            Ok(identity) => identity,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        };
        if !comparison.matches(&observed, expected) {
            return Ok(false);
        }
        before_quarantine()?;

        let cleanup = self.cleanup_directory()?;
        let quarantine_name =
            move_to_unique_quarantine(self.fd.as_raw_fd(), name, cleanup.fd.as_raw_fd())?;
        let quarantined = stable_identity_at(cleanup.fd.as_raw_fd(), &quarantine_name);
        match quarantined {
            Ok(identity) if comparison.matches(&identity, expected) => {
                // The source pathname was captured atomically by rename. The cleanup directory
                // is private (owner-only) and retained. Synchronize the move before the final
                // unlink so an interruption can leave only a bounded, inspectable remnant.
                cleanup.sync()?;
                self.sync()?;
                after_quarantine()?;
                let final_identity = stable_identity_at(cleanup.fd.as_raw_fd(), &quarantine_name)?;
                if !comparison.matches(&final_identity, expected) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "cleanup quarantine entry changed before unlink; the replacement was preserved",
                    ));
                }
                unlink_at(cleanup.fd.as_raw_fd(), &quarantine_name)?;
                cleanup.sync()?;
                self.sync()?;
                Ok(true)
            }
            _ => {
                // A writer replaced the source between the identity check and rename. Restore
                // it without overwriting anything that may now occupy the original name. If
                // restoration races, the foreign entry remains preserved in quarantine.
                let _ = rename_noreplace(
                    cleanup.fd.as_raw_fd(),
                    &quarantine_name,
                    self.fd.as_raw_fd(),
                    name,
                );
                let _ = cleanup.sync();
                let _ = self.sync();
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "cleanup entry changed before identity-atomic quarantine; it was preserved",
                ))
            }
        }
    }

    fn unlink_if_identity(&self, name: &OsStr, expected: &StableFileIdentity) -> io::Result<bool> {
        self.quarantine_unlink_if_identity(
            name,
            expected,
            IdentityComparison::SameContents,
            || Ok(()),
            || Ok(()),
        )
    }
}

#[derive(Clone, Copy)]
enum IdentityComparison {
    SameContents,
}

impl IdentityComparison {
    fn matches(self, observed: &StableFileIdentity, expected: &StableFileIdentity) -> bool {
        match self {
            Self::SameContents => observed.same_file_contents(expected),
        }
    }
}

/// An opened regular file bound to its retained parent directory and entry name.
#[derive(Debug)]
pub struct AnchoredFile {
    file: File,
    parent: AnchoredDir,
    name: OsString,
    display: PathBuf,
}

impl AnchoredFile {
    /// Opens a regular file without following any path component symlink.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid, missing, symlinked, or non-regular path.
    pub fn open_existing(path: &Path) -> io::Result<Self> {
        let (parent, name) = AnchoredDir::open_parent(path, false)?;
        match parent.entry_kind(&name)? {
            Some(EntryKind::File) => parent.open_regular(&name),
            Some(_) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "path is not a regular file",
            )),
            None => Err(io::Error::new(io::ErrorKind::NotFound, "file not found")),
        }
    }

    /// Returns the underlying file for APIs such as advisory locking.
    #[must_use]
    pub fn file(&self) -> &File {
        &self.file
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

    /// Hashes a retained descriptor while requiring its metadata and length to remain stable.
    ///
    /// Unlike [`Self::stable_identity`], this operation intentionally does not require the
    /// original pathname to remain present. It is used to bind content before a temporary hard
    /// link is removed and later revalidate the final publication path.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor exceeds `maximum_bytes`, changes while hashing, or
    /// cannot be read exactly once.
    pub fn retained_sha256(&self, maximum_bytes: u64) -> io::Result<([u8; 32], u64)> {
        let before = StableFileIdentity::from_metadata(&self.file.metadata()?);
        if before.length > maximum_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "retained file exceeds its hashing limit",
            ));
        }
        let sha256 = digest_file_exact(&self.file, before.length)?;
        let after = StableFileIdentity::from_metadata(&self.file.metadata()?);
        if after != before {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "retained file changed while it was hashed",
            ));
        }
        Ok((sha256, before.length))
    }

    /// Reports whether both retained pathnames still name the same regular file.
    ///
    /// # Errors
    ///
    /// Returns an error if either retained pathname was replaced or cannot be inspected.
    pub fn is_same_file_as(&self, other: &Self) -> io::Result<bool> {
        let left = self.stable_identity()?;
        let right = other.stable_identity()?;
        Ok(left.same_file_contents(&right))
    }

    /// Publishes this pathname-bound file as a new hard link in another retained directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the source changed, the target exists, or linking/syncing fails.
    pub fn publish_as_new(&self, target_directory: &AnchoredDir, target: &OsStr) -> io::Result<()> {
        self.publish_as_new_with_hook(target_directory, target, || Ok(()))
    }

    fn publish_as_new_with_hook(
        &self,
        target_directory: &AnchoredDir,
        target: &OsStr,
        after_link: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        validate_publication_name(target)?;
        let source_identity = self.stable_identity()?;
        let source_digest = digest_file_exact(&self.file, source_identity.length)?;
        if target_directory.entry_kind(target)?.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "hard-link target already exists",
            ));
        }
        link_at(
            self.parent.fd.as_raw_fd(),
            &self.name,
            target_directory.fd.as_raw_fd(),
            target,
        )?;
        after_link()?;
        target_directory.sync()?;
        let source_path_identity = stable_identity_at(self.parent.fd.as_raw_fd(), &self.name).ok();
        let target_identity = stable_identity_at(target_directory.fd.as_raw_fd(), target).ok();
        let descriptor_stable = self
            .stable_identity()
            .is_ok_and(|identity| identity.same_file_contents(&source_identity));
        let digest_stable = digest_file_exact(&self.file, source_identity.length)
            .is_ok_and(|digest| digest == source_digest);
        let valid = descriptor_stable
            && digest_stable
            && source_path_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity))
            && target_identity
                .as_ref()
                .is_some_and(|identity| identity.same_file_contents(&source_identity));
        if !valid {
            rollback_untrusted_link(
                target_directory,
                target,
                target_identity.as_ref(),
                &source_identity,
                source_path_identity.as_ref(),
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "hard-link source was relinked during publication",
            ));
        }
        Ok(())
    }

    /// Returns the diagnostic path captured while opening the descriptor.
    #[must_use]
    pub fn display(&self) -> &Path {
        &self.display
    }

    /// Captures one identity after verifying that the retained pathname still names this file.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata cannot be read or descriptor/path identity differs.
    pub fn stable_identity(&self) -> io::Result<StableFileIdentity> {
        let descriptor = StableFileIdentity::from_metadata(&self.file.metadata()?);
        let pathname = stable_identity_at(self.parent.fd.as_raw_fd(), &self.name)?;
        if descriptor != pathname {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "opened descriptor no longer matches its anchored pathname",
            ));
        }
        Ok(descriptor)
    }

    /// Verifies descriptor/path identity and the exact read count after a bounded read.
    ///
    /// # Errors
    ///
    /// Returns an error if the file changed, was relinked, or was not read exactly once.
    pub fn verify_stable_read(
        &self,
        before: &StableFileIdentity,
        bytes_read: u64,
    ) -> io::Result<()> {
        let after = self.stable_identity()?;
        if &after != before || bytes_read != before.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file changed while it was being read",
            ));
        }
        Ok(())
    }
}

fn file_matches_bytes(file: &File, expected: &[u8]) -> io::Result<bool> {
    if file.metadata()?.len() != u64::try_from(expected.len()).unwrap_or(u64::MAX) {
        return Ok(false);
    }
    let mut reader = file.try_clone()?;
    reader.seek(io::SeekFrom::Start(0))?;
    let mut offset = 0_usize;
    let mut buffer = [0_u8; 16 * 1024];
    while offset < expected.len() {
        let length = buffer.len().min(expected.len() - offset);
        reader.read_exact(&mut buffer[..length])?;
        if buffer[..length] != expected[offset..offset + length] {
            return Ok(false);
        }
        offset += length;
    }
    let mut trailing = [0_u8; 1];
    Ok(reader.read(&mut trailing)? == 0)
}

fn digest_file_exact(file: &File, expected_length: u64) -> io::Result<[u8; 32]> {
    let mut reader = file.try_clone()?;
    reader.seek(io::SeekFrom::Start(0))?;
    let mut limited = reader.take(expected_length.saturating_add(1));
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut total = 0_u64;
    loop {
        let read = limited.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(read).expect("buffer length fits u64"))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "file length overflow"))?;
        digest.update(&buffer[..read]);
    }
    if total != expected_length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file length changed during content verification",
        ));
    }
    Ok(digest.finalize().into())
}

fn rollback_untrusted_link(
    target_directory: &AnchoredDir,
    target: &OsStr,
    target_identity: Option<&StableFileIdentity>,
    expected_source: &StableFileIdentity,
    current_source: Option<&StableFileIdentity>,
) {
    let Some(target_identity) = target_identity else {
        return;
    };
    let provably_linked = target_identity.same_file_contents(expected_source)
        || current_source.is_some_and(|source| target_identity.same_file_contents(source));
    if provably_linked
        && target_directory
            .unlink_if_identity(target, target_identity)
            .is_ok_and(std::convert::identity)
    {
        let _ = target_directory.sync();
    }
}

/// Same-descriptor and retained-path identity used around bounded reads.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StableFileIdentity {
    length: u64,
    device: u64,
    inode: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

/// Device/inode authority for one retained directory descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StableDirectoryIdentity {
    device: u64,
    inode: u64,
}

impl StableDirectoryIdentity {
    fn from_fd(fd: RawFd) -> io::Result<Self> {
        let mut stat = MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `fd` is live and `stat` points to sufficient writable storage. The value is
        // assumed initialized only after `fstat` reports success.
        if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `fstat` succeeded and initialized the complete structure.
        let stat = unsafe { stat.assume_init() };
        if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "retained descriptor is not a directory",
            ));
        }
        #[cfg_attr(target_os = "linux", allow(clippy::useless_conversion))]
        let device = u64::try_from(stat.st_dev)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid device id"))?;
        Ok(Self {
            device,
            inode: stat.st_ino,
        })
    }
}

impl StableFileIdentity {
    fn same_file_contents(&self, other: &Self) -> bool {
        self.length == other.length
            && self.device == other.device
            && self.inode == other.inode
            && self.modified_seconds == other.modified_seconds
            && self.modified_nanoseconds == other.modified_nanoseconds
    }

    fn from_metadata(metadata: &Metadata) -> Self {
        use std::os::unix::fs::MetadataExt as _;

        Self {
            length: metadata.len(),
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    fn from_stat(stat: &libc::stat) -> io::Result<Self> {
        // `dev_t` is already `u64` on Linux, where the wide conversion would be a
        // useless-conversion lint; it stays a checked widening on macOS.
        #[cfg_attr(target_os = "linux", allow(clippy::useless_conversion))]
        let device = u64::try_from(stat.st_dev)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid device id"))?;
        Ok(Self {
            length: u64::try_from(stat.st_size)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "negative file size"))?,
            device,
            inode: stat.st_ino,
            modified_seconds: stat_modified_seconds(stat),
            modified_nanoseconds: stat_modified_nanoseconds(stat),
            changed_seconds: stat_changed_seconds(stat),
            changed_nanoseconds: stat_changed_nanoseconds(stat),
        })
    }

    /// Returns the descriptor length captured by this identity.
    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }
}

fn validate_name(name: &OsStr) -> io::Result<()> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes == b"." || bytes == b".." || bytes.contains(&b'/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "entry name must be one ordinary path component",
        ));
    }
    if bytes.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "entry name contains NUL",
        ));
    }
    Ok(())
}

fn validate_publication_name(name: &OsStr) -> io::Result<()> {
    validate_name(name)?;
    if name == OsStr::new(CLEANUP_DIRECTORY_NAME) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "entry name is reserved for identity-atomic cleanup",
        ));
    }
    Ok(())
}

fn validate_cleanup_quarantine_entries(entries: &[DirectoryEntry]) -> io::Result<()> {
    if entries
        .iter()
        .any(|entry| entry.kind != EntryKind::File || !is_cleanup_quarantine_name(&entry.name))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cleanup quarantine contains an unexpected entry; all entries were preserved for manual inspection",
        ));
    }
    Ok(())
}

fn list_cleanup_quarantine_entries(
    cleanup: &AnchoredDir,
    maximum: usize,
) -> io::Result<Vec<DirectoryEntry>> {
    // Independent publications in the same private parent can finish between `readdir` and
    // `fstatat`. A disappeared name consumes no storage and is safe to omit; every name still
    // present is validated without following symlinks.
    list_directory(cleanup.fd.as_raw_fd(), maximum)?
        .into_iter()
        .filter_map(|name| match cleanup.entry_kind(&name) {
            Ok(Some(kind)) => Some(Ok(DirectoryEntry { name, kind })),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

fn is_cleanup_quarantine_name(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let Some(suffix) = name.strip_prefix(".entry-") else {
        return false;
    };
    let Some((process, sequence)) = suffix.split_once('-') else {
        return false;
    };
    !process.is_empty()
        && !sequence.is_empty()
        && process.bytes().all(|byte| byte.is_ascii_digit())
        && sequence.bytes().all(|byte| byte.is_ascii_digit())
}

fn cleanup_capacity_error(cleanup: &AnchoredDir) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "cleanup quarantine reached its {MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES}-entry safety cap at {}; every remnant was preserved; remove entries manually after inspection",
            cleanup.display().display()
        ),
    )
}

fn lock_cleanup_quarantine(cleanup: &AnchoredDir) -> io::Result<()> {
    let started = Instant::now();
    loop {
        match cleanup.try_lock_exclusive() {
            Ok(()) => return Ok(()),
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    && started.elapsed() < CLEANUP_LOCK_TIMEOUT =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "cleanup quarantine remained busy for the bounded 5-second lock timeout",
                ));
            }
            Err(error) => return Err(error),
        }
    }
}

fn c_name(name: &OsStr) -> io::Result<CString> {
    validate_name(name)?;
    CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "entry name contains NUL"))
}

fn open_directory_path(path: &Path) -> io::Result<OwnedFd> {
    let encoded = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
    // SAFETY: `encoded` is a live NUL-terminated string. The returned descriptor is
    // checked for failure and immediately transferred into `OwnedFd` exactly once.
    let fd = unsafe {
        libc::open(
            encoded.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    owned_fd(fd)
}

fn open_directory_at(parent: RawFd, name: &OsStr) -> io::Result<OwnedFd> {
    let name = c_name(name)?;
    // SAFETY: `parent` is a live directory descriptor owned by the caller and `name`
    // is NUL-terminated. `O_NOFOLLOW|O_DIRECTORY` rejects symlinks and non-directories.
    let fd = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    owned_fd(fd)
}

fn open_file_at(parent: RawFd, name: &OsStr, flags: i32, mode: libc::mode_t) -> io::Result<File> {
    let name = c_name(name)?;
    // SAFETY: the directory descriptor and C string are live for this call. The fd is
    // checked before single ownership is transferred to `File`.
    let fd = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            libc::c_uint::from(mode),
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` was returned successfully above and has not been transferred or closed.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn mkdir_at(parent: RawFd, name: &OsStr) -> io::Result<()> {
    mkdir_at_mode(parent, name, 0o755)
}

fn mkdir_at_mode(parent: RawFd, name: &OsStr, mode: libc::mode_t) -> io::Result<()> {
    let name = c_name(name)?;
    // SAFETY: `parent` and `name` are valid for the duration of the syscall.
    if unsafe { libc::mkdirat(parent, name.as_ptr(), mode) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn validate_private_directory(fd: RawFd) -> io::Result<()> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fd` is live and the output buffer is large enough for `fstat`.
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fstat` succeeded and initialized the complete structure.
    let stat = unsafe { stat.assume_init() };
    // SAFETY: `geteuid` has no preconditions.
    let effective_user = unsafe { libc::geteuid() };
    if stat.st_mode & libc::S_IFMT != libc::S_IFDIR
        || stat.st_uid != effective_user
        || stat.st_mode & 0o077 != 0
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "cleanup quarantine must be an owner-only directory",
        ));
    }
    Ok(())
}

fn rename_at(
    source_directory: RawFd,
    source: &OsStr,
    target_directory: RawFd,
    target: &OsStr,
) -> io::Result<()> {
    let source = c_name(source)?;
    let target = c_name(target)?;
    // SAFETY: both descriptors and both C strings are valid for this syscall.
    if unsafe {
        libc::renameat(
            source_directory,
            source.as_ptr(),
            target_directory,
            target.as_ptr(),
        )
    } == 0
    {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos"
))]
fn rename_noreplace(
    source_directory: RawFd,
    source: &OsStr,
    target_directory: RawFd,
    target: &OsStr,
) -> io::Result<()> {
    let source = c_name(source)?;
    let target = c_name(target)?;
    // SAFETY: both retained descriptors and both C strings are valid for this syscall.
    let result = unsafe {
        libc::renameatx_np(
            source_directory,
            source.as_ptr(),
            target_directory,
            target.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn rename_noreplace(
    source_directory: RawFd,
    source: &OsStr,
    target_directory: RawFd,
    target: &OsStr,
) -> io::Result<()> {
    let source = c_name(source)?;
    let target = c_name(target)?;
    // SAFETY: both retained descriptors and both C strings are valid for this syscall.
    let result = unsafe {
        libc::renameat2(
            source_directory,
            source.as_ptr(),
            target_directory,
            target.as_ptr(),
            libc::RENAME_NOREPLACE as libc::c_uint,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn link_at(
    source_directory: RawFd,
    source: &OsStr,
    target_directory: RawFd,
    target: &OsStr,
) -> io::Result<()> {
    let source = c_name(source)?;
    let target = c_name(target)?;
    // SAFETY: both descriptors and both C strings are valid; flags=0 does not follow a
    // source symlink. Callers require and re-open regular artifacts with `O_NOFOLLOW`.
    if unsafe {
        libc::linkat(
            source_directory,
            source.as_ptr(),
            target_directory,
            target.as_ptr(),
            0,
        )
    } == 0
    {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn unlink_at(parent: RawFd, name: &OsStr) -> io::Result<()> {
    let name = c_name(name)?;
    // SAFETY: `parent` and `name` are valid. With flags=0, unlink removes only this
    // directory entry and never traverses a symlink target.
    if unsafe { libc::unlinkat(parent, name.as_ptr(), 0) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn stat_at(parent: RawFd, name: &OsStr) -> io::Result<Option<EntryKind>> {
    let Some(stat) = full_stat_at(parent, name)? else {
        return Ok(None);
    };
    let mode = stat.st_mode & libc::S_IFMT;
    Ok(Some(if mode == libc::S_IFREG {
        EntryKind::File
    } else if mode == libc::S_IFDIR {
        EntryKind::Directory
    } else if mode == libc::S_IFLNK {
        EntryKind::Symlink
    } else {
        EntryKind::Other
    }))
}

fn stable_identity_at(parent: RawFd, name: &OsStr) -> io::Result<StableFileIdentity> {
    let stat = full_stat_at(parent, name)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "anchored pathname disappeared"))?;
    if stat.st_mode & libc::S_IFMT != libc::S_IFREG {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "anchored pathname is not a regular file",
        ));
    }
    StableFileIdentity::from_stat(&stat)
}

fn full_stat_at(parent: RawFd, name: &OsStr) -> io::Result<Option<libc::stat>> {
    let name = c_name(name)?;
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `stat` points to enough writable memory, and the descriptor/C string are
    // valid. The value is assumed initialized only after a successful return.
    let result = unsafe {
        libc::fstatat(
            parent,
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::NotFound {
            return Ok(None);
        }
        return Err(error);
    }
    // SAFETY: `fstatat` succeeded and initialized the complete struct.
    Ok(Some(unsafe { stat.assume_init() }))
}

const fn stat_modified_seconds(stat: &libc::stat) -> i64 {
    stat.st_mtime
}

const fn stat_modified_nanoseconds(stat: &libc::stat) -> i64 {
    stat.st_mtime_nsec
}

const fn stat_changed_seconds(stat: &libc::stat) -> i64 {
    stat.st_ctime
}

const fn stat_changed_nanoseconds(stat: &libc::stat) -> i64 {
    stat.st_ctime_nsec
}

fn list_directory(fd: RawFd, maximum: usize) -> io::Result<Vec<OsString>> {
    let current = c".";
    // SAFETY: `fd` is a live directory descriptor and the static C string is valid.
    // `openat(".")` creates a new open file description, avoiding the shared directory
    // offset that `dup` would introduce across repeated listings.
    let duplicate = unsafe {
        libc::openat(
            fd,
            current.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if duplicate < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `duplicate` is a newly owned directory descriptor.
    let stream = unsafe { libc::fdopendir(duplicate) };
    if stream.is_null() {
        // SAFETY: ownership was not transferred when `fdopendir` failed.
        unsafe { libc::close(duplicate) };
        return Err(io::Error::last_os_error());
    }
    let stream = DirectoryStream(stream);
    let mut names = Vec::new();
    loop {
        set_errno(0);
        // SAFETY: the stream remains live and is used only by this loop.
        let entry = unsafe { libc::readdir(stream.0) };
        if entry.is_null() {
            let error = get_errno();
            if error == 0 {
                break;
            }
            return Err(io::Error::from_raw_os_error(error));
        }
        // SAFETY: POSIX guarantees `d_name` is NUL-terminated for a successful entry.
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
        if name != b"." && name != b".." {
            if names.len() >= maximum {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "directory exceeds its bounded entry limit",
                ));
            }
            names.push(OsString::from_vec(name.to_vec()));
        }
    }
    Ok(names)
}

struct DirectoryStream(*mut libc::DIR);

impl Drop for DirectoryStream {
    fn drop(&mut self) {
        // SAFETY: this wrapper exclusively owns the stream and drops it exactly once.
        unsafe { libc::closedir(self.0) };
    }
}

fn fsync_fd(fd: RawFd) -> io::Result<()> {
    // SAFETY: the descriptor is live for the duration of the syscall.
    if unsafe { libc::fsync(fd) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn owned_fd(fd: RawFd) -> io::Result<OwnedFd> {
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        // SAFETY: this successful raw fd has not been transferred or closed.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

fn temporary_name(target: &OsStr) -> io::Result<OsString> {
    let target = target
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file name is not UTF-8"))?;
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(OsString::from(format!(
        ".{target}.tmp-{}-{sequence}",
        std::process::id()
    )))
}

fn quarantine_name() -> OsString {
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    OsString::from(format!(".entry-{}-{sequence}", std::process::id()))
}

fn move_to_unique_quarantine(
    source_directory: RawFd,
    source: &OsStr,
    quarantine_directory: RawFd,
) -> io::Result<OsString> {
    for _ in 0..TEMPORARY_CREATE_ATTEMPTS {
        let name = quarantine_name();
        match rename_noreplace(source_directory, source, quarantine_directory, &name) {
            Ok(()) => return Ok(name),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "cleanup quarantine exhausted its bounded collision retries",
    ))
}

fn open_unique_temporary(parent: RawFd, target: &OsStr) -> io::Result<(OsString, File)> {
    open_unique_temporary_with(parent, || temporary_name(target))
}

fn open_unique_temporary_with(
    parent: RawFd,
    mut next_name: impl FnMut() -> io::Result<OsString>,
) -> io::Result<(OsString, File)> {
    for _ in 0..TEMPORARY_CREATE_ATTEMPTS {
        let name = next_name()?;
        match open_file_at(
            parent,
            &name,
            // Publication validates the retained descriptor's complete contents after the
            // namespace commit. Keep this descriptor readable as well as writable.
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
            0o600,
        ) {
            Ok(file) => return Ok((name, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "cannot create a unique bounded temporary entry",
    ))
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos"
))]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc returns the calling thread's valid errno storage pointer.
    unsafe { libc::__error() }
}

#[cfg(target_os = "linux")]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc returns the calling thread's valid errno storage pointer.
    unsafe { libc::__errno_location() }
}

#[cfg(target_os = "android")]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc returns the calling thread's valid errno storage pointer.
    unsafe { libc::__errno() }
}

fn set_errno(value: libc::c_int) {
    // SAFETY: `errno_pointer` returns writable thread-local errno storage.
    unsafe { *errno_pointer() = value };
}

fn get_errno() -> libc::c_int {
    // SAFETY: `errno_pointer` returns readable thread-local errno storage.
    unsafe { *errno_pointer() }
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::{OsStr, OsString},
        io,
        os::fd::AsRawFd,
        os::unix::fs::symlink,
        path::{Path, PathBuf},
    };

    use super::{AnchoredDir, EntryKind};

    fn temporary_directory(label: &str) -> PathBuf {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-{label}-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn retained_directory_descriptor_is_not_redirected_by_a_symlink_swap() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let external = root.with_extension("external");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let moved = root.with_extension("moved");
        std::fs::rename(&root, &moved).unwrap();
        symlink(&external, &root).unwrap();

        anchored
            .publish_new_atomic(OsStr::new("proof"), b"anchored")
            .unwrap();
        assert_eq!(std::fs::read(moved.join("proof")).unwrap(), b"anchored");
        assert!(!external.join("proof").exists());
        assert_eq!(
            anchored.entry_kind(OsStr::new("proof")).unwrap(),
            Some(EntryKind::File)
        );
    }

    #[test]
    fn final_symlink_is_never_followed_for_read_or_write() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-final-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("external"), b"outside").unwrap();
        symlink(root.join("external"), root.join("target")).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        assert!(anchored.open_regular(OsStr::new("target")).is_err());
        assert!(
            anchored
                .replace_atomic(OsStr::new("target"), b"inside")
                .is_err()
        );
        assert_eq!(std::fs::read(root.join("external")).unwrap(), b"outside");
    }

    #[test]
    fn regular_open_rejects_a_fifo_without_waiting_for_a_writer() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-fifo-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let name = super::c_name(OsStr::new("pipe")).unwrap();
        // SAFETY: `anchored` retains a live directory descriptor, `name` is NUL-terminated,
        // and the test supplies an ordinary owner-only permission mode.
        assert_eq!(
            unsafe { libc::mkfifoat(anchored.fd.as_raw_fd(), name.as_ptr(), 0o600) },
            0
        );

        let error = anchored.open_regular(OsStr::new("pipe")).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        drop(anchored);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cloned_directory_descriptor_remains_close_on_exec() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-cloexec-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let cloned = anchored.try_clone().unwrap();

        // SAFETY: the clone retains a live descriptor and `F_GETFD` has no pointer argument.
        let flags = unsafe { libc::fcntl(cloned.fd.as_raw_fd(), libc::F_GETFD) };

        assert!(flags >= 0);
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
        drop(cloned);
        drop(anchored);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ancestor_symlink_is_never_followed_when_opening_a_file() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-ancestor-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let external = root.with_extension("external");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        std::fs::write(external.join("model"), b"outside").unwrap();
        symlink(&external, root.join("models")).unwrap();

        assert!(
            super::AnchoredFile::open_existing(&root.join("models/model")).is_err(),
            "the ancestor symlink must not redirect the open"
        );
    }

    #[test]
    fn parent_components_do_not_lexically_bypass_a_symlink_check() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-parent-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let external = root.with_extension("external");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        std::fs::write(root.join("model"), b"inside").unwrap();
        symlink(&external, root.join("redirect")).unwrap();

        assert!(
            super::AnchoredFile::open_existing(&root.join("redirect/../model")).is_err(),
            "the redirect component must be opened and rejected before its parent component"
        );
    }

    #[test]
    fn leading_parent_component_fails_closed() {
        assert!(AnchoredDir::open_existing(std::path::Path::new("..")).is_err());
    }

    #[test]
    fn parent_component_uses_the_retained_descriptor_stack_after_reparent() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-parent-reparent-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let external = root.with_extension("external");
        std::fs::create_dir_all(root.join("child")).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        std::fs::write(root.join("proof"), b"inside").unwrap();
        std::fs::write(external.join("proof"), b"outside").unwrap();

        let mut moved = false;
        let anchored =
            AnchoredDir::open_path_with_hook(&root.join("child/.."), false, |component| {
                if component == OsStr::new("child") {
                    std::fs::rename(root.join("child"), external.join("child"))?;
                    moved = true;
                }
                Ok(())
            })
            .unwrap();
        assert!(moved);
        let mut proof = anchored.open_regular(OsStr::new("proof")).unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(proof.reader(), &mut bytes).unwrap();
        assert_eq!(bytes, b"inside");
    }

    #[test]
    fn anchored_file_detects_path_relink_after_open() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-relink-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("model");
        let moved = root.join("model-original");
        std::fs::write(&path, b"original").unwrap();
        let file = super::AnchoredFile::open_existing(&path).unwrap();
        let identity = file.stable_identity().unwrap();

        std::fs::rename(&path, &moved).unwrap();
        std::fs::write(&path, b"replacement").unwrap();

        assert!(
            file.verify_stable_read(&identity, identity.length())
                .is_err()
        );
    }

    #[test]
    fn immutable_publication_preserves_a_replaced_temporary_entry() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-temp-relink-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let mut replacement = None;

        let result = anchored.publish_new_atomic_with_hooks(
            OsStr::new("result"),
            b"trusted",
            |temporary| {
                let path = root.join(temporary);
                std::fs::remove_file(&path)?;
                std::fs::write(&path, b"foreign")?;
                replacement = Some(path);
                Ok(())
            },
            || Ok(()),
            || Ok(()),
        );

        assert!(result.is_err());
        assert!(!root.join("result").exists());
        assert_eq!(std::fs::read(replacement.unwrap()).unwrap(), b"foreign");
    }

    #[test]
    fn immutable_publication_preserves_a_replaced_target_entry() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-target-relink-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();

        let result = anchored.publish_new_atomic_with_hooks(
            OsStr::new("result"),
            b"trusted",
            |_| Ok(()),
            || {
                std::fs::remove_file(root.join("result"))?;
                std::fs::write(root.join("result"), b"foreign")
            },
            || Ok(()),
        );

        assert!(result.is_err());
        assert_eq!(std::fs::read(root.join("result")).unwrap(), b"foreign");
    }

    #[test]
    fn hard_link_publication_rejects_and_rolls_back_a_relinked_source() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-source-relink-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("source"), b"trusted").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();

        let result = anchored.link_new_to_with_hooks(
            OsStr::new("source"),
            &anchored,
            OsStr::new("target"),
            || {
                std::fs::remove_file(root.join("source"))?;
                std::fs::write(root.join("source"), b"foreign")
            },
            || Ok(()),
        );

        assert!(result.is_err());
        assert!(!root.join("target").exists());
        assert_eq!(std::fs::read(root.join("source")).unwrap(), b"foreign");
    }

    fn rewrite_same_length_and_restore_mtime(path: &Path, bytes: &[u8]) -> io::Result<()> {
        let modified = std::fs::metadata(path)?.modified()?;
        std::fs::write(path, bytes)?;
        std::fs::File::options()
            .write(true)
            .open(path)?
            .set_times(std::fs::FileTimes::new().set_modified(modified))
    }

    #[test]
    fn mutable_publication_rejects_same_inode_foreign_bytes_after_rename() {
        let root = temporary_directory("mutable-content-rewrite");
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let result = anchored.replace_atomic_with_hook(OsStr::new("result"), b"trusted", || {
            rewrite_same_length_and_restore_mtime(&root.join("result"), b"foreign")
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(root.join("result")).unwrap(), b"foreign");
    }

    #[test]
    fn immutable_byte_publication_rejects_same_inode_foreign_bytes_after_link() {
        let root = temporary_directory("immutable-content-rewrite");
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let mut temporary = None;
        let result = anchored.publish_new_atomic_with_hooks(
            OsStr::new("result"),
            b"trusted",
            |name| {
                temporary = Some(root.join(name));
                Ok(())
            },
            || rewrite_same_length_and_restore_mtime(&root.join("result"), b"foreign"),
            || Ok(()),
        );
        assert!(result.is_err());
        assert!(!root.join("result").exists());
        assert_eq!(std::fs::read(temporary.unwrap()).unwrap(), b"foreign");
    }

    #[test]
    fn immutable_publication_rejects_a_target_replaced_after_temporary_unlink() {
        let root = temporary_directory("immutable-final-target-relink");
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let result = anchored.publish_new_atomic_with_hooks(
            OsStr::new("result"),
            b"trusted",
            |_| Ok(()),
            || Ok(()),
            || {
                std::fs::remove_file(root.join("result"))?;
                std::fs::write(root.join("result"), b"foreign")
            },
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read(root.join("result")).unwrap(), b"foreign");
    }

    #[test]
    fn directory_link_publication_rejects_same_inode_foreign_bytes_after_link() {
        let root = temporary_directory("directory-link-content-rewrite");
        std::fs::write(root.join("source"), b"trusted").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let result = anchored.link_new_to_with_hooks(
            OsStr::new("source"),
            &anchored,
            OsStr::new("target"),
            || Ok(()),
            || rewrite_same_length_and_restore_mtime(&root.join("source"), b"foreign"),
        );
        assert!(result.is_err());
        assert!(!root.join("target").exists());
        assert_eq!(std::fs::read(root.join("source")).unwrap(), b"foreign");
    }

    #[test]
    fn retained_file_publication_rejects_same_inode_foreign_bytes_after_link() {
        let root = temporary_directory("retained-link-content-rewrite");
        std::fs::write(root.join("source"), b"trusted").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let source = anchored.open_regular(OsStr::new("source")).unwrap();
        let result = source.publish_as_new_with_hook(&anchored, OsStr::new("target"), || {
            rewrite_same_length_and_restore_mtime(&root.join("source"), b"foreign")
        });
        assert!(result.is_err());
        assert!(!root.join("target").exists());
        assert_eq!(std::fs::read(root.join("source")).unwrap(), b"foreign");
    }

    #[test]
    fn temporary_creation_skips_bounded_name_collisions_without_deleting_them() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-temp-collision-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("collision"), b"preserve").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let mut names = [OsStr::new("collision"), OsStr::new("available")].into_iter();

        let (name, file) = super::open_unique_temporary_with(anchored.fd.as_raw_fd(), || {
            Ok(names.next().unwrap().to_os_string())
        })
        .unwrap();

        assert_eq!(name, OsStr::new("available"));
        drop(file);
        assert_eq!(std::fs::read(root.join("collision")).unwrap(), b"preserve");
        assert!(root.join("available").exists());
    }

    #[test]
    fn temporary_retry_budget_exceeds_the_arena_preservation_cap() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-temp-pid-reuse-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let collisions = 256_usize;
        let names = (0..collisions)
            .map(|sequence| OsString::from(format!("collision-{sequence}")))
            .collect::<Vec<_>>();
        for name in &names {
            std::fs::write(root.join(name), b"foreign").unwrap();
        }
        let available = OsString::from("available");
        let mut candidates = names
            .iter()
            .cloned()
            .chain(std::iter::once(available.clone()));

        let (name, file) = super::open_unique_temporary_with(anchored.fd.as_raw_fd(), || {
            Ok(candidates.next().unwrap())
        })
        .unwrap();

        assert_eq!(name, available);
        drop(file);
        for collision in names {
            assert_eq!(std::fs::read(root.join(collision)).unwrap(), b"foreign");
        }
    }

    #[test]
    fn temporary_creation_exhaustion_preserves_every_colliding_entry() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-temp-exhaustion-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let names = (0..super::TEMPORARY_CREATE_ATTEMPTS)
            .map(|sequence| OsString::from(format!("collision-{sequence}")))
            .collect::<Vec<_>>();
        for (sequence, name) in names.iter().enumerate() {
            std::fs::write(root.join(name), format!("foreign-{sequence}")).unwrap();
        }
        let mut candidates = names.iter();

        let error = super::open_unique_temporary_with(anchored.fd.as_raw_fd(), || {
            Ok(candidates.next().unwrap().clone())
        })
        .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        for (sequence, name) in names.iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(root.join(name)).unwrap(),
                format!("foreign-{sequence}")
            );
        }
    }

    #[test]
    fn cleanup_authority_name_is_reserved_for_publication_targets() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-reserved-cleanup-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();

        let error = anchored
            .publish_new_atomic(OsStr::new(super::CLEANUP_DIRECTORY_NAME), b"collision")
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(std::fs::read_dir(root).unwrap().next().is_none());
    }

    #[test]
    fn retained_cleanup_preserves_a_replaced_path() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-cleanup-relink-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("pending");
        std::fs::write(&path, b"owned").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let retained = anchored.open_regular(OsStr::new("pending")).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"foreign").unwrap();

        assert!(anchored.remove_anchored_file(&retained).is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"foreign");
    }

    #[test]
    fn cleanup_quarantine_restores_a_between_check_and_remove_replacement() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-cleanup-race-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("pending");
        std::fs::write(&path, b"owned").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let retained = anchored.open_regular(OsStr::new("pending")).unwrap();

        let result = anchored.remove_anchored_file_with_hook(&retained, || {
            std::fs::remove_file(&path)?;
            std::fs::write(&path, b"foreign")
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"foreign");
        assert!(
            std::fs::read_dir(root.join(super::CLEANUP_DIRECTORY_NAME))
                .unwrap()
                .next()
                .is_none(),
            "the restored foreign entry must not remain duplicated in quarantine"
        );
    }

    #[test]
    fn interrupted_quarantine_is_preserved_bounded_and_not_reused() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-cleanup-crash-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("pending"), b"owned-before-crash").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let retained = anchored.open_regular(OsStr::new("pending")).unwrap();
        let identity = retained.stable_identity().unwrap();

        let error = anchored
            .quarantine_unlink_if_identity(
                OsStr::new("pending"),
                &identity,
                super::IdentityComparison::SameContents,
                || Ok(()),
                || {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "simulated process interruption after durable quarantine",
                    ))
                },
            )
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
        assert!(!root.join("pending").exists());
        assert_eq!(anchored.cleanup_quarantine_entry_count(1).unwrap(), 1);
        let cleanup = root.join(super::CLEANUP_DIRECTORY_NAME);
        let preserved = std::fs::read_dir(&cleanup)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert_eq!(std::fs::read(&preserved).unwrap(), b"owned-before-crash");

        std::fs::write(root.join("later"), b"later").unwrap();
        let later = anchored.open_regular(OsStr::new("later")).unwrap();
        anchored.remove_anchored_file(&later).unwrap();
        assert_eq!(anchored.cleanup_quarantine_entry_count(1).unwrap(), 1);
        assert_eq!(std::fs::read(preserved).unwrap(), b"owned-before-crash");
    }

    #[test]
    fn cleanup_never_unlinks_a_replacement_inside_the_private_quarantine() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-cleanup-private-race-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("pending"), b"owned").unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let retained = anchored.open_regular(OsStr::new("pending")).unwrap();
        let identity = retained.stable_identity().unwrap();
        let cleanup = root.join(super::CLEANUP_DIRECTORY_NAME);
        let mut replacement = None;

        let error = anchored
            .quarantine_unlink_if_identity(
                OsStr::new("pending"),
                &identity,
                super::IdentityComparison::SameContents,
                || Ok(()),
                || {
                    let path = std::fs::read_dir(&cleanup)?
                        .next()
                        .ok_or_else(|| std::io::Error::other("missing quarantine entry"))??
                        .path();
                    std::fs::remove_file(&path)?;
                    std::fs::write(&path, b"foreign")?;
                    replacement = Some(path);
                    Ok(())
                },
            )
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        let replacement = replacement.unwrap();
        assert_eq!(std::fs::read(&replacement).unwrap(), b"foreign");
        assert_eq!(anchored.cleanup_quarantine_entry_count(1).unwrap(), 1);
    }

    #[test]
    fn full_quarantine_fails_closed_until_manual_recovery_without_deleting_entries() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-anchored-cleanup-cap-{}-{}",
            std::process::id(),
            super::TEMPORARY_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let anchored = AnchoredDir::open_existing(&root).unwrap();
        let cleanup = anchored.cleanup_directory().unwrap();
        let cleanup_path = cleanup.display().to_path_buf();
        drop(cleanup);
        let mut entries = Vec::new();
        for sequence in 0..super::MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES {
            let name = OsString::from(format!(".entry-4242-{sequence}"));
            std::fs::write(cleanup_path.join(&name), format!("foreign-{sequence}")).unwrap();
            entries.push(name);
        }
        std::fs::write(root.join("pending"), b"pending").unwrap();
        let retained = anchored.open_regular(OsStr::new("pending")).unwrap();

        let error = anchored.remove_anchored_file(&retained).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("remove entries manually"));
        assert_eq!(std::fs::read(root.join("pending")).unwrap(), b"pending");
        assert_eq!(
            anchored
                .cleanup_quarantine_entry_count(super::MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES)
                .unwrap(),
            super::MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES
        );
        assert_eq!(
            std::fs::read(cleanup_path.join(&entries[1])).unwrap(),
            b"foreign-1"
        );

        // Recovery is explicit: once an operator removes one inspected remnant, cleanup can
        // proceed without consuming any other preserved entry.
        std::fs::remove_file(cleanup_path.join(&entries[0])).unwrap();
        anchored.remove_anchored_file(&retained).unwrap();
        assert_eq!(
            anchored
                .cleanup_quarantine_entry_count(super::MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES)
                .unwrap(),
            super::MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES - 1
        );
        assert_eq!(
            std::fs::read(cleanup_path.join(&entries[1])).unwrap(),
            b"foreign-1"
        );
    }
}
