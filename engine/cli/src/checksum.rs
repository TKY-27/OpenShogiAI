use std::{io::Read, path::Path};

use open_shogi_core::{AnchoredFile, StableFileIdentity};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct FileArtifact {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub size: u64,
}

pub fn read_file_artifact(path: &Path, maximum_bytes: u64) -> Result<FileArtifact, String> {
    let mut file = open_regular_readonly(path)?;
    read_file_artifact_from_open_file(&mut file, path, maximum_bytes, || Ok(()))
}

pub(crate) fn read_open_file_artifact(
    mut file: AnchoredFile,
    display_path: &Path,
    maximum_bytes: u64,
) -> Result<FileArtifact, String> {
    read_file_artifact_from_open_file(&mut file, display_path, maximum_bytes, || Ok(()))
}

pub(crate) fn read_open_file_artifact_with_check(
    mut file: AnchoredFile,
    display_path: &Path,
    maximum_bytes: u64,
    check: impl FnMut() -> Result<(), String>,
) -> Result<FileArtifact, String> {
    read_file_artifact_from_open_file_with_checks(
        &mut file,
        display_path,
        maximum_bytes,
        || Ok(()),
        check,
    )
}

pub(crate) fn read_retained_file_artifact(
    file: &mut AnchoredFile,
    display_path: &Path,
    maximum_bytes: u64,
) -> Result<FileArtifact, String> {
    read_file_artifact_from_open_file(file, display_path, maximum_bytes, || Ok(()))
}

fn read_file_artifact_from_open_file(
    file: &mut AnchoredFile,
    path: &Path,
    maximum_bytes: u64,
    before_read: impl FnOnce() -> Result<(), String>,
) -> Result<FileArtifact, String> {
    read_file_artifact_from_open_file_with_checks(file, path, maximum_bytes, before_read, || Ok(()))
}

fn read_file_artifact_from_open_file_with_checks(
    file: &mut AnchoredFile,
    path: &Path,
    maximum_bytes: u64,
    before_read: impl FnOnce() -> Result<(), String>,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<FileArtifact, String> {
    let identity = stable_identity(file, path)?;
    if identity.length() > maximum_bytes {
        return Err(format!(
            "{} exceeds the {maximum_bytes}-byte read limit",
            path.display()
        ));
    }
    before_read()?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(identity.length().min(maximum_bytes)).unwrap_or_default(),
    );
    {
        let mut reader = file.reader().take(maximum_bytes.saturating_add(1));
        let mut buffer = [0_u8; 8 * 1024];
        loop {
            check()?;
            let read = reader
                .read(&mut buffer)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..read]);
        }
    }
    check()?;
    let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if size > maximum_bytes {
        return Err(format!(
            "{} exceeds the {maximum_bytes}-byte read limit",
            path.display()
        ));
    }
    verify_stable_file(file, &identity, size, path)?;
    Ok(FileArtifact {
        sha256: sha256_bytes(&bytes),
        bytes,
        size,
    })
}

pub(crate) fn sha256_open_file_with_check(
    file: AnchoredFile,
    display_path: &Path,
    maximum_bytes: u64,
    check: impl FnMut() -> Result<(), String>,
) -> Result<(String, u64), String> {
    sha256_open_file_with_checks(file, display_path, maximum_bytes, || Ok(()), check)
}

#[cfg(test)]
fn sha256_open_file_with_size(
    file: AnchoredFile,
    path: &Path,
    maximum_bytes: u64,
    before_read: impl FnOnce() -> Result<(), String>,
) -> Result<(String, u64), String> {
    sha256_open_file_with_checks(file, path, maximum_bytes, before_read, || Ok(()))
}

fn sha256_open_file_with_checks(
    mut file: AnchoredFile,
    path: &Path,
    maximum_bytes: u64,
    before_read: impl FnOnce() -> Result<(), String>,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<(String, u64), String> {
    let identity = stable_identity(&file, path)?;
    if identity.length() > maximum_bytes {
        return Err(format!(
            "{} exceeds the {maximum_bytes}-byte hashing limit",
            path.display()
        ));
    }
    before_read()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];
    let mut total = 0_u64;
    loop {
        check()?;
        let read = file
            .reader()
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        if total > maximum_bytes {
            return Err(format!(
                "{} exceeds the {maximum_bytes}-byte hashing limit",
                path.display()
            ));
        }
        hasher.update(&buffer[..read]);
    }
    check()?;
    verify_stable_file(&file, &identity, total, path)?;
    Ok((format!("{:x}", hasher.finalize()), total))
}

pub fn sha256_bytes(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

pub fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
}

fn open_regular_readonly(path: &Path) -> Result<AnchoredFile, String> {
    AnchoredFile::open_existing(path).map_err(|error| {
        format!(
            "cannot open {} without following links: {error}",
            path.display()
        )
    })
}

fn verify_stable_file(
    file: &AnchoredFile,
    before: &StableFileIdentity,
    bytes_read: u64,
    path: &Path,
) -> Result<(), String> {
    file.verify_stable_read(before, bytes_read)
        .map_err(|error| {
            format!(
                "{} changed while it was being read: {error}",
                path.display()
            )
        })
}

fn stable_identity(file: &AnchoredFile, path: &Path) -> Result<StableFileIdentity, String> {
    file.stable_identity()
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{
        open_regular_readonly, read_file_artifact, read_file_artifact_from_open_file,
        sha256_open_file_with_check, sha256_open_file_with_size, sha256_text,
    };

    #[test]
    fn text_hash_is_canonical_sha256() {
        assert_eq!(
            sha256_text("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[cfg(unix)]
    #[test]
    fn artifact_reader_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory =
            std::env::temp_dir().join(format!("open-shogi-checksum-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("target");
        let link = directory.join("link");
        std::fs::write(&target, b"model").unwrap();
        let _ = std::fs::remove_file(&link);
        symlink(&target, &link).unwrap();

        assert!(read_file_artifact(&link, 100).is_err());

        std::fs::remove_file(link).unwrap();
        std::fs::remove_file(target).unwrap();
    }

    #[test]
    fn artifact_readers_reject_same_descriptor_mutation() {
        let directory = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-checksum-mutation-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("artifact");
        std::fs::write(&path, b"first-version").unwrap();

        let mut file = open_regular_readonly(&path).unwrap();
        let result = read_file_artifact_from_open_file(&mut file, &path, 100, || {
            std::fs::write(&path, b"other-version")
                .map_err(|error| format!("fixture mutation failed: {error}"))
        });
        assert!(
            result
                .unwrap_err()
                .contains("changed while it was being read")
        );

        std::fs::write(&path, b"first-version").unwrap();
        let file = open_regular_readonly(&path).unwrap();
        let result = sha256_open_file_with_size(file, &path, 100, || {
            std::fs::write(&path, b"other-version")
                .map_err(|error| format!("fixture mutation failed: {error}"))
        });
        assert!(
            result
                .unwrap_err()
                .contains("changed while it was being read")
        );

        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn streaming_hash_honors_a_bounded_progress_check() {
        let directory = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("open-shogi-checksum-budget-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("artifact");
        std::fs::write(&path, vec![7_u8; 32 * 1024]).unwrap();
        let file = open_regular_readonly(&path).unwrap();
        let mut checks = 0;

        let error = sha256_open_file_with_check(file, &path, 64 * 1024, || {
            checks += 1;
            if checks >= 3 {
                Err("verification deadline reached".to_owned())
            } else {
                Ok(())
            }
        })
        .unwrap_err();

        assert_eq!(error, "verification deadline reached");
        assert_eq!(checks, 3);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
