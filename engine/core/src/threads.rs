//! Search-worker sizing: topology- and load-aware automatic counts with a manual
//! override. Discovery failures fall back to topology rather than invented load numbers.

use std::num::NonZeroUsize;

/// Where the automatic worker count came from, for operator-visible diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerBasis {
    /// Logical CPUs usable by this process.
    pub logical: usize,
    /// Physical cores when discoverable, otherwise `logical`.
    pub physical: Option<usize>,
    /// One-minute load average when the platform reports one.
    pub load_average: Option<f64>,
}

impl WorkerBasis {
    /// Brief `key=value` description for `info string` output.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut parts = vec![format!("logical {}", self.logical)];
        if let Some(physical) = self.physical {
            parts.push(format!("physical {physical}"));
        }
        if let Some(load) = self.load_average {
            parts.push(format!("load {load:.2}"));
        }
        parts.join(" ")
    }

    /// Automatic worker count: physical cores with headroom for the host, shrunk under
    /// observed load. Never exceeds the usable logical count.
    ///
    /// The headroom is deliberate: a busy development machine keeps its desktop, build
    /// tools and GUI responsive while the engine searches.
    #[must_use]
    pub fn automatic_workers(&self) -> NonZeroUsize {
        let base = self.physical.unwrap_or(self.logical).min(self.logical);
        let mut workers = base.saturating_sub(1).max(1);
        if let Some(load) = self.load_average {
            let saturation =
                load / f64::from(u16::try_from(self.logical.max(1)).unwrap_or(u16::MAX));
            if saturation > 0.9 {
                workers = 1;
            } else if saturation >= 0.5 {
                workers = workers.div_ceil(2).max(1);
            }
        }
        NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN)
    }
}

/// Upper bound for manual `Threads` values: what the process can actually use.
#[must_use]
pub fn usable_logical_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

/// Topology and load snapshot for the automatic policy. Load discovery is best effort:
/// unsupported platforms simply report `None` and the policy uses topology alone.
#[must_use]
pub fn probe_worker_basis() -> WorkerBasis {
    WorkerBasis {
        logical: usable_logical_cpus(),
        physical: physical_core_count(),
        load_average: one_minute_load_average(),
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn physical_core_count() -> Option<usize> {
    let name = b"hw.perflevel0.physicalcpu\0";
    let mut value: libc::c_int = 0;
    let mut length = std::mem::size_of::<libc::c_int>();
    // `perflevel0` counts performance cores; efficiency cores add latency variance that
    // hurts parallel search more than the extra width helps.
    let ok = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast::<libc::c_char>(),
            (&raw mut value).cast(),
            (&raw mut length).cast::<usize>(),
            std::ptr::null_mut(),
            0,
        )
    };
    (ok == 0 && value > 0).then(|| usize::try_from(value).unwrap_or_default())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn physical_core_count() -> Option<usize> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut cores = std::collections::BTreeSet::new();
    let mut saw_topology = false;
    let mut physical_id = String::new();
    let mut core_id = String::new();
    for line in cpuinfo.lines() {
        let (key, value) = line.split_once(':')?;
        let key = key.trim();
        let value = value.trim();
        match (key, !physical_id.is_empty() && !core_id.is_empty()) {
            ("physical id", _) => value.clone_into(&mut physical_id),
            ("core id", _) => value.clone_into(&mut core_id),
            ("processor", true) => {
                saw_topology = true;
                cores.insert((physical_id.clone(), core_id.clone()));
            }
            _ => {}
        }
    }
    (saw_topology && !cores.is_empty()).then_some(cores.len())
}

#[cfg(not(unix))]
fn physical_core_count() -> Option<usize> {
    None
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn one_minute_load_average() -> Option<f64> {
    let mut averages = [0.0_f64; 3];
    let loaded = unsafe { libc::getloadavg(averages.as_mut_ptr(), 3) };
    (loaded >= 1).then(|| averages[0])
}

#[cfg(not(unix))]
fn one_minute_load_average() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_policy_leaves_headroom_and_never_exceeds_logical() {
        let basis = |logical, physical, load| WorkerBasis {
            logical,
            physical,
            load_average: load,
        };
        // Single logical CPU: one worker.
        assert_eq!(basis(1, None, None).automatic_workers().get(), 1);
        // Typical laptop: 10 physical / 20 logical leaves one logical of headroom.
        assert_eq!(basis(20, Some(10), None).automatic_workers().get(), 9);
        // SMT-only discovery falls back to logical minus one.
        assert_eq!(basis(8, None, None).automatic_workers().get(), 7);
        // At half saturation the plan halves; near saturation it drops to one worker.
        assert_eq!(basis(8, Some(8), Some(4.0)).automatic_workers().get(), 4);
        assert_eq!(basis(8, Some(8), Some(5.5)).automatic_workers().get(), 4);
        assert_eq!(basis(8, Some(8), Some(7.5)).automatic_workers().get(), 1);
        // Light load keeps the topology plan.
        assert_eq!(basis(8, Some(8), Some(3.0)).automatic_workers().get(), 7);
        // Physical discovery never invents workers beyond the usable logical count.
        assert_eq!(basis(4, Some(16), None).automatic_workers().get(), 3);
    }

    #[test]
    fn probe_reports_the_host_without_panicking() {
        let basis = probe_worker_basis();
        assert!(basis.logical >= 1);
        if let Some(physical) = basis.physical {
            assert!(physical >= 1);
        }
        if let Some(load) = basis.load_average {
            assert!(load >= 0.0);
        }
    }
}
