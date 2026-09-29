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

    /// Topology baseline: physical cores with one logical CPU of headroom for the host,
    /// shrunk to the usable logical count. The headroom is deliberate: a busy development
    /// machine keeps its desktop, build tools and GUI responsive while the engine searches.
    #[must_use]
    pub fn automatic_workers(&self) -> NonZeroUsize {
        let base = self.physical.unwrap_or(self.logical).min(self.logical);
        NonZeroUsize::new(base.saturating_sub(1).max(1)).unwrap_or(NonZeroUsize::MIN)
    }
}

/// External-load bands for the automatic policy, as a fraction of the usable logical
/// CPUs. Each adjacent pair of bands is separated by a deadband: one borderline sample
/// must never flip the plan between adjacent moves.
const LOAD_BUSY_SATURATION: f64 = 0.5;
const LOAD_RECOVER_SATURATION: f64 = 0.25;
const LOAD_SATURATED_SATURATION: f64 = 0.9;
const LOAD_UNLOAD_SATURATION: f64 = 0.75;

/// Sticky automatic worker plan across a session's searches.
///
/// A search that just used many workers raises the one-minute load average itself, so
/// the raw load is never interpreted as external contention: the estimate subtracts the
/// worker count this plan used on the previous search. Our own contribution to the load
/// is bounded by that count, so the subtraction can only under-detect real contention,
/// never manufacture it, and a heavy search cannot read back as an external load spike.
///
/// Band transitions are a Schmitt trigger with a wide deadband: downscaling reacts at
/// one threshold, recovery needs clearly unloaded conditions, and recovery from the
/// saturated floor is stepped (`saturated -> busy -> baseline`). Plans are recomputed
/// only at search boundaries, so adjacent moves never oscillate.
#[derive(Clone, Debug)]
pub struct AutoWorkerPolicy {
    workers: usize,
    band: LoadBand,
    /// Whether at least one search boundary has passed. Until then none of our searches
    /// has contributed to the load average, so nothing may be subtracted for one.
    searched: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LoadBand {
    Baseline,
    Busy,
    Saturated,
}

impl AutoWorkerPolicy {
    /// Initializes the plan from the first probe, treating the whole observed load as
    /// external because no search of ours has run yet.
    #[must_use]
    pub fn new(basis: &WorkerBasis) -> Self {
        let mut policy = Self {
            workers: 1,
            band: LoadBand::Baseline,
            searched: false,
        };
        policy.workers = policy.next_workers(basis).0;
        // The constructor's probe is not a search: the first real search boundary must
        // still subtract nothing.
        policy.searched = false;
        policy
    }

    /// The worker count chosen for the previous search.
    #[must_use]
    pub fn current_workers(&self) -> usize {
        self.workers
    }

    /// Chooses this search's worker count from a fresh probe. Manual overrides bypass
    /// this policy entirely and never update its state.
    #[must_use]
    pub fn next_workers(&mut self, basis: &WorkerBasis) -> (usize, String) {
        let baseline = basis.automatic_workers().get();
        let busy = baseline.div_ceil(2).max(1);
        let own = if self.searched { self.workers } else { 0 };
        let saturation = external_saturation(basis, own);
        self.band = match saturation {
            None => LoadBand::Baseline,
            Some(value) if value >= LOAD_SATURATED_SATURATION => LoadBand::Saturated,
            Some(value) => match self.band {
                LoadBand::Baseline if value >= LOAD_BUSY_SATURATION => LoadBand::Busy,
                LoadBand::Busy if value < LOAD_RECOVER_SATURATION => LoadBand::Baseline,
                // Stepped recovery: the saturated floor returns to busy first, so a
                // transient load spike cannot produce a two-step thrash.
                LoadBand::Saturated if value < LOAD_UNLOAD_SATURATION => LoadBand::Busy,
                _ => self.band,
            },
        };
        self.workers = match self.band {
            LoadBand::Baseline => baseline,
            LoadBand::Busy => busy,
            LoadBand::Saturated => 1,
        };
        self.searched = true;
        let label = match saturation {
            Some(value) => format!("{} external {value:.2}", basis.describe()),
            None => basis.describe(),
        };
        (self.workers, label)
    }
}

/// One-minute load with the engine's own previous worker count removed, divided by the
/// usable logical CPUs. Our workers contributed at most that count to the load, so the
/// remainder is a lower bound on genuinely external contention.
fn external_saturation(basis: &WorkerBasis, own_workers: usize) -> Option<f64> {
    if basis.logical == 0 {
        return None;
    }
    let external = basis.load_average? - f64::from(u16::try_from(own_workers).unwrap_or(u16::MAX));
    Some((external.max(0.0)) / f64::from(u16::try_from(basis.logical).unwrap_or(u16::MAX)))
}

/// Upper bound for manual `Threads` values: what the process can actually use.
#[must_use]
pub fn usable_logical_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

/// Topology and load snapshot for the automatic policy. Discovery is best effort:
/// unsupported platforms simply report `None` and the policy uses topology alone.
#[must_use]
pub fn probe_worker_basis() -> WorkerBasis {
    let logical = usable_logical_cpus();
    WorkerBasis {
        logical,
        // Usable logical CPUs bound the topology claim: a cpuset-restricted process on a
        // large host must not report more cores than it can actually run on.
        physical: physical_core_count().map(|physical| physical.min(logical)),
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

/// Counts unique physical-package/core pairs in `/proc/cpuinfo` text.
///
/// Real files separate per-processor blocks with blank lines and mix in dozens of
/// unrelated keys, so blank or colon-less lines are noise, never parse failure. A block
/// contributes a core only when it carries both a `physical id` and a `core id`; kernels
/// that omit either (some ARM systems, unusual containers) prove no topology and yield
/// `None`, so callers fall back to the logical count instead of inventing cores. Blocks
/// end at a blank line or at the next `processor` line, whichever comes first, so both
/// the real layout and separator-less variants parse identically.
#[cfg(unix)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn parse_physical_core_count(cpuinfo: &str) -> Option<usize> {
    let mut cores = std::collections::BTreeSet::<(String, String)>::new();
    let mut physical_id = String::new();
    let mut core_id = String::new();
    let mut in_block = false;
    let flush_block = |cores: &mut std::collections::BTreeSet<(String, String)>,
                       physical_id: &mut String,
                       core_id: &mut String,
                       in_block: &mut bool| {
        if !physical_id.is_empty() && !core_id.is_empty() {
            cores.insert((std::mem::take(physical_id), std::mem::take(core_id)));
        }
        physical_id.clear();
        core_id.clear();
        *in_block = false;
    };
    for line in cpuinfo.lines() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            flush_block(&mut cores, &mut physical_id, &mut core_id, &mut in_block);
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            // A keyless line is noise between fields, not a parse failure.
            continue;
        };
        match key.trim() {
            "processor" => {
                if in_block {
                    flush_block(&mut cores, &mut physical_id, &mut core_id, &mut in_block);
                }
                in_block = true;
            }
            "physical id" => (value.trim()).clone_into(&mut physical_id),
            "core id" => (value.trim()).clone_into(&mut core_id),
            _ => {}
        }
    }
    flush_block(&mut cores, &mut physical_id, &mut core_id, &mut in_block);
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn physical_core_count() -> Option<usize> {
    parse_physical_core_count(&std::fs::read_to_string("/proc/cpuinfo").ok()?)
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
        // Physical discovery never invents workers beyond the usable logical count.
        assert_eq!(basis(4, Some(16), None).automatic_workers().get(), 3);
    }

    #[test]
    fn probe_reports_the_host_without_panicking() {
        let basis = probe_worker_basis();
        assert!(basis.logical >= 1);
        if let Some(physical) = basis.physical {
            assert!(physical >= 1);
            assert!(physical <= basis.logical);
        }
        if let Some(load) = basis.load_average {
            assert!(load >= 0.0);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_probe_discovers_real_topology() {
        // GitHub and developer x86_64 hosts expose `physical id`/`core id`; a regression
        // back to `None` here is exactly the blank-line parse failure this module fixed.
        let basis = probe_worker_basis();
        let physical = basis
            .physical
            .unwrap_or_else(|| panic!("no physical topology on a real Linux host"));
        assert!(physical >= 1 && physical <= basis.logical);
    }

    fn basis(logical: usize, physical: Option<usize>, load: Option<f64>) -> WorkerBasis {
        WorkerBasis {
            logical,
            physical,
            load_average: load,
        }
    }

    #[test]
    fn own_previous_search_is_never_external_contention() {
        // 12 physical / 24 logical host; the engine searched with 11 workers and the
        // one-minute load reads 11.9. All of that is ours: the plan must stand.
        let mut policy = AutoWorkerPolicy::new(&basis(24, Some(12), None));
        assert_eq!(policy.next_workers(&basis(24, Some(12), None)).0, 11);
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(9.0))).0, 11);
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(11.0))).0, 11);
        // Even a load equal to our whole worker count changes nothing.
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(11.9))).0, 11);
    }

    #[test]
    fn first_search_treats_pre_existing_load_as_fully_external() {
        // Session construction on a quiet host, then a build saturates the machine
        // before the first go: nothing of ours has run, so the raw load decides and
        // the plan must hit the one-worker floor instead of subtracting a search that
        // never happened.
        let mut policy = AutoWorkerPolicy::new(&basis(24, Some(12), None));
        assert_eq!(policy.current_workers(), 11);
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(23.0))).0, 1);
    }

    #[test]
    fn external_contention_scales_down_fast_and_recovers_in_steps() {
        let mut policy = AutoWorkerPolicy::new(&basis(24, Some(12), None));
        assert_eq!(policy.next_workers(&basis(24, Some(12), None)).0, 11);
        // A 23-load build on top of our 11 workers leaves 12 external runnables: 0.5
        // saturation drops us to busy immediately.
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(23.0))).0, 6);
        // Six of ours change the subtraction: 17.5 now reads 0.48, inside the deadband,
        // so the reduced plan stands until conditions clearly change.
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(17.5))).0, 6);
        // A genuinely saturated host drops the plan to one worker.
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(33.0))).0, 1);
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(33.0))).0, 1);
        // Recovery steps busy first and then needs clearly unloaded conditions.
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(16.9))).0, 6);
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(17.5))).0, 6);
        assert_eq!(policy.next_workers(&basis(24, Some(12), Some(11.9))).0, 11);
    }

    #[test]
    fn adjacent_moves_never_oscillate_under_alternating_borderline_load() {
        let mut policy = AutoWorkerPolicy::new(&basis(8, Some(8), None));
        assert_eq!(policy.next_workers(&basis(8, Some(8), None)).0, 7);
        // With seven of eight logicals ours, external saturation stays under the busy
        // band no matter how the raw load wiggles below 11: the plan must not move.
        for load in [7.0, 9.5, 10.9, 8.0, 10.0, 9.0, 8.5, 10.99] {
            assert_eq!(
                policy.next_workers(&basis(8, Some(8), Some(load))).0,
                7,
                "load {load} must not move the plan"
            );
        }
    }

    #[test]
    fn unavailable_load_and_topology_fall_back_to_topology_alone() {
        // No load signal: pure topology, stable across moves.
        let mut policy = AutoWorkerPolicy::new(&basis(24, Some(12), None));
        assert_eq!(policy.next_workers(&basis(24, Some(12), None)).0, 11);
        assert_eq!(policy.next_workers(&basis(24, Some(12), None)).0, 11);
        // No physical topology either (ARM-style /proc/cpuinfo): logical minus one.
        let mut policy = AutoWorkerPolicy::new(&basis(8, None, None));
        assert_eq!(policy.next_workers(&basis(8, None, None)).0, 7);
        // A superseded session's worker count cannot make the subtraction go negative
        // and freeze the plan at one worker forever.
        let mut policy = AutoWorkerPolicy::new(&basis(8, None, None));
        policy.searched = true;
        policy.workers = 64;
        assert_eq!(policy.next_workers(&basis(8, None, Some(7.5))).0, 7);
    }

    #[cfg(unix)]
    fn pair_block(processor: &str, physical: &str, core: &str) -> String {
        format!(
            "processor\t: {processor}\nvendor_id\t: AuthenticAMD\ncpu family\t: 25\n\
             model name\t: AMD Ryzen 9 7900X 12-Core Processor\nphysical id\t: {physical}\n\
             siblings\t: 24\ncore id\t\t: {core}\ncpu cores\t: 12\nflags\t\t: fpu vme de pse\n\
             power management: ts ttp tm hwpstate cpb\n\n"
        )
    }

    #[cfg(unix)]
    fn single_socket_12c24t_cpuinfo() -> String {
        // SMT siblings share a core id: processors 0..24 map to core ids 0..11 twice,
        // exactly the layout that used to abort parsing on its first blank line.
        (0..24)
            .map(|processor| pair_block(&processor.to_string(), "0", &(processor / 2).to_string()))
            .collect()
    }

    #[cfg(unix)]
    fn dual_socket_2x4c_cpuinfo() -> String {
        (0..16)
            .map(|processor| {
                pair_block(
                    &processor.to_string(),
                    &(processor / 8).to_string(),
                    &((processor % 8) / 2).to_string(),
                )
            })
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn realistic_cpuinfo_counts_unique_pairs_across_blank_lines() {
        assert_eq!(
            parse_physical_core_count(&single_socket_12c24t_cpuinfo()),
            Some(12)
        );
        assert_eq!(
            parse_physical_core_count(&dual_socket_2x4c_cpuinfo()),
            Some(8)
        );
        let dual = dual_socket_2x4c_cpuinfo().replace("physical id\t: 1", "physical id\t: 0");
        // Same core ids on both sockets collapse when the packages are indistinguishable.
        assert_eq!(parse_physical_core_count(&dual), Some(4));
    }

    #[cfg(unix)]
    #[test]
    fn separator_less_blocks_still_parse_through_the_processor_boundary() {
        let mut flat = String::new();
        for processor in 0..4 {
            let block = pair_block(&processor.to_string(), "0", &(processor / 2).to_string());
            flat.push_str(block.trim_end_matches('\n'));
            flat.push('\n');
        }
        assert_eq!(parse_physical_core_count(&flat), Some(2));
    }

    #[cfg(unix)]
    #[test]
    fn incomplete_topology_proves_nothing() {
        // ARM-style kernels list processors without package/core fields.
        let arm = "processor\t: 0\nmodel name\t: ARMv8 Processor rev 4\n\n\
                   processor\t: 1\nmodel name\t: ARMv8 Processor rev 4\n\n";
        assert_eq!(parse_physical_core_count(arm), None);
        // Core ids without packages cannot distinguish sockets, so they prove nothing.
        let cores_only = "processor\t: 0\ncore id\t\t: 0\n\nprocessor\t: 1\ncore id\t\t: 1\n\n";
        assert_eq!(parse_physical_core_count(cores_only), None);
        // A package field without core ids is equally unprovable.
        let packages_only =
            "processor\t: 0\nphysical id\t: 0\n\nprocessor\t: 1\nphysical id\t: 0\n\n";
        assert_eq!(parse_physical_core_count(packages_only), None);
        assert_eq!(parse_physical_core_count(""), None);
    }

    #[cfg(unix)]
    #[test]
    fn keyless_noise_and_crlf_never_abort_the_parse() {
        let mut noisy = String::from("\n# garbled kernel noise without a colon\n\r\n");
        noisy.push_str(&pair_block("0", "0", "0"));
        noisy.push_str("unexpected line without colon\n");
        noisy.push_str(&pair_block("1", "0", "0"));
        noisy.push_str(&pair_block("2", "0", "1"));
        assert_eq!(parse_physical_core_count(&noisy), Some(2));
        // A single processor block counts exactly one core.
        assert_eq!(
            parse_physical_core_count(&pair_block("0", "0", "3")),
            Some(1)
        );
    }
}
