"""Rights-gated acquisition primitives for audited public game records."""

from open_shogi_training.data.downloader import (
    AcquisitionPlan,
    Downloader,
    DownloadOutcome,
    plan_acquisition,
)
from open_shogi_training.data.external_audit import (
    ExternalAuditFormatError,
    measure_exact_overlap,
    measure_split_contamination,
    parse_csa_sample,
    parse_hcpe3_sample,
    parse_hcpe_sample,
    parse_kif_sample,
    parse_packed_sfen_value_sample,
    sample_summary,
)
from open_shogi_training.data.manifest import CompletedObject, EvidenceSnapshot, ManifestStore
from open_shogi_training.data.phase10r_acquisition import (
    DownloadResult,
    Phase10RAcquisitionError,
    acquire_artifact,
    check_free_space,
    data_root_from_environment,
)
from open_shogi_training.data.phase10r_acquisition import (
    dry_run as phase10r_dry_run,
)
from open_shogi_training.data.phase10r_adapters import (
    PHASE10R_NORMALIZED_SCHEMA,
    Phase10RAdapterError,
    normalize_source_bytes,
    normalize_source_file,
    validate_csa_with_engine,
    validate_kif_with_engine,
    write_normalized_jsonl,
)
from open_shogi_training.data.phase10r_archive import (
    ArchiveEntry,
    ArchiveInventory,
    Phase10RArchiveError,
    extract_zip,
    inventory_archive,
)
from open_shogi_training.data.phase10r_kif import KifConversionError, convert_kif_to_csa
from open_shogi_training.data.phase10r_overlap import (
    deduplicate_records,
    measure_phase10r_overlap,
    split_name,
)
from open_shogi_training.data.phase10r_registry import (
    DECISION_STATES,
    Phase10RArtifact,
    Phase10RRegistry,
    Phase10RRegistryError,
    Phase10RSource,
    load_phase10r_registry,
)
from open_shogi_training.data.registry import (
    CatalogObject,
    DataSource,
    EvidenceObject,
    LicenseEvidence,
    SourceRegistry,
    load_source_registry,
)

__all__ = [
    "DECISION_STATES",
    "PHASE10R_NORMALIZED_SCHEMA",
    "AcquisitionPlan",
    "ArchiveEntry",
    "ArchiveInventory",
    "CatalogObject",
    "CompletedObject",
    "DataSource",
    "DownloadOutcome",
    "DownloadResult",
    "Downloader",
    "EvidenceObject",
    "EvidenceSnapshot",
    "ExternalAuditFormatError",
    "KifConversionError",
    "LicenseEvidence",
    "ManifestStore",
    "Phase10RAcquisitionError",
    "Phase10RAdapterError",
    "Phase10RArchiveError",
    "Phase10RArtifact",
    "Phase10RRegistry",
    "Phase10RRegistryError",
    "Phase10RSource",
    "SourceRegistry",
    "acquire_artifact",
    "check_free_space",
    "convert_kif_to_csa",
    "data_root_from_environment",
    "deduplicate_records",
    "extract_zip",
    "inventory_archive",
    "load_phase10r_registry",
    "load_source_registry",
    "measure_exact_overlap",
    "measure_phase10r_overlap",
    "measure_split_contamination",
    "normalize_source_bytes",
    "normalize_source_file",
    "parse_csa_sample",
    "parse_hcpe3_sample",
    "parse_hcpe_sample",
    "parse_kif_sample",
    "parse_packed_sfen_value_sample",
    "phase10r_dry_run",
    "plan_acquisition",
    "sample_summary",
    "split_name",
    "validate_csa_with_engine",
    "validate_kif_with_engine",
    "write_normalized_jsonl",
]
