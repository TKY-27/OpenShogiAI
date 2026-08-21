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
from open_shogi_training.data.registry import (
    CatalogObject,
    DataSource,
    EvidenceObject,
    LicenseEvidence,
    SourceRegistry,
    load_source_registry,
)

__all__ = [
    "AcquisitionPlan",
    "CatalogObject",
    "CompletedObject",
    "DataSource",
    "DownloadOutcome",
    "Downloader",
    "EvidenceObject",
    "EvidenceSnapshot",
    "ExternalAuditFormatError",
    "LicenseEvidence",
    "ManifestStore",
    "SourceRegistry",
    "load_source_registry",
    "measure_exact_overlap",
    "measure_split_contamination",
    "parse_csa_sample",
    "parse_hcpe3_sample",
    "parse_hcpe_sample",
    "parse_kif_sample",
    "parse_packed_sfen_value_sample",
    "plan_acquisition",
    "sample_summary",
]
