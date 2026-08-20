"""Rights-gated acquisition primitives for audited public game records."""

from open_shogi_training.data.downloader import (
    AcquisitionPlan,
    Downloader,
    DownloadOutcome,
    plan_acquisition,
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
    "LicenseEvidence",
    "ManifestStore",
    "SourceRegistry",
    "load_source_registry",
    "plan_acquisition",
]
