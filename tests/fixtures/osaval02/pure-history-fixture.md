# Synthetic history regression fixture

`pure-history.osaval02.gz` is a deterministic, untrained OSAVAL02 sparse-pair
int8 fixture. Its only active path maps repetition count (scalar index 57) to
the win logit. This makes history changes observable in the actual leaf score.
It is not an incumbent, candidate, teacher, or training-data artifact.

Recreate from the repository root with the existing Python runtime:

```python
import gzip
from pathlib import Path
from open_shogi_training.phase10r_model import (
    deterministic_test_tensors, serialize_osaval02, VARIANT_PAIR,
)
weights = deterministic_test_tensors(VARIANT_PAIR)
for tensor in weights.values():
    tensor.fill(0)
weights["scalar_projection.weight"][0, 57] = 1
weights["trunk.0.weight"][0, 16] = 1
weights["trunk.1.weight"][0, 0] = 1
weights["value_heads.weight"][2, 0] = 1
data = serialize_osaval02(
    weights, variant_id=VARIANT_PAIR, quantization="int8",
    training_run_reference="pure-history-regression-synthetic",
    git_commit="0" * 40,
)
Path("tests/fixtures/osaval02/pure-history.osaval02.gz").write_bytes(
    gzip.compress(data, mtime=0)
)
```

Tests decompress the 3.3 KB fixture into memory and run the normal strict loader,
including schema, descriptor and full checksum validation. The synthetic
zero-valued Git metadata is intentionally not a provenance claim.
