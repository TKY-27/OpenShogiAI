# OpenShogiAI opening-book format

Version: `open_shogi_opening_book/v2`

The book is a deterministic gzip-compressed JSON Lines artifact. Each line is one canonical
position record described by [`opening-book.schema.json`](opening-book.schema.json). The position
identity is the SHA-256 of `stateSfen`, where `stateSfen` omits the move number but retains board,
side to move, and hands. The only accepted rule profile is `standard-shogi/v1`.

Every candidate binds legal USI notation, sample and result counts, source distribution, teacher
score and uncertainty, teacher work, opening classification, and sorted SHA-256 provenance
references. A record's provenance is the union that covers every candidate. `recordChecksum` is
SHA-256 over compact UTF-8 JSON with sorted keys after removing `recordChecksum`.

The runtime fully decompresses within fixed byte/record limits, validates every checksum and count,
parses every canonical position, and confirms every move is legal before making any entry visible.
Duplicate positions or moves, incompatible rules, missing teacher evidence, corrupt gzip data, and
provenance mismatches reject the complete snapshot.

Maximum-strength selection is deterministic: candidates are sorted by teacher score, sample count,
then USI notation. The selected move must meet the minimum sample count, remain within the allowed
teacher loss from the best candidate, and satisfy the opening profile. Variety is not part of this
mode. A miss or unsafe entry falls back to unrestricted legal search.

`ibisha_strict` accepts only `ibisha` and `ibisha-vs-furibisha` classifications during the
configured opening phase. `ibisha_preferred` first selects the strongest safe candidate in those
classes, then permits the strongest safe unrestricted book candidate if none exists.
`unrestricted` accepts any teacher-safe classification. Profiles never modify move generation;
after a miss, all legal moves—including necessary later rook movement and all opponent Furibisha
replies—remain available to normal search.

Verify an artifact with:

```sh
cargo run --locked -p open-shogi-cli -- opening-book verify --book ARTIFACT.jsonl.gz
```
