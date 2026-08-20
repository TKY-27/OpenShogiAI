use open_shogi_core::ReproducibilitySeed;
use proptest::prelude::*;

proptest! {
    #[test]
    fn seed_encoding_round_trips(value: u64) {
        let encoded = ReproducibilitySeed::new(value).to_le_bytes();
        let decoded = ReproducibilitySeed::from_le_bytes(encoded);

        prop_assert_eq!(decoded.value(), value);
    }
}
