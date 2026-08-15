from __future__ import annotations

from perfect_blocking_upsampling.checks import verify_sha256_manifest


def test_frozen_checkpoint_hashes_match():
    rows = verify_sha256_manifest("checkpoints/frozen/lam0p022_kappa0p2705_small3_refine/sha256_checksums.txt", root="checkpoints/frozen/lam0p022_kappa0p2705_small3_refine")
    assert rows
    assert all(r["matches"] for r in rows)

