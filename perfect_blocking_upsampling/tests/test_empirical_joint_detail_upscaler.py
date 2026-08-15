from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking_upsampling" / "src"))

from perfect_blocking_upsampling.empirical_joint_detail_upscaler import (  # noqa: E402
    EmpiricalJointDetailUpscaler,
    _detail_vectors,
    _metadata,
    apply_radial_calibration,
    load_config,
)


CONFIG_PATH = ROOT / "perfect_blocking_upsampling/configs/lam1p0/calibrated_empirical_upscaler.json"


def test_joint_detail_flattening_is_sector_then_dy_dx() -> None:
    detail = np.zeros((1, 3, 4, 4), dtype=np.float32)
    for sector in range(3):
        for y in range(4):
            for x in range(4):
                detail[0, sector, y, x] = 100 * sector + 10 * y + x
    vectors = _detail_vectors(detail, _metadata(1, 4))
    np.testing.assert_array_equal(vectors[0], np.array([0, 1, 10, 11, 100, 101, 110, 111, 200, 201, 210, 211]))


def test_radial_calibration_scales_only_edge_sectors_and_uses_volume_rule() -> None:
    detail16 = np.ones((2, 3, 16, 16), dtype=np.float32)
    calibrated16, gamma16 = apply_radial_calibration(detail16, np.array([0.0, 1.0], dtype=np.float32), mean_scale=.97, sigma_coefficient=.32)
    np.testing.assert_allclose(gamma16[0], .97)
    np.testing.assert_allclose(gamma16[1], .97 * np.exp(.02), rtol=1.e-6)
    np.testing.assert_allclose(calibrated16[:, 2], detail16[:, 2])

    detail32 = np.ones((1, 3, 32, 32), dtype=np.float32)
    _, gamma32 = apply_radial_calibration(detail32, np.array([1.0], dtype=np.float32), mean_scale=.97, sigma_coefficient=.32)
    np.testing.assert_allclose(gamma32[0], .97 * np.exp(.01), rtol=1.e-6)


def test_empirical_upscaler_is_deterministic_and_reblocks_at_l16_and_l32() -> None:
    config = load_config(CONFIG_PATH, ROOT)
    upscaler = EmpiricalJointDetailUpscaler(config)
    for lc in (16, 32):
        coarse = np.zeros((2, lc, lc), dtype=np.float32)
        coarse[0, lc // 2, lc // 2] = .1
        coarse[1, lc // 3, lc // 4] = -.2
        first = upscaler.sample(coarse, np.random.default_rng(1234))
        second = upscaler.sample(coarse, np.random.default_rng(1234))
        fine, detail, z, metadata = first
        np.testing.assert_array_equal(fine, second[0])
        np.testing.assert_array_equal(detail, second[1])
        np.testing.assert_array_equal(z, second[2])
        assert fine.shape == (2, 2 * lc, 2 * lc)
        assert detail.shape == (2, 3, lc, lc)
        assert metadata["reblocking_error"] <= config.reblocking_tolerance
        assert metadata["radial_sigma"] == .32 / lc
        assert np.all((metadata["selected_donor_blocks"] >= 0) & (metadata["selected_donor_blocks"] < len(upscaler.donor_d)))

