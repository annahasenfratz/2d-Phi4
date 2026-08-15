from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSchedule:
    detail_patch_size: int
    coarse_patch_size: int
    detail_passes: int
    coarse_passes: int
    random_origin: bool = True


def from_config(config: dict) -> PatchSchedule:
    patch = config.get("patch", {})
    return PatchSchedule(
        detail_patch_size=int(patch["detail_patch_size"]),
        coarse_patch_size=int(patch["coarse_patch_size"]),
        detail_passes=int(patch["detail_passes"]),
        coarse_passes=int(patch["coarse_passes"]),
        random_origin=bool(patch.get("random_origin", True)),
    )

