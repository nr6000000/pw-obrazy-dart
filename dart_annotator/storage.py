"""JSON persistence for dartboard homography calibration and hit annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from dart_annotator.geometry import (
    CALIBRATION_TARGETS,
    RADII_MM,
    SCHEMATIC_CENTER_PX,
    SCHEMATIC_PX_PER_MM,
    SCHEMATIC_SIZE_PX,
)


@dataclass
class HitRecord:
    """A single registered dart hit with float coordinates across spaces."""

    id: int
    image_xy: tuple[float, float]
    schematic_xy: tuple[float, float]
    board_mm: tuple[float, float]
    source: str  # "image" or "schematic"
    sector: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "image_xy": [float(self.image_xy[0]), float(self.image_xy[1])],
            "schematic_xy": [float(self.schematic_xy[0]), float(self.schematic_xy[1])],
            "board_mm": [float(self.board_mm[0]), float(self.board_mm[1])],
            "source": str(self.source),
            "sector": str(self.sector),
            "score": int(self.score),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HitRecord:
        return cls(
            id=int(data["id"]),
            image_xy=(float(data["image_xy"][0]), float(data["image_xy"][1])),
            schematic_xy=(float(data["schematic_xy"][0]), float(data["schematic_xy"][1])),
            board_mm=(float(data["board_mm"][0]), float(data["board_mm"][1])),
            source=str(data.get("source", "image")),
            sector=str(data.get("sector", "")),
            score=int(data.get("score", 0)),
        )


@dataclass
class ImageAnnotation:
    """Annotation state for one dartboard image."""

    image_path: str
    image_size: tuple[int, int]  # (width, height)
    calibration_image_pts: list[tuple[float, float]] = field(default_factory=list)
    h_image_to_schematic: np.ndarray | None = None
    h_schematic_to_image: np.ndarray | None = None
    h_image_to_board_mm: np.ndarray | None = None
    hits: list[HitRecord] = field(default_factory=list)

    @property
    def is_calibrated(self) -> bool:
        return (
            len(self.calibration_image_pts) == 4
            and self.h_image_to_schematic is not None
            and self.h_schematic_to_image is not None
        )

    def to_dict(self) -> dict[str, Any]:
        cal_points = []
        for idx, img_pt in enumerate(self.calibration_image_pts):
            target = CALIBRATION_TARGETS[idx]
            sch_pt = target.schematic_xy()
            mm_pt = target.board_mm
            cal_points.append(
                {
                    "name": target.name,
                    "description": target.description,
                    "image_xy": [float(img_pt[0]), float(img_pt[1])],
                    "schematic_xy": [float(sch_pt[0]), float(sch_pt[1])],
                    "board_mm": [float(mm_pt[0]), float(mm_pt[1])],
                }
            )

        homography_data: dict[str, Any] | None = None
        if self.is_calibrated:
            homography_data = {
                "calibration_points": cal_points,
                "H_image_to_schematic": [
                    [float(v) for v in row] for row in self.h_image_to_schematic.tolist()
                ],
                "H_schematic_to_image": [
                    [float(v) for v in row] for row in self.h_schematic_to_image.tolist()
                ],
                "H_image_to_board_mm": (
                    [[float(v) for v in row] for row in self.h_image_to_board_mm.tolist()]
                    if self.h_image_to_board_mm is not None
                    else None
                ),
            }
        elif cal_points:
            homography_data = {
                "calibration_points": cal_points,
                "H_image_to_schematic": None,
                "H_schematic_to_image": None,
                "H_image_to_board_mm": None,
            }

        return {
            "image_path": self.image_path,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "homography": homography_data,
            "hits": [h.to_dict() for h in self.hits],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageAnnotation:
        hom = data.get("homography") or {}
        cal_pts: list[tuple[float, float]] = []
        for cp in hom.get("calibration_points") or []:
            xy = cp["image_xy"]
            cal_pts.append((float(xy[0]), float(xy[1])))

        def parse_mat(key: str) -> np.ndarray | None:
            raw = hom.get(key)
            if raw is None:
                return None
            return np.asarray(raw, dtype=np.float64).reshape(3, 3)

        size = data.get("image_size", [0, 0])
        hits = [HitRecord.from_dict(h) for h in (data.get("hits") or [])]

        return cls(
            image_path=str(data["image_path"]),
            image_size=(int(size[0]), int(size[1])),
            calibration_image_pts=cal_pts,
            h_image_to_schematic=parse_mat("H_image_to_schematic"),
            h_schematic_to_image=parse_mat("H_schematic_to_image"),
            h_image_to_board_mm=parse_mat("H_image_to_board_mm"),
            hits=hits,
        )


def build_schematic_spec() -> dict[str, Any]:
    """Return metadata describing the canonical schematic and mm coordinate frame."""
    return {
        "width_px": SCHEMATIC_SIZE_PX,
        "height_px": SCHEMATIC_SIZE_PX,
        "center_xy": [float(SCHEMATIC_CENTER_PX[0]), float(SCHEMATIC_CENTER_PX[1])],
        "px_per_mm": float(SCHEMATIC_PX_PER_MM),
        "board_mm_axes": "+x right (sector 6), +y down (sector 3), origin at bullseye center",
        "radii_mm": {k: float(v) for k, v in RADII_MM.items()},
        "calibration_targets": [
            {
                "name": t.name,
                "description": t.description,
                "schematic_xy": [float(t.schematic_xy()[0]), float(t.schematic_xy()[1])],
                "board_mm": [float(t.board_mm[0]), float(t.board_mm[1])],
            }
            for t in CALIBRATION_TARGETS
        ],
    }


def load_annotations(json_path: str | Path) -> dict[str, ImageAnnotation]:
    """Load existing annotations from JSON if present."""
    path = Path(json_path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    images_section = raw.get("images", {})
    result: dict[str, ImageAnnotation] = {}
    for key, entry in images_section.items():
        if "image_path" not in entry:
            entry = {**entry, "image_path": key}
        ann = ImageAnnotation.from_dict(entry)
        result[key] = ann
    return result


def save_annotations(
    json_path: str | Path,
    annotations: dict[str, ImageAnnotation],
) -> None:
    """Save all annotations to JSON atomically with full floating-point precision."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schematic_spec": build_schematic_spec(),
        "images": {key: ann.to_dict() for key, ann in annotations.items()},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
