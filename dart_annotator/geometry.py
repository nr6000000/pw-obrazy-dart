"""Standard dartboard geometry, homography math, scoring, and schematic rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

# Clockwise sector numbers starting from 12 o'clock (top)
SECTOR_ORDER: tuple[int, ...] = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)

# Official BDO / WDF / PDC steel-tip dartboard wire radii in millimeters
RADII_MM: dict[str, float] = {
    "inner_bull": 6.35,
    "outer_bull": 15.9,
    "triple_inner": 99.0,
    "triple_outer": 107.0,
    "double_inner": 162.0,
    "double_outer": 170.0,
    "number_ring_inner": 195.0,
    "board_outer": 225.0,
}

# Schematic canvas configuration: crisp high-res canvas with exact linear mm mapping
SCHEMATIC_SIZE_PX: int = 900
SCHEMATIC_CENTER_PX: tuple[float, float] = (450.0, 450.0)
SCHEMATIC_PX_PER_MM: float = 1.92  # 225 mm * 1.92 = 432 px radius (fits comfortably in 900x900)


@dataclass(frozen=True)
class CalibrationTarget:
    """One of the 4 predefined calibration points on the dartboard."""

    name: str
    description: str
    angle_deg_cw_from_top: float
    radius_mm: float

    @property
    def board_mm(self) -> tuple[float, float]:
        """Return (x_mm, y_mm) with +x right (sector 6) and +y down (sector 3)."""
        theta = math.radians(self.angle_deg_cw_from_top - 90.0)
        return (
            self.radius_mm * math.cos(theta),
            self.radius_mm * math.sin(theta),
        )

    def schematic_xy(
        self,
        center_px: tuple[float, float] = SCHEMATIC_CENTER_PX,
        px_per_mm: float = SCHEMATIC_PX_PER_MM,
    ) -> tuple[float, float]:
        """Return exact floating-point (x, y) on the schematic image."""
        return board_mm_to_schematic_xy(self.board_mm, center_px=center_px, px_per_mm=px_per_mm)


# 4 orthogonal wire intersections on the outer Double ring (r = 170.0 mm).
# Exact 90-degree spacing makes the homography maximally well-conditioned.
CALIBRATION_TARGETS: tuple[CalibrationTarget, ...] = (
    CalibrationTarget(
        name="outer_double_5_20",
        description="Outer DOUBLE wire between 5 and 20 (top)",
        angle_deg_cw_from_top=-9.0,
        radius_mm=RADII_MM["double_outer"],
    ),
    CalibrationTarget(
        name="outer_double_13_6",
        description="Outer DOUBLE wire between 13 and 6 (right)",
        angle_deg_cw_from_top=81.0,
        radius_mm=RADII_MM["double_outer"],
    ),
    CalibrationTarget(
        name="outer_double_17_3",
        description="Outer DOUBLE wire between 17 and 3 (bottom)",
        angle_deg_cw_from_top=171.0,
        radius_mm=RADII_MM["double_outer"],
    ),
    CalibrationTarget(
        name="outer_double_8_11",
        description="Outer DOUBLE wire between 8 and 11 (left)",
        angle_deg_cw_from_top=261.0,
        radius_mm=RADII_MM["double_outer"],
    ),
)


def board_mm_to_schematic_xy(
    pt_mm: Sequence[float],
    center_px: tuple[float, float] = SCHEMATIC_CENTER_PX,
    px_per_mm: float = SCHEMATIC_PX_PER_MM,
) -> tuple[float, float]:
    """Convert board mm (+x right, +y down) to float schematic pixel coordinates."""
    return (
        float(center_px[0] + float(pt_mm[0]) * px_per_mm),
        float(center_px[1] + float(pt_mm[1]) * px_per_mm),
    )


def schematic_xy_to_board_mm(
    pt_sch: Sequence[float],
    center_px: tuple[float, float] = SCHEMATIC_CENTER_PX,
    px_per_mm: float = SCHEMATIC_PX_PER_MM,
) -> tuple[float, float]:
    """Convert float schematic pixel coordinates to board mm (+x right, +y down)."""
    return (
        float((float(pt_sch[0]) - center_px[0]) / px_per_mm),
        float((float(pt_sch[1]) - center_px[1]) / px_per_mm),
    )


def compute_homographies(
    image_pts: Sequence[Sequence[float]],
    schematic_pts: Sequence[Sequence[float]],
    center_px: tuple[float, float] = SCHEMATIC_CENTER_PX,
    px_per_mm: float = SCHEMATIC_PX_PER_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact float64 homographies from 4 point correspondences.

    Returns (H_image_to_schematic, H_schematic_to_image, H_image_to_board_mm).
    """
    if len(image_pts) != 4 or len(schematic_pts) != 4:
        raise ValueError("Exactly 4 point correspondences are required.")

    src = np.asarray(image_pts, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(schematic_pts, dtype=np.float64).reshape(4, 2)

    h_img_to_sch, _ = cv2.findHomography(src, dst, method=0)
    if h_img_to_sch is None or abs(float(np.linalg.det(h_img_to_sch))) < 1e-12:
        raise ValueError("Calibration points are degenerate or collinear; cannot compute homography.")
    h_img_to_sch = np.asarray(h_img_to_sch, dtype=np.float64)
    h_sch_to_img = np.linalg.inv(h_img_to_sch)

    # Schematic -> board_mm is pure translation + uniform scale
    s = 1.0 / px_per_mm
    t_sch_to_mm = np.array(
        [
            [s, 0.0, -center_px[0] * s],
            [0.0, s, -center_px[1] * s],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h_img_to_mm = t_sch_to_mm @ h_img_to_sch
    return h_img_to_sch, h_sch_to_img, h_img_to_mm


def project_point(H: np.ndarray, pt: Sequence[float]) -> tuple[float, float] | None:
    """Project a single 2D float point through a 3x3 homography matrix."""
    vec = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
    out = H @ vec
    w = float(out[2])
    if abs(w) < 1e-12:
        return None
    return (float(out[0] / w), float(out[1] / w))


def classify_board_mm(x_mm: float, y_mm: float) -> tuple[str, int]:
    """Return (sector_label, score) for a board coordinate in millimeters."""
    r = math.hypot(x_mm, y_mm)
    if r <= RADII_MM["inner_bull"]:
        return ("DBULL", 50)
    if r <= RADII_MM["outer_bull"]:
        return ("BULL", 25)
    if r > RADII_MM["double_outer"]:
        return ("MISS", 0)

    # Angle clockwise from 12 o'clock (0 deg = top, 90 deg = right)
    angle_cw = (math.degrees(math.atan2(y_mm, x_mm)) + 90.0) % 360.0
    sector_idx = int(((angle_cw + 9.0) % 360.0) // 18.0)
    base = SECTOR_ORDER[sector_idx]

    if RADII_MM["triple_inner"] <= r <= RADII_MM["triple_outer"]:
        return (f"T{base}", 3 * base)
    if RADII_MM["double_inner"] <= r <= RADII_MM["double_outer"]:
        return (f"D{base}", 2 * base)
    return (f"S{base}", base)


def render_schematic(
    size_px: int = SCHEMATIC_SIZE_PX,
    center_px: tuple[float, float] = SCHEMATIC_CENTER_PX,
    px_per_mm: float = SCHEMATIC_PX_PER_MM,
) -> np.ndarray:
    """Render a clean black-on-white standard dartboard schematic image."""
    # Render at 2x supersampled resolution and downsample for crisp anti-aliased wires & text
    ss = 2
    img = np.full((size_px * ss, size_px * ss, 3), 255, dtype=np.uint8)
    cx = center_px[0] * ss
    cy = center_px[1] * ss
    scale = px_per_mm * ss

    shift = 4
    factor = 1 << shift

    def to_subpix(x: float, y: float) -> tuple[int, int]:
        return (int(round(x * factor)), int(round(y * factor)))

    black = (25, 25, 25)
    subtle_fill = (242, 242, 242)

    # Subtle shading on sector 20's double/triple/wedge? Keep clean white with crisp wires
    # like the reference schematic, plus subtle light tint on double/triple rings?
    # Actually keeping pure white background with crisp black lines matches the schematic cleanly.
    _ = subtle_fill

    # Concentric circles
    ring_keys = (
        ("inner_bull", 2),
        ("outer_bull", 2),
        ("triple_inner", 2),
        ("triple_outer", 2),
        ("double_inner", 2),
        ("double_outer", 3),
        ("board_outer", 2),
    )
    center_sp = to_subpix(cx, cy)
    for key, thickness in ring_keys:
        radius_sp = int(round(RADII_MM[key] * scale * factor))
        cv2.circle(
            img,
            center_sp,
            radius_sp,
            black,
            thickness=thickness * ss // 2 + 1,
            lineType=cv2.LINE_AA,
            shift=shift,
        )

    # 20 radial wires from outer bull to outer double
    r_start = RADII_MM["outer_bull"] * scale
    r_end = RADII_MM["double_outer"] * scale
    for k in range(20):
        angle_rad = math.radians(-90.0 - 9.0 + 18.0 * k)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        p1 = to_subpix(cx + r_start * cos_a, cy + r_start * sin_a)
        p2 = to_subpix(cx + r_end * cos_a, cy + r_end * sin_a)
        cv2.line(img, p1, p2, black, thickness=2, lineType=cv2.LINE_AA, shift=shift)

    # Sector numbers centered in each sector between double_outer and board_outer
    r_text = (RADII_MM["double_outer"] + 27.0) * scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.45
    font_thickness = 3
    for k, num in enumerate(SECTOR_ORDER):
        angle_rad = math.radians(-90.0 + 18.0 * k)
        tx = cx + r_text * math.cos(angle_rad)
        ty = cy + r_text * math.sin(angle_rad)
        label = str(num)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        org = (int(round(tx - tw / 2.0)), int(round(ty + th / 2.0)))
        cv2.putText(img, label, org, font, font_scale, black, font_thickness, cv2.LINE_AA)

    return cv2.resize(img, (size_px, size_px), interpolation=cv2.INTER_AREA)


def generate_wireframe_polylines_sch(
    center_px: tuple[float, float] = SCHEMATIC_CENTER_PX,
    px_per_mm: float = SCHEMATIC_PX_PER_MM,
    circle_segments: int = 120,
) -> list[np.ndarray]:
    """Return schematic polylines (float64 Nx2) for projecting board wires onto the photo."""
    polylines: list[np.ndarray] = []
    angles = np.linspace(0.0, 2.0 * math.pi, circle_segments, endpoint=True)
    for key in ("inner_bull", "outer_bull", "triple_inner", "triple_outer", "double_inner", "double_outer"):
        r_px = RADII_MM[key] * px_per_mm
        xs = center_px[0] + r_px * np.cos(angles)
        ys = center_px[1] + r_px * np.sin(angles)
        polylines.append(np.column_stack([xs, ys]))

    r_start = RADII_MM["outer_bull"] * px_per_mm
    r_end = RADII_MM["double_outer"] * px_per_mm
    for k in range(20):
        a = math.radians(-90.0 - 9.0 + 18.0 * k)
        ca, sa = math.cos(a), math.sin(a)
        polylines.append(
            np.array(
                [
                    [center_px[0] + r_start * ca, center_px[1] + r_start * sa],
                    [center_px[0] + r_end * ca, center_px[1] + r_end * sa],
                ],
                dtype=np.float64,
            )
        )
    return polylines
