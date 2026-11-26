"""Zoom/pan viewport mapping between panel coordinates and float source coordinates."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class PanelViewport:
    """Manages aspect-preserving fit, zoom, pan, and cached rendering for one panel."""

    MIN_ZOOM = 1.0
    MAX_ZOOM = 24.0

    def __init__(self, src_width: int, src_height: int, panel_width: int, panel_height: int) -> None:
        self.src_width = int(src_width)
        self.src_height = int(src_height)
        self.panel_width = max(1, int(panel_width))
        self.panel_height = max(1, int(panel_height))
        self.zoom = 1.0
        self.center_x = self.src_width * 0.5
        self.center_y = self.src_height * 0.5

        self._cached_key: tuple[Any, ...] | None = None
        self._cached_img: np.ndarray | None = None

    @property
    def fit_scale(self) -> float:
        return min(
            self.panel_width / max(1, self.src_width),
            self.panel_height / max(1, self.src_height),
        )

    @property
    def total_scale(self) -> float:
        return self.fit_scale * self.zoom

    def set_source_size(self, src_width: int, src_height: int) -> None:
        self.src_width = int(src_width)
        self.src_height = int(src_height)
        self.reset_view()

    def set_panel_size(self, panel_width: int, panel_height: int) -> None:
        panel_width = max(1, int(panel_width))
        panel_height = max(1, int(panel_height))
        if panel_width != self.panel_width or panel_height != self.panel_height:
            self.panel_width = panel_width
            self.panel_height = panel_height
            self._clamp_center()
            self._cached_key = None

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.center_x = self.src_width * 0.5
        self.center_y = self.src_height * 0.5
        self._cached_key = None

    def panel_to_source(self, px: float, py: float) -> tuple[float, float]:
        """Convert panel coordinates to float source image coordinates."""
        scale = self.total_scale
        sx = self.center_x + (float(px) - self.panel_width * 0.5) / scale
        sy = self.center_y + (float(py) - self.panel_height * 0.5) / scale
        return (float(sx), float(sy))

    def source_to_panel(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert float source image coordinates to panel coordinates."""
        scale = self.total_scale
        px = self.panel_width * 0.5 + (float(sx) - self.center_x) * scale
        py = self.panel_height * 0.5 + (float(sy) - self.center_y) * scale
        return (float(px), float(py))

    def contains_source(self, sx: float, sy: float, margin: float = 0.0) -> bool:
        return (-margin <= sx <= self.src_width + margin) and (-margin <= sy <= self.src_height + margin)

    def zoom_at(self, px: float, py: float, factor: float) -> None:
        """Zoom around the given panel coordinate so the point under cursor stays fixed."""
        sx_before, sy_before = self.panel_to_source(px, py)
        new_zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        self.zoom = new_zoom
        if self.zoom <= 1.0001:
            self.reset_view()
            return
        scale = self.total_scale
        self.center_x = sx_before - (px - self.panel_width * 0.5) / scale
        self.center_y = sy_before - (py - self.panel_height * 0.5) / scale
        self._clamp_center()
        self._cached_key = None

    def pan_by_panel_delta(self, dx_panel: float, dy_panel: float) -> None:
        """Pan the source center by a screen-space pixel drag delta."""
        if self.zoom <= 1.0001:
            return
        scale = self.total_scale
        self.center_x -= dx_panel / scale
        self.center_y -= dy_panel / scale
        self._clamp_center()
        self._cached_key = None

    def _clamp_center(self) -> None:
        if self.zoom <= 1.0001:
            self.center_x = self.src_width * 0.5
            self.center_y = self.src_height * 0.5
            return
        self.center_x = min(float(self.src_width), max(0.0, self.center_x))
        self.center_y = min(float(self.src_height), max(0.0, self.center_y))

    def render(self, src_img: np.ndarray, img_token: int) -> np.ndarray:
        """Return a cached (panel_height, panel_width, 3) rendered view of src_img."""
        key = (
            img_token,
            self.src_width,
            self.src_height,
            self.panel_width,
            self.panel_height,
            round(self.zoom, 6),
            round(self.center_x, 4),
            round(self.center_y, 4),
        )
        if self._cached_key == key and self._cached_img is not None:
            return self._cached_img.copy()

        scale = self.total_scale
        tx = self.panel_width * 0.5 - self.center_x * scale
        ty = self.panel_height * 0.5 - self.center_y * scale
        mat = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        rendered = cv2.warpAffine(
            src_img,
            mat,
            (self.panel_width, self.panel_height),
            flags=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(32, 32, 36),
        )
        self._cached_key = key
        self._cached_img = rendered
        return rendered.copy()
