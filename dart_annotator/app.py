"""Interactive OpenCV side-by-side dartboard annotator application."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import re
import time
from typing import Sequence

import cv2
import numpy as np

from dart_annotator.geometry import (
    CALIBRATION_TARGETS,
    SCHEMATIC_SIZE_PX,
    classify_board_mm,
    compute_homographies,
    generate_wireframe_polylines_sch,
    project_point,
    render_schematic,
    schematic_xy_to_board_mm,
)
from dart_annotator.storage import (
    HitRecord,
    ImageAnnotation,
    load_annotations,
    save_annotations,
)
from dart_annotator.viewport import PanelViewport

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

WINDOW_NAME = "Dartboard Hit Annotator"
HEADER_HEIGHT = 68
FOOTER_HEIGHT = 36
DIVIDER_WIDTH = 4
DEFAULT_WIDTH = 1560
DEFAULT_HEIGHT = 860
SUBPIXEL_SHIFT = 4
SUBPIXEL_FACTOR = 1 << SUBPIXEL_SHIFT


CURSOR_STYLES = ("crosshair", "dot", "system_cross")


class _QtCursorHelper:
    """Safely overrides the Qt5 window cursor when running OpenCV's Qt HighGUI backend."""

    QT_CROSS_CURSOR = 2
    QT_BLANK_CURSOR = 10

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._ctor = None
        self._dtor = None
        self._set_override = None
        self._restore_override = None
        self._initialized = False
        self._active_shape: int | None = None

    def _init_bindings(self) -> bool:
        if self._initialized:
            return self._set_override is not None
        self._initialized = True
        try:
            maps_text = Path("/proc/self/maps").read_text(encoding="utf-8", errors="ignore")
            qt_gui_path: str | None = None
            for line in maps_text.splitlines():
                if "libQt5Gui" in line and ".so" in line:
                    qt_gui_path = line.split()[-1]
                    break
            if not qt_gui_path:
                return False

            lib = ctypes.CDLL(qt_gui_path)
            for prefix in ("_ZN14QtOpenCVPython", "_ZN"):
                ctor_name = f"{prefix}7QCursorC1EN2Qt11CursorShapeE" if prefix == "_ZN14QtOpenCVPython" else "_ZN7QCursorC1EN2Qt11CursorShapeE"
                dtor_name = f"{prefix}7QCursorD1Ev" if prefix == "_ZN14QtOpenCVPython" else "_ZN7QCursorD1Ev"
                set_name = f"{prefix}15QGuiApplication17setOverrideCursorERKNS_7QCursorE" if prefix == "_ZN14QtOpenCVPython" else "_ZN15QGuiApplication17setOverrideCursorERK7QCursor"
                rst_name = f"{prefix}15QGuiApplication21restoreOverrideCursorEv" if prefix == "_ZN14QtOpenCVPython" else "_ZN15QGuiApplication21restoreOverrideCursorEv"
                try:
                    self._ctor = lib[ctor_name]
                    self._dtor = lib[dtor_name]
                    self._set_override = lib[set_name]
                    self._restore_override = lib[rst_name]
                    self._lib = lib
                    return True
                except AttributeError:
                    continue
        except Exception:
            return False
        return False

    def set_shape(self, shape: int | None) -> bool:
        """Set Qt override cursor shape (or None to restore default arrow). Returns True if active."""
        if shape == self._active_shape:
            return self._active_shape is not None
        if not self._init_bindings():
            return False
        try:
            if self._active_shape is not None and self._restore_override is not None:
                self._restore_override()
                self._active_shape = None
            if shape is not None and self._ctor and self._dtor and self._set_override:
                buf = ctypes.create_string_buffer(64)
                self._ctor(buf, ctypes.c_int(shape))
                self._set_override(buf)
                self._dtor(buf)
                self._active_shape = shape
                return True
        except Exception:
            self._active_shape = None
        return False

    def restore(self) -> None:
        self.set_shape(None)


def _to_subpix(x: float, y: float) -> tuple[int, int]:
    return (int(round(x * SUBPIXEL_FACTOR)), int(round(y * SUBPIXEL_FACTOR)))


def _draw_subpix_circle(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius_px: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    radius_sp = max(1, int(round(radius_px * SUBPIXEL_FACTOR)))
    if thickness < 0:
        cv2.circle(
            canvas,
            _to_subpix(x, y),
            radius_sp,
            color,
            -1,
            cv2.LINE_AA,
            SUBPIXEL_SHIFT,
        )
    else:
        cv2.circle(
            canvas,
            _to_subpix(x, y),
            radius_sp,
            color,
            thickness,
            cv2.LINE_AA,
            SUBPIXEL_SHIFT,
        )


def _draw_crosshair(
    canvas: np.ndarray,
    x: float,
    y: float,
    arm_px: int,
    color: tuple[int, int, int],
    thickness: int = 1,
    gap_px: int = 0,
) -> None:
    if gap_px <= 0:
        cv2.line(
            canvas,
            _to_subpix(x - arm_px, y),
            _to_subpix(x + arm_px, y),
            color,
            thickness,
            cv2.LINE_AA,
            SUBPIXEL_SHIFT,
        )
        cv2.line(
            canvas,
            _to_subpix(x, y - arm_px),
            _to_subpix(x, y + arm_px),
            color,
            thickness,
            cv2.LINE_AA,
            SUBPIXEL_SHIFT,
        )
        return

    for x0, y0, x1, y1 in (
        (x - arm_px, y, x - gap_px, y),
        (x + gap_px, y, x + arm_px, y),
        (x, y - arm_px, x, y - gap_px),
        (x, y + gap_px, x, y + arm_px),
    ):
        cv2.line(
            canvas,
            _to_subpix(x0, y0),
            _to_subpix(x1, y1),
            color,
            thickness,
            cv2.LINE_AA,
            SUBPIXEL_SHIFT,
        )


def _draw_precision_reticle(
    canvas: np.ndarray,
    x: float,
    y: float,
    color: tuple[int, int, int],
    style: str,
) -> None:
    """Draw a high-precision small dot and/or thin gapped crosshair."""
    outline = (10, 10, 10)
    if style == "dot":
        _draw_subpix_circle(canvas, x, y, 2.8, outline, -1)
        _draw_subpix_circle(canvas, x, y, 1.8, color, -1)
        return

    if style == "crosshair":
        _draw_crosshair(canvas, x, y, arm_px=11, color=outline, thickness=2, gap_px=4)
        _draw_crosshair(canvas, x, y, arm_px=11, color=color, thickness=1, gap_px=4)
        _draw_subpix_circle(canvas, x, y, 1.8, outline, -1)
        _draw_subpix_circle(canvas, x, y, 1.1, color, -1)
        return

    # "system_cross": OS crosshair handles the active cursor; draw just a tiny center dot
    _draw_subpix_circle(canvas, x, y, 1.5, color, -1)


def normalize_image_key(path: Path) -> str:
    """Return a clean, consistent path string relative to cwd when possible."""
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def _natural_sort_key(path: Path) -> list[int | str]:
    """Natural sort key so numeric suffixes and timestamps sort in human order."""
    parts = re.split(r"(\d+)", normalize_image_key(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def discover_images(inputs: Sequence[str | Path]) -> list[Path]:
    """Collect image file paths from file or directory arguments in sorted order."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for child in p.iterdir():
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    res = child.resolve()
                    if res not in seen:
                        seen.add(res)
                        paths.append(child)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            res = p.resolve()
            if res not in seen:
                seen.add(res)
                paths.append(p)
    return sorted(paths, key=_natural_sort_key)


class DartAnnotatorApp:
    """OpenCV GUI for 4-point homography calibration and synchronized hit annotation."""

    def __init__(
        self,
        inputs: Sequence[str | Path],
        output_json: str | Path = "annotations.json",
    ) -> None:
        self.inputs = list(inputs)
        self.output_json = Path(output_json)
        self.image_paths: list[Path] = discover_images(self.inputs)
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in: {', '.join(str(p) for p in self.inputs)}"
            )

        self.annotations: dict[str, ImageAnnotation] = load_annotations(self.output_json)
        self.schematic_img: np.ndarray = render_schematic()
        self.wireframe_polylines = generate_wireframe_polylines_sch()

        self.win_width = DEFAULT_WIDTH
        self.win_height = DEFAULT_HEIGHT
        self.panel_w = (self.win_width - DIVIDER_WIDTH) // 2
        self.panel_h = self.win_height - HEADER_HEIGHT - FOOTER_HEIGHT

        self.left_vp = PanelViewport(100, 100, self.panel_w, self.panel_h)
        self.right_vp = PanelViewport(
            SCHEMATIC_SIZE_PX, SCHEMATIC_SIZE_PX, self.panel_w, self.panel_h
        )

        self.current_idx = self._initial_image_index()
        self.current_img: np.ndarray | None = None
        self.current_key: str = ""
        self.img_token = 0

        self.show_grid = True
        self.edit_homography = False
        self.cursor_style_idx = 0
        self._qt_cursor = _QtCursorHelper()
        self._window_active = False
        self.status_message = ""
        self.status_until = 0.0

        # Hover & drag state in full window coordinates
        self.hover_panel: str | None = None  # "left" or "right"
        self.hover_panel_xy: tuple[float, float] | None = None
        self.dragging_panel: str | None = None
        self.drag_last_xy: tuple[int, int] | None = None
        self.dragging_cal_idx: int | None = None

        inherit = None
        if self.current_idx > 0:
            prev_key = normalize_image_key(self.image_paths[self.current_idx - 1])
            inherit = self.annotations.get(prev_key)
        self._load_current_image(inherit_from=inherit)

    @property
    def cursor_style(self) -> str:
        return CURSOR_STYLES[self.cursor_style_idx]

    def _cycle_cursor_style(self) -> None:
        self.cursor_style_idx = (self.cursor_style_idx + 1) % len(CURSOR_STYLES)
        self._sync_os_cursor()
        self._set_status(f"Cursor style: {self.cursor_style}")

    def _sync_os_cursor(self) -> None:
        if not self._window_active:
            return
        if self.hover_panel is not None and self.dragging_panel is None:
            shape = (
                _QtCursorHelper.QT_CROSS_CURSOR
                if self.cursor_style == "system_cross"
                else _QtCursorHelper.QT_BLANK_CURSOR
            )
            self._qt_cursor.set_shape(shape)
        else:
            self._qt_cursor.restore()

    def _initial_image_index(self) -> int:
        for i, p in enumerate(self.image_paths):
            key = normalize_image_key(p)
            ann = self.annotations.get(key)
            if ann is None or not ann.is_calibrated:
                return i
        return 0

    def _set_status(self, message: str, duration_s: float = 3.0) -> None:
        self.status_message = message
        self.status_until = time.monotonic() + duration_s

    def _current_annotation(self) -> ImageAnnotation:
        return self.annotations[self.current_key]

    def _load_current_image(self, inherit_from: ImageAnnotation | None = None) -> None:
        path = self.image_paths[self.current_idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")

        self.current_img = img
        self.current_key = normalize_image_key(path)
        self.img_token += 1
        h, w = img.shape[:2]

        if self.current_key not in self.annotations:
            self.annotations[self.current_key] = ImageAnnotation(
                image_path=self.current_key,
                image_size=(w, h),
            )
        else:
            self.annotations[self.current_key].image_size = (w, h)

        ann = self.annotations[self.current_key]
        if (
            inherit_from is not None
            and inherit_from.is_calibrated
            and inherit_from.image_size == (w, h)
            and not ann.is_calibrated
            and not ann.calibration_image_pts
            and not ann.hits
        ):
            ann.calibration_image_pts = list(inherit_from.calibration_image_pts)
            ann.h_image_to_schematic = (
                inherit_from.h_image_to_schematic.copy()
                if inherit_from.h_image_to_schematic is not None
                else None
            )
            ann.h_schematic_to_image = (
                inherit_from.h_schematic_to_image.copy()
                if inherit_from.h_schematic_to_image is not None
                else None
            )
            ann.h_image_to_board_mm = (
                inherit_from.h_image_to_board_mm.copy()
                if inherit_from.h_image_to_board_mm is not None
                else None
            )
            ann.hits = [
                HitRecord(
                    id=hit.id,
                    image_xy=hit.image_xy,
                    schematic_xy=hit.schematic_xy,
                    board_mm=hit.board_mm,
                    source=hit.source,
                    sector=hit.sector,
                    score=hit.score,
                )
                for hit in inherit_from.hits
            ]
            self._save()
            self._set_status(
                f"Inherited homography & {len(ann.hits)} hit(s) from previous image (C: clear hits, R: recalibrate).",
                duration_s=3.5,
            )

        self.left_vp.set_source_size(w, h)
        self.right_vp.reset_view()
        self.dragging_cal_idx = None
        if not ann.is_calibrated:
            self.edit_homography = False

    def _save(self, notify: bool = False) -> None:
        save_annotations(self.output_json, self.annotations)
        if notify:
            self._set_status(f"Saved annotations to {self.output_json}")

    def _update_layout(self, width: int, height: int) -> None:
        width = max(800, int(width))
        height = max(540, int(height))
        if width == self.win_width and height == self.win_height:
            return
        self.win_width = width
        self.win_height = height
        self.panel_w = (self.win_width - DIVIDER_WIDTH) // 2
        self.panel_h = max(200, self.win_height - HEADER_HEIGHT - FOOTER_HEIGHT)
        self.left_vp.set_panel_size(self.panel_w, self.panel_h)
        self.right_vp.set_panel_size(self.panel_w, self.panel_h)

    def _hit_test_panel(self, wx: int, wy: int) -> tuple[str | None, float, float]:
        py = float(wy - HEADER_HEIGHT)
        if py < 0 or py >= self.panel_h:
            return (None, 0.0, 0.0)
        if 0 <= wx < self.panel_w:
            return ("left", float(wx), py)
        right_x0 = self.panel_w + DIVIDER_WIDTH
        if right_x0 <= wx < right_x0 + self.panel_w:
            return ("right", float(wx - right_x0), py)
        return (None, 0.0, 0.0)

    def _finalize_homography(self, ann: ImageAnnotation) -> None:
        sch_pts = [t.schematic_xy() for t in CALIBRATION_TARGETS]
        try:
            h_i2s, h_s2i, h_i2m = compute_homographies(ann.calibration_image_pts, sch_pts)
        except ValueError:
            ann.calibration_image_pts.pop()
            self._set_status("Degenerate points! Please re-click point 4 accurately.", duration_s=4.0)
            return
        ann.h_image_to_schematic = h_i2s
        ann.h_schematic_to_image = h_s2i
        ann.h_image_to_board_mm = h_i2m

        # Re-project any existing hits if user recalibrated
        self._reproject_hits(ann)

        self._save()
        self._set_status("Homography established! Click on either image to register dart hits.")

    def _handle_left_click(self, panel: str, px: float, py: float) -> None:
        ann = self._current_annotation()

        if self.edit_homography:
            self._set_status("Edit Homography: drag C1-C4 on left image, or press E when done.")
            return

        if not ann.is_calibrated:
            if panel != "left":
                self._set_status("Calibration: click the highlighted point on the LEFT (photo) image.")
                return
            sx, sy = self.left_vp.panel_to_source(px, py)
            if not self.left_vp.contains_source(sx, sy):
                return
            ann.calibration_image_pts.append((float(sx), float(sy)))
            if len(ann.calibration_image_pts) == 4:
                self._finalize_homography(ann)
            else:
                self._save()
            return

        # Phase 2: register dart hit from EITHER panel
        assert ann.h_image_to_schematic is not None
        assert ann.h_schematic_to_image is not None

        if panel == "left":
            img_xy = self.left_vp.panel_to_source(px, py)
            if not self.left_vp.contains_source(img_xy[0], img_xy[1]):
                return
            sch_xy = project_point(ann.h_image_to_schematic, img_xy)
            if sch_xy is None:
                return
            source = "image"
        else:
            sch_xy = self.right_vp.panel_to_source(px, py)
            if not self.right_vp.contains_source(sch_xy[0], sch_xy[1]):
                return
            img_xy = project_point(ann.h_schematic_to_image, sch_xy)
            if img_xy is None:
                return
            source = "schematic"

        board_mm = schematic_xy_to_board_mm(sch_xy)
        sector, score = classify_board_mm(*board_mm)
        next_id = (ann.hits[-1].id + 1) if ann.hits else 1
        ann.hits.append(
            HitRecord(
                id=next_id,
                image_xy=(float(img_xy[0]), float(img_xy[1])),
                schematic_xy=(float(sch_xy[0]), float(sch_xy[1])),
                board_mm=(float(board_mm[0]), float(board_mm[1])),
                source=source,
                sector=sector,
                score=score,
            )
        )
        self._save()
        self._set_status(
            f"Hit #{next_id}: {sector} ({score} pts) at img=({img_xy[0]:.2f}, {img_xy[1]:.2f})",
            duration_s=2.5,
        )

    def _undo_last(self) -> None:
        ann = self._current_annotation()
        if not ann.is_calibrated:
            if ann.calibration_image_pts:
                ann.calibration_image_pts.pop()
                self._save()
                self._set_status("Undid last calibration point.")
        else:
            if ann.hits:
                removed = ann.hits.pop()
                self._save()
                self._set_status(f"Removed hit #{removed.id} ({removed.sector}).")

    def _delete_nearest_hit(self) -> None:
        ann = self._current_annotation()
        if not ann.is_calibrated or not ann.hits or self.hover_panel is None or self.hover_panel_xy is None:
            return
        hx, hy = self.hover_panel_xy
        best_idx = -1
        best_dist = 28.0  # screen-pixel threshold
        for idx, hit in enumerate(ann.hits):
            if self.hover_panel == "left":
                px, py = self.left_vp.source_to_panel(*hit.image_xy)
            else:
                px, py = self.right_vp.source_to_panel(*hit.schematic_xy)
            d = math.hypot(px - hx, py - hy)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx >= 0:
            removed = ann.hits.pop(best_idx)
            for new_i, hit in enumerate(ann.hits, start=1):
                hit.id = new_i
            self._save()
            self._set_status(f"Deleted hit ({removed.sector}).")

    def _reproject_hits(self, ann: ImageAnnotation) -> None:
        if not ann.hits or ann.h_image_to_schematic is None or ann.h_schematic_to_image is None:
            return
        for hit in ann.hits:
            if hit.source == "schematic":
                new_img = project_point(ann.h_schematic_to_image, hit.schematic_xy)
                if new_img is not None:
                    hit.image_xy = new_img
            else:
                new_sch = project_point(ann.h_image_to_schematic, hit.image_xy)
                if new_sch is not None:
                    hit.schematic_xy = new_sch
                    hit.board_mm = schematic_xy_to_board_mm(new_sch)
                    hit.sector, hit.score = classify_board_mm(*hit.board_mm)

    def _toggle_edit_homography(self) -> None:
        ann = self._current_annotation()
        if not ann.is_calibrated:
            self._set_status("Complete 4-point calibration first before editing homography.")
            return
        self.edit_homography = not self.edit_homography
        self.dragging_cal_idx = None
        if self.edit_homography:
            self._set_status("Edit Homography ON: drag C1-C4 points on left image; press E when done.", duration_s=4.0)
        else:
            self._save()
            self._set_status("Edit Homography OFF.")

    def _find_cal_handle(self, px: float, py: float, radius_px: float = 18.0) -> int | None:
        ann = self._current_annotation()
        if not ann.is_calibrated:
            return None
        best_idx: int | None = None
        best_dist = radius_px
        for idx, (ix, iy) in enumerate(ann.calibration_image_pts):
            hx, hy = self.left_vp.source_to_panel(ix, iy)
            d = math.hypot(hx - px, hy - py)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _update_dragged_cal_point(self, px: float, py: float) -> None:
        if self.dragging_cal_idx is None:
            return
        ann = self._current_annotation()
        if not ann.is_calibrated:
            return
        sx, sy = self.left_vp.panel_to_source(px, py)
        w, h = ann.image_size
        sx = min(float(w), max(0.0, float(sx)))
        sy = min(float(h), max(0.0, float(sy)))

        idx = self.dragging_cal_idx
        old_pt = ann.calibration_image_pts[idx]
        ann.calibration_image_pts[idx] = (sx, sy)
        sch_pts = [t.schematic_xy() for t in CALIBRATION_TARGETS]
        try:
            h_i2s, h_s2i, h_i2m = compute_homographies(ann.calibration_image_pts, sch_pts)
        except ValueError:
            ann.calibration_image_pts[idx] = old_pt
            return

        ann.h_image_to_schematic = h_i2s
        ann.h_schematic_to_image = h_s2i
        ann.h_image_to_board_mm = h_i2m
        self._reproject_hits(ann)

    def _reset_calibration(self) -> None:
        ann = self._current_annotation()
        ann.calibration_image_pts.clear()
        ann.h_image_to_schematic = None
        ann.h_schematic_to_image = None
        ann.h_image_to_board_mm = None
        self.edit_homography = False
        self.dragging_cal_idx = None
        self._save()
        self._set_status("Calibration reset. Click the 4 predefined points on the left image.")

    def _rescan_images(self) -> None:
        new_list = discover_images(self.inputs)
        if not new_list:
            self._set_status("No images found during rescan.")
            return
        added = len(new_list) - len(self.image_paths)
        current_resolved = self.image_paths[self.current_idx].resolve()
        self.image_paths = new_list
        for i, p in enumerate(self.image_paths):
            if p.resolve() == current_resolved:
                self.current_idx = i
                break
        else:
            self.current_idx = 0
            inherit = None
        if self.current_idx > 0:
            prev_key = normalize_image_key(self.image_paths[self.current_idx - 1])
            inherit = self.annotations.get(prev_key)
        self._load_current_image(inherit_from=inherit)
        self._set_status(f"Rescanned images ({len(self.image_paths)} total, {added:+d}).")

    def _switch_image(self, delta: int) -> None:
        prev_ann = self._current_annotation()
        self.edit_homography = False
        self.dragging_cal_idx = None
        self._save()
        self.current_idx = (self.current_idx + delta) % len(self.image_paths)
        self._load_current_image(inherit_from=prev_ann)

    def _zoom_hovered(self, factor: float) -> None:
        if self.hover_panel == "left" and self.hover_panel_xy is not None:
            self.left_vp.zoom_at(self.hover_panel_xy[0], self.hover_panel_xy[1], factor)
        elif self.hover_panel == "right" and self.hover_panel_xy is not None:
            self.right_vp.zoom_at(self.hover_panel_xy[0], self.hover_panel_xy[1], factor)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        panel, px, py = self._hit_test_panel(x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_cal_idx is not None:
                self._update_dragged_cal_point(px, py)
            elif self.dragging_panel is not None and self.drag_last_xy is not None:
                dx = x - self.drag_last_xy[0]
                dy = y - self.drag_last_xy[1]
                self.drag_last_xy = (x, y)
                if self.dragging_panel == "left":
                    self.left_vp.pan_by_panel_delta(dx, dy)
                else:
                    self.right_vp.pan_by_panel_delta(dx, dy)
            self.hover_panel = panel
            self.hover_panel_xy = (px, py) if panel is not None else None
            self._sync_os_cursor()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if panel is not None:
                if self.edit_homography and panel == "left":
                    handle_idx = self._find_cal_handle(px, py)
                    if handle_idx is not None:
                        self.dragging_cal_idx = handle_idx
                        self._update_dragged_cal_point(px, py)
                        return
                self._handle_left_click(panel, px, py)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if self.dragging_cal_idx is not None:
                self.dragging_cal_idx = None
                self._save()
            return

        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            if panel is not None:
                self.dragging_panel = panel
                self.drag_last_xy = (x, y)
                self._sync_os_cursor()
            return

        if event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.dragging_panel = None
            self.drag_last_xy = None
            self._sync_os_cursor()
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            raw = (flags >> 16) & 0xFFFF
            if raw >= 0x8000:
                raw -= 0x10000
            delta = raw if raw != 0 else flags
            factor = 1.25 if delta > 0 else (1.0 / 1.25)
            if panel == "left":
                self.left_vp.zoom_at(px, py, factor)
            elif panel == "right":
                self.right_vp.zoom_at(px, py, factor)
            return

    def _compute_hover_info(
        self, ann: ImageAnnotation
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None, tuple[float, float] | None, str, int]:
        """Return (img_xy, sch_xy, board_mm, sector, score) for current cursor position."""
        if self.hover_panel is None or self.hover_panel_xy is None:
            return (None, None, None, "", 0)

        px, py = self.hover_panel_xy
        if self.hover_panel == "left":
            img_xy = self.left_vp.panel_to_source(px, py)
            if not self.left_vp.contains_source(img_xy[0], img_xy[1]):
                return (None, None, None, "", 0)
            if not ann.is_calibrated or ann.h_image_to_schematic is None:
                return (img_xy, None, None, "", 0)
            sch_xy = project_point(ann.h_image_to_schematic, img_xy)
            if sch_xy is None:
                return (img_xy, None, None, "", 0)
            mm = schematic_xy_to_board_mm(sch_xy)
            sector, score = classify_board_mm(*mm)
            return (img_xy, sch_xy, mm, sector, score)

        # Hovered over schematic panel
        sch_xy = self.right_vp.panel_to_source(px, py)
        if not self.right_vp.contains_source(sch_xy[0], sch_xy[1]):
            return (None, None, None, "", 0)
        mm = schematic_xy_to_board_mm(sch_xy)
        sector, score = classify_board_mm(*mm)
        if not ann.is_calibrated or ann.h_schematic_to_image is None:
            return (None, sch_xy, mm, sector, score)
        img_xy = project_point(ann.h_schematic_to_image, sch_xy)
        return (img_xy, sch_xy, mm, sector, score)

    def _render_frame(self) -> np.ndarray:
        assert self.current_img is not None
        ann = self._current_annotation()

        left_panel = self.left_vp.render(self.current_img, self.img_token)
        right_panel = self.right_vp.render(self.schematic_img, img_token=0)

        # Projected wireframe grid on original image (always shown while editing homography)
        if ann.is_calibrated and (self.show_grid or self.edit_homography) and ann.h_schematic_to_image is not None:
            overlay = left_panel.copy()
            h_mat = ann.h_schematic_to_image
            for poly in self.wireframe_polylines:
                pts_h = np.column_stack([poly, np.ones((len(poly), 1), dtype=np.float64)])
                proj = (h_mat @ pts_h.T).T
                w = proj[:, 2:3]
                if np.any(np.abs(w) < 1e-9):
                    continue
                img_pts = proj[:, :2] / w
                panel_pts = np.array(
                    [self.left_vp.source_to_panel(pt[0], pt[1]) for pt in img_pts],
                    dtype=np.float64,
                )
                sp_pts = np.round(panel_pts * SUBPIXEL_FACTOR).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    overlay,
                    [sp_pts],
                    isClosed=False,
                    color=(255, 210, 60),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    shift=SUBPIXEL_SHIFT,
                )
            alpha = 0.65 if self.edit_homography else 0.45
            cv2.addWeighted(overlay, alpha, left_panel, 1.0 - alpha, 0.0, dst=left_panel)

        # Draw calibration markers
        hovered_cal_idx: int | None = None
        if self.edit_homography and self.hover_panel == "left" and self.hover_panel_xy is not None:
            hovered_cal_idx = self._find_cal_handle(*self.hover_panel_xy)

        step_idx = len(ann.calibration_image_pts)
        for idx, target in enumerate(CALIBRATION_TARGETS):
            sch_tx, sch_ty = target.schematic_xy()
            rpx, rpy = self.right_vp.source_to_panel(sch_tx, sch_ty)

            if idx < step_idx:
                # Completed calibration point
                ix, iy = ann.calibration_image_pts[idx]
                lpx, lpy = self.left_vp.source_to_panel(ix, iy)
                is_active_handle = self.edit_homography and (
                    idx == self.dragging_cal_idx or idx == hovered_cal_idx
                )
                if is_active_handle:
                    cal_color = (0, 140, 255)
                    radius = 9.0
                elif self.edit_homography:
                    cal_color = (0, 225, 255)
                    radius = 7.5
                elif ann.is_calibrated:
                    cal_color = (255, 190, 0)
                    radius = 5.0
                else:
                    cal_color = (60, 220, 90)
                    radius = 7.0

                _draw_subpix_circle(left_panel, lpx, lpy, radius, (0, 0, 0), 3)
                _draw_subpix_circle(left_panel, lpx, lpy, radius, cal_color, 2)
                if self.edit_homography:
                    _draw_subpix_circle(left_panel, lpx, lpy, 1.8, cal_color, -1)
                _draw_subpix_circle(right_panel, rpx, rpy, radius, cal_color, 2)
                if not ann.is_calibrated or self.edit_homography:
                    cv2.putText(
                        left_panel,
                        f"C{idx + 1}",
                        (int(round(lpx + 10)), int(round(lpy - 7))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        left_panel,
                        f"C{idx + 1}",
                        (int(round(lpx + 10)), int(round(lpy - 7))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        cal_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        right_panel,
                        f"C{idx + 1}",
                        (int(round(rpx + 10)), int(round(rpy - 7))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        cal_color,
                        2,
                        cv2.LINE_AA,
                    )
            elif not ann.is_calibrated and idx == step_idx:
                # Active calibration target to click next
                pulse = 12 + int(round(3.0 * math.sin(time.monotonic() * 6.0)))
                target_color = (0, 110, 255)  # vivid orange-red in BGR
                _draw_subpix_circle(right_panel, rpx, rpy, pulse, target_color, 2)
                _draw_subpix_circle(right_panel, rpx, rpy, 4, target_color, -1)
                _draw_crosshair(right_panel, rpx, rpy, 18, target_color, 2)
                cv2.putText(
                    right_panel,
                    f"TARGET {idx + 1}/4",
                    (int(round(rpx + 14)), int(round(rpy - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    target_color,
                    2,
                    cv2.LINE_AA,
                )

        # Draw registered dart hits on BOTH panels
        for hit in ann.hits:
            lpx, lpy = self.left_vp.source_to_panel(*hit.image_xy)
            rpx, rpy = self.right_vp.source_to_panel(*hit.schematic_xy)
            hit_color = (50, 230, 90)
            outline = (15, 15, 15)

            for pnl, hx, hy in ((left_panel, lpx, lpy), (right_panel, rpx, rpy)):
                _draw_subpix_circle(pnl, hx, hy, 5.0, outline, 2)
                _draw_subpix_circle(pnl, hx, hy, 5.0, hit_color, 1)
                _draw_subpix_circle(pnl, hx, hy, 1.4, (255, 255, 255), -1)
                label = f"{hit.id}:{hit.sector}"
                cv2.putText(
                    pnl,
                    label,
                    (int(round(hx + 8)), int(round(hy - 6))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    outline,
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    pnl,
                    label,
                    (int(round(hx + 8)), int(round(hy - 6))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    hit_color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw precision reticle under active cursor and corresponding small dot on opposite panel
        hover_img, hover_sch, hover_mm, hover_sector, hover_score = self._compute_hover_info(ann)
        if self.hover_panel == "left" and hover_img is not None:
            lpx, lpy = self.left_vp.source_to_panel(*hover_img)
            _draw_precision_reticle(left_panel, lpx, lpy, (0, 255, 255), self.cursor_style)
            if ann.is_calibrated and hover_sch is not None:
                rpx, rpy = self.right_vp.source_to_panel(*hover_sch)
                opp_style = "dot" if self.cursor_style == "dot" else "crosshair"
                _draw_precision_reticle(right_panel, rpx, rpy, (255, 0, 255), opp_style)
        elif self.hover_panel == "right" and hover_sch is not None:
            rpx, rpy = self.right_vp.source_to_panel(*hover_sch)
            _draw_precision_reticle(right_panel, rpx, rpy, (255, 0, 255), self.cursor_style)
            if ann.is_calibrated and hover_img is not None:
                lpx, lpy = self.left_vp.source_to_panel(*hover_img)
                opp_style = "dot" if self.cursor_style == "dot" else "crosshair"
                _draw_precision_reticle(left_panel, lpx, lpy, (0, 255, 255), opp_style)

        # Compose full window canvas
        canvas = np.full((self.win_height, self.win_width, 3), 24, dtype=np.uint8)
        y0 = HEADER_HEIGHT
        y1 = HEADER_HEIGHT + self.panel_h
        canvas[y0:y1, 0 : self.panel_w] = left_panel
        canvas[y0:y1, self.panel_w : self.panel_w + DIVIDER_WIDTH] = (70, 70, 75)
        right_x0 = self.panel_w + DIVIDER_WIDTH
        canvas[y0:y1, right_x0 : right_x0 + self.panel_w] = right_panel

        # Header HUD
        img_name = self.image_paths[self.current_idx].name
        line1 = f"[{self.current_idx + 1}/{len(self.image_paths)}] {img_name}"
        if not ann.is_calibrated:
            target = CALIBRATION_TARGETS[len(ann.calibration_image_pts)]
            mode_str = f"CALIBRATION ({len(ann.calibration_image_pts) + 1}/4): Click {target.description}"
            mode_color = (80, 180, 255)
        elif self.edit_homography:
            mode_str = "EDIT HOMOGRAPHY - Drag C1..C4 points on left image to align wireframe (E: Done)"
            mode_color = (0, 180, 255)
        else:
            mode_str = f"ANNOTATE HITS (Hits: {len(ann.hits)}) - Click either panel to add hit"
            mode_color = (90, 235, 130)

        (tw1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        cv2.putText(
            canvas, line1, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 235), 1, cv2.LINE_AA
        )
        mode_x = 14 + tw1 + 22
        cv2.putText(
            canvas, "|", (mode_x - 14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (110, 115, 125), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, mode_str, (mode_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 2, cv2.LINE_AA
        )

        # Second header line: live coords or status banner
        if time.monotonic() < self.status_until and self.status_message:
            line2 = self.status_message
            line2_color = (140, 240, 255)
        else:
            parts = [f"Zoom L:{self.left_vp.zoom:.1f}x R:{self.right_vp.zoom:.1f}x"]
            if hover_img is not None:
                parts.append(f"Img: ({hover_img[0]:.2f}, {hover_img[1]:.2f})")
            if hover_sch is not None:
                parts.append(f"Sch: ({hover_sch[0]:.2f}, {hover_sch[1]:.2f})")
            if hover_mm is not None and hover_sector:
                parts.append(f"Board: ({hover_mm[0]:.1f}, {hover_mm[1]:.1f}) mm [{hover_sector} = {hover_score}]")
            line2 = "  |  ".join(parts)
            line2_color = (185, 190, 200)

        cv2.putText(
            canvas, line2, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, line2_color, 1, cv2.LINE_AA
        )

        # Footer controls legend
        footer_y = self.win_height - 12
        legend = (
            f"LMB: Click/Drag | Wheel: Zoom | RMB-drag: Pan | V: Reset | M: Cursor [{self.cursor_style}] | "
            "E: Edit H | R: Recalibrate | G: Grid | Z: Undo | X: Del | C: Clear | N/P: Next/Prev | S: Save | Q: Quit"
        )
        cv2.putText(
            canvas, legend, (12, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 180, 190), 1, cv2.LINE_AA
        )
        return canvas

    def run(self) -> None:
        """Start the interactive OpenCV event loop."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.win_width, self.win_height)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._window_active = True

        try:
            while True:
                # Check if window was closed by user via titlebar button
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                try:
                    _, _, rect_w, rect_h = cv2.getWindowImageRect(WINDOW_NAME)
                    if rect_w >= 800 and rect_h >= 540:
                        self._update_layout(rect_w, rect_h)
                except cv2.error:
                    pass

                frame = self._render_frame()
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(8)
                if key == -1:
                    continue
                ch = key & 0xFF
                if ch in (ord("q"), ord("Q"), 27):  # q or Esc
                    break
                elif ch in (ord("n"), ord("N"), 32):  # n, Space
                    self._switch_image(+1)
                elif ch == 13:  # Enter confirms edit mode or advances to next image
                    if self.edit_homography:
                        self._toggle_edit_homography()
                    else:
                        self._switch_image(+1)
                elif ch in (ord("e"), ord("E")):
                    self._toggle_edit_homography()
                elif ch in (ord("p"), ord("P")):
                    self._switch_image(-1)
                elif ch in (ord("z"), ord("Z"), 8):  # z or Backspace
                    self._undo_last()
                elif ch in (ord("x"), ord("X"), 127):  # x or Delete
                    self._delete_nearest_hit()
                elif ch in (ord("c"), ord("C")):
                    ann = self._current_annotation()
                    ann.hits.clear()
                    self._save()
                    self._set_status("Cleared all hits on current image.")
                elif ch in (ord("r"), ord("R")):
                    self._reset_calibration()
                elif ch in (ord("g"), ord("G")):
                    self.show_grid = not self.show_grid
                    self._set_status(f"Projected grid: {'ON' if self.show_grid else 'OFF'}")
                elif ch in (ord("m"), ord("M")):
                    self._cycle_cursor_style()
                elif ch in (ord("v"), ord("V")):
                    self.left_vp.reset_view()
                    self.right_vp.reset_view()
                    self._set_status("Reset zoom & pan.")
                elif ch in (ord("+"), ord("=")):
                    self._zoom_hovered(1.25)
                elif ch in (ord("-"), ord("_")):
                    self._zoom_hovered(1.0 / 1.25)
                elif ch in (ord("a"), ord("A")):
                    self._rescan_images()
                elif ch in (ord("s"), ord("S")):
                    self._save(notify=True)
        finally:
            self._window_active = False
            self._qt_cursor.restore()
            self._save()
            cv2.destroyAllWindows()
