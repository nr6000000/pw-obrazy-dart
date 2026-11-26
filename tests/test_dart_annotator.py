"""Automated tests for dart_annotator geometry, viewport, storage, and app logic."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dart_annotator.app import DartAnnotatorApp, discover_images
from dart_annotator.geometry import (
    CALIBRATION_TARGETS,
    SCHEMATIC_CENTER_PX,
    SCHEMATIC_PX_PER_MM,
    SCHEMATIC_SIZE_PX,
    classify_board_mm,
    compute_homographies,
    project_point,
    render_schematic,
)
from dart_annotator.storage import load_annotations
from dart_annotator.viewport import PanelViewport


class TestDartAnnotator(unittest.TestCase):
    def test_schematic_render_dimensions(self) -> None:
        sch = render_schematic()
        self.assertEqual(sch.shape, (SCHEMATIC_SIZE_PX, SCHEMATIC_SIZE_PX, 3))
        self.assertEqual(sch.dtype, np.uint8)
        # Background is white, lines are dark
        self.assertGreater(int(sch[10, 10, 0]), 240)
        self.assertLess(int(sch.min()), 60)

    def test_classify_board_mm(self) -> None:
        self.assertEqual(classify_board_mm(0.0, 0.0), ("DBULL", 50))
        self.assertEqual(classify_board_mm(10.0, 0.0), ("BULL", 25))
        self.assertEqual(classify_board_mm(0.0, -103.0), ("T20", 60))
        self.assertEqual(classify_board_mm(0.0, -166.0), ("D20", 40))
        self.assertEqual(classify_board_mm(0.0, -60.0), ("S20", 20))
        self.assertEqual(classify_board_mm(130.0, 0.0), ("S6", 6))
        self.assertEqual(classify_board_mm(0.0, 166.0), ("D3", 6))
        self.assertEqual(classify_board_mm(-103.0, 0.0), ("T11", 33))
        self.assertEqual(classify_board_mm(180.0, 0.0), ("MISS", 0))

    def test_homography_float_round_trip(self) -> None:
        sch_pts = [t.schematic_xy() for t in CALIBRATION_TARGETS]
        # Apply a known perspective warp to create synthetic image points with float precision
        true_h = np.array(
            [
                [1.85, 0.14, 412.37],
                [-0.08, 2.05, 528.91],
                [0.00012, -0.00009, 1.0],
            ],
            dtype=np.float64,
        )
        img_pts = [project_point(true_h, p) for p in sch_pts]
        self.assertTrue(all(pt is not None for pt in img_pts))

        h_i2s, h_s2i, h_i2m = compute_homographies(img_pts, sch_pts)  # type: ignore[arg-type]

        probe_sch = (483.275, 291.642)
        probe_img = project_point(h_s2i, probe_sch)
        self.assertIsNotNone(probe_img)
        back_sch = project_point(h_i2s, probe_img)  # type: ignore[arg-type]
        self.assertIsNotNone(back_sch)
        self.assertAlmostEqual(back_sch[0], probe_sch[0], places=6)
        self.assertAlmostEqual(back_sch[1], probe_sch[1], places=6)

        probe_mm = project_point(h_i2m, probe_img)  # type: ignore[arg-type]
        self.assertIsNotNone(probe_mm)
        expected_mm_x = (probe_sch[0] - SCHEMATIC_CENTER_PX[0]) / SCHEMATIC_PX_PER_MM
        expected_mm_y = (probe_sch[1] - SCHEMATIC_CENTER_PX[1]) / SCHEMATIC_PX_PER_MM
        self.assertAlmostEqual(probe_mm[0], expected_mm_x, places=6)
        self.assertAlmostEqual(probe_mm[1], expected_mm_y, places=6)

    def test_viewport_float_zoom_and_pan(self) -> None:
        vp = PanelViewport(3072, 4096, 778, 756)
        # Zooming around a panel point keeps the corresponding float source point fixed
        px, py = 312.0, 245.0
        src_before = vp.panel_to_source(px, py)
        vp.zoom_at(px, py, 3.0)
        src_after = vp.panel_to_source(px, py)
        self.assertAlmostEqual(src_before[0], src_after[0], places=5)
        self.assertAlmostEqual(src_before[1], src_after[1], places=5)

        # Panning shifts center smoothly and source_to_panel is exact inverse of panel_to_source
        vp.pan_by_panel_delta(40.0, -25.0)
        sx, sy = vp.panel_to_source(500.0, 300.0)
        px2, py2 = vp.source_to_panel(sx, sy)
        self.assertAlmostEqual(px2, 500.0, places=6)
        self.assertAlmostEqual(py2, 300.0, places=6)
        # Verify fractional coordinates are preserved (not rounded to int)
        self.assertNotEqual(sx, round(sx))

    def test_app_end_to_end_calibration_hover_both_panels_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_json = tmp_dir / "annotations.json"

            # Use real example image if available, otherwise synthetic
            sample_dir = Path("images")
            sample_images = sorted(sample_dir.glob("*.jpg"))
            if sample_images:
                inputs = [sample_images[0]]
            else:
                synth = np.full((1200, 1000, 3), 200, dtype=np.uint8)
                synth_path = tmp_dir / "board.png"
                cv2.imwrite(str(synth_path), synth)
                inputs = [synth_path]

            app = DartAnnotatorApp(inputs=inputs, output_json=out_json)
            ann = app._current_annotation()
            self.assertFalse(ann.is_calibrated)

            # Clicking right panel during calibration must NOT advance calibration
            app._handle_left_click("right", 300.0, 300.0)
            self.assertEqual(len(ann.calibration_image_pts), 0)

            # Click 4 calibration points on left panel (mapped from a square on the photo)
            img_w, img_h = ann.image_size
            cal_src = [
                (img_w * 0.45, img_h * 0.15),
                (img_w * 0.85, img_h * 0.48),
                (img_w * 0.52, img_h * 0.85),
                (img_w * 0.15, img_h * 0.52),
            ]
            for sx, sy in cal_src:
                px, py = app.left_vp.source_to_panel(sx, sy)
                app._handle_left_click("left", px, py)

            self.assertTrue(ann.is_calibrated)

            # Hover on left panel -> projects float dot to right schematic panel
            app.hover_panel = "left"
            app.hover_panel_xy = app.left_vp.source_to_panel(img_w * 0.5, img_h * 0.5)
            h_img, h_sch, h_mm, _, _ = app._compute_hover_info(ann)
            self.assertIsNotNone(h_img)
            self.assertIsNotNone(h_sch)
            self.assertIsNotNone(h_mm)

            # Hover on right panel -> projects float dot to left image panel
            app.hover_panel = "right"
            app.hover_panel_xy = app.right_vp.source_to_panel(450.0, 450.0)
            h_img2, h_sch2, _, sector2, score2 = app._compute_hover_info(ann)
            self.assertIsNotNone(h_img2)
            self.assertIsNotNone(h_sch2)
            self.assertEqual((sector2, score2), ("DBULL", 50))

            # Click left panel to register hit #1 (zoomed in so coordinates are clearly non-integer)
            app.left_vp.zoom_at(350.0, 350.0, 3.0)
            app._handle_left_click("left", 351.0, 352.0)

            # Click right panel (schematic) to register hit #2
            rpx, rpy = app.right_vp.source_to_panel(450.0, 450.0 - 103.0 * SCHEMATIC_PX_PER_MM)
            app._handle_left_click("right", rpx, rpy)

            self.assertEqual(len(ann.hits), 2)
            self.assertEqual(ann.hits[0].source, "image")
            self.assertEqual(ann.hits[1].source, "schematic")
            self.assertEqual(ann.hits[1].sector, "T20")
            self.assertEqual(ann.hits[1].score, 60)

            # Render side-by-side frame across all cursor styles without errors
            for expected_style in ("crosshair", "dot", "system_cross"):
                self.assertEqual(app.cursor_style, expected_style)
                frame = app._render_frame()
                self.assertEqual(frame.shape, (app.win_height, app.win_width, 3))
                app._cycle_cursor_style()
            self.assertEqual(app.cursor_style, "crosshair")

            # Verify JSON file contents and float preservation
            raw = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("schematic_spec", raw)
            self.assertIn(app.current_key, raw["images"])
            saved_entry = raw["images"][app.current_key]
            self.assertIsNotNone(saved_entry["homography"]["H_image_to_schematic"])
            self.assertIsNotNone(saved_entry["homography"]["H_schematic_to_image"])
            self.assertEqual(len(saved_entry["hits"]), 2)
            # Coordinates are floats, not rounded to integer pixels
            hit0_x, hit0_y = saved_entry["hits"][0]["image_xy"]
            self.assertIsInstance(hit0_x, float)
            self.assertIsInstance(hit0_y, float)
            self.assertTrue((hit0_x % 1.0) != 0.0 or (hit0_y % 1.0) != 0.0)

            # Verify load_annotations restores everything accurately
            reloaded = load_annotations(out_json)
            self.assertIn(app.current_key, reloaded)
            self.assertTrue(reloaded[app.current_key].is_calibrated)
            self.assertEqual(len(reloaded[app.current_key].hits), 2)
            self.assertAlmostEqual(
                reloaded[app.current_key].hits[0].image_xy[0], hit0_x, places=6
            )

    def test_sorted_image_order_and_inherit_homography_and_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_json = tmp_dir / "annotations.json"
            # Create files intentionally out of alphabetical/natural order
            for name in ("shot_10.png", "shot_2.png", "shot_1.png"):
                img = np.full((600, 500, 3), 180, dtype=np.uint8)
                cv2.imwrite(str(tmp_dir / name), img)

            discovered = discover_images([tmp_dir])
            self.assertEqual(
                [p.name for p in discovered],
                ["shot_1.png", "shot_2.png", "shot_10.png"],
            )

            app = DartAnnotatorApp(inputs=[tmp_dir], output_json=out_json)
            self.assertEqual(app.image_paths[app.current_idx].name, "shot_1.png")

            # Calibrate shot_1 and register 1 hit
            ann1 = app._current_annotation()
            w, h = ann1.image_size
            for sx, sy in [(w * 0.45, h * 0.15), (w * 0.85, h * 0.48), (w * 0.52, h * 0.85), (w * 0.15, h * 0.52)]:
                app._handle_left_click("left", *app.left_vp.source_to_panel(sx, sy))
            app._handle_left_click("right", *app.right_vp.source_to_panel(450.0, 450.0))
            self.assertEqual(len(ann1.hits), 1)

            # Move to shot_2 -> inherits homography and hit #1
            app._switch_image(+1)
            self.assertEqual(app.image_paths[app.current_idx].name, "shot_2.png")
            ann2 = app._current_annotation()
            self.assertTrue(ann2.is_calibrated)
            self.assertEqual(len(ann2.hits), 1)
            np.testing.assert_allclose(ann2.h_image_to_schematic, ann1.h_image_to_schematic)

            # Register hit #2 on shot_2; shot_1 must remain at 1 hit
            app._handle_left_click("right", *app.right_vp.source_to_panel(450.0, 300.0))
            self.assertEqual(len(ann2.hits), 2)
            self.assertEqual(len(ann1.hits), 1)

            # Going back to shot_1 preserves shot_1's 1 hit, and returning to shot_2 keeps shot_2's 2 hits
            app._switch_image(-1)
            self.assertEqual(len(app._current_annotation().hits), 1)
            app._switch_image(+1)
            self.assertEqual(len(app._current_annotation().hits), 2)

    def test_edit_homography_by_dragging_calibration_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_json = tmp_dir / "annotations.json"
            img = np.full((800, 600, 3), 180, dtype=np.uint8)
            img_path = tmp_dir / "board.png"
            cv2.imwrite(str(img_path), img)

            app = DartAnnotatorApp(inputs=[img_path], output_json=out_json)
            ann = app._current_annotation()
            w, h = ann.image_size
            for sx, sy in [(w * 0.45, h * 0.15), (w * 0.85, h * 0.48), (w * 0.52, h * 0.85), (w * 0.15, h * 0.52)]:
                app._handle_left_click("left", *app.left_vp.source_to_panel(sx, sy))

            # Register a hit from image before editing homography
            app._handle_left_click("left", *app.left_vp.source_to_panel(w * 0.5, h * 0.5))
            old_h = ann.h_image_to_schematic.copy()
            old_sch_xy = ann.hits[0].schematic_xy

            # Turn off grid, enter Edit Homography mode (wireframe must still render), grab & drag C1
            app.show_grid = False
            app._toggle_edit_homography()
            self.assertTrue(app.edit_homography)

            c1_px, c1_py = app.left_vp.source_to_panel(*ann.calibration_image_pts[0])
            # Simulate mouse down on C1 in left panel
            wx = int(round(c1_px))
            wy = int(round(c1_py)) + 68  # HEADER_HEIGHT = 68
            app._on_mouse(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)
            self.assertEqual(app.dragging_cal_idx, 0)

            # Drag C1 right & down by 25 panel pixels
            app._on_mouse(cv2.EVENT_MOUSEMOVE, wx + 25, wy + 18, 0, None)
            frame = app._render_frame()
            self.assertEqual(frame.shape, (app.win_height, app.win_width, 3))
            app._on_mouse(cv2.EVENT_LBUTTONUP, wx + 25, wy + 18, 0, None)
            self.assertIsNone(app.dragging_cal_idx)

            # Verify homography & hit schematic projection updated and persisted
            self.assertFalse(np.allclose(ann.h_image_to_schematic, old_h))
            self.assertNotEqual(ann.hits[0].schematic_xy, old_sch_xy)
            self.assertEqual(len(ann.hits), 1)  # stray/drag clicks must not create extra hits

            reloaded = load_annotations(out_json)
            np.testing.assert_allclose(
                reloaded[app.current_key].h_image_to_schematic,
                ann.h_image_to_schematic,
            )


if __name__ == "__main__":
    unittest.main()
