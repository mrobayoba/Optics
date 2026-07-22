"""Tests for Fresnel dashboard visuals and widget behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np

DIFFRACTION_DIR = Path(__file__).resolve().parents[1]
if str(DIFFRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFRACTION_DIR))

import fresnel_config as config
import fresnel_dashboard_tools as visuals
import fresnel_engine as engine
import fresnel_screen_tools as screen
import fresnel_style


class FresnelDrawingTests(unittest.TestCase):
    def setUp(self):
        self.wavelength = 633e-9
        self.slit_report = engine.evaluate_fresnel_regime(
            "slit",
            self.wavelength,
            0.5,
            slit_width=1e-3,
        )
        self.edge_report = engine.evaluate_fresnel_regime(
            "edge", self.wavelength, 0.5
        )

    def tearDown(self):
        plt.close("all")

    def test_schematic_and_both_aperture_previews_draw(self):
        figure, axes = plt.subplots(1, 3)
        visuals.draw_optical_schematic(
            axes[0],
            "slit",
            "plane",
            0.5,
            None,
            633.0,
            self.slit_report,
        )
        slit_extent = visuals.draw_aperture_preview(
            axes[1], "slit", slit_width=1e-3, orientation="vertical"
        )
        edge_extent = visuals.draw_aperture_preview(
            axes[2],
            "edge",
            orientation="horizontal",
            shadow_side="negative",
        )

        self.assertGreater(len(axes[0].patches), 0)
        self.assertEqual(len(slit_extent), 4)
        self.assertEqual(len(edge_extent), 4)
        self.assertGreaterEqual(len(axes[1].patches), 1)
        self.assertGreaterEqual(len(axes[2].patches), 1)

    def test_fresnel_camera_draws_wavelength_image(self):
        result = screen.simulate_pattern(
            "edge", self.wavelength, 0.5, resolution=101
        )
        figure, axis = plt.subplots()
        extent = visuals.draw_fresnel_preview(
            axis, result.X, result.Y, result.intensity, 633.0
        )

        self.assertEqual(len(extent), 4)
        self.assertEqual(len(axis.images), 1)
        self.assertEqual(axis.get_title(), "FRESNEL PATTERN")

    def test_slit_cornu_chord_matches_intensity(self):
        probe = 0.35e-3
        figure, axis = plt.subplots()
        start, finish = visuals.draw_cornu_construction(
            axis,
            "slit",
            probe,
            self.wavelength,
            0.5,
            slit_width=1e-3,
        )
        chord_intensity = abs(finish - start) ** 2 / 2.0
        expected = engine.slit_intensity(
            probe, 1e-3, self.wavelength, 0.5
        )

        self.assertAlmostEqual(chord_intensity, float(expected), places=13)
        self.assertGreaterEqual(len(axis.lines), 2)

    def test_edge_cornu_chord_matches_intensity_and_shadow_direction(self):
        probe = 0.2e-3
        figure, axes = plt.subplots(1, 2)
        positive = visuals.draw_cornu_construction(
            axes[0],
            "edge",
            probe,
            self.wavelength,
            0.5,
            shadow_side="positive",
        )
        negative = visuals.draw_cornu_construction(
            axes[1],
            "edge",
            probe,
            self.wavelength,
            0.5,
            shadow_side="negative",
        )

        positive_intensity = abs(positive[1] - positive[0]) ** 2 / 2.0
        negative_intensity = abs(negative[1] - negative[0]) ** 2 / 2.0
        self.assertNotAlmostEqual(positive_intensity, negative_intensity)


class FresnelWidgetTests(unittest.TestCase):
    def test_dashboard_builds_both_case_models(self):
        app = fresnel_style.FresnelDashboard()
        app.auto_update.value = False

        self.assertIsNotNone(app.widget)
        self.assertEqual(
            app.aperture_preview_output.layout.justify_content, "center"
        )
        self.assertEqual(
            app.diffraction_preview_output.layout.justify_content, "center"
        )
        for case in ("slit", "edge"):
            app.case.value = case
            result = screen.simulate_pattern(**app._simulation_parameters())
            self.assertEqual(result.intensity.shape, (401, 401))

    def test_source_mode_and_case_controls_switch(self):
        app = fresnel_style.FresnelDashboard()
        app.auto_update.value = False

        self.assertNotIn(app.source_distance_box, app.source_controls.children)
        self.assertIn(app.slit_width_box, app.case_controls.children)
        app.illumination.value = "point"
        app.case.value = "edge"

        self.assertIn(app.source_distance_box, app.source_controls.children)
        self.assertIn(app.shadow_side, app.case_controls.children)
        self.assertNotIn(app.slit_width_box, app.case_controls.children)

    def test_logarithmic_near_field_control_propagates_and_disables_for_edge(self):
        app = fresnel_style.FresnelDashboard()
        app.auto_update.value = False

        self.assertIsInstance(app.near_field_limit, widgets.FloatLogSlider)
        self.assertAlmostEqual(app.near_field_limit.min, -4.0)
        self.assertAlmostEqual(app.near_field_limit.max, 1.0)
        self.assertEqual(
            app.near_field_limit.value, config.NEAR_FIELD_LIMIT.default
        )
        app.near_field_limit.value = 2.5
        parameters = app._simulation_parameters()
        self.assertEqual(parameters["near_field_limit"], 2.5)
        self.assertTrue(
            all(
                not control.disabled
                for control in app.near_field_limit_box.children
            )
        )

        app.case.value = "edge"
        self.assertTrue(
            all(
                control.disabled
                for control in app.near_field_limit_box.children
            )
        )

    def test_status_reports_selected_condition_and_physical_regime(self):
        app = fresnel_style.FresnelDashboard()
        app.auto_update.value = False
        report = engine.evaluate_fresnel_regime(
            "slit",
            633e-9,
            0.5,
            slit_width=1e-3,
            near_field_limit=10.0,
        )
        app._set_status(report=report)

        self.assertIn("NOT MET", app.validity_status.value)
        self.assertIn("N<sub>F,min</sub>=10", app.validity_status.value)
        self.assertIn(report.regime, app.validity_status.value)

    def test_cornu_toggle_exposes_probe_and_reset_restores_defaults(self):
        app = fresnel_style.FresnelDashboard()
        app.auto_update.value = False
        self.assertEqual(app.cornu_probe_box.layout.display, "none")

        app.extra_view_buttons["cornu"].value = True
        self.assertEqual(app.cornu_probe_box.layout.display, "")
        app.case.value = "edge"
        app.illumination.value = "point"
        app.shadow_side.value = "negative"
        app.near_field_limit.value = 4.0
        app.cornu_probe.value = 1.5
        app.reset()

        self.assertEqual(app.case.value, config.DEFAULT_CASE)
        self.assertEqual(
            app.illumination.value, config.DEFAULT_ILLUMINATION
        )
        self.assertEqual(app.shadow_side.value, config.DEFAULT_SHADOW_SIDE)
        self.assertEqual(app.cornu_probe.value, config.CORNU_PROBE_MM.default)
        self.assertEqual(
            app.near_field_limit.value, config.NEAR_FIELD_LIMIT.default
        )
        self.assertFalse(app.extra_view_buttons["cornu"].value)


if __name__ == "__main__":
    unittest.main()
