"""Tests for the notebook dashboard presentation helpers."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DIFFRACTION_DIR = Path(__file__).resolve().parents[1]
if str(DIFFRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFRACTION_DIR))

import dashboard_tools as dashboard
import diffraction_config as config
import diffraction_engine as engine
import diffraction_style
import fourier_transform as fourier


class WavelengthColorTests(unittest.TestCase):
    def test_wavelength_control_has_hundredth_nanometre_precision(self):
        self.assertEqual(config.WAVELENGTH_NM.step, 0.01)

    def test_profiles_is_the_only_optional_dashboard_view(self):
        self.assertEqual(config.EXTRA_VIEW_OPTIONS, (("Profiles", "profiles"),))

    def test_spectrum_marker_uses_visible_wavelength_scale(self):
        self.assertEqual(
            dashboard.wavelength_to_spectrum_position(380.0), 0.0
        )
        self.assertEqual(
            dashboard.wavelength_to_spectrum_position(580.0), 50.0
        )
        self.assertEqual(
            dashboard.wavelength_to_spectrum_position(780.0), 100.0
        )
        self.assertEqual(
            dashboard.wavelength_to_spectrum_position(1_000.0), 100.0
        )

    def test_visible_wavelengths_have_expected_dominant_channels(self):
        violet = dashboard.wavelength_to_rgb(420.0)
        green = dashboard.wavelength_to_rgb(530.0)
        red = dashboard.wavelength_to_rgb(650.0)

        self.assertGreater(violet[2], violet[1])
        self.assertGreater(green[1], green[0])
        self.assertGreater(red[0], red[1])
        for color in (violet, green, red):
            self.assertTrue(all(0.0 <= channel <= 1.0 for channel in color))

    def test_out_of_visible_range_is_clamped(self):
        np.testing.assert_allclose(
            dashboard.wavelength_to_rgb(1_000.0),
            dashboard.wavelength_to_rgb(780.0),
        )

    def test_invalid_wavelength_is_rejected(self):
        with self.assertRaises(ValueError):
            dashboard.wavelength_to_rgb(0.0)

    def test_intensity_rgb_is_normalized_and_keeps_shape(self):
        intensity = np.array([[0.0, 0.25], [0.5, 2.0]])
        rgb = dashboard.intensity_to_rgb(intensity, 633.0, gamma=1.0)

        self.assertEqual(rgb.shape, (2, 2, 3))
        np.testing.assert_allclose(rgb[0, 0], 0.0)
        np.testing.assert_allclose(
            rgb[1, 1], dashboard.wavelength_to_rgb(633.0)
        )
        self.assertTrue(np.all((0.0 <= rgb) & (rgb <= 1.0)))

    def test_zero_intensity_produces_black_image(self):
        rgb = dashboard.intensity_to_rgb(np.zeros((3, 4)), 532.0)
        np.testing.assert_array_equal(rgb, np.zeros((3, 4, 3)))

    def test_lower_gamma_reveals_weak_secondary_intensity(self):
        intensity = np.array([[1.0, 0.01]])
        linear = dashboard.intensity_to_rgb(intensity, 633.0, gamma=1.0)
        boosted = dashboard.intensity_to_rgb(intensity, 633.0, gamma=0.25)

        self.assertGreater(boosted[0, 1, 0], linear[0, 1, 0])
        np.testing.assert_allclose(boosted[0, 0], linear[0, 0])


class DashboardDrawingTests(unittest.TestCase):
    def setUp(self):
        self.apertures = [
            engine.Aperture.slit(40e-6, 180e-6, center_x=-120e-6),
            engine.Aperture.circle(55e-6, center_x=120e-6),
        ]
        self.wavelength = 633e-9
        self.valid_report = engine.evaluate_far_field(
            self.apertures,
            self.wavelength,
            distance=5.0,
        )
        self.invalid_report = engine.evaluate_far_field(
            self.apertures,
            self.wavelength,
            distance=self.valid_report.required_distance / 2.0,
        )

    def tearDown(self):
        plt.close("all")

    def test_nice_scale_length_uses_readable_increment(self):
        self.assertEqual(dashboard.nice_scale_length(10.0), 2.0)
        self.assertEqual(dashboard.nice_scale_length(0.9), 0.1)

    def test_schematic_and_aperture_preview_draw(self):
        figure, axes = plt.subplots(1, 2)
        dashboard.draw_optical_schematic(
            axes[0],
            self.apertures,
            distance=5.0,
            wavelength_nm=633.0,
            report=self.valid_report,
        )
        extent = dashboard.draw_aperture_preview(axes[1], self.apertures)

        self.assertEqual(len(extent), 4)
        self.assertGreater(len(axes[0].patches), 0)
        source_color = axes[0].collections[0].get_facecolor()[0, :3]
        np.testing.assert_allclose(
            source_color, dashboard.wavelength_to_rgb(633.0)
        )
        self.assertEqual(len(axes[1].patches), len(self.apertures))

    def test_diffraction_and_invalid_previews_draw(self):
        coordinates = np.linspace(-0.01, 0.01, 21)
        X, Y = np.meshgrid(coordinates, coordinates)
        intensity = np.sinc(X / 0.002) ** 2 * np.sinc(Y / 0.003) ** 2
        figure, axes = plt.subplots(1, 2)

        extent = dashboard.draw_diffraction_preview(
            axes[0], X, Y, intensity, 532.0
        )
        dashboard.draw_far_field_unavailable(axes[1], self.invalid_report)

        self.assertEqual(len(extent), 4)
        self.assertEqual(len(axes[0].images), 1)
        self.assertGreaterEqual(len(axes[1].texts), 2)

    def test_custom_raster_preview_is_centered_and_scaled(self):
        aperture = fourier.CustomAperture.from_expression(
            "(abs(x) < 0.15) & (abs(y) < 0.05)",
            width=0.5e-3,
            height=0.3e-3,
            resolution=64,
        )
        figure, axis = plt.subplots()
        extent = dashboard.draw_custom_aperture_preview(axis, aperture)

        self.assertEqual(len(axis.images), 1)
        self.assertAlmostEqual(extent[0], -extent[1])
        self.assertAlmostEqual(extent[2], -extent[3])
        self.assertEqual(axis.get_aspect(), 1.0)


class DashboardWidgetTests(unittest.TestCase):
    def test_dashboard_module_builds_widget_and_all_case_models(self):
        app = diffraction_style.FraunhoferDashboard()
        app.auto_update.value = False

        self.assertIsNotNone(app.widget)
        self.assertEqual(
            app.aperture_preview_output.layout.justify_content, "center"
        )
        self.assertEqual(
            app.diffraction_preview_output.layout.justify_content, "center"
        )
        for case_name in ("slit", "rectangle", "circle", "multiple"):
            app.case.value = case_name
            self.assertGreater(len(app._build_apertures()), 0)
        app.case.value = "custom"
        self.assertEqual(
            app._build_custom_aperture().transmittance.shape, (256, 256)
        )

    def test_custom_mode_switches_controls_and_reset_restores_defaults(self):
        app = diffraction_style.FraunhoferDashboard()
        app.auto_update.value = False
        app.case.value = "custom"
        self.assertIn(app.custom_expression, app.custom_mode_controls.children)
        self.assertIn(app.custom_invert, app.custom_mode_controls.children)
        normal = app._build_custom_aperture()
        app.custom_invert.value = True
        inverted = app._build_custom_aperture()
        np.testing.assert_array_equal(
            inverted.transmittance, 1.0 - normal.transmittance
        )

        app.custom_mode.value = "image"
        self.assertIn(app.custom_upload, app.custom_mode_controls.children)
        app.custom_width.value = 1.25
        app.custom_invert.value = True
        app.reset()

        self.assertEqual(app.case.value, "slit")
        self.assertEqual(app.custom_mode.value, "expression")
        self.assertEqual(app.custom_width.value, config.CUSTOM_WIDTH_MM.default)
        self.assertFalse(app.custom_invert.value)

    def test_uploaded_image_bytes_are_parsed_for_custom_builder(self):
        pixels = np.zeros((16, 16), dtype=np.uint8)
        pixels[4:12, 4:12] = 255
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        data = buffer.getvalue()
        upload = SimpleNamespace(
            value=({"name": "mask.png", "content": memoryview(data)},)
        )
        self.assertEqual(
            diffraction_style.FraunhoferDashboard._uploaded_file_bytes(upload),
            data,
        )

        app = diffraction_style.FraunhoferDashboard()
        app.auto_update.value = False
        app.case.value = "custom"
        app.custom_mode.value = "image"
        app.custom_upload.value = (
            {
                "name": "mask.png",
                "type": "image/png",
                "size": len(data),
                "content": memoryview(data),
                "last_modified": datetime.now(timezone.utc),
            },
        )
        custom = app._build_custom_aperture()
        self.assertGreater(custom.area, 0.0)


if __name__ == "__main__":
    unittest.main()
