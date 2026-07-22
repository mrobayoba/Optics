"""Regression tests for analytical slit and edge Fresnel diffraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

DIFFRACTION_DIR = Path(__file__).resolve().parents[1]
if str(DIFFRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFRACTION_DIR))

import fresnel_engine as engine
import fresnel_screen_tools as fresnel_screen
import screen_tools as shared_screen


class FresnelIntegralTests(unittest.TestCase):
    def test_pdf_finite_source_slit_example(self):
        intensity = engine.slit_intensity(
            0.0,
            slit_width=0.58e-3,
            wavelength_vacuum=592e-9,
            distance=1.0,
            illumination="point",
            source_distance=1.0,
        )
        self.assertAlmostEqual(float(intensity), 1.0584, places=4)

    def test_pdf_finite_source_edge_examples(self):
        scale = engine.screen_fresnel_scale(
            592e-9,
            1.0,
            illumination="point",
            source_distance=1.0,
        )
        shadow = engine.edge_intensity(
            0.7 / scale,
            592e-9,
            1.0,
            illumination="point",
            source_distance=1.0,
        )
        illuminated = engine.edge_intensity(
            -0.8 / scale,
            592e-9,
            1.0,
            illumination="point",
            source_distance=1.0,
        )

        self.assertAlmostEqual(float(shadow), 0.06649, places=4)
        self.assertAlmostEqual(float(illuminated), 1.02843, places=4)

    def test_edge_boundary_and_deep_region_limits(self):
        wavelength, distance = 633e-9, 1.0
        scale = engine.screen_fresnel_scale(wavelength, distance)
        boundary = engine.edge_intensity(0.0, wavelength, distance)
        limits = engine.edge_intensity(
            np.array([-100.0, 100.0]) / scale,
            wavelength,
            distance,
        )

        self.assertEqual(float(boundary), 0.25)
        np.testing.assert_allclose(limits, [1.0, 0.0], atol=0.004)

    def test_slit_intensity_is_symmetric(self):
        coordinates = np.linspace(-4e-3, 4e-3, 501)
        intensity = engine.slit_intensity(
            coordinates, 1e-3, 633e-9, 0.5
        )
        np.testing.assert_allclose(intensity, intensity[::-1], atol=1e-13)

    def test_point_source_converges_to_plane_wave(self):
        coordinates = np.linspace(-2e-3, 2e-3, 101)
        plane = engine.slit_intensity(
            coordinates, 0.8e-3, 532e-9, 0.7
        )
        distant_point = engine.slit_intensity(
            coordinates,
            0.8e-3,
            532e-9,
            0.7,
            illumination="point",
            source_distance=1e9,
        )
        np.testing.assert_allclose(distant_point, plane, rtol=1e-8, atol=1e-9)


class FresnelGeometryTests(unittest.TestCase):
    def test_effective_distance_and_magnification(self):
        self.assertEqual(engine.effective_distance(2.0), 2.0)
        self.assertEqual(
            engine.effective_distance(
                2.0, illumination="point", source_distance=1.0
            ),
            2.0 / 3.0,
        )
        self.assertEqual(
            engine.geometric_magnification(
                2.0, illumination="point", source_distance=1.0
            ),
            3.0,
        )

    def test_fresnel_number_uses_medium_and_effective_distance(self):
        width, wavelength = 1e-3, 600e-9
        actual = engine.fresnel_number(
            width,
            wavelength,
            1.0,
            n=1.5,
            illumination="point",
            source_distance=1.0,
        )
        expected = width**2 / (4.0 * (wavelength / 1.5) * 0.5)
        self.assertEqual(actual, expected)

    def test_refractive_index_contracts_characteristic_length(self):
        vacuum = engine.characteristic_screen_length(600e-9, 1.0, n=1.0)
        glass = engine.characteristic_screen_length(600e-9, 1.0, n=1.5)
        self.assertAlmostEqual(glass / vacuum, 1.0 / np.sqrt(1.5))

    def test_report_labels_slit_and_edge_without_blocking(self):
        slit = engine.evaluate_fresnel_regime(
            "slit", 633e-9, 100.0, slit_width=0.1e-3
        )
        edge = engine.evaluate_fresnel_regime("edge", 633e-9, 0.5)

        self.assertEqual(slit.regime, "far-field limit")
        self.assertFalse(slit.is_near_field)
        self.assertIsNotNone(slit.fresnel_number)
        self.assertEqual(edge.regime, "edge diffraction")
        self.assertIsNone(edge.is_near_field)
        self.assertIsNone(edge.fresnel_number)

        near_field = engine.evaluate_fresnel_regime(
            "slit", 633e-9, 0.05, slit_width=1e-3
        )
        self.assertTrue(near_field.is_near_field)

    def test_adjustable_condition_does_not_change_physical_regime(self):
        permissive = engine.evaluate_fresnel_regime(
            "slit",
            633e-9,
            100.0,
            slit_width=0.1e-3,
            near_field_limit=1e-6,
        )
        strict = engine.evaluate_fresnel_regime(
            "slit",
            633e-9,
            100.0,
            slit_width=0.1e-3,
            near_field_limit=10.0,
        )

        self.assertEqual(permissive.regime, "far-field limit")
        self.assertEqual(strict.regime, permissive.regime)
        self.assertTrue(permissive.is_near_field)
        self.assertFalse(strict.is_near_field)
        self.assertEqual(permissive.near_field_limit, 1e-6)
        self.assertEqual(strict.near_field_limit, 10.0)

    def test_near_field_limit_must_be_positive_and_finite(self):
        for invalid in (0.0, -0.1, np.inf, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    engine.evaluate_fresnel_regime(
                        "slit",
                        633e-9,
                        0.5,
                        slit_width=1e-3,
                        near_field_limit=invalid,
                    )

    def test_screen_result_carries_selected_condition(self):
        result = fresnel_screen.simulate_pattern(
            "slit",
            633e-9,
            0.5,
            slit_width=1e-3,
            near_field_limit=2.5,
            resolution=51,
        )
        self.assertEqual(result.report.near_field_limit, 2.5)
        self.assertFalse(result.report.is_near_field)

    def test_far_field_limit_converges_to_normalized_sinc_squared(self):
        width, wavelength, distance = 0.1e-3, 633e-9, 100.0
        coordinates = np.linspace(-0.5, 0.5, 501)
        fresnel = engine.slit_intensity(
            coordinates, width, wavelength, distance
        )
        fresnel /= fresnel[len(fresnel) // 2]
        expected = np.sinc(width * coordinates / (wavelength * distance)) ** 2

        np.testing.assert_allclose(fresnel, expected, rtol=3e-4, atol=3e-5)

    def test_invalid_geometry_fails_before_screen_grid_allocation(self):
        with mock.patch.object(shared_screen, "observation_grid") as grid:
            with self.assertRaises(ValueError):
                fresnel_screen.simulate_pattern(
                    "slit",
                    633e-9,
                    -1.0,
                    slit_width=1e-3,
                )
        grid.assert_not_called()


class FresnelScreenTests(unittest.TestCase):
    def test_slit_orientation_selects_varying_axis(self):
        vertical = fresnel_screen.simulate_pattern(
            "slit",
            633e-9,
            0.5,
            slit_width=1e-3,
            orientation="vertical",
            resolution=101,
        )
        horizontal = fresnel_screen.simulate_pattern(
            "slit",
            633e-9,
            0.5,
            slit_width=1e-3,
            orientation="horizontal",
            resolution=101,
        )

        np.testing.assert_allclose(vertical.intensity[0], vertical.intensity[50])
        np.testing.assert_allclose(
            horizontal.intensity[:, 0], horizontal.intensity[:, 50]
        )
        np.testing.assert_allclose(horizontal.intensity, vertical.intensity.T)

    def test_edge_shadow_side_mirrors_pattern(self):
        positive = fresnel_screen.simulate_pattern(
            "edge",
            633e-9,
            0.5,
            shadow_side="positive",
            resolution=101,
        )
        negative = fresnel_screen.simulate_pattern(
            "edge",
            633e-9,
            0.5,
            shadow_side="negative",
            resolution=101,
        )

        np.testing.assert_allclose(
            positive.intensity, np.fliplr(negative.intensity), atol=1e-13
        )
        self.assertEqual(positive.intensity[50, 50], 0.25)


if __name__ == "__main__":
    unittest.main()
