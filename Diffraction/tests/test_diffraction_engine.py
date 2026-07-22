"""Regression tests for the analytical Fraunhofer simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.special import jn_zeros

DIFFRACTION_DIR = Path(__file__).resolve().parents[1]
if str(DIFFRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFRACTION_DIR))

import diffraction_config as config
import diffraction_engine as engine
import screen_tools as screen


class ApertureAmplitudeTests(unittest.TestCase):
    def test_medium_wavelength_uses_refractive_index(self):
        wavelength_vacuum = 633e-9
        self.assertEqual(
            engine.medium_wavelength(wavelength_vacuum, 1.5),
            wavelength_vacuum / 1.5,
        )

    def test_rectangle_center_and_first_zeros(self):
        rectangle = engine.Aperture.rectangle(80e-6, 200e-6)
        fx = np.array([0.0, 1.0 / rectangle.width, 0.0])
        fy = np.array([0.0, 0.0, 1.0 / rectangle.height])
        amplitude = engine.aperture_amplitude(fx, fy, rectangle)

        np.testing.assert_allclose(amplitude[0], rectangle.area)
        np.testing.assert_allclose(
            amplitude[1:], 0.0, atol=rectangle.area * 1e-12
        )

    def test_circle_center_limit_and_first_zero(self):
        circle = engine.Aperture.circle(125e-6)
        root = jn_zeros(1, 1)[0]
        first_zero_frequency = root / (2.0 * np.pi * circle.radius)
        amplitude = engine.aperture_amplitude(
            np.array([0.0, first_zero_frequency]), np.zeros(2), circle
        )

        np.testing.assert_allclose(amplitude[0], circle.area)
        np.testing.assert_allclose(
            amplitude[1], 0.0, atol=circle.area * 1e-12
        )

    def test_translation_changes_phase_not_single_aperture_intensity(self):
        centered = engine.Aperture.rectangle(50e-6, 90e-6)
        shifted = engine.Aperture.rectangle(
            50e-6, 90e-6, center_x=120e-6, center_y=-75e-6
        )
        fx = np.array([1_000.0, 2_000.0, 3_000.0])
        fy = np.array([-500.0, 750.0, 1_250.0])
        centered_field = engine.aperture_amplitude(fx, fy, centered)
        shifted_field = engine.aperture_amplitude(fx, fy, shifted)
        expected_phase = np.exp(
            -2j
            * np.pi
            * (fx * shifted.center_x + fy * shifted.center_y)
        )

        np.testing.assert_allclose(shifted_field, centered_field * expected_phase)
        np.testing.assert_allclose(
            np.abs(shifted_field) ** 2, np.abs(centered_field) ** 2
        )

    def test_two_slits_match_cosine_interference_factor(self):
        width, length, separation = 30e-6, 150e-6, 180e-6
        apertures = [
            engine.Aperture.slit(width, length, -separation / 2.0, 0.0),
            engine.Aperture.slit(width, length, separation / 2.0, 0.0),
        ]
        fx = np.linspace(-20_000.0, 20_000.0, 101)
        fy = np.zeros_like(fx)

        actual = engine.composite_amplitude(fx, fy, apertures)
        single_envelope = width * length * np.sinc(width * fx)
        expected = (
            2.0
            * single_envelope
            * np.cos(np.pi * separation * fx)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-20)

    def test_obstacle_is_negative_of_matching_opening(self):
        opening = engine.Aperture.rectangle(60e-6, 90e-6, center_x=40e-6)
        obstacle = engine.Aperture.rectangle(
            60e-6, 90e-6, center_x=40e-6, is_obstacle=True
        )
        fx = np.linspace(-15_000.0, 15_000.0, 41)
        fy = np.linspace(-8_000.0, 8_000.0, 41)
        FX, FY = np.meshgrid(fx, fy)

        opening_field = engine.aperture_amplitude(FX, FY, opening)
        obstacle_field = engine.aperture_amplitude(FX, FY, obstacle)

        np.testing.assert_allclose(obstacle_field, -opening_field)
        np.testing.assert_allclose(
            np.abs(obstacle_field) ** 2, np.abs(opening_field) ** 2
        )
        self.assertEqual(obstacle.signed_area, -opening.area)

    def test_opening_plus_matching_obstacle_cancels(self):
        apertures = [
            engine.Aperture.circle(80e-6),
            engine.Aperture.circle(80e-6, is_obstacle=True),
        ]
        fx = np.linspace(-10_000.0, 10_000.0, 31)
        fy = np.zeros_like(fx)
        field = engine.composite_amplitude(fx, fy, apertures)
        np.testing.assert_allclose(field, 0.0, atol=1e-18)

    def test_mixed_aperture_intensity_is_finite_and_normalized(self):
        apertures = [
            engine.Aperture.slit(30e-6, 120e-6, center_x=-200e-6),
            engine.Aperture.circle(45e-6, center_x=200e-6),
        ]
        wavelength = 532e-9
        probe = engine.evaluate_far_field(
            apertures,
            wavelength,
            distance=1.0,
            max_fresnel_number=config.MAX_FRESNEL_NUMBER.default,
        )
        coordinates = np.linspace(-0.01, 0.01, 51)
        X, Y = np.meshgrid(coordinates, coordinates)
        intensity, _ = engine.fraunhofer_intensity(
            X,
            Y,
            apertures,
            wavelength,
            2.0 * probe.required_distance,
        )

        self.assertTrue(np.isfinite(intensity).all())
        np.testing.assert_allclose(intensity[25, 25], 1.0)


class FirstMinimumTests(unittest.TestCase):
    def test_first_minima_scale_with_medium_wavelength(self):
        wavelength, distance, n = 600e-9, 3.0, 1.5
        width, height, radius = 100e-6, 250e-6, 75e-6
        wavelength_medium = wavelength / n

        self.assertEqual(
            engine.slit_first_minimum(wavelength, distance, width, n),
            wavelength_medium * distance / width,
        )
        self.assertEqual(
            engine.rectangle_first_minima(
                wavelength, distance, width, height, n
            ),
            (
                wavelength_medium * distance / width,
                wavelength_medium * distance / height,
            ),
        )
        self.assertEqual(
            engine.circular_first_minimum(
                wavelength, distance, radius, n
            ),
            0.61 * wavelength_medium * distance / radius,
        )


class FarFieldGateTests(unittest.TestCase):
    def setUp(self):
        self.apertures = [
            engine.Aperture.rectangle(
                100e-6, 250e-6, center_x=80e-6, center_y=-40e-6
            )
        ]
        self.wavelength = 550e-9
        self.threshold = 0.1
        probe = engine.evaluate_far_field(
            self.apertures,
            self.wavelength,
            distance=1.0,
            n=1.2,
            max_fresnel_number=self.threshold,
        )
        self.required_distance = probe.required_distance

    def test_gate_passes_above_and_fails_below_required_distance(self):
        valid = engine.require_far_field(
            self.apertures,
            self.wavelength,
            1.01 * self.required_distance,
            n=1.2,
            max_fresnel_number=self.threshold,
        )
        self.assertTrue(valid.is_valid)

        with self.assertRaises(engine.FarFieldError) as context:
            engine.require_far_field(
                self.apertures,
                self.wavelength,
                0.99 * self.required_distance,
                n=1.2,
                max_fresnel_number=self.threshold,
            )
        self.assertFalse(context.exception.report.is_valid)

    def test_screen_grid_is_not_created_when_gate_fails(self):
        with mock.patch.object(screen, "observation_grid") as grid:
            with self.assertRaises(engine.FarFieldError):
                screen.simulate_pattern(
                    self.apertures,
                    self.wavelength,
                    0.5 * self.required_distance,
                    n=1.2,
                    max_fresnel_number=self.threshold,
                )
        grid.assert_not_called()

    def test_guarded_screen_simulation_returns_profiles(self):
        result = screen.simulate_pattern(
            self.apertures,
            self.wavelength,
            2.0 * self.required_distance,
            n=1.2,
            screen_half_width=0.01,
            resolution=101,
            max_fresnel_number=self.threshold,
        )
        profiles = screen.extract_central_profiles(
            result.X, result.Y, result.intensity
        )

        self.assertEqual(result.intensity.shape, (101, 101))
        self.assertTrue(result.far_field.is_valid)
        np.testing.assert_allclose(result.intensity[50, 50], 1.0)
        self.assertEqual(profiles.horizontal.shape, (101,))
        self.assertEqual(profiles.vertical.shape, (101,))


if __name__ == "__main__":
    unittest.main()
