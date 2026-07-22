"""Tests for arbitrary-aperture numerical Fraunhofer diffraction."""

from __future__ import annotations

import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

DIFFRACTION_DIR = Path(__file__).resolve().parents[1]
if str(DIFFRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFRACTION_DIR))

import diffraction_engine as engine
import fourier_transform as fourier
import screen_tools as screen


class CustomApertureConstructionTests(unittest.TestCase):
    def test_callable_receives_si_coordinates(self):
        aperture = fourier.CustomAperture.from_callable(
            lambda X, Y: (np.abs(X) <= 0.1e-3)
            & (np.abs(Y) <= 0.15e-3),
            width=0.5e-3,
            height=0.5e-3,
            resolution=200,
        )

        self.assertEqual(aperture.transmittance.shape, (200, 200))
        self.assertAlmostEqual(aperture.width, 0.5e-3)
        self.assertAlmostEqual(aperture.height, 0.5e-3)
        self.assertAlmostEqual(aperture.area, 0.2e-3 * 0.3e-3, places=12)
        self.assertGreater(aperture.support_radius, 0.15e-3)

    def test_expression_uses_millimetres_and_boolean_operators(self):
        aperture = fourier.CustomAperture.from_expression(
            "(r <= 0.16) & ~((x > 0.05) & (abs(y) < 0.03))",
            width=0.5e-3,
            height=0.5e-3,
            resolution=128,
        )

        center = len(aperture.x_axis) // 2
        self.assertEqual(aperture.transmittance[center, center], 1.0)
        self.assertEqual(aperture.transmittance[center, -1], 0.0)
        self.assertTrue(set(np.unique(aperture.transmittance)) <= {0.0, 1.0})

    def test_expression_invert_swaps_open_and_closed_regions(self):
        normal = fourier.CustomAperture.from_expression(
            "r <= 0.15",
            width=0.5e-3,
            height=0.5e-3,
            resolution=64,
        )
        inverted = fourier.CustomAperture.from_expression(
            "r <= 0.15",
            width=0.5e-3,
            height=0.5e-3,
            resolution=64,
            invert=True,
        )

        np.testing.assert_array_equal(
            inverted.transmittance, 1.0 - normal.transmittance
        )

    def test_unsafe_expression_constructs_are_rejected(self):
        forbidden = (
            "x.__class__",
            "__import__('os')",
            "x[0]",
            "(lambda q: q)(x)",
            "[q for q in x]",
            "unknown(x)",
            "x ** 100",
        )
        coordinates = np.zeros((4, 4))
        for expression in forbidden:
            with self.subTest(expression=expression):
                with self.assertRaises(fourier.UnsafeExpressionError):
                    fourier.evaluate_expression(
                        expression, coordinates, coordinates
                    )

    def test_expression_transmittance_must_be_in_unit_interval(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            fourier.CustomAperture.from_expression(
                "2 * (r <= 0.1)",
                width=0.5e-3,
                height=0.5e-3,
                resolution=64,
            )

    def test_image_threshold_flip_and_invert(self):
        image = np.zeros((2, 2), dtype=np.uint8)
        image[0, 0] = 255
        aperture = fourier.CustomAperture.from_image_array(
            image,
            width=0.4e-3,
            height=0.4e-3,
            resolution=8,
            threshold=0.5,
        )
        inverted = fourier.CustomAperture.from_image_array(
            image,
            width=0.4e-3,
            height=0.4e-3,
            resolution=8,
            threshold=0.5,
            invert=True,
        )

        self.assertEqual(aperture.transmittance[-1, 0], 1.0)
        self.assertEqual(aperture.transmittance[0, 0], 0.0)
        np.testing.assert_array_equal(
            inverted.transmittance, 1.0 - aperture.transmittance
        )

    def test_image_bytes_decode_and_empty_mask_rejection(self):
        source = np.zeros((12, 16), dtype=np.uint8)
        source[3:9, 5:11] = 255
        buffer = BytesIO()
        Image.fromarray(source).save(buffer, format="PNG")

        aperture = fourier.CustomAperture.from_image_bytes(
            buffer.getvalue(),
            width=0.8e-3,
            height=0.6e-3,
            resolution=(96, 128),
        )
        self.assertEqual(aperture.transmittance.shape, (96, 128))
        self.assertGreater(aperture.area, 0.0)

        with self.assertRaisesRegex(ValueError, "no open pixels"):
            fourier.CustomAperture.from_image_array(
                np.zeros((8, 8)),
                width=1e-3,
                height=1e-3,
                resolution=8,
            )


class NumericalFourierTests(unittest.TestCase):
    wavelength = 633e-9

    def test_fft_continuous_scaling_normalization_and_padding(self):
        aperture = fourier.CustomAperture.from_expression(
            "(abs(x) <= 0.1) & (abs(y) <= 0.15)",
            width=0.5e-3,
            height=0.5e-3,
            resolution=128,
        )
        raw = fourier.fraunhofer_fft(
            aperture,
            self.wavelength,
            distance=20.0,
            padding_factor=2.0,
            normalize=False,
        )
        normalized = fourier.fraunhofer_fft(
            aperture,
            self.wavelength,
            distance=20.0,
            padding_factor=2.0,
        )
        center = tuple(length // 2 for length in raw.intensity.shape)

        self.assertEqual(raw.intensity.shape, (256, 256))
        self.assertAlmostEqual(abs(raw.field[center]), aperture.area, places=14)
        self.assertAlmostEqual(normalized.intensity[center], 1.0, places=12)
        self.assertTrue(normalized.far_field.is_valid)

    def test_rectangle_matches_analytical_sinc_pattern(self):
        opening_width = 0.2e-3
        opening_height = 0.3e-3
        aperture = fourier.CustomAperture.from_callable(
            lambda X, Y: (np.abs(X) <= opening_width / 2.0)
            & (np.abs(Y) <= opening_height / 2.0),
            width=0.8e-3,
            height=0.8e-3,
            resolution=256,
        )
        distance = 20.0
        numerical = fourier.simulate_pattern(
            aperture,
            self.wavelength,
            distance,
            screen_half_width=0.08,
            resolution=301,
            padding_factor=8.0,
        )
        analytical = screen.simulate_pattern(
            [engine.Aperture.rectangle(opening_width, opening_height)],
            self.wavelength,
            distance,
            screen_half_width=0.08,
            resolution=301,
        )
        center = numerical.intensity.shape[0] // 2
        numerical_profile = numerical.intensity[center]
        analytical_profile = analytical.intensity[center]
        mean_error = float(np.mean(np.abs(numerical_profile - analytical_profile)))

        self.assertLess(mean_error, 0.025)
        expected_zero = self.wavelength * distance / opening_width
        zero_index = int(
            np.argmin(np.abs(numerical.X[center] - expected_zero))
        )
        self.assertLess(numerical_profile[zero_index], 0.02)

    def test_circle_first_minimum_matches_airy_pattern(self):
        radius = 0.15e-3
        aperture = fourier.CustomAperture.from_callable(
            lambda X, Y: np.hypot(X, Y) <= radius,
            width=0.5e-3,
            height=0.5e-3,
            resolution=256,
        )
        distance = 10.0
        result = fourier.simulate_pattern(
            aperture,
            self.wavelength,
            distance,
            screen_half_width=0.05,
            resolution=401,
            padding_factor=8.0,
        )
        analytical = screen.simulate_pattern(
            [engine.Aperture.circle(radius)],
            self.wavelength,
            distance,
            screen_half_width=0.05,
            resolution=401,
        )
        center = result.intensity.shape[0] // 2
        expected_zero = engine.circular_first_minimum(
            self.wavelength, distance, radius
        )
        zero_index = int(
            np.argmin(np.abs(result.X[center] - expected_zero))
        )

        self.assertAlmostEqual(result.intensity[center, center], 1.0, places=12)
        self.assertLess(result.intensity[center, zero_index], 0.02)
        self.assertLess(
            float(
                np.mean(
                    np.abs(
                        result.intensity[center]
                        - analytical.intensity[center]
                    )
                )
            ),
            0.025,
        )

    def test_far_field_failure_happens_before_fft_or_screen_allocation(self):
        aperture = fourier.CustomAperture.from_expression(
            "r <= 0.2",
            width=0.5e-3,
            height=0.5e-3,
            resolution=64,
        )
        report = engine.evaluate_far_field_radius(
            aperture.support_radius,
            self.wavelength,
            distance=1.0,
        )
        with mock.patch.object(fourier, "_fft_unchecked") as transform:
            with mock.patch.object(screen, "observation_grid") as grid:
                with self.assertRaises(engine.FarFieldError):
                    fourier.simulate_pattern(
                        aperture,
                        self.wavelength,
                        distance=report.required_distance / 2.0,
                    )
        transform.assert_not_called()
        grid.assert_not_called()


if __name__ == "__main__":
    unittest.main()
