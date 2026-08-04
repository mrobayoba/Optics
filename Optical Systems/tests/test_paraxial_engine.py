"""Regression tests for the paraxial optical-system module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

OPTICAL_SYSTEMS_DIR = Path(__file__).resolve().parents[1]
if str(OPTICAL_SYSTEMS_DIR) not in sys.path:
    sys.path.insert(0, str(OPTICAL_SYSTEMS_DIR))

import paraxial_engine as engine
import paraxial_tools as tools


class ElementalMatrixTests(unittest.TestCase):
    def test_source_interface_power_and_translation_examples(self):
        self.assertAlmostEqual(engine.interface_power(1.3, 1.0, 0.50), -0.6)
        np.testing.assert_allclose(
            engine.translation_matrix(1.0, 1.5),
            [[1.0, 0.0], [1.5, 1.0]],
        )

    def test_refraction_and_reflection_match_course_convention(self):
        np.testing.assert_allclose(
            engine.refraction_matrix(2.5),
            [[1.0, -2.5], [0.0, 1.0]],
        )
        np.testing.assert_allclose(
            engine.reflection_matrix(1.5, -0.3),
            [[1.0, -10.0], [0.0, 1.0]],
        )

    def test_thin_lens_is_two_refractions_and_thick_limit(self):
        thin = engine.thin_lens_matrix(3.0, 7.0)
        np.testing.assert_allclose(thin, engine.refraction_matrix(10.0))
        thick_at_zero = engine.thick_lens_matrix(1.7, 0.0, 3.0, 7.0)
        np.testing.assert_allclose(thick_at_zero, thin)

    def test_invalid_indices_and_radii_are_rejected(self):
        with self.assertRaises(ValueError):
            engine.translation_matrix(0.0, 1.0)
        with self.assertRaises(ValueError):
            engine.interface_power(1.0, 1.5, 0.0)


class CourseExerciseTests(unittest.TestCase):
    @staticmethod
    def _surface_powers():
        return (
            engine.interface_power(1.0, 1.812, 11.5),
            engine.interface_power(1.812, 1.0, -127.0),
            engine.interface_power(1.0, 1.695, -23.5),
            engine.interface_power(1.695, 1.0, 10.2),
            engine.interface_power(1.0, 1.812, 30.0),
            engine.interface_power(1.812, 1.0, -15.0),
        )

    @staticmethod
    def _matlab_thick_lens(n, d, p_s, p_f):
        """MATLAB notebook order: Ra(Ps)*T*Ra(Pf) with first factor leftmost."""
        return engine.cascade(
            (
                engine.refraction_matrix(p_s),
                engine.translation_matrix(n, d),
                engine.refraction_matrix(p_f),
            )
        )

    @classmethod
    def _course_matrices(cls):
        p1, p2, p3, p4, p5, p6 = cls._surface_powers()
        m1 = engine.thick_lens_matrix(1.812, 5.00, p1, p2)
        m2 = engine.thick_lens_matrix(1.695, 1.55, p3, p4)
        m3 = engine.thick_lens_matrix(1.812, 5.00, p5, p6)
        matlab_m1 = cls._matlab_thick_lens(1.812, 5.00, p1, p2)
        matlab_m2 = cls._matlab_thick_lens(1.695, 1.55, p3, p4)
        matlab_m3 = cls._matlab_thick_lens(1.812, 5.00, p5, p6)
        matlab_total = engine.cascade(
            (
                matlab_m1,
                engine.translation_matrix(1.0, 1.25),
                matlab_m2,
                engine.translation_matrix(1.0, 2.50),
                matlab_m3,
            )
        )
        physical_total = engine.cascade(
            (
                m3,
                engine.translation_matrix(1.0, 2.50),
                m2,
                engine.translation_matrix(1.0, 1.25),
                m1,
            )
        )
        return m1, m2, m3, matlab_m1, matlab_m2, matlab_m3, matlab_total, physical_total

    def test_each_thick_lens_matches_elemental_rightmost_first_product(self):
        """Interactive thick_lens: optical order, then cascade(reversed(...))."""
        p1, p2, p3, p4, p5, p6 = self._surface_powers()
        m1, m2, m3, *_ = self._course_matrices()
        optical_m1 = (
            engine.refraction_matrix(p1),
            engine.translation_matrix(1.812, 5.00),
            engine.refraction_matrix(p2),
        )
        optical_m2 = (
            engine.refraction_matrix(p3),
            engine.translation_matrix(1.695, 1.55),
            engine.refraction_matrix(p4),
        )
        optical_m3 = (
            engine.refraction_matrix(p5),
            engine.translation_matrix(1.812, 5.00),
            engine.refraction_matrix(p6),
        )
        np.testing.assert_allclose(m1, engine.cascade(reversed(optical_m1)))
        np.testing.assert_allclose(m2, engine.cascade(reversed(optical_m2)))
        np.testing.assert_allclose(m3, engine.cascade(reversed(optical_m3)))

    def test_matlab_written_thick_lenses_and_total_match_source(self):
        *_, matlab_m1, matlab_m2, matlab_m3, matlab_total, _ = self._course_matrices()
        np.testing.assert_allclose(
            matlab_m1,
            [[0.8052, -0.0758], [2.7594, 0.9824]],
            atol=5e-5,
        )
        np.testing.assert_allclose(
            matlab_m2,
            [[1.0270, 0.0996], [0.9145, 1.0623]],
            atol=5e-5,
        )
        np.testing.assert_allclose(
            matlab_m3,
            [[0.9253, -0.0772], [2.7594, 0.8506]],
            atol=5e-5,
        )
        np.testing.assert_allclose(
            matlab_total,
            [[0.5616, -0.0574], [11.9278, 0.5622]],
            atol=5e-5,
        )
        self.assertAlmostEqual(engine.assert_unit_determinant(matlab_total), 1.0)

    def test_course_preset_uses_rightmost_first_product(self):
        elements = tools.preset_elements("course_exercise")
        report = tools.build_system(elements)
        *_, physical_total = self._course_matrices()
        np.testing.assert_allclose(report.matrix, physical_total)
        self.assertEqual(len(report.factors), 5)
        self.assertEqual(
            report.product_expression,
            "Mtk[L3] @ T[Gap2] @ Mtk[L2] @ T[Gap1] @ Mtk[L1]",
        )
        self.assertTrue(report.product_expression.endswith("Mtk[L1]"))

    def test_thick_lens_component_matches_separate_ra_t_ra_stack(self):
        elemental = tools.build_system(
            [
                tools.OpticalElement.refraction("Ra1", 1.0, 1.5, 0.10),
                tools.OpticalElement.translation("T1", 1.5, 0.02),
                tools.OpticalElement.refraction("Ra2", 1.5, 1.0, -0.12),
            ]
        )
        p1 = engine.interface_power(1.0, 1.5, 0.10)
        p2 = engine.interface_power(1.5, 1.0, -0.12)
        composite = tools.build_system(
            [
                tools.OpticalElement.thick_lens("L1", 1.5, 0.02, p1, p2),
            ]
        )
        np.testing.assert_allclose(composite.matrix, elemental.matrix)
        np.testing.assert_allclose(
            composite.matrix,
            engine.thick_lens_matrix(1.5, 0.02, p1, p2),
        )


class MatrixOrderTests(unittest.TestCase):
    def test_first_component_is_rightmost_in_the_product(self):
        translation = tools.OpticalElement.translation("T1", 1.0, 0.2)
        lens = tools.OpticalElement.thin_lens("L1", 10.0)
        report = tools.build_system((translation, lens))

        self.assertEqual(report.product_expression, "Mtn[L1] @ T[T1]")
        np.testing.assert_allclose(
            report.matrix,
            lens.matrix @ translation.matrix,
        )
        self.assertFalse(
            np.allclose(
                translation.matrix @ lens.matrix,
                lens.matrix @ translation.matrix,
            )
        )

    def test_thin_and_thick_lenses_contribute_one_matrix_each(self):
        thin = tools.OpticalElement.thin_lens("L1", 5.0, 6.0)
        thick = tools.OpticalElement.thick_lens("L2", 1.5, 0.01, 5.0, 6.0)
        self.assertEqual(len(thin.matrix_factors()), 1)
        self.assertEqual(len(thick.matrix_factors()), 1)
        self.assertEqual(thin.matrix_factors()[0].label, "Mtn[L1]")
        self.assertEqual(thick.matrix_factors()[0].label, "Mtk[L2]")
        np.testing.assert_allclose(
            thin.matrix,
            engine.thin_lens_matrix(5.0, 6.0),
        )
        np.testing.assert_allclose(
            thick.matrix,
            engine.thick_lens_matrix(1.5, 0.01, 5.0, 6.0),
        )


class PrincipalAndConjugatePlaneTests(unittest.TestCase):
    def test_principal_plane_reduction_resolves_equivalent_thin_lens(self):
        base = engine.thick_lens_matrix(1.5, 0.02, 5.0, 8.0)
        cardinal = engine.principal_planes(base, 1.0, 1.0)

        self.assertAlmostEqual(cardinal.power, -base[0, 1])
        self.assertAlmostEqual(
            cardinal.object_principal_offset,
            (1.0 - base[0, 0]) / base[0, 1],
        )
        self.assertAlmostEqual(
            cardinal.image_principal_offset,
            (1.0 - base[1, 1]) / base[0, 1],
        )
        np.testing.assert_allclose(
            cardinal.reduction_matrix,
            cardinal.equivalent_matrix,
            atol=1e-12,
        )
        self.assertLess(cardinal.reduction_residual, 1e-12)

    def test_conjugate_matrix_has_zero_m21_and_source_magnifications(self):
        cardinal = engine.principal_planes(
            engine.thin_lens_matrix(10.0),
            1.0,
            1.5,
        )
        result = engine.conjugate_planes(cardinal, 0.2, 1.0, 1.5)

        self.assertAlmostEqual(result.image_distance, 0.3)
        self.assertAlmostEqual(result.matrix[1, 0], 0.0)
        self.assertAlmostEqual(result.lateral_magnification, result.matrix[1, 1])
        self.assertAlmostEqual(
            result.angular_magnification,
            result.matrix[0, 0] / 1.5,
        )
        self.assertAlmostEqual(result.lateral_magnification, -1.0)
        self.assertAlmostEqual(result.angular_magnification, -2.0 / 3.0)
        self.assertAlmostEqual(result.lagrange_invariant, 1.0)
        self.assertTrue(result.is_real)
        self.assertFalse(result.is_erect)

    def test_gaussian_imaging_equation_is_satisfied(self):
        cardinal = engine.principal_planes(engine.thin_lens_matrix(4.0))
        result = engine.conjugate_planes(cardinal, 0.5)
        lhs = 1.0 / result.image_distance + 1.0 / result.object_distance
        self.assertAlmostEqual(lhs, cardinal.power)

    def test_afocal_and_focal_plane_cases_are_explicit(self):
        with self.assertRaises(engine.AfocalSystemError):
            engine.principal_planes(np.eye(2))

        cardinal = engine.principal_planes(engine.thin_lens_matrix(10.0))
        with self.assertRaises(engine.ConjugateAtInfinityError):
            engine.conjugate_planes(cardinal, 0.1)


class RaySamplingTests(unittest.TestCase):
    def test_translation_updates_height_in_row_state_form(self):
        element = tools.OpticalElement.translation("T1", 2.0, 0.5)
        trace = tools.trace_rays((element,), (0.1,), (0.2,))
        self.assertAlmostEqual(trace.path_positions[-1], 0.5)
        self.assertAlmostEqual(trace.heights[0, -1], 0.15)
        self.assertAlmostEqual(trace.reduced_angles[0, -1], 0.2)


if __name__ == "__main__":
    unittest.main()
