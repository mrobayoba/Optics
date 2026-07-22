"""Numerical Fraunhofer diffraction for arbitrary sampled apertures.

This module implements the spatial-Fourier relation from the course notes:

``I(x', y') ∝ |F{t(x_tilde, y_tilde)}|²``

The aperture transmittance can come from a Python callable, a restricted
mathematical expression, or a binary/grayscale image. All physical lengths use
SI metres. Dashboard expressions use millimetres for friendlier input.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftfreq, fftshift, ifftshift, next_fast_len
from scipy.interpolate import RegularGridInterpolator

import diffraction_config as config
import diffraction_engine as engine
import screen_tools as screen

__all__ = [
    "UnsafeExpressionError",
    "CustomAperture",
    "FourierGridResult",
    "FourierScreenResult",
    "build_aperture_grid",
    "evaluate_expression",
    "fraunhofer_fft",
    "simulate_pattern",
]


class UnsafeExpressionError(ValueError):
    """Raised when a custom expression contains forbidden syntax."""


@dataclass(frozen=True)
class CustomAperture:
    """Sampled aperture-plane amplitude transmittance."""

    transmittance: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    source: str = "custom"

    def __post_init__(self) -> None:
        values = np.asarray(self.transmittance, dtype=float)
        x_axis = np.asarray(self.x_axis, dtype=float)
        y_axis = np.asarray(self.y_axis, dtype=float)
        if values.ndim != 2:
            raise ValueError("transmittance must be a two-dimensional array.")
        if x_axis.ndim != 1 or y_axis.ndim != 1:
            raise ValueError("x_axis and y_axis must be one-dimensional.")
        if values.shape != (len(y_axis), len(x_axis)):
            raise ValueError(
                "transmittance shape must be (len(y_axis), len(x_axis))."
            )
        if len(x_axis) < 8 or len(y_axis) < 8:
            raise ValueError("Custom aperture resolution must be at least 8x8.")
        if not np.isfinite(values).all():
            raise ValueError("transmittance contains non-finite values.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("transmittance values must lie in [0, 1].")
        _validate_uniform_axis(x_axis, "x_axis")
        _validate_uniform_axis(y_axis, "y_axis")
        if not np.any(values > 0.0):
            raise ValueError("Custom aperture has no open pixels.")

        values = values.copy()
        x_axis = x_axis.copy()
        y_axis = y_axis.copy()
        values.setflags(write=False)
        x_axis.setflags(write=False)
        y_axis.setflags(write=False)
        object.__setattr__(self, "transmittance", values)
        object.__setattr__(self, "x_axis", x_axis)
        object.__setattr__(self, "y_axis", y_axis)

    @property
    def dx(self) -> float:
        return float(self.x_axis[1] - self.x_axis[0])

    @property
    def dy(self) -> float:
        return float(self.y_axis[1] - self.y_axis[0])

    @property
    def width(self) -> float:
        return self.dx * len(self.x_axis)

    @property
    def height(self) -> float:
        return self.dy * len(self.y_axis)

    @property
    def area(self) -> float:
        """Effective open area, including partial transmittance."""
        return float(self.transmittance.sum() * self.dx * self.dy)

    @property
    def support_radius(self) -> float:
        """Maximum radius containing nonzero transmittance."""
        Y, X = np.meshgrid(self.y_axis, self.x_axis, indexing="ij")
        open_pixels = self.transmittance > 0.0
        half_dx = self.dx / 2.0
        half_dy = self.dy / 2.0
        radii = np.hypot(
            np.abs(X[open_pixels]) + half_dx,
            np.abs(Y[open_pixels]) + half_dy,
        )
        return float(radii.max())

    @classmethod
    def from_callable(
        cls,
        function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        width: float,
        height: float,
        resolution: int | tuple[int, int] = 256,
        *,
        source: str = "callable",
    ) -> "CustomAperture":
        """Sample ``function(X, Y)`` on a physical SI-coordinate grid."""
        x_axis, y_axis, X, Y = build_aperture_grid(
            width, height, resolution
        )
        values = function(X, Y)
        mask = _coerce_transmittance(values, X.shape)
        return cls(mask, x_axis, y_axis, source)

    @classmethod
    def from_expression(
        cls,
        expression: str,
        width: float,
        height: float,
        resolution: int | tuple[int, int] = 256,
        *,
        invert: bool = False,
    ) -> "CustomAperture":
        """Build, and optionally invert, a millimetre-based safe expression."""
        x_axis, y_axis, X, Y = build_aperture_grid(
            width, height, resolution
        )
        millimetres = 1e-3
        values = evaluate_expression(
            expression, X / millimetres, Y / millimetres
        )
        mask = _coerce_transmittance(values, X.shape)
        if invert:
            mask = 1.0 - mask
        return cls(
            mask,
            x_axis,
            y_axis,
            f"expression: {expression}",
        )

    @classmethod
    def from_image_bytes(
        cls,
        data: bytes,
        width: float,
        height: float,
        resolution: int | tuple[int, int] = 256,
        *,
        threshold: float = 0.5,
        invert: bool = False,
    ) -> "CustomAperture":
        """Decode image bytes and create a white=open binary aperture."""
        if not data:
            raise ValueError("Uploaded image is empty.")
        try:
            with Image.open(BytesIO(data)) as image:
                grayscale = image.convert("L")
                array = np.asarray(grayscale, dtype=float) / 255.0
        except Exception as exc:
            raise ValueError("Could not decode the uploaded image.") from exc
        return cls.from_image_array(
            array,
            width,
            height,
            resolution,
            threshold=threshold,
            invert=invert,
            source="uploaded image",
        )

    @classmethod
    def from_image_array(
        cls,
        image: np.ndarray,
        width: float,
        height: float,
        resolution: int | tuple[int, int] = 256,
        *,
        threshold: float = 0.5,
        invert: bool = False,
        source: str = "image array",
    ) -> "CustomAperture":
        """Threshold and resize an array into a physical binary aperture."""
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        raw_values = np.asarray(image)
        if raw_values.ndim == 3:
            raw_values = raw_values[..., :3].astype(float).mean(axis=2)
        if raw_values.ndim != 2 or 0 in raw_values.shape:
            raise ValueError("image must be a non-empty 2D or RGB array.")
        values = raw_values.astype(float)
        if not np.isfinite(values).all():
            raise ValueError("image contains non-finite values.")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < 0.0 or maximum > 1.0:
            if minimum >= 0.0 and maximum <= 255.0:
                values = values / 255.0
            else:
                raise ValueError(
                    "image values must lie in [0, 1] or [0, 255]."
                )

        ny, nx = _resolution_pair(resolution)
        image_u8 = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
        resized = Image.fromarray(image_u8).resize(
            (nx, ny), resample=Image.Resampling.NEAREST
        )
        normalized = np.asarray(resized, dtype=float) / 255.0
        mask = normalized >= threshold
        if invert:
            mask = ~mask
        # Image row zero is the top; aperture row zero represents negative y.
        mask = np.flipud(mask).astype(float)
        x_axis, y_axis, _, _ = build_aperture_grid(
            width, height, (ny, nx)
        )
        return cls(mask, x_axis, y_axis, source)


@dataclass(frozen=True)
class FourierGridResult:
    """Natural spatial-frequency output of the guarded FFT."""

    fx: np.ndarray
    fy: np.ndarray
    field: np.ndarray
    intensity: np.ndarray
    far_field: engine.FarFieldReport


@dataclass(frozen=True)
class FourierScreenResult:
    """FFT intensity resampled onto the observation plane."""

    X: np.ndarray
    Y: np.ndarray
    intensity: np.ndarray
    far_field: engine.FarFieldReport


def _validate_uniform_axis(axis: np.ndarray, name: str) -> None:
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} contains non-finite values.")
    differences = np.diff(axis)
    if np.any(differences <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    if not np.allclose(differences, differences[0], rtol=1e-9, atol=0.0):
        raise ValueError(f"{name} must be uniformly spaced.")


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _resolution_pair(
    resolution: int | tuple[int, int],
) -> tuple[int, int]:
    if isinstance(resolution, tuple):
        if len(resolution) != 2:
            raise ValueError("resolution tuple must be (ny, nx).")
        ny, nx = resolution
    else:
        ny = nx = resolution
    if isinstance(ny, bool) or isinstance(nx, bool):
        raise ValueError("resolution values must be integers.")
    if int(ny) != ny or int(nx) != nx:
        raise ValueError("resolution values must be integers.")
    ny, nx = int(ny), int(nx)
    if ny < 8 or nx < 8:
        raise ValueError("resolution must be at least 8 pixels per axis.")
    return ny, nx


def build_aperture_grid(
    width: float,
    height: float,
    resolution: int | tuple[int, int] = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel-centred aperture axes and ``X, Y`` meshgrids in metres."""
    width = _positive_finite(width, "width")
    height = _positive_finite(height, "height")
    ny, nx = _resolution_pair(resolution)
    dx = width / nx
    dy = height / ny
    x_axis = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y_axis = (np.arange(ny) - (ny - 1) / 2.0) * dy
    X, Y = np.meshgrid(x_axis, y_axis)
    return x_axis, y_axis, X, Y


def _coerce_transmittance(
    values: np.ndarray | float | bool,
    shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.full(shape, array, dtype=float)
    else:
        try:
            array = np.broadcast_to(array, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(
                "Aperture function result cannot broadcast to the grid."
            ) from exc
    if not np.isfinite(array).all():
        raise ValueError("Aperture function returned non-finite values.")
    tolerance = 10.0 * np.finfo(float).eps
    if np.any((array < -tolerance) | (array > 1.0 + tolerance)):
        raise ValueError(
            "Aperture function values must lie in [0, 1]."
        )
    return np.clip(array, 0.0, 1.0)


_ALLOWED_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "sinc": np.sinc,
    "sign": np.sign,
    "hypot": np.hypot,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
}

_BINARY_OPERATORS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.BitAnd: np.logical_and,
    ast.BitOr: np.logical_or,
}

_COMPARISON_OPERATORS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


def evaluate_expression(
    expression: str,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate a restricted vectorized aperture expression.

    ``x`` and ``y`` are supplied by the caller; the dashboard passes them in
    millimetres. Available names are ``x``, ``y``, ``r``, ``pi`` and ``e``.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("Expression cannot be empty.")
    if len(expression) > config.CUSTOM_EXPRESSION_MAX_LENGTH:
        raise UnsafeExpressionError("Expression is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression: {exc.msg}.") from exc
    if sum(1 for _ in ast.walk(tree)) > config.CUSTOM_EXPRESSION_MAX_NODES:
        raise UnsafeExpressionError("Expression is too complex.")

    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    names: dict[str, np.ndarray | float] = {
        "x": x_array,
        "y": y_array,
        "r": np.hypot(x_array, y_array),
        "pi": np.pi,
        "e": np.e,
    }
    with np.errstate(all="ignore"):
        result = _evaluate_ast(tree.body, names)
    result_array = np.asarray(result)
    if result_array.ndim == 0:
        result_array = np.full(x_array.shape, result_array)
    else:
        try:
            result_array = np.broadcast_to(result_array, x_array.shape)
        except ValueError as exc:
            raise ValueError(
                "Expression result cannot broadcast to the aperture grid."
            ) from exc
    if not np.isfinite(result_array.astype(float)).all():
        raise ValueError("Expression returned non-finite values.")
    return result_array


def _evaluate_ast(
    node: ast.AST,
    names: dict[str, np.ndarray | float],
):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError("Only numeric constants are allowed.")
        if not np.isfinite(node.value):
            raise UnsafeExpressionError("Constants must be finite.")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise UnsafeExpressionError(f"Unknown name: {node.id!r}.")
        return names[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand, names)
        if isinstance(node.op, ast.USub):
            return np.negative(operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, (ast.Invert, ast.Not)):
            return np.logical_not(operand)
        raise UnsafeExpressionError("Unary operator is not allowed.")
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left, names)
        right = _evaluate_ast(node.right, names)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(
                node.right.value, (int, float)
            ):
                raise UnsafeExpressionError(
                    "Exponent must be a numeric constant."
                )
            if abs(float(node.right.value)) > config.CUSTOM_MAX_EXPONENT:
                raise UnsafeExpressionError("Exponent magnitude is too large.")
            return np.power(left, right)
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise UnsafeExpressionError("Binary operator is not allowed.")
        return operator(left, right)
    if isinstance(node, ast.BoolOp):
        values = [_evaluate_ast(value, names) for value in node.values]
        operation = (
            np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        )
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result
    if isinstance(node, ast.Compare):
        left = _evaluate_ast(node.left, names)
        result = True
        for operator_node, comparator in zip(node.ops, node.comparators):
            right = _evaluate_ast(comparator, names)
            operator = _COMPARISON_OPERATORS.get(type(operator_node))
            if operator is None:
                raise UnsafeExpressionError(
                    "Comparison operator is not allowed."
                )
            result = np.logical_and(result, operator(left, right))
            left = right
        return result
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError(
                "Only direct calls to approved functions are allowed."
            )
        function = _ALLOWED_FUNCTIONS.get(node.func.id)
        if function is None:
            raise UnsafeExpressionError(
                f"Unknown function: {node.func.id!r}."
            )
        if node.keywords:
            raise UnsafeExpressionError(
                "Keyword arguments are not allowed."
            )
        arguments = [_evaluate_ast(argument, names) for argument in node.args]
        return function(*arguments)
    if isinstance(node, ast.IfExp):
        return np.where(
            _evaluate_ast(node.test, names),
            _evaluate_ast(node.body, names),
            _evaluate_ast(node.orelse, names),
        )
    raise UnsafeExpressionError(
        f"Forbidden expression syntax: {type(node).__name__}."
    )


def _padded_shape(
    shape: tuple[int, int],
    padding_factor: float,
) -> tuple[int, int]:
    padding_factor = _positive_finite(padding_factor, "padding_factor")
    if padding_factor < 1.0:
        raise ValueError("padding_factor must be at least 1.")
    return tuple(
        next_fast_len(int(np.ceil(length * padding_factor)))
        for length in shape
    )


def _pad_centered(
    values: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    pads = []
    for current, target in zip(values.shape, target_shape):
        total = target - current
        if total < 0:
            raise ValueError("target_shape cannot crop the aperture.")
        before = total // 2
        pads.append((before, total - before))
    return np.pad(values, pads, mode="constant")


def _fft_unchecked(
    aperture: CustomAperture,
    padding_factor: float,
    normalize: bool,
    report: engine.FarFieldReport,
) -> FourierGridResult:
    target_shape = _padded_shape(
        aperture.transmittance.shape, padding_factor
    )
    padded = _pad_centered(aperture.transmittance, target_shape)
    field = (
        fftshift(fft2(ifftshift(padded))) * aperture.dx * aperture.dy
    )
    fy = fftshift(fftfreq(target_shape[0], d=aperture.dy))
    fx = fftshift(fftfreq(target_shape[1], d=aperture.dx))
    intensity = np.abs(field) ** 2
    if normalize:
        intensity = intensity / aperture.area**2
    return FourierGridResult(
        fx=np.asarray(fx),
        fy=np.asarray(fy),
        field=np.asarray(field),
        intensity=np.asarray(intensity, dtype=float),
        far_field=report,
    )


def fraunhofer_fft(
    aperture: CustomAperture,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    padding_factor: float = 4.0,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
    normalize: bool = True,
) -> FourierGridResult:
    """Run the mandatory far-field gate, then calculate the centered FFT."""
    report = engine.require_far_field_radius(
        aperture.support_radius,
        wavelength_vacuum,
        distance,
        n,
        max_fresnel_number,
    )
    return _fft_unchecked(aperture, padding_factor, normalize, report)


def simulate_pattern(
    aperture: CustomAperture,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    screen_half_width: float = config.SCREEN_HALF_WIDTH_MM.to_si(
        config.SCREEN_HALF_WIDTH_MM.default
    ),
    resolution: int = int(config.RESOLUTION.default),
    zoom: float = config.ZOOM.default,
    padding_factor: float = 4.0,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
    normalize: bool = True,
    interpolation: str = "linear",
) -> FourierScreenResult:
    """Calculate and interpolate an arbitrary-aperture Fraunhofer pattern."""
    report = engine.require_far_field_radius(
        aperture.support_radius,
        wavelength_vacuum,
        distance,
        n,
        max_fresnel_number,
    )
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest'.")
    X, Y = screen.observation_grid(screen_half_width, resolution, zoom)
    grid_result = _fft_unchecked(
        aperture, padding_factor, normalize, report
    )
    target_fx, target_fy = engine.spatial_frequencies(
        X, Y, wavelength_vacuum, distance, n
    )
    interpolator = RegularGridInterpolator(
        (grid_result.fy, grid_result.fx),
        grid_result.intensity,
        method=interpolation,
        bounds_error=False,
        fill_value=0.0,
    )
    points = np.column_stack((target_fy.ravel(), target_fx.ravel()))
    intensity = interpolator(points).reshape(X.shape)
    return FourierScreenResult(
        X=X,
        Y=Y,
        intensity=np.asarray(intensity, dtype=float),
        far_field=report,
    )
