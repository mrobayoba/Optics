"""Interactive system builder for the paraxial-optics notebook."""

from __future__ import annotations

from typing import Any, Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import paraxial_config as config
import paraxial_dashboard_tools as visuals
import paraxial_engine as engine
import paraxial_tools as tools

__all__ = ["ParaxialDashboard"]

LengthUnit = Literal["m", "mm"]


def _saved_system_options() -> list[tuple[str, str]]:
    names = tools.list_saved_systems()
    if not names:
        return [("(none)", "")]
    return [(name, name) for name in names]


def _quantity(value: float, unit: str = "", decimals: int = 6) -> str:
    suffix = f" {unit}" if unit else ""
    digits = int(decimals)
    return f"{value:.{digits}f}{suffix}"


class ParaxialDashboard:
    """Self-contained ipywidgets dashboard for matrix-system construction."""

    def __init__(self) -> None:
        self._rendering = False
        self._suspend_updates = True
        self.length_unit: LengthUnit = config.DEFAULT_LENGTH_UNIT  # type: ignore[assignment]
        self._length_controls: list[tuple[widgets.FloatText, str]] = []
        self._power_controls: list[tuple[widgets.FloatText, str]] = []
        self.element_rows: list[dict[str, Any]] = []
        self._create_controls()
        self._create_outputs()
        self.element_box = widgets.VBox()
        self._load_preset(config.DEFAULT_PRESET)
        self._create_layout()
        self._wire_events()
        self._suspend_updates = False
        self.refresh()

    def _length_scale(self) -> float:
        return config.length_scale(self.length_unit)

    def _display_digits(self) -> int:
        digits = int(self.display_decimals.value)
        return max(
            config.MIN_DISPLAY_DECIMALS,
            min(config.MAX_DISPLAY_DECIMALS, digits),
        )

    def _length_unit_label(self) -> str:
        return self.length_unit

    def _power_unit_label(self) -> str:
        return f"{self.length_unit}⁻¹"

    def _to_si_length(self, display_value: float) -> float:
        return float(display_value) / self._length_scale()

    def _from_si_length(self, si_value: float) -> float:
        return float(si_value) * self._length_scale()

    def _to_si_power(self, display_value: float) -> float:
        return float(display_value) * self._length_scale()

    def _from_si_power(self, si_value: float) -> float:
        return float(si_value) / self._length_scale()

    def _scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Convert a SI course matrix into the active length unit."""
        scale = self._length_scale()
        values = np.array(matrix, dtype=float, copy=True)
        values[0, 1] /= scale
        values[1, 0] *= scale
        return values

    def _display_factors(
        self,
        factors: tuple[engine.MatrixFactor, ...],
    ) -> tuple[engine.MatrixFactor, ...]:
        return tuple(
            engine.MatrixFactor(
                factor.label,
                self._scale_matrix(factor.matrix),
                factor.description,
                style_key=factor.style_key,
            )
            for factor in factors
        )

    @staticmethod
    def _numeric(
        spec: config.NumericControl,
        description: str,
        value: float | None = None,
    ) -> widgets.FloatText:
        """Unbounded float control so typed/calculated values are never clipped."""
        return widgets.FloatText(
            value=spec.default if value is None else value,
            step=spec.step,
            description=description,
            style={"description_width": "initial"},
        )

    def _register_length(
        self,
        control: widgets.FloatText,
        base_description: str,
    ) -> widgets.FloatText:
        self._length_controls.append((control, base_description))
        return control

    def _register_power(
        self,
        control: widgets.FloatText,
        base_description: str,
    ) -> widgets.FloatText:
        self._power_controls.append((control, base_description))
        return control

    def _length_control(
        self,
        spec: config.NumericControl,
        base_description: str,
        value_si: float | None = None,
        observe: bool = True,
    ) -> widgets.FloatText:
        scale = self._length_scale()
        selected = spec.default if value_si is None else value_si
        control = widgets.FloatText(
            value=selected * scale,
            step=max(spec.step * scale, 1e-15),
            description=f"{base_description} ({self._length_unit_label()})",
            style={"description_width": "initial"},
        )
        self._register_length(control, base_description)
        if observe:
            control.observe(self._on_value_change, names="value")
        return control

    def _power_control(
        self,
        spec: config.NumericControl,
        base_description: str,
        value_si: float | None = None,
        observe: bool = True,
    ) -> widgets.FloatText:
        scale = self._length_scale()
        selected = spec.default if value_si is None else value_si
        control = widgets.FloatText(
            value=selected / scale,
            step=max(spec.step / scale, 1e-15),
            description=f"{base_description} ({self._power_unit_label()})",
            style={"description_width": "initial"},
        )
        self._register_power(control, base_description)
        if observe:
            control.observe(self._on_value_change, names="value")
        return control

    def _update_unit_chrome(self) -> None:
        unit = self._length_unit_label()
        other = "mm" if unit == "m" else "m"
        self.unit_toggle.description = f"Units: {unit} -> {other}"
        self.unit_badge.value = (
            f"<span class='paraxial-unit-badge'>Window units: {unit} "
            f"/ {unit}⁻¹</span>"
        )

    def _rescale_float_control(
        self,
        control: widgets.FloatText,
        factor: float,
        base_description: str,
        unit_label: str,
    ) -> None:
        control.step = max(float(control.step) * factor, 1e-15)
        control.value = float(control.value) * factor
        control.description = f"{base_description} ({unit_label})"

    def _toggle_length_unit(self, _button: widgets.Button | None = None) -> None:
        """Switch the whole dashboard between metres and millimetres."""
        old_unit = self.length_unit
        new_unit: LengthUnit = "mm" if old_unit == "m" else "m"
        length_factor = config.length_scale(new_unit) / config.length_scale(old_unit)
        power_factor = 1.0 / length_factor

        self._suspend_updates = True
        try:
            self.length_unit = new_unit
            for control, base in self._length_controls:
                self._rescale_float_control(
                    control,
                    length_factor,
                    base,
                    self._length_unit_label(),
                )
            for control, base in self._power_controls:
                self._rescale_float_control(
                    control,
                    power_factor,
                    base,
                    self._power_unit_label(),
                )
            self._update_unit_chrome()
        finally:
            self._suspend_updates = False
        self._sync_vertex_distance_label()
        self.refresh()

    def _assign_control_value(
        self,
        control: widgets.FloatText,
        value: float,
    ) -> float:
        """Write the exact finite value with no clipping or rounding."""
        exact = float(value)
        if not np.isfinite(exact):
            raise ValueError("Control value must be finite.")
        control.value = exact
        return exact

    def _create_controls(self) -> None:
        self.preset = widgets.Dropdown(
            options=config.PRESET_OPTIONS,
            value=config.DEFAULT_PRESET,
            description="Preset",
            style={"description_width": "initial"},
        )
        self.load_preset_button = widgets.Button(
            description="Load preset",
            icon="folder-open",
            button_style="info",
        )
        self.save_name = widgets.Text(
            value="",
            placeholder="system name",
            description="Save as",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="14rem"),
        )
        self.save_system_button = widgets.Button(
            description="Save system",
            icon="save",
            button_style="success",
            tooltip="Write the current stack and analysis to Optical Systems/saves/",
        )
        self.saved_system = widgets.Dropdown(
            options=_saved_system_options(),
            description="Saved",
            style={"description_width": "initial"},
        )
        self.load_system_button = widgets.Button(
            description="Load system",
            icon="upload",
            button_style="info",
            tooltip="Replace the stack and analysis from a saved JSON file",
        )
        self.new_element_kind = widgets.Dropdown(
            options=config.ELEMENT_OPTIONS,
            value=config.DEFAULT_ELEMENT_KIND,
            description="Add",
            style={"description_width": "initial"},
        )
        self.add_element_button = widgets.Button(
            description="Add matrix",
            icon="plus",
            button_style="success",
        )
        self.new_element_kind_footer = widgets.Dropdown(
            options=config.ELEMENT_OPTIONS,
            value=config.DEFAULT_ELEMENT_KIND,
            description="Add",
            style={"description_width": "initial"},
        )
        self.add_element_button_footer = widgets.Button(
            description="Add matrix",
            icon="plus",
            button_style="success",
        )
        self.refresh_button = widgets.Button(
            description="Recalculate",
            icon="refresh",
            button_style="primary",
        )
        self.auto_update = widgets.Checkbox(
            value=config.DEFAULT_AUTO_UPDATE,
            description="Auto update",
            indent=False,
        )
        self.unit_toggle = widgets.Button(
            description="Units: m -> mm",
            icon="exchange",
            button_style="warning",
            tooltip=(
                "Toggle every length and power control in this window "
                "between metres and millimetres. Physics stays in SI."
            ),
        )
        self.unit_badge = widgets.HTML()

        ambient_default = config.REFRACTIVE_INDEX.minimum
        self.input_index = self._numeric(
            config.REFRACTIVE_INDEX,
            "Input n",
            ambient_default,
        )
        self.output_index = self._numeric(
            config.REFRACTIVE_INDEX,
            "Output n′",
            ambient_default,
        )
        self.conjugate_mode = widgets.Dropdown(
            options=config.CONJUGATE_MODE_OPTIONS,
            value=config.DEFAULT_CONJUGATE_MODE,
            description="Known distance",
            style={"description_width": "initial"},
            tooltip=(
                "Enter Object→V (x) or V′→image (x′). "
                "s and s′ follow from D, D′ on M_VV′ and the imaging equation."
            ),
        )
        self.vertex_distance = self._length_control(
            config.VERTEX_DISTANCE_M,
            "Object→V",
        )
        self.object_height = self._length_control(
            config.OBJECT_HEIGHT_M,
            "Object height",
        )
        self.ray_half_angle = self._numeric(
            config.RAY_ANGLE,
            "Ray half-angle (rad)",
        )
        self.ray_count = widgets.BoundedIntText(
            min=int(config.RAY_COUNT.minimum),
            max=int(config.RAY_COUNT.maximum),
            step=int(config.RAY_COUNT.step),
            value=int(config.RAY_COUNT.default),
            description="Rays",
            style={"description_width": "initial"},
        )
        self.display_decimals = widgets.BoundedIntText(
            min=config.MIN_DISPLAY_DECIMALS,
            max=config.MAX_DISPLAY_DECIMALS,
            step=1,
            value=config.DEFAULT_DISPLAY_DECIMALS,
            description="Digits after decimal",
            style={"description_width": "initial"},
            tooltip="Display precision for matrices and reported values (0 to 16).",
        )
        self.zoom_scale = widgets.FloatSlider(
            value=config.ZOOM_SCALE.default,
            min=config.ZOOM_SCALE.minimum,
            max=config.ZOOM_SCALE.maximum,
            step=config.ZOOM_SCALE.step,
            description="Zoom",
            readout=True,
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="16rem"),
        )
        self.zoom_pan = widgets.FloatSlider(
            value=config.ZOOM_PAN.default,
            min=config.ZOOM_PAN.minimum,
            max=config.ZOOM_PAN.maximum,
            step=config.ZOOM_PAN.step,
            description="Pan",
            readout=True,
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="16rem"),
        )
        self.zoom_reset_button = widgets.Button(
            description="Reset view",
            icon="search-minus",
            tooltip="Reset schematic zoom and pan to the full system",
        )
        self._update_unit_chrome()
        self._sync_vertex_distance_label()

    def _create_outputs(self) -> None:
        self.status = widgets.HTML()
        self.base_matrix = widgets.HTML()
        self.principal_values = widgets.HTML()
        self.principal_matrix = widgets.HTML()
        self.conjugate_values = widgets.HTML()
        self.conjugate_matrix = widgets.HTML()
        self.figure_output = widgets.Output(
            layout=widgets.Layout(width="100%", overflow_x="auto")
        )

    def _create_layout(self) -> None:
        title = widgets.HTML(
            config.DASHBOARD_CSS
            + """
            <h2 class="paraxial-title">Paraxial optical-system laboratory</h2>
            <p class="paraxial-note">
            Build the vertex-to-vertex matrix from stack components. Each thin
            or thick lens contributes <strong>one</strong> lens matrix. The first
            component added is rightmost in the product and multiplies the input
            ray first; later components pile on the left
            (<code>M = M<sub>n</sub> @ … @ M<sub>1</sub></code>). Matrices do not
            commute. Use the unit toggle for the whole window: lengths and
            powers stay consistent, while the engine always works in SI metres.
            Input fields are unbounded so calculated and typed values are kept
            exactly (no clipping to a control range). Conjugate analysis uses a
            known <strong>Object→V</strong> or <strong>V′→image</strong> distance;
            <code>s</code> and <code>s′</code> follow from <code>D</code>,
            <code>D′</code> on <code>M<sub>VV′</sub></code>.
            </p>
            """
        )
        builder_toolbar = widgets.HBox(
            [
                self.preset,
                self.load_preset_button,
                self.save_name,
                self.save_system_button,
                self.saved_system,
                self.load_system_button,
                self.new_element_kind,
                self.add_element_button,
                self.unit_toggle,
                self.unit_badge,
            ],
            layout=widgets.Layout(flex_flow="row wrap", gap="0.5rem", align_items="center"),
        )
        builder_footer = widgets.HBox(
            [
                self.new_element_kind_footer,
                self.add_element_button_footer,
            ],
            layout=widgets.Layout(flex_flow="row wrap", gap="0.5rem", align_items="center"),
        )
        analysis_controls = widgets.HBox(
            [
                self.input_index,
                self.output_index,
                self.conjugate_mode,
                self.vertex_distance,
                self.object_height,
                self.ray_half_angle,
                self.ray_count,
                self.display_decimals,
                self.auto_update,
                self.refresh_button,
            ],
            layout=widgets.Layout(flex_flow="row wrap", gap="0.5rem"),
        )
        zoom_controls = widgets.HBox(
            [
                self.zoom_scale,
                self.zoom_pan,
                self.zoom_reset_button,
            ],
            layout=widgets.Layout(flex_flow="row wrap", gap="0.5rem", align_items="center"),
        )
        matrices = widgets.VBox(
            [
                self.base_matrix,
                self.principal_values,
                self.principal_matrix,
                self.conjugate_values,
                self.conjugate_matrix,
            ],
            layout=widgets.Layout(gap="0.75rem"),
        )
        schematic_tab = widgets.VBox(
            [zoom_controls, self.figure_output],
            layout=widgets.Layout(gap="0.45rem", width="100%"),
        )
        tabs = widgets.Tab(children=[matrices, schematic_tab])
        tabs.set_title(0, "Matrix resolution")
        tabs.set_title(1, "Optical schematic")
        self.root = widgets.VBox(
            [
                title,
                builder_toolbar,
                widgets.HTML("<h3>Element stack: V → V′</h3>"),
                self.element_box,
                builder_footer,
                widgets.HTML("<h3>Analysis controls</h3>"),
                analysis_controls,
                self.status,
                tabs,
            ],
            layout=widgets.Layout(gap="0.65rem", width="100%"),
        )

    def _wire_events(self) -> None:
        self.load_preset_button.on_click(
            lambda _: self._load_preset(self.preset.value, refresh=True)
        )
        self.save_system_button.on_click(lambda _: self._save_current_system())
        self.load_system_button.on_click(lambda _: self._load_saved_system())
        self.add_element_button.on_click(lambda _: self._add_default_element())
        self.add_element_button_footer.on_click(lambda _: self._add_default_element())
        self.refresh_button.on_click(lambda _: self.refresh())
        self.unit_toggle.on_click(self._toggle_length_unit)
        self.new_element_kind.observe(self._sync_add_kind_from_top, names="value")
        self.new_element_kind_footer.observe(
            self._sync_add_kind_from_footer,
            names="value",
        )
        self.conjugate_mode.observe(self._on_conjugate_mode_change, names="value")
        self.zoom_reset_button.on_click(lambda _: self._reset_zoom_view())
        for control in (
            self.input_index,
            self.output_index,
            self.ray_half_angle,
            self.ray_count,
            self.display_decimals,
            self.zoom_scale,
            self.zoom_pan,
        ):
            control.observe(self._on_value_change, names="value")

    def _sync_vertex_distance_label(self) -> None:
        unit = self._length_unit_label()
        if self.conjugate_mode.value == "v_prime_to_image":
            base = "V′→image"
        else:
            base = "Object→V"
        # Update registered length-control description base so unit toggles stay correct.
        for index, (control, registered) in enumerate(self._length_controls):
            if control is self.vertex_distance:
                self._length_controls[index] = (control, base)
                break
        self.vertex_distance.description = f"{base} ({unit})"

    def _on_conjugate_mode_change(self, _change: object = None) -> None:
        self._sync_vertex_distance_label()
        self._on_value_change()

    def _reset_zoom_view(self) -> None:
        self._suspend_updates = True
        try:
            self.zoom_scale.value = config.ZOOM_SCALE.default
            self.zoom_pan.value = config.ZOOM_PAN.default
        finally:
            self._suspend_updates = False
        self._on_value_change()

    def _row_numeric(
        self,
        spec: config.NumericControl,
        description: str,
        value: float,
    ) -> widgets.FloatText:
        control = self._numeric(spec, description, value)
        control.observe(self._on_value_change, names="value")
        return control

    def _make_element_row(
        self,
        element: tools.OpticalElement,
    ) -> dict[str, Any]:
        kind = widgets.Dropdown(
            options=config.ELEMENT_OPTIONS,
            value=element.kind.value,
            layout=widgets.Layout(width="10rem"),
        )
        label = widgets.Text(
            value=element.label,
            description="Label",
            layout=widgets.Layout(width="12rem"),
            style={"description_width": "initial"},
        )
        up = widgets.Button(icon="arrow-up", tooltip="Move earlier")
        down = widgets.Button(icon="arrow-down", tooltip="Move later")
        remove = widgets.Button(
            icon="trash",
            tooltip="Remove",
            button_style="danger",
        )

        fields = {
            "refractive_index": self._row_numeric(
                config.REFRACTIVE_INDEX,
                "n",
                element.refractive_index,
            ),
            "distance": self._length_control(
                config.DISTANCE_M,
                "d",
                element.distance,
            ),
            "incident_index": self._row_numeric(
                config.REFRACTIVE_INDEX,
                "nᵢ",
                element.incident_index,
            ),
            "transmitted_index": self._row_numeric(
                config.REFRACTIVE_INDEX,
                "nₜ",
                element.transmitted_index,
            ),
            "radius": self._length_control(
                config.RADIUS_M,
                "R",
                element.radius,
            ),
            "lens_index": self._row_numeric(
                config.REFRACTIVE_INDEX,
                "nₗ",
                element.lens_index,
            ),
            "first_power": self._power_control(
                config.SURFACE_POWER,
                "P₁",
                element.first_power,
            ),
            "second_power": self._power_control(
                config.SURFACE_POWER,
                "P₂",
                element.second_power,
            ),
            "m11": self._row_numeric(
                config.MATRIX_ENTRY,
                "M11",
                element.m11,
            ),
            "m12": self._power_control(
                config.MATRIX_ENTRY,
                "M12",
                element.m12,
            ),
            "m21": self._length_control(
                config.MATRIX_ENTRY,
                "M21",
                element.m21,
            ),
            "m22": self._row_numeric(
                config.MATRIX_ENTRY,
                "M22",
                element.m22,
            ),
        }

        ambient = config.REFRACTIVE_INDEX.minimum
        if element.kind == tools.ElementKind.THIN_LENS:
            glass = (
                config.REFRACTIVE_INDEX.default
                if element.lens_index == 1.0
                else element.lens_index
            )
            fields["lens_index"].value = glass
        elif element.kind == tools.ElementKind.THICK_LENS:
            glass = element.lens_index
            fields["lens_index"].value = glass
        else:
            glass = config.REFRACTIVE_INDEX.default

        power_helpers = {
            "first_power": self._power_calculator_controls(
                fields["first_power"],
                fields["lens_index"],
                surface="first",
                ambient_default=ambient,
            ),
            "second_power": self._power_calculator_controls(
                fields["second_power"],
                fields["lens_index"],
                surface="second",
                ambient_default=ambient,
            ),
        }
        lens_error = widgets.HTML()

        parameter_boxes = {
            tools.ElementKind.TRANSLATION.value: widgets.HBox(
                [fields["refractive_index"], fields["distance"]]
            ),
            tools.ElementKind.REFRACTION.value: widgets.HBox(
                [
                    fields["incident_index"],
                    fields["transmitted_index"],
                    fields["radius"],
                ]
            ),
            tools.ElementKind.REFLECTION.value: widgets.HBox(
                [fields["incident_index"], fields["radius"]]
            ),
            tools.ElementKind.THIN_LENS.value: widgets.VBox(
                [
                    fields["lens_index"],
                    power_helpers["first_power"],
                    power_helpers["second_power"],
                    lens_error,
                ]
            ),
            tools.ElementKind.THICK_LENS.value: widgets.VBox(
                [
                    widgets.HBox(
                        [fields["lens_index"], fields["distance"]],
                        layout=widgets.Layout(flex_flow="row wrap", gap="0.35rem"),
                    ),
                    power_helpers["first_power"],
                    power_helpers["second_power"],
                    lens_error,
                ]
            ),
            tools.ElementKind.SYSTEM_MATRIX.value: widgets.VBox(
                [
                    widgets.HBox(
                        [fields["m11"], fields["m12"]],
                        layout=widgets.Layout(flex_flow="row wrap", gap="0.35rem"),
                    ),
                    widgets.HBox(
                        [fields["m21"], fields["m22"]],
                        layout=widgets.Layout(flex_flow="row wrap", gap="0.35rem"),
                    ),
                ]
            ),
        }
        for box in parameter_boxes.values():
            if isinstance(box, widgets.HBox):
                box.layout = widgets.Layout(flex_flow="row wrap", gap="0.35rem")
            else:
                box.layout = widgets.Layout(flex_flow="column", gap="0.35rem")

        container = widgets.VBox(
            [
                widgets.HBox(
                    [kind, label, up, down, remove],
                    layout=widgets.Layout(flex_flow="row wrap", gap="0.35rem"),
                ),
                widgets.VBox(list(parameter_boxes.values())),
            ],
            layout=widgets.Layout(
                border="2px solid #cbd5e1",
                padding="0.45rem",
                margin="0 0 0.35rem 0",
            ),
        )
        row: dict[str, Any] = {
            "kind": kind,
            "label": label,
            "fields": fields,
            "parameter_boxes": parameter_boxes,
            "power_helpers": power_helpers,
            "lens_error": lens_error,
            "up": up,
            "down": down,
            "remove": remove,
            "container": container,
        }
        self._apply_row_accent(row)

        kind.observe(lambda change: self._on_kind_change(row, change), names="value")
        label.observe(self._on_value_change, names="value")
        up.on_click(lambda _: self._move_row(row, -1))
        down.on_click(lambda _: self._move_row(row, 1))
        remove.on_click(lambda _: self._remove_row(row))
        self._refresh_parameter_visibility(row)
        return row

    @staticmethod
    def _apply_row_accent(row: dict[str, Any]) -> None:
        """Tint the element-stack row to match its matrix type color."""
        style = config.factor_card_style(row["kind"].value)
        row["container"].layout.border = f"2px solid {style['border']}"
        row["container"].layout.background_color = style["background"]

    def _power_calculator_controls(
        self,
        power_field: widgets.FloatText,
        lens_index_field: widgets.FloatText,
        surface: str,
        ambient_default: float,
    ) -> widgets.HBox:
        """Build a surface power calculator locked to the lens index nₗ.

        For the first surface, ``nₜ = nₗ``. For the second surface, ``nᵢ = nₗ``.
        """
        if surface not in {"first", "second"}:
            raise ValueError("surface must be 'first' or 'second'.")

        if surface == "first":
            incident = self._numeric(
                config.REFRACTIVE_INDEX,
                "nᵢ",
                ambient_default,
            )
            transmitted = self._numeric(
                config.REFRACTIVE_INDEX,
                "nₜ=nₗ",
                lens_index_field.value,
            )
            transmitted.disabled = True
            linked = transmitted
            relation = "nₜ = nₗ"
        else:
            incident = self._numeric(
                config.REFRACTIVE_INDEX,
                "nᵢ=nₗ",
                lens_index_field.value,
            )
            incident.disabled = True
            transmitted = self._numeric(
                config.REFRACTIVE_INDEX,
                "nₜ",
                ambient_default,
            )
            linked = incident
            relation = "nᵢ = nₗ"

        radius = self._length_control(
            config.RADIUS_M,
            "R",
            config.RADIUS_M.default,
            observe=False,
        )
        button = widgets.Button(
            description="Calc P",
            tooltip=(
                f"Set power from P = (nₜ − nᵢ) / R with {relation} "
                "using the window units"
            ),
            icon="calculator",
            button_style="info",
            layout=widgets.Layout(width="6.5rem"),
        )
        status = widgets.HTML()

        def _sync_lens_index(_change: object = None) -> None:
            linked.value = lens_index_field.value

        def _relation_holds() -> bool:
            return bool(
                np.isclose(
                    linked.value,
                    lens_index_field.value,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

        def _apply(_button: widgets.Button = button, *, notify: bool = True) -> bool:
            _sync_lens_index()
            if not _relation_holds():
                status.value = (
                    "<span style='color:#991b1b;font-size:0.8rem'>"
                    f"Lens error: {relation} is required for this surface."
                    "</span>"
                )
                return False
            try:
                power_si = engine.interface_power(
                    incident.value,
                    transmitted.value,
                    self._to_si_length(radius.value),
                )
            except ValueError as exc:
                status.value = (
                    f"<span style='color:#991b1b;font-size:0.8rem'>{exc}</span>"
                )
                return False
            power_display = self._from_si_power(power_si)
            exact = self._assign_control_value(power_field, power_display)
            unit = self._power_unit_label()
            status.value = (
                f"<span style='color:#166534;font-size:0.8rem'>"
                f"P=({transmitted.value:g}−{incident.value:g})/"
                f"{radius.value:g}={exact:.{self._display_digits()}f} {unit} "
                f"({relation})</span>"
            )
            if notify:
                self._on_value_change()
            return True

        lens_index_field.observe(_sync_lens_index, names="value")
        _sync_lens_index()
        button.on_click(lambda _btn: _apply(notify=True))
        box = widgets.HBox(
            [power_field, incident, transmitted, radius, button, status],
            layout=widgets.Layout(
                flex_flow="row wrap",
                gap="0.35rem",
                align_items="center",
            ),
        )
        # Expose calculator widgets for lens validation and Recalculate sync.
        box._paraxial_surface = surface  # type: ignore[attr-defined]
        box._paraxial_incident = incident  # type: ignore[attr-defined]
        box._paraxial_transmitted = transmitted  # type: ignore[attr-defined]
        box._paraxial_linked = linked  # type: ignore[attr-defined]
        box._paraxial_relation_holds = _relation_holds  # type: ignore[attr-defined]
        box._paraxial_apply_power = lambda: _apply(notify=False)  # type: ignore[attr-defined]
        return box

    def _validate_lens_row(self, row: dict[str, Any]) -> None:
        """Require P₁: nₜ=nₗ and P₂: nᵢ=nₗ for thin/thick lenses."""
        kind = row["kind"].value
        if kind not in {
            tools.ElementKind.THIN_LENS.value,
            tools.ElementKind.THICK_LENS.value,
        }:
            row["lens_error"].value = ""
            return

        lens_index = row["fields"]["lens_index"].value
        helpers = row["power_helpers"]
        first = helpers["first_power"]
        second = helpers["second_power"]

        errors: list[str] = []
        if not first._paraxial_relation_holds():  # type: ignore[attr-defined]
            errors.append(
                f"{row['label'].value}: P1 requires nt = nl "
                f"(got nt={first._paraxial_transmitted.value:g}, "  # type: ignore[attr-defined]
                f"nl={lens_index:g})."
            )
        if not second._paraxial_relation_holds():  # type: ignore[attr-defined]
            errors.append(
                f"{row['label'].value}: P2 requires ni = nl "
                f"(got ni={second._paraxial_incident.value:g}, "  # type: ignore[attr-defined]
                f"nl={lens_index:g})."
            )
        if errors:
            row["lens_error"].value = (
                "<div class='paraxial-error'>" + "<br>".join(errors) + "</div>"
            )
            raise ValueError(" ".join(errors))
        row["lens_error"].value = ""

    def _refresh_parameter_visibility(self, row: dict[str, Any]) -> None:
        selected = row["kind"].value
        for kind, box in row["parameter_boxes"].items():
            box.layout.display = "" if kind == selected else "none"

    def _on_kind_change(
        self,
        row: dict[str, Any],
        _change: dict[str, Any],
    ) -> None:
        self._apply_row_accent(row)
        self._refresh_parameter_visibility(row)
        self._on_value_change()

    def _sync_add_kind_from_top(self, change: dict[str, Any]) -> None:
        if self.new_element_kind_footer.value != change["new"]:
            self.new_element_kind_footer.value = change["new"]

    def _sync_add_kind_from_footer(self, change: dict[str, Any]) -> None:
        if self.new_element_kind.value != change["new"]:
            self.new_element_kind.value = change["new"]

    def _sync_lens_powers_from_calculators(self) -> None:
        """Rewrite P₁/P₂ from each lens Calc-P inputs before matrix rebuild."""
        for row in self.element_rows:
            kind = row["kind"].value
            if kind not in {
                tools.ElementKind.THIN_LENS.value,
                tools.ElementKind.THICK_LENS.value,
            }:
                continue
            helpers = row["power_helpers"]
            for key in ("first_power", "second_power"):
                apply = getattr(helpers[key], "_paraxial_apply_power", None)
                if callable(apply):
                    apply()

    def _on_value_change(self, _change: object = None) -> None:
        if (
            not self._suspend_updates
            and self.auto_update.value
            and not self._rendering
        ):
            self.refresh()

    def _sync_element_box(self) -> None:
        self.element_box.children = tuple(
            row["container"] for row in self.element_rows
        )
        last = len(self.element_rows) - 1
        for index, row in enumerate(self.element_rows):
            row["up"].disabled = index == 0
            row["down"].disabled = index == last

    def _forget_row_unit_controls(self, row: dict[str, Any]) -> None:
        live = {id(control) for control, _ in row.get("unit_controls", ())}
        if not live:
            fields = row["fields"]
            live = {
                id(fields["distance"]),
                id(fields["radius"]),
                id(fields["first_power"]),
                id(fields["second_power"]),
                id(fields["m12"]),
                id(fields["m21"]),
            }
            # Power calculators also register radius controls.
            for box in row["parameter_boxes"].values():
                for child in getattr(box, "children", ()):
                    if isinstance(child, widgets.HBox):
                        for widget in child.children:
                            if isinstance(widget, widgets.FloatText):
                                live.add(id(widget))
                    elif isinstance(child, widgets.FloatText):
                        live.add(id(child))
                    elif isinstance(child, widgets.VBox):
                        for nested in child.children:
                            if isinstance(nested, widgets.HBox):
                                for widget in nested.children:
                                    if isinstance(widget, widgets.FloatText):
                                        live.add(id(widget))

        self._length_controls = [
            item for item in self._length_controls if id(item[0]) not in live
        ]
        self._power_controls = [
            item for item in self._power_controls if id(item[0]) not in live
        ]

    def _load_preset(self, name: str, refresh: bool = False) -> None:
        self._suspend_updates = True
        try:
            for row in self.element_rows:
                self._forget_row_unit_controls(row)
            # Keep analysis length controls; rebuild only stack-linked ones.
            analysis_ids = {id(self.vertex_distance), id(self.object_height)}
            self._length_controls = [
                item for item in self._length_controls if id(item[0]) in analysis_ids
            ]
            self._power_controls = []
            self.element_rows = [
                self._make_element_row(element)
                for element in tools.preset_elements(name)
            ]
            self._sync_element_box()
        finally:
            self._suspend_updates = False
        if refresh:
            self.refresh()

    def _refresh_saved_system_list(self, select: str | None = None) -> None:
        options = _saved_system_options()
        self.saved_system.options = options
        values = {value for _, value in options}
        if select and select in values:
            self.saved_system.value = select
        elif self.saved_system.value not in values:
            self.saved_system.value = options[0][1]

    def _current_elements(self) -> list[tools.OpticalElement]:
        return [self._element_from_row(row) for row in self.element_rows]

    def _save_current_system(self) -> None:
        name = str(self.save_name.value).strip() or "untitled"
        try:
            path = tools.save_system(
                name,
                self._current_elements(),
                {
                    "input_index": float(self.input_index.value),
                    "output_index": float(self.output_index.value),
                    "conjugate_mode": str(self.conjugate_mode.value),
                    "vertex_distance_m": self._to_si_length(self.vertex_distance.value),
                    "object_height_m": self._to_si_length(self.object_height.value),
                    "ray_half_angle": float(self.ray_half_angle.value),
                    "ray_count": int(self.ray_count.value),
                    "display_decimals": int(self.display_decimals.value),
                },
                length_unit=self.length_unit,
            )
        except Exception as exc:
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return
        self._refresh_saved_system_list(select=path.stem)
        self.save_name.value = path.stem
        self._set_status(f"Saved system to {path.name} (overwrites same name).")

    def _load_saved_system(self) -> None:
        name = str(self.saved_system.value).strip()
        if not name:
            self._set_status("No saved system selected.", error=True)
            return
        try:
            payload = tools.load_system(name)
        except Exception as exc:
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return

        analysis = payload["analysis"]
        target_unit = str(payload["length_unit"])
        self._suspend_updates = True
        try:
            for row in self.element_rows:
                self._forget_row_unit_controls(row)
            analysis_ids = {id(self.vertex_distance), id(self.object_height)}
            self._length_controls = [
                item for item in self._length_controls if id(item[0]) in analysis_ids
            ]
            self._power_controls = []
            self.element_rows = [
                self._make_element_row(element) for element in payload["elements"]
            ]
            self._sync_element_box()
            self.input_index.value = float(analysis["input_index"])
            self.output_index.value = float(analysis["output_index"])
            mode = str(analysis["conjugate_mode"])
            mode_values = {value for _, value in config.CONJUGATE_MODE_OPTIONS}
            if mode not in mode_values:
                raise ValueError(f"Unknown conjugate_mode {mode!r}.")
            self.conjugate_mode.value = mode
            self._sync_vertex_distance_label()
            self._assign_control_value(
                self.vertex_distance,
                self._from_si_length(float(analysis["vertex_distance_m"])),
            )
            self._assign_control_value(
                self.object_height,
                self._from_si_length(float(analysis["object_height_m"])),
            )
            self.ray_half_angle.value = float(analysis["ray_half_angle"])
            self.ray_count.value = int(analysis["ray_count"])
            self.display_decimals.value = int(analysis["display_decimals"])
            self.save_name.value = str(payload["name"])
        except Exception as exc:
            self._suspend_updates = False
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return
        finally:
            self._suspend_updates = False

        if target_unit != self.length_unit:
            self._toggle_length_unit()
        else:
            self.refresh()
        self._set_status(f"Loaded system {payload['name']!r}.")

    def _default_element(self, kind: str) -> tools.OpticalElement:
        index = len(self.element_rows) + 1
        if kind == tools.ElementKind.TRANSLATION.value:
            return tools.OpticalElement.translation(
                f"T{index}",
                config.REFRACTIVE_INDEX.default,
                config.DISTANCE_M.default,
            )
        if kind == tools.ElementKind.REFRACTION.value:
            return tools.OpticalElement.refraction(
                f"Ra{index}",
                config.REFRACTIVE_INDEX.minimum,
                config.REFRACTIVE_INDEX.default,
                config.RADIUS_M.default,
            )
        if kind == tools.ElementKind.REFLECTION.value:
            return tools.OpticalElement.reflection(
                f"Re{index}",
                config.REFRACTIVE_INDEX.minimum,
                config.RADIUS_M.default,
            )
        if kind == tools.ElementKind.THIN_LENS.value:
            return tools.OpticalElement.thin_lens(
                f"L{index}",
                config.SURFACE_POWER.default,
                config.SURFACE_POWER.default,
                lens_index=config.REFRACTIVE_INDEX.default,
            )
        if kind == tools.ElementKind.SYSTEM_MATRIX.value:
            return tools.OpticalElement.system_matrix(
                f"M{index}",
                np.eye(2, dtype=float),
            )
        return tools.OpticalElement.thick_lens(
            f"L{index}",
            config.REFRACTIVE_INDEX.default,
            config.DISTANCE_M.default,
            config.SURFACE_POWER.default,
            config.SURFACE_POWER.default,
        )

    def _add_default_element(self) -> None:
        self.element_rows.append(
            self._make_element_row(
                self._default_element(self.new_element_kind.value)
            )
        )
        self._sync_element_box()
        self._on_value_change()

    def _remove_row(self, row: dict[str, Any]) -> None:
        self._forget_row_unit_controls(row)
        self.element_rows.remove(row)
        self._sync_element_box()
        self._on_value_change()

    def _move_row(self, row: dict[str, Any], offset: int) -> None:
        old_index = self.element_rows.index(row)
        new_index = max(0, min(old_index + offset, len(self.element_rows) - 1))
        if new_index == old_index:
            return
        self.element_rows.pop(old_index)
        self.element_rows.insert(new_index, row)
        self._sync_element_box()
        self._on_value_change()

    def _element_from_row(self, row: dict[str, Any]) -> tools.OpticalElement:
        kind = tools.ElementKind(row["kind"].value)
        label = row["label"].value
        fields = row["fields"]
        if kind == tools.ElementKind.TRANSLATION:
            return tools.OpticalElement.translation(
                label,
                fields["refractive_index"].value,
                self._to_si_length(fields["distance"].value),
            )
        if kind == tools.ElementKind.REFRACTION:
            return tools.OpticalElement.refraction(
                label,
                fields["incident_index"].value,
                fields["transmitted_index"].value,
                self._to_si_length(fields["radius"].value),
            )
        if kind == tools.ElementKind.REFLECTION:
            return tools.OpticalElement.reflection(
                label,
                fields["incident_index"].value,
                self._to_si_length(fields["radius"].value),
            )
        if kind == tools.ElementKind.THIN_LENS:
            return tools.OpticalElement.thin_lens(
                label,
                self._to_si_power(fields["first_power"].value),
                self._to_si_power(fields["second_power"].value),
                lens_index=fields["lens_index"].value,
            )
        if kind == tools.ElementKind.SYSTEM_MATRIX:
            return tools.OpticalElement.system_matrix(
                label,
                [
                    [
                        fields["m11"].value,
                        self._to_si_power(fields["m12"].value),
                    ],
                    [
                        self._to_si_length(fields["m21"].value),
                        fields["m22"].value,
                    ],
                ],
            )
        return tools.OpticalElement.thick_lens(
            label,
            fields["lens_index"].value,
            self._to_si_length(fields["distance"].value),
            self._to_si_power(fields["first_power"].value),
            self._to_si_power(fields["second_power"].value),
        )

    def _set_status(self, message: str = "", error: bool = False) -> None:
        if not message:
            self.status.value = ""
            return
        css_class = "paraxial-error" if error else "paraxial-note"
        self.status.value = f"<div class='{css_class}'>{message}</div>"

    def _unit_note(self) -> str:
        unit = self._length_unit_label()
        return (
            f"All lengths and powers in this window use {unit} / {unit}⁻¹. "
            "Toggle units any time; the engine converts to SI metres internally."
        )

    def _render_base(self, report: engine.SystemMatrixReport) -> None:
        factors = self._display_factors(report.factors)
        matrix = self._scale_matrix(report.matrix)
        digits = self._display_digits()
        self.base_matrix.value = visuals.factor_product_html(
            "1. Stack matrices → base matrix (rightmost = first on the input)",
            factors,
            matrix,
            "M_VV′",
            (
                ("order", report.product_expression or "I"),
                ("det(M_VV′)", f"{report.determinant:.{digits}f}"),
                (
                    "unit determinant",
                    "yes" if report.is_unit_determinant else "no",
                ),
                ("display units", f"{self._length_unit_label()} / {self._power_unit_label()}"),
                ("digits", str(digits)),
            ),
            decimals=digits,
        )

    def _render_principal(self, cardinal: engine.CardinalPoints) -> None:
        length = self._length_unit_label()
        power = self._power_unit_label()
        digits = self._display_digits()
        self.principal_values.value = visuals.values_html(
            "2. Principal-plane values from M_VV′",
            (
                (
                    "Power P",
                    _quantity(self._from_si_power(cardinal.power), power, digits),
                ),
                (
                    "H→V: D",
                    _quantity(
                        self._from_si_length(cardinal.object_principal_offset),
                        length,
                        digits,
                    ),
                ),
                (
                    "V′→H′: D′",
                    _quantity(
                        self._from_si_length(cardinal.image_principal_offset),
                        length,
                        digits,
                    ),
                ),
                (
                    "Front focal f",
                    _quantity(
                        self._from_si_length(cardinal.front_focal_length),
                        length,
                        digits,
                    ),
                ),
                (
                    "Back focal f′",
                    _quantity(
                        self._from_si_length(cardinal.back_focal_length),
                        length,
                        digits,
                    ),
                ),
            ),
            self._unit_note(),
        )
        self.principal_matrix.value = visuals.factor_product_html(
            "Principal-plane matrix reduction",
            self._display_factors(cardinal.reduction_factors),
            self._scale_matrix(cardinal.reduction_matrix),
            "M_HH′",
            (
                ("target", "[[1, -P], [0, 1]]"),
                (
                    "max residual",
                    f"{cardinal.reduction_residual:.{digits}f}",
                ),
            ),
            decimals=digits,
        )

    def _render_conjugate(
        self,
        result: engine.ConjugateReport,
        cardinal: engine.CardinalPoints,
    ) -> None:
        image_kind = "real" if result.is_real else "virtual"
        orientation = "erect" if result.is_erect else "inverted"
        length = self._length_unit_label()
        power = self._power_unit_label()
        digits = self._display_digits()
        p_check = engine.gaussian_power(
            result.object_distance,
            result.image_distance,
            self.input_index.value,
            self.output_index.value,
        )
        mode_label = (
            "Object→V"
            if self.conjugate_mode.value == "object_to_v"
            else "V′→image"
        )
        self.conjugate_values.value = visuals.values_html(
            "3. Conjugate-plane values",
            (
                ("Known", mode_label),
                (
                    "Object→V (x)",
                    _quantity(
                        self._from_si_length(result.object_to_vertex),
                        length,
                        digits,
                    ),
                ),
                (
                    "V′→image (x′)",
                    _quantity(
                        self._from_si_length(result.exit_vertex_to_image),
                        length,
                        digits,
                    ),
                ),
                (
                    "Object s (from H)",
                    _quantity(
                        self._from_si_length(result.object_distance),
                        length,
                        digits,
                    ),
                ),
                (
                    "Image s′ (from H′)",
                    _quantity(
                        self._from_si_length(result.image_distance),
                        length,
                        digits,
                    ),
                ),
                (
                    "P (from M_VV′)",
                    _quantity(
                        self._from_si_power(cardinal.power),
                        power,
                        digits,
                    ),
                ),
                (
                    "P check n′/s′+n/s",
                    _quantity(self._from_si_power(p_check), power, digits),
                ),
                (
                    "Lateral mₓ",
                    _quantity(result.lateral_magnification, "", digits),
                ),
                (
                    "Angular mα",
                    _quantity(result.angular_magnification, "", digits),
                ),
                (
                    "Lagrange check",
                    _quantity(result.lagrange_invariant, "", digits),
                ),
                (
                    "Image",
                    f"{image_kind}, {orientation}, {result.size_classification}",
                ),
            ),
            (
                f"s = x − D and s′ = x′ − D′ with D, D′ from M_VV′. "
                f"{self._unit_note()}"
            ),
        )
        self.conjugate_matrix.value = visuals.factor_product_html(
            "Conjugate object → image matrix",
            self._display_factors(result.factors),
            self._scale_matrix(result.matrix),
            "M_oi",
            (
                ("order", result.product_expression),
                (
                    "M₂₁",
                    f"{self._from_si_length(result.conjugacy_residual):.{digits}f}",
                ),
                ("conjugate", "yes" if result.is_conjugate else "no"),
            ),
            decimals=digits,
        )

    def _clear_principal(self, message: str) -> None:
        self.principal_values.value = visuals.values_html(
            "2. Principal planes unavailable",
            (),
            message,
        )
        self.principal_matrix.value = ""
        self.conjugate_values.value = ""
        self.conjugate_matrix.value = ""

    def _clear_conjugate(self, message: str) -> None:
        self.conjugate_values.value = visuals.values_html(
            "3. Conjugate planes unavailable",
            (),
            message,
        )
        self.conjugate_matrix.value = ""

    def _clear_all_results(self, message: str) -> None:
        self.base_matrix.value = visuals.values_html(
            "Matrix calculation unavailable",
            (),
            message,
        )
        self.principal_values.value = ""
        self.principal_matrix.value = ""
        self.conjugate_values.value = ""
        self.conjugate_matrix.value = ""
        with self.figure_output:
            self.figure_output.clear_output(wait=True)

    def _render_figure(
        self,
        elements: list[tools.OpticalElement],
        cardinal: engine.CardinalPoints | None,
        conjugate: engine.ConjugateReport | None,
    ) -> None:
        object_height_si = self._to_si_length(self.object_height.value)
        with self.figure_output:
            self.figure_output.clear_output(wait=True)
            fig, ax = plt.subplots(
                figsize=config.figure_size_for_elements(max(len(elements), 1))
            )
            visuals.draw_optical_schematic(
                ax,
                elements,
                cardinal,
                conjugate,
                object_height_si,
                trace=None,
                length_scale=self._length_scale(),
                length_unit=self._length_unit_label(),
            )
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def refresh(self) -> None:
        """Resolve all matrices and redraw every output panel."""
        if self._rendering:
            return
        self._rendering = True
        try:
            self._set_status()
            self._sync_lens_powers_from_calculators()
            elements = []
            for row in self.element_rows:
                self._validate_lens_row(row)
                elements.append(self._element_from_row(row))
            base = tools.build_system(elements)
            self._render_base(base)

            cardinal: engine.CardinalPoints | None = None
            conjugate: engine.ConjugateReport | None = None
            try:
                cardinal = engine.principal_planes(
                    base.matrix,
                    self.input_index.value,
                    self.output_index.value,
                    config.MATRIX_TOLERANCE,
                )
                self._render_principal(cardinal)
            except engine.AfocalSystemError as exc:
                self._clear_principal(str(exc))

            if cardinal is not None:
                try:
                    conjugate = engine.conjugate_from_vertex_distance(
                        cardinal,
                        self.conjugate_mode.value,
                        self._to_si_length(self.vertex_distance.value),
                        self.input_index.value,
                        self.output_index.value,
                        config.MATRIX_TOLERANCE,
                    )
                    self._render_conjugate(conjugate, cardinal)
                except (engine.ConjugateAtInfinityError, ValueError) as exc:
                    self._clear_conjugate(str(exc))

            self._render_figure(elements, cardinal, conjugate)
        except Exception as exc:
            self._clear_all_results(str(exc))
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
        finally:
            self._rendering = False

    def show(self) -> None:
        """Display the complete dashboard in the current notebook."""
        display(self.root)
