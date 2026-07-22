"""Friendly ipywidgets dashboard for the Fraunhofer simulator.

The notebook only needs to instantiate :class:`FraunhoferDashboard` and call
``show()``. Physics remains in :mod:`diffraction_engine` and
:mod:`screen_tools`; reusable plotting primitives remain in
:mod:`dashboard_tools`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

import dashboard_tools as visuals
import diffraction_config as config
import diffraction_engine as engine
import fourier_transform as fourier
import screen_tools as screen

__all__ = ["FraunhoferDashboard"]


class FraunhoferDashboard:
    """Self-contained interactive Fraunhofer notebook dashboard."""

    def __init__(self) -> None:
        self._rendering = False
        self._suspend_updates = False
        self.aperture_rows: list[dict[str, widgets.Widget]] = []

        self._create_controls()
        self._create_outputs()
        self._create_layout()
        self._wire_events()
        self._refresh_case_controls()

    @staticmethod
    def _linked_float(
        spec: config.NumericControl,
        description: str,
        value: float | None = None,
    ) -> tuple[widgets.FloatSlider, widgets.HBox]:
        selected = spec.default if value is None else value
        slider = widgets.FloatSlider(
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=selected,
            description=description,
            readout=False,
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=config.DASHBOARD_SLIDER_WIDTH),
        )
        text = widgets.BoundedFloatText(
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=selected,
            layout=widgets.Layout(width=config.DASHBOARD_TEXT_WIDTH),
        )
        widgets.link((slider, "value"), (text, "value"))
        box = widgets.HBox(
            [slider, text],
            layout=widgets.Layout(width="100%", flex_flow="row wrap"),
        )
        return slider, box

    @staticmethod
    def _linked_int(
        spec: config.NumericControl,
        description: str,
    ) -> tuple[widgets.IntSlider, widgets.HBox]:
        slider = widgets.IntSlider(
            min=int(spec.minimum),
            max=int(spec.maximum),
            step=int(spec.step),
            value=int(spec.default),
            description=description,
            readout=False,
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=config.DASHBOARD_SLIDER_WIDTH),
        )
        text = widgets.BoundedIntText(
            min=int(spec.minimum),
            max=int(spec.maximum),
            step=int(spec.step),
            value=int(spec.default),
            layout=widgets.Layout(width=config.DASHBOARD_TEXT_WIDTH),
        )
        widgets.link((slider, "value"), (text, "value"))
        box = widgets.HBox(
            [slider, text],
            layout=widgets.Layout(width="100%", flex_flow="row wrap"),
        )
        return slider, box

    @staticmethod
    def _bounded_float(
        spec: config.NumericControl,
        description: str,
        value: float | None = None,
    ) -> widgets.BoundedFloatText:
        return widgets.BoundedFloatText(
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=spec.default if value is None else value,
            description=description,
            style={"description_width": "initial"},
        )

    def _create_controls(self) -> None:
        self.case = widgets.ToggleButtons(
            options=config.DASHBOARD_CASE_OPTIONS,
            value="slit",
            description="Aperture",
            icons=("pause", "square-o", "circle-o", "th", "paint-brush"),
            tooltips=(
                "Single slit",
                "Rectangle",
                "Circle",
                "Multiple mixed openings",
                "Expression or binary-image opening",
            ),
            style={"description_width": "initial"},
        )
        self.slit_orientation = widgets.ToggleButtons(
            options=config.SLIT_ORIENTATION_OPTIONS,
            value="vertical",
            description="Slit orientation",
            style={"description_width": "initial"},
        )
        self.auto_update = widgets.Checkbox(
            value=config.DEFAULT_AUTO_UPDATE,
            description="Auto update",
            indent=False,
        )
        self.refresh_button = widgets.Button(
            description="Refresh", icon="refresh", button_style="primary"
        )
        self.reset_button = widgets.Button(description="Reset", icon="undo")
        self.add_aperture_button = widgets.Button(
            description="Add opening", icon="plus", button_style="info"
        )
        self.extra_view_buttons = {
            key: widgets.ToggleButton(value=False, description=label)
            for label, key in config.EXTRA_VIEW_OPTIONS
        }

        self.wavelength, self.wavelength_box = self._linked_float(
            config.WAVELENGTH_NM, "lambda_0 (nm)"
        )
        self.refractive_index, self.refractive_index_box = self._linked_float(
            config.REFRACTIVE_INDEX, "n"
        )
        self.distance, self.distance_box = self._linked_float(
            config.DISTANCE_M, "distance z (m)"
        )
        self.screen_half_width, self.screen_half_width_box = self._linked_float(
            config.SCREEN_HALF_WIDTH_MM, "screen half-width (mm)"
        )
        self.zoom, self.zoom_box = self._linked_float(config.ZOOM, "zoom")
        self.resolution, self.resolution_box = self._linked_int(
            config.RESOLUTION, "resolution"
        )
        self.max_fresnel, self.max_fresnel_box = self._linked_float(
            config.MAX_FRESNEL_NUMBER, "maximum N_F"
        )
        self.secondary_boost, self.secondary_boost_box = self._linked_float(
            config.SECONDARY_MAXIMA_BOOST, "secondary maxima boost"
        )

        self.slit_width, self.slit_width_box = self._linked_float(
            config.APERTURE_WIDTH_UM,
            "slit width (um)",
            config.DEFAULT_SLIT_WIDTH_M
            / config.APERTURE_WIDTH_UM.display_to_si,
        )
        self.slit_length, self.slit_length_box = self._linked_float(
            config.APERTURE_HEIGHT_UM,
            "slit length (um)",
            config.DEFAULT_SLIT_LENGTH_M
            / config.APERTURE_HEIGHT_UM.display_to_si,
        )
        self.rectangle_width, self.rectangle_width_box = self._linked_float(
            config.APERTURE_WIDTH_UM,
            "width (um)",
            config.DEFAULT_RECTANGLE_WIDTH_M
            / config.APERTURE_WIDTH_UM.display_to_si,
        )
        self.rectangle_height, self.rectangle_height_box = self._linked_float(
            config.APERTURE_HEIGHT_UM,
            "height (um)",
            config.DEFAULT_RECTANGLE_HEIGHT_M
            / config.APERTURE_HEIGHT_UM.display_to_si,
        )
        self.circle_radius, self.circle_radius_box = self._linked_float(
            config.CIRCLE_RADIUS_UM,
            "radius (um)",
            config.DEFAULT_CIRCLE_RADIUS_M
            / config.CIRCLE_RADIUS_UM.display_to_si,
        )

        self.custom_mode = widgets.ToggleButtons(
            options=config.CUSTOM_INPUT_MODE_OPTIONS,
            value="expression",
            description="Input",
            style={"description_width": "initial"},
        )
        self.custom_width, self.custom_width_box = self._linked_float(
            config.CUSTOM_WIDTH_MM, "physical width (mm)"
        )
        self.custom_height, self.custom_height_box = self._linked_float(
            config.CUSTOM_HEIGHT_MM, "physical height (mm)"
        )
        self.custom_mask_resolution, self.custom_mask_resolution_box = (
            self._linked_int(
                config.CUSTOM_MASK_RESOLUTION, "mask pixels / axis"
            )
        )
        self.custom_threshold, self.custom_threshold_box = self._linked_float(
            config.CUSTOM_IMAGE_THRESHOLD, "white-open threshold"
        )
        self.custom_fft_padding, self.custom_fft_padding_box = (
            self._linked_float(config.CUSTOM_FFT_PADDING, "FFT padding")
        )
        self.custom_expression = widgets.Textarea(
            value=config.DEFAULT_CUSTOM_EXPRESSION,
            description="Shape",
            placeholder="Example: r <= 0.15",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%", height="105px"),
        )
        self.custom_expression_help = widgets.HTML(
            f"<div class='custom-expression-help'>"
            f"{config.CUSTOM_EXPRESSION_HELP}</div>"
        )
        self.custom_upload = widgets.FileUpload(
            accept=".png,.jpg,.jpeg,.bmp,.gif",
            multiple=False,
            description="Upload mask",
        )
        self.custom_invert = widgets.ToggleButton(
            value=False,
            description="Invert white / black",
            icon="adjust",
            tooltip="Swap open and closed regions",
        )
        self.custom_mode_controls = widgets.VBox(
            layout=widgets.Layout(width="100%")
        )

        self.case_controls = widgets.VBox(layout=widgets.Layout(width="100%"))
        self.aperture_rows_box = widgets.VBox(
            layout=widgets.Layout(width="100%")
        )
        for center in config.DEFAULT_MULTIPLE_CENTERS_MM:
            self._add_aperture_row("slit", center, render=False)

    def _create_outputs(self) -> None:
        self.schematic_output = widgets.Output(
            layout=widgets.Layout(width="100%", min_height="210px")
        )
        self.aperture_preview_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="390px",
                display="flex",
                justify_content="center",
                align_items="center",
            )
        )
        self.diffraction_preview_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="390px",
                display="flex",
                justify_content="center",
                align_items="center",
            )
        )
        self.aperture_preview_output.add_class("dashboard-preview-output")
        self.diffraction_preview_output.add_class("dashboard-preview-output")
        self.extra_output = widgets.Output(
            layout=widgets.Layout(width="100%")
        )
        self.validity_status = widgets.HTML()
        self.spectrum_indicator = widgets.HTML()

    def _card(
        self,
        title: str,
        children: list[widgets.Widget],
        width: str = "100%",
    ) -> widgets.VBox:
        card = widgets.VBox(
            [
                widgets.HTML(
                    f"<div class='dashboard-card-title'>{title}</div>"
                ),
                *children,
            ],
            layout=widgets.Layout(
                width=width,
                border=config.DASHBOARD_CARD_BORDER,
                padding=config.DASHBOARD_CARD_PADDING,
                margin=config.DASHBOARD_CARD_MARGIN,
            ),
        )
        card.add_class("dashboard-card")
        return card

    def _create_layout(self) -> None:
        source_section = self._card(
            "SOURCE",
            [
                self.spectrum_indicator,
                self.wavelength_box,
                self.refractive_index_box,
            ],
        )
        aperture_section = self._card(
            "APERTURE", [self.case, self.case_controls]
        )
        propagation_section = self._card(
            "PROPAGATION", [self.distance_box]
        )
        extra_selector = widgets.HBox(
            tuple(self.extra_view_buttons.values())
        )
        advanced = widgets.Accordion(
            children=[
                widgets.VBox(
                    [self.resolution_box, self.max_fresnel_box]
                )
            ]
        )
        advanced.set_title(0, "Advanced sampling and validity")
        display_section = self._card(
            "DISPLAY",
            [
                self.screen_half_width_box,
                self.zoom_box,
                self.secondary_boost_box,
                advanced,
                widgets.HTML("<b>Additional views</b>"),
                extra_selector,
            ],
        )
        control_panel = self._card(
            "CONTROLS",
            [
                source_section,
                aperture_section,
                propagation_section,
                display_section,
            ],
            width=config.DASHBOARD_CONTROL_WIDTH,
        )
        aperture_panel = self._card(
            "SELECTED OPENING", [self.aperture_preview_output]
        )
        diffraction_panel = self._card(
            "CAMERA / OBSERVATION", [self.diffraction_preview_output]
        )
        preview_column = widgets.VBox(
            [aperture_panel, diffraction_panel],
            layout=widgets.Layout(width="100%"),
        )
        schematic_panel = self._card(
            "OPTICAL PATH", [self.schematic_output]
        )
        self.extras_card = self._card(
            "ANALYTICAL DETAILS", [self.extra_output]
        )
        self.extras_card.layout.display = "none"

        header = widgets.HBox(
            [
                widgets.HTML(
                    "<div class='dashboard-title'>"
                    "Fraunhofer Diffraction Workbench"
                    "<span>Analytical and numerical far-field simulator"
                    "</span></div>"
                ),
                widgets.HBox(
                    [
                        self.auto_update,
                        self.refresh_button,
                        self.reset_button,
                    ]
                ),
            ],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
                flex_flow="row wrap",
            ),
        )
        main_area = widgets.GridBox(
            [control_panel, preview_column],
            layout=widgets.Layout(
                width="100%",
                grid_template_columns=config.DASHBOARD_MAIN_COLUMNS,
                grid_gap=config.DASHBOARD_COLUMN_GAP,
                align_items="flex-start",
            ),
        )
        main_area.add_class("dashboard-main-grid")

        style = widgets.HTML(self._style_html())
        self.widget = widgets.VBox(
            [
                style,
                header,
                self.validity_status,
                schematic_panel,
                main_area,
                self.extras_card,
            ],
            layout=widgets.Layout(width="100%"),
        )
        self.widget.add_class("fraunhofer-dashboard")

    @staticmethod
    def _style_html() -> str:
        return f"""
        <style>
          .fraunhofer-dashboard {{
            max-width: {config.DASHBOARD_MAX_WIDTH}; margin: 0 auto;
          }}
          .fraunhofer-dashboard .dashboard-card {{
            border-radius: 10px; background: #ffffff;
            box-shadow: 0 2px 8px rgba(16,24,40,.08);
            box-sizing: border-box;
          }}
          .fraunhofer-dashboard .dashboard-card-title {{
            color: #344054; font-size: 12px; font-weight: 700;
            letter-spacing: .08em; margin-bottom: 7px;
          }}
          .fraunhofer-dashboard .dashboard-title {{
            color: #101828; font-size: 23px; font-weight: 700;
            padding: 8px 4px;
          }}
          .fraunhofer-dashboard .dashboard-title span {{
            display: block; color: #667085; font-size: 12px;
            font-weight: 400;
          }}
          .fraunhofer-dashboard .status-badge {{
            color: white; border-radius: 6px; padding: 8px 12px;
            font-weight: 600; margin: 4px 6px 8px 6px;
          }}
          .fraunhofer-dashboard .spectrum-track {{
            position: relative; height: 16px; border-radius: 4px;
            margin: 7px 0; border: 1px solid #98a2b3;
          }}
          .fraunhofer-dashboard .spectrum-marker {{
            position: absolute; top: -4px; width: 3px; height: 24px;
            background: white; border: 1px solid #101828;
            box-shadow: 0 0 0 1px white;
          }}
          .fraunhofer-dashboard .spectrum-labels {{
            display: flex; justify-content: space-between; color: #667085;
            font-size: 10px; margin-top: -3px;
          }}
          .fraunhofer-dashboard .dashboard-preview-output .jp-OutputArea-child,
          .fraunhofer-dashboard .dashboard-preview-output .output_area {{
            display: flex; justify-content: center; width: 100%;
          }}
          .fraunhofer-dashboard .dashboard-preview-output img,
          .fraunhofer-dashboard .dashboard-preview-output canvas {{
            display: block; margin-left: auto; margin-right: auto;
          }}
          .fraunhofer-dashboard .widget-label {{ color: #344054; }}
          .fraunhofer-dashboard .custom-expression-help {{
            color: #475467; font-size: 11px; line-height: 1.35;
          }}
          @media (max-width: 900px) {{
            .fraunhofer-dashboard .dashboard-main-grid {{
              grid-template-columns: 1fr !important;
            }}
            .fraunhofer-dashboard .dashboard-card {{
              width: 100% !important;
            }}
          }}
        </style>
        """

    def _wire_events(self) -> None:
        self.case.observe(self._case_changed, names="value")
        self.custom_mode.observe(self._custom_mode_changed, names="value")
        for control in (
            self.wavelength,
            self.refractive_index,
            self.distance,
            self.slit_orientation,
            self.slit_width,
            self.slit_length,
            self.rectangle_width,
            self.rectangle_height,
            self.circle_radius,
            self.screen_half_width,
            self.zoom,
            self.resolution,
            self.max_fresnel,
            self.secondary_boost,
            self.custom_width,
            self.custom_height,
            self.custom_mask_resolution,
            self.custom_threshold,
            self.custom_fft_padding,
            self.custom_expression,
            self.custom_upload,
            self.custom_invert,
        ):
            control.observe(self._request_update, names="value")
        for toggle in self.extra_view_buttons.values():
            toggle.observe(self._request_update, names="value")
        self.auto_update.observe(self._auto_update_changed, names="value")
        self.add_aperture_button.on_click(self._add_button_clicked)
        self.refresh_button.on_click(self.render)
        self.reset_button.on_click(self.reset)

    def _add_aperture_row(
        self,
        kind: str = "slit",
        center_mm: float | None = None,
        render: bool = True,
    ) -> None:
        center_value = (
            config.CENTER_MM.default if center_mm is None else center_mm
        )
        shape = widgets.Dropdown(
            options=config.SHAPE_OPTIONS, value=kind, description="Shape"
        )
        orientation = widgets.Dropdown(
            options=config.SLIT_ORIENTATION_OPTIONS,
            value="vertical",
            description="orientation",
        )
        width = self._bounded_float(
            config.APERTURE_WIDTH_UM, "width (um)"
        )
        height = self._bounded_float(
            config.APERTURE_HEIGHT_UM, "height/length (um)"
        )
        radius = self._bounded_float(
            config.CIRCLE_RADIUS_UM, "radius (um)"
        )
        center_x = self._bounded_float(
            config.CENTER_MM, "x center (mm)", center_value
        )
        center_y = self._bounded_float(
            config.CENTER_MM, "y center (mm)"
        )
        remove = widgets.Button(description="Remove", icon="trash")
        dimensions = widgets.HBox()
        row_box = widgets.VBox()
        row: dict[str, widgets.Widget] = {
            "shape": shape,
            "orientation": orientation,
            "width": width,
            "height": height,
            "radius": radius,
            "center_x": center_x,
            "center_y": center_y,
            "remove": remove,
            "box": row_box,
        }

        def refresh_dimensions(change=None) -> None:
            if shape.value == "circle":
                dimensions.children = (radius,)
            elif shape.value == "slit":
                dimensions.children = (width, height, orientation)
            else:
                dimensions.children = (width, height)
            row_box.children = (
                widgets.HBox([shape, remove]),
                dimensions,
                widgets.HBox([center_x, center_y]),
            )

        def remove_row(button=None) -> None:
            if row in self.aperture_rows:
                self.aperture_rows.remove(row)
                self._refresh_aperture_rows()
                self._request_update()

        shape.observe(refresh_dimensions, names="value")
        for control in (
            shape,
            orientation,
            width,
            height,
            radius,
            center_x,
            center_y,
        ):
            control.observe(self._request_update, names="value")
        remove.on_click(remove_row)
        refresh_dimensions()
        self.aperture_rows.append(row)
        self._refresh_aperture_rows()
        if render:
            self._request_update()

    def _refresh_aperture_rows(self) -> None:
        self.aperture_rows_box.children = tuple(
            row["box"] for row in self.aperture_rows
        )

    def _refresh_case_controls(self) -> None:
        if self.case.value == "slit":
            self.case_controls.children = (
                self.slit_orientation,
                self.slit_width_box,
                self.slit_length_box,
            )
        elif self.case.value == "rectangle":
            self.case_controls.children = (
                self.rectangle_width_box,
                self.rectangle_height_box,
            )
        elif self.case.value == "circle":
            self.case_controls.children = (self.circle_radius_box,)
        elif self.case.value == "multiple":
            self.case_controls.children = (
                self.add_aperture_button,
                self.aperture_rows_box,
            )
        else:
            self._refresh_custom_controls()
            self.case_controls.children = (
                self.custom_mode,
                self.custom_width_box,
                self.custom_height_box,
                self.custom_mask_resolution_box,
                self.custom_fft_padding_box,
                self.custom_mode_controls,
            )

    def _refresh_custom_controls(self) -> None:
        if self.custom_mode.value == "expression":
            self.custom_mode_controls.children = (
                self.custom_expression,
                self.custom_invert,
                self.custom_expression_help,
            )
        else:
            self.custom_mode_controls.children = (
                self.custom_upload,
                self.custom_threshold_box,
                self.custom_invert,
                widgets.HTML(
                    "<small>White pixels are open by default. PNG and JPEG "
                    "images are resized with nearest-neighbour sampling.</small>"
                ),
            )

    def _build_apertures(self) -> list[engine.Aperture]:
        if self.case.value == "slit":
            width = config.APERTURE_WIDTH_UM.to_si(self.slit_width.value)
            length = config.APERTURE_HEIGHT_UM.to_si(
                self.slit_length.value
            )
            if self.slit_orientation.value == "horizontal":
                width, length = length, width
            return [engine.Aperture.slit(width, length)]
        if self.case.value == "rectangle":
            return [
                engine.Aperture.rectangle(
                    config.APERTURE_WIDTH_UM.to_si(
                        self.rectangle_width.value
                    ),
                    config.APERTURE_HEIGHT_UM.to_si(
                        self.rectangle_height.value
                    ),
                )
            ]
        if self.case.value == "circle":
            return [
                engine.Aperture.circle(
                    config.CIRCLE_RADIUS_UM.to_si(
                        self.circle_radius.value
                    )
                )
            ]
        if self.case.value == "custom":
            raise ValueError(
                "Custom apertures use the numerical Fourier backend."
            )

        apertures: list[engine.Aperture] = []
        for row in self.aperture_rows:
            center_x = config.CENTER_MM.to_si(row["center_x"].value)
            center_y = config.CENTER_MM.to_si(row["center_y"].value)
            if row["shape"].value == "circle":
                apertures.append(
                    engine.Aperture.circle(
                        config.CIRCLE_RADIUS_UM.to_si(row["radius"].value),
                        center_x,
                        center_y,
                    )
                )
                continue
            width = config.APERTURE_WIDTH_UM.to_si(row["width"].value)
            height = config.APERTURE_HEIGHT_UM.to_si(row["height"].value)
            if (
                row["shape"].value == "slit"
                and row["orientation"].value == "horizontal"
            ):
                width, height = height, width
            constructor = (
                engine.Aperture.slit
                if row["shape"].value == "slit"
                else engine.Aperture.rectangle
            )
            apertures.append(
                constructor(width, height, center_x, center_y)
            )
        return apertures

    @staticmethod
    def _uploaded_file_bytes(upload: widgets.FileUpload) -> bytes | None:
        value = upload.value
        if not value:
            return None
        if isinstance(value, dict):
            item = next(iter(value.values()))
        else:
            item = value[0]
        content = item.get("content") if isinstance(item, dict) else None
        if content is None:
            return None
        if isinstance(content, memoryview):
            return content.tobytes()
        return bytes(content)

    def _build_custom_aperture(self) -> fourier.CustomAperture:
        width = config.CUSTOM_WIDTH_MM.to_si(self.custom_width.value)
        height = config.CUSTOM_HEIGHT_MM.to_si(self.custom_height.value)
        resolution = int(self.custom_mask_resolution.value)
        if self.custom_mode.value == "expression":
            return fourier.CustomAperture.from_expression(
                self.custom_expression.value,
                width,
                height,
                resolution,
                invert=self.custom_invert.value,
            )
        image_data = self._uploaded_file_bytes(self.custom_upload)
        if image_data is None:
            raise ValueError("Upload a binary or grayscale aperture image.")
        return fourier.CustomAperture.from_image_bytes(
            image_data,
            width,
            height,
            resolution,
            threshold=self.custom_threshold.value,
            invert=self.custom_invert.value,
        )

    def _first_minima(
        self,
        apertures: list[engine.Aperture],
        wavelength_vacuum: float,
        distance: float,
        n: float,
    ) -> tuple[float, float] | None:
        if self.case.value in ("slit", "rectangle"):
            aperture = apertures[0]
            return engine.rectangle_first_minima(
                wavelength_vacuum,
                distance,
                aperture.width,
                aperture.height,
                n,
            )
        if self.case.value == "circle":
            radius = engine.circular_first_minimum(
                wavelength_vacuum,
                distance,
                apertures[0].radius,
                n,
            )
            return radius, radius
        return None

    def _update_spectrum_indicator(self) -> None:
        marker = visuals.wavelength_to_spectrum_position(
            self.wavelength.value
        )
        self.spectrum_indicator.value = f"""
        <div class="spectrum-readout">
          <div><b>Vacuum wavelength:</b>
            {self.wavelength.value:.2f} nm
          </div>
          <div class="spectrum-track"
               style="background:{config.WAVELENGTH_GRADIENT_CSS};">
            <span class="spectrum-marker"
                  style="left:{marker:.2f}%;"></span>
          </div>
          <div class="spectrum-labels">
            <span>{config.VISIBLE_WAVELENGTH_MIN_NM:g} nm</span>
            <span>{config.VISIBLE_WAVELENGTH_MAX_NM:g}+ nm</span>
          </div>
        </div>
        """

    def _set_status(
        self,
        report: engine.FarFieldReport | None = None,
        message: str | None = None,
    ) -> None:
        if message is not None:
            color = config.STATUS_NEUTRAL_COLOR
            text = message
        elif report is not None and report.is_valid:
            color = config.STATUS_PASS_COLOR
            text = (
                f"FAR FIELD READY &nbsp; "
                f"N<sub>F</sub>={report.fresnel_number:.3g} "
                f"&le; {report.max_fresnel_number:.3g} &nbsp; | &nbsp; "
                f"&lambda;<sub>medium</sub>="
                f"{report.wavelength_medium / config.WAVELENGTH_NM.display_to_si:.2f} nm"
            )
        elif report is not None:
            color = config.STATUS_FAIL_COLOR
            text = (
                f"FRAUNHOFER BLOCKED &nbsp; "
                f"N<sub>F</sub>={report.fresnel_number:.3g} "
                f"&gt; {report.max_fresnel_number:.3g} &nbsp; | &nbsp; "
                f"use z &ge; {report.required_distance:.3g} m"
            )
        else:
            color = config.STATUS_NEUTRAL_COLOR
            text = "Waiting for input"
        self.validity_status.value = (
            f"<div class='status-badge' style='background:{color};'>"
            f"{text}</div>"
        )

    @staticmethod
    def _clear_with_message(
        output: widgets.Output, message: str
    ) -> None:
        with output:
            output.clear_output(wait=True)
            print(message)

    def _draw_profiles(
        self,
        result: screen.SimulationResult,
        apertures: list[engine.Aperture],
        wavelength_vacuum: float,
        distance: float,
        n: float,
    ) -> None:
        profiles = screen.extract_central_profiles(
            result.X, result.Y, result.intensity
        )
        millimetres = config.SCREEN_HALF_WIDTH_MM.display_to_si
        figure, axis = plt.subplots(
            figsize=config.DASHBOARD_PROFILE_FIGURE_SIZE
        )
        axis.plot(
            profiles.x / millimetres,
            profiles.horizontal,
            color=config.PROFILE_COLORS[0],
            label="horizontal (y'=0)",
        )
        axis.plot(
            profiles.y / millimetres,
            profiles.vertical,
            color=config.PROFILE_COLORS[1],
            label="vertical (x'=0)",
        )
        minima = self._first_minima(
            apertures, wavelength_vacuum, distance, n
        )
        if minima is not None:
            for position in (-minima[0], minima[0]):
                axis.axvline(
                    position / millimetres,
                    color=config.PROFILE_COLORS[0],
                    linestyle="--",
                    alpha=0.45,
                )
            for position in (-minima[1], minima[1]):
                axis.axvline(
                    position / millimetres,
                    color=config.PROFILE_COLORS[1],
                    linestyle=":",
                    alpha=0.45,
                )
        axis.set(
            title="Central intensity profiles",
            xlabel="observation-plane coordinate (mm)",
            ylabel="normalized intensity",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plt.show()

    def render(self, button=None) -> None:
        """Refresh schematic, aperture, camera, status, and optional views."""
        if self._rendering:
            return
        self._rendering = True
        self._update_spectrum_indicator()
        try:
            wavelength_vacuum = config.WAVELENGTH_NM.to_si(
                self.wavelength.value
            )
            distance = float(self.distance.value)
            n = float(self.refractive_index.value)
            custom_aperture = None
            if self.case.value == "custom":
                apertures = []
                custom_aperture = self._build_custom_aperture()
                report = engine.evaluate_far_field_radius(
                    custom_aperture.support_radius,
                    wavelength_vacuum,
                    distance,
                    n=n,
                    max_fresnel_number=self.max_fresnel.value,
                )
            else:
                apertures = self._build_apertures()
                if not apertures:
                    raise ValueError(
                        "Add at least one opening in Multiple mode."
                    )
                report = engine.evaluate_far_field(
                    apertures,
                    wavelength_vacuum,
                    distance,
                    n=n,
                    max_fresnel_number=self.max_fresnel.value,
                )
            self._set_status(report=report)

            with self.schematic_output:
                self.schematic_output.clear_output(wait=True)
                figure, axis = plt.subplots(
                    figsize=config.SCHEMATIC_FIGURE_SIZE
                )
                visuals.draw_optical_schematic(
                    axis,
                    apertures if custom_aperture is None else None,
                    distance,
                    self.wavelength.value,
                    report,
                    opening_count=1 if custom_aperture is not None else None,
                )
                figure.tight_layout(pad=0.4)
                plt.show()

            with self.aperture_preview_output:
                self.aperture_preview_output.clear_output(wait=True)
                figure, axis = plt.subplots(
                    figsize=config.DASHBOARD_PREVIEW_FIGURE_SIZE,
                    facecolor=config.DASHBOARD_BACKGROUND,
                )
                if custom_aperture is None:
                    visuals.draw_aperture_preview(axis, apertures)
                else:
                    visuals.draw_custom_aperture_preview(
                        axis, custom_aperture
                    )
                figure.tight_layout(pad=0.5)
                plt.show()

            result = None
            with self.diffraction_preview_output:
                self.diffraction_preview_output.clear_output(wait=True)
                figure, axis = plt.subplots(
                    figsize=config.DASHBOARD_PREVIEW_FIGURE_SIZE,
                    facecolor=config.DASHBOARD_BACKGROUND,
                )
                if report.is_valid:
                    if custom_aperture is None:
                        result = screen.simulate_pattern(
                            apertures,
                            wavelength_vacuum,
                            distance,
                            n=n,
                            screen_half_width=(
                                config.SCREEN_HALF_WIDTH_MM.to_si(
                                    self.screen_half_width.value
                                )
                            ),
                            resolution=self.resolution.value,
                            zoom=self.zoom.value,
                            max_fresnel_number=self.max_fresnel.value,
                        )
                    else:
                        result = fourier.simulate_pattern(
                            custom_aperture,
                            wavelength_vacuum,
                            distance,
                            n=n,
                            screen_half_width=(
                                config.SCREEN_HALF_WIDTH_MM.to_si(
                                    self.screen_half_width.value
                                )
                            ),
                            resolution=self.resolution.value,
                            zoom=self.zoom.value,
                            padding_factor=self.custom_fft_padding.value,
                            max_fresnel_number=self.max_fresnel.value,
                        )
                    visuals.draw_diffraction_preview(
                        axis,
                        result.X,
                        result.Y,
                        result.intensity,
                        self.wavelength.value,
                        gamma=1.0 / self.secondary_boost.value,
                    )
                else:
                    visuals.draw_far_field_unavailable(axis, report)
                figure.tight_layout(pad=0.5)
                plt.show()

            enabled = {
                key
                for key, toggle in self.extra_view_buttons.items()
                if toggle.value
            }
            self.extras_card.layout.display = "" if enabled else "none"
            with self.extra_output:
                self.extra_output.clear_output(wait=True)
                if "profiles" in enabled:
                    if result is None:
                        print(
                            "Profiles are unavailable until the "
                            "far-field check passes."
                        )
                    else:
                        self._draw_profiles(
                            result,
                            apertures,
                            wavelength_vacuum,
                            distance,
                            n,
                        )
        except (TypeError, ValueError) as error:
            self._set_status(message=f"Input error: {error}")
            self._clear_with_message(self.schematic_output, str(error))
            self._clear_with_message(
                self.aperture_preview_output, str(error)
            )
            self._clear_with_message(
                self.diffraction_preview_output, str(error)
            )
        finally:
            self._rendering = False

    def reset(self, button=None) -> None:
        """Restore configured defaults and redraw the dashboard."""
        self._suspend_updates = True
        try:
            self.case.value = "slit"
            self.slit_orientation.value = "vertical"
            self.wavelength.value = config.WAVELENGTH_NM.default
            self.refractive_index.value = config.REFRACTIVE_INDEX.default
            self.distance.value = config.DISTANCE_M.default
            self.slit_width.value = (
                config.DEFAULT_SLIT_WIDTH_M
                / config.APERTURE_WIDTH_UM.display_to_si
            )
            self.slit_length.value = (
                config.DEFAULT_SLIT_LENGTH_M
                / config.APERTURE_HEIGHT_UM.display_to_si
            )
            self.rectangle_width.value = (
                config.DEFAULT_RECTANGLE_WIDTH_M
                / config.APERTURE_WIDTH_UM.display_to_si
            )
            self.rectangle_height.value = (
                config.DEFAULT_RECTANGLE_HEIGHT_M
                / config.APERTURE_HEIGHT_UM.display_to_si
            )
            self.circle_radius.value = (
                config.DEFAULT_CIRCLE_RADIUS_M
                / config.CIRCLE_RADIUS_UM.display_to_si
            )
            self.custom_mode.value = "expression"
            self.custom_width.value = config.CUSTOM_WIDTH_MM.default
            self.custom_height.value = config.CUSTOM_HEIGHT_MM.default
            self.custom_mask_resolution.value = int(
                config.CUSTOM_MASK_RESOLUTION.default
            )
            self.custom_threshold.value = config.CUSTOM_IMAGE_THRESHOLD.default
            self.custom_fft_padding.value = config.CUSTOM_FFT_PADDING.default
            self.custom_expression.value = config.DEFAULT_CUSTOM_EXPRESSION
            self.custom_invert.value = False
            self.custom_upload.value = ()
            self.screen_half_width.value = (
                config.SCREEN_HALF_WIDTH_MM.default
            )
            self.zoom.value = config.ZOOM.default
            self.resolution.value = int(config.RESOLUTION.default)
            self.max_fresnel.value = config.MAX_FRESNEL_NUMBER.default
            self.secondary_boost.value = (
                config.SECONDARY_MAXIMA_BOOST.default
            )
            for toggle in self.extra_view_buttons.values():
                toggle.value = False

            self.aperture_rows.clear()
            self._refresh_aperture_rows()
            for center in config.DEFAULT_MULTIPLE_CENTERS_MM:
                self._add_aperture_row(
                    "slit", center, render=False
                )
            self._refresh_case_controls()
        finally:
            self._suspend_updates = False
        self.render()

    def show(self) -> None:
        """Display and initially render the dashboard in a notebook."""
        display(self.widget)
        self.render()

    def _request_update(self, change=None) -> None:
        if self._suspend_updates or not self.auto_update.value:
            return
        self.render()

    def _case_changed(self, change=None) -> None:
        self._refresh_case_controls()
        self._request_update()

    def _custom_mode_changed(self, change=None) -> None:
        self._refresh_custom_controls()
        self._request_update()

    def _auto_update_changed(self, change) -> None:
        if change["new"]:
            self.render()

    def _add_button_clicked(self, button=None) -> None:
        self._add_aperture_row()
