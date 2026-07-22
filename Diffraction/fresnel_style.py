"""Friendly ipywidgets dashboard for slit and edge Fresnel diffraction."""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import dashboard_tools as shared_visuals
import diffraction_config as shared_config
import fresnel_config as config
import fresnel_dashboard_tools as visuals
import fresnel_engine as engine
import fresnel_screen_tools as screen

__all__ = ["FresnelDashboard"]


class FresnelDashboard:
    """Self-contained Fresnel slit/edge notebook dashboard."""

    def __init__(self) -> None:
        self._rendering = False
        self._suspend_updates = False
        self._create_controls()
        self._create_outputs()
        self._create_layout()
        self._wire_events()
        self._refresh_dynamic_controls()

    @staticmethod
    def _linked_float(
        spec: shared_config.NumericControl,
        description: str,
    ) -> tuple[widgets.FloatSlider, widgets.HBox]:
        slider = widgets.FloatSlider(
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=spec.default,
            description=description,
            readout=False,
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=shared_config.DASHBOARD_SLIDER_WIDTH),
        )
        text = widgets.BoundedFloatText(
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=spec.default,
            layout=widgets.Layout(width=shared_config.DASHBOARD_TEXT_WIDTH),
        )
        widgets.link((slider, "value"), (text, "value"))
        return slider, widgets.HBox(
            [slider, text],
            layout=widgets.Layout(width="100%", flex_flow="row wrap"),
        )

    @staticmethod
    def _linked_int(
        spec: shared_config.NumericControl,
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
            layout=widgets.Layout(width=shared_config.DASHBOARD_SLIDER_WIDTH),
        )
        text = widgets.BoundedIntText(
            min=int(spec.minimum),
            max=int(spec.maximum),
            step=int(spec.step),
            value=int(spec.default),
            layout=widgets.Layout(width=shared_config.DASHBOARD_TEXT_WIDTH),
        )
        widgets.link((slider, "value"), (text, "value"))
        return slider, widgets.HBox(
            [slider, text],
            layout=widgets.Layout(width="100%", flex_flow="row wrap"),
        )

    @staticmethod
    def _linked_log_float(
        spec: shared_config.NumericControl,
        description: str,
    ) -> tuple[widgets.FloatLogSlider, widgets.HBox]:
        log_base = np.log(config.NEAR_FIELD_LOG_BASE)
        slider = widgets.FloatLogSlider(
            base=config.NEAR_FIELD_LOG_BASE,
            min=float(np.log(spec.minimum) / log_base),
            max=float(np.log(spec.maximum) / log_base),
            step=config.NEAR_FIELD_LOG_STEP,
            value=spec.default,
            description=description,
            readout=False,
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=shared_config.DASHBOARD_SLIDER_WIDTH),
        )
        text = widgets.BoundedFloatText(
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=spec.default,
            layout=widgets.Layout(width=shared_config.DASHBOARD_TEXT_WIDTH),
        )
        widgets.link((slider, "value"), (text, "value"))
        return slider, widgets.HBox(
            [slider, text],
            layout=widgets.Layout(width="100%", flex_flow="row wrap"),
        )

    def _create_controls(self) -> None:
        self.case = widgets.ToggleButtons(
            options=config.CASE_OPTIONS,
            value=config.DEFAULT_CASE,
            description="Obstacle",
            icons=("pause", "align-left"),
            tooltips=("Single slit", "Opaque straight edge"),
            style={"description_width": "initial"},
        )
        self.illumination = widgets.ToggleButtons(
            options=config.ILLUMINATION_OPTIONS,
            value=config.DEFAULT_ILLUMINATION,
            description="Illumination",
            icons=("arrows-h", "circle"),
            style={"description_width": "initial"},
        )
        self.orientation = widgets.ToggleButtons(
            options=config.ORIENTATION_OPTIONS,
            value=config.DEFAULT_ORIENTATION,
            description="Orientation",
            style={"description_width": "initial"},
        )
        self.shadow_side = widgets.ToggleButtons(
            options=config.SHADOW_SIDE_OPTIONS,
            value=config.DEFAULT_SHADOW_SIDE,
            description="Shadow",
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
        self.extra_view_buttons = {
            key: widgets.ToggleButton(value=False, description=label)
            for label, key in config.EXTRA_VIEW_OPTIONS
        }

        self.wavelength, self.wavelength_box = self._linked_float(
            shared_config.WAVELENGTH_NM, "lambda_0 (nm)"
        )
        self.refractive_index, self.refractive_index_box = self._linked_float(
            shared_config.REFRACTIVE_INDEX, "n"
        )
        self.source_distance, self.source_distance_box = self._linked_float(
            config.SOURCE_DISTANCE_M, "source distance d (m)"
        )
        self.distance, self.distance_box = self._linked_float(
            config.DISTANCE_M, "screen distance D (m)"
        )
        self.slit_width, self.slit_width_box = self._linked_float(
            config.SLIT_WIDTH_MM, "slit width (mm)"
        )
        self.screen_half_width, self.screen_half_width_box = self._linked_float(
            config.SCREEN_HALF_WIDTH_MM, "screen half-width (mm)"
        )
        self.zoom, self.zoom_box = self._linked_float(config.ZOOM, "zoom")
        self.resolution, self.resolution_box = self._linked_int(
            config.RESOLUTION, "resolution"
        )
        self.secondary_boost, self.secondary_boost_box = self._linked_float(
            config.SECONDARY_MAXIMA_BOOST, "secondary maxima boost"
        )
        self.near_field_limit, self.near_field_limit_box = (
            self._linked_log_float(
                config.NEAR_FIELD_LIMIT, "near-field N_F,min"
            )
        )
        self.cornu_probe, self.cornu_probe_box = self._linked_float(
            config.CORNU_PROBE_MM, "Cornu probe (mm)"
        )

        self.source_controls = widgets.VBox(
            layout=widgets.Layout(width="100%")
        )
        self.case_controls = widgets.VBox(
            layout=widgets.Layout(width="100%")
        )

    def _create_outputs(self) -> None:
        self.schematic_output = widgets.Output(
            layout=widgets.Layout(width="100%", min_height="210px")
        )
        centered = widgets.Layout(
            width="100%",
            min_height="390px",
            display="flex",
            justify_content="center",
            align_items="center",
        )
        self.aperture_preview_output = widgets.Output(layout=centered)
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
        self.extra_output = widgets.Output(layout=widgets.Layout(width="100%"))
        self.validity_status = widgets.HTML()
        self.spectrum_indicator = widgets.HTML()

    @staticmethod
    def _card(
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
                border=shared_config.DASHBOARD_CARD_BORDER,
                padding=shared_config.DASHBOARD_CARD_PADDING,
                margin=shared_config.DASHBOARD_CARD_MARGIN,
            ),
        )
        card.add_class("dashboard-card")
        return card

    def _create_layout(self) -> None:
        source_section = self._card(
            "SOURCE",
            [self.spectrum_indicator, self.source_controls],
        )
        aperture_section = self._card(
            "APERTURE / OBSTACLE", [self.case, self.case_controls]
        )
        propagation_section = self._card(
            "PROPAGATION", [self.distance_box]
        )
        extra_selector = widgets.HBox(
            tuple(self.extra_view_buttons.values()),
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        advanced = widgets.Accordion(
            children=[
                widgets.VBox(
                    [self.resolution_box, self.near_field_limit_box]
                )
            ]
        )
        advanced.set_title(0, "Advanced sampling")
        display_section = self._card(
            "DISPLAY",
            [
                self.screen_half_width_box,
                self.zoom_box,
                self.secondary_boost_box,
                advanced,
                widgets.HTML("<b>Additional views</b>"),
                extra_selector,
                self.cornu_probe_box,
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
            width=shared_config.DASHBOARD_CONTROL_WIDTH,
        )
        aperture_panel = self._card(
            "SELECTED OPENING / EDGE", [self.aperture_preview_output]
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
            "FRESNEL DETAILS", [self.extra_output]
        )
        self.extras_card.layout.display = "none"

        header = widgets.HBox(
            [
                widgets.HTML(
                    "<div class='dashboard-title'>"
                    "Fresnel Diffraction Workbench"
                    "<span>Slit and straight-edge near-field simulator"
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
                grid_template_columns=shared_config.DASHBOARD_MAIN_COLUMNS,
                grid_gap=shared_config.DASHBOARD_COLUMN_GAP,
                align_items="flex-start",
            ),
        )
        main_area.add_class("dashboard-main-grid")
        self.widget = widgets.VBox(
            [
                widgets.HTML(self._style_html()),
                header,
                self.validity_status,
                schematic_panel,
                main_area,
                self.extras_card,
            ],
            layout=widgets.Layout(width="100%"),
        )
        self.widget.add_class("fresnel-dashboard")

    @staticmethod
    def _style_html() -> str:
        return f"""
        <style>
          .fresnel-dashboard {{
            max-width: {shared_config.DASHBOARD_MAX_WIDTH}; margin: 0 auto;
          }}
          .fresnel-dashboard .dashboard-card {{
            border-radius: 10px; background: #ffffff;
            box-shadow: 0 2px 8px rgba(16,24,40,.08);
            box-sizing: border-box;
          }}
          .fresnel-dashboard .dashboard-card-title {{
            color: #344054; font-size: 12px; font-weight: 700;
            letter-spacing: .08em; margin-bottom: 7px;
          }}
          .fresnel-dashboard .dashboard-title {{
            color: #101828; font-size: 23px; font-weight: 700;
            padding: 8px 4px;
          }}
          .fresnel-dashboard .dashboard-title span {{
            display: block; color: #667085; font-size: 12px;
            font-weight: 400;
          }}
          .fresnel-dashboard .status-badge {{
            color: white; border-radius: 6px; padding: 8px 12px;
            font-weight: 600; margin: 4px 6px 8px 6px;
          }}
          .fresnel-dashboard .spectrum-track {{
            position: relative; height: 16px; border-radius: 4px;
            margin: 7px 0; border: 1px solid #98a2b3;
          }}
          .fresnel-dashboard .spectrum-marker {{
            position: absolute; top: -4px; width: 3px; height: 24px;
            background: white; border: 1px solid #101828;
            box-shadow: 0 0 0 1px white;
          }}
          .fresnel-dashboard .spectrum-labels {{
            display: flex; justify-content: space-between; color: #667085;
            font-size: 10px; margin-top: -3px;
          }}
          .fresnel-dashboard .dashboard-preview-output .jp-OutputArea-child,
          .fresnel-dashboard .dashboard-preview-output .output_area {{
            display: flex; justify-content: center; width: 100%;
          }}
          @media (max-width: 900px) {{
            .fresnel-dashboard .dashboard-main-grid {{
              grid-template-columns: 1fr !important;
            }}
            .fresnel-dashboard .dashboard-card {{
              width: 100% !important;
            }}
          }}
        </style>
        """

    def _wire_events(self) -> None:
        self.case.observe(self._dynamic_control_changed, names="value")
        self.illumination.observe(
            self._dynamic_control_changed, names="value"
        )
        for control in (
            self.wavelength,
            self.refractive_index,
            self.source_distance,
            self.distance,
            self.slit_width,
            self.orientation,
            self.shadow_side,
            self.screen_half_width,
            self.zoom,
            self.resolution,
            self.secondary_boost,
            self.near_field_limit,
            self.cornu_probe,
        ):
            control.observe(self._request_update, names="value")
        for key, toggle in self.extra_view_buttons.items():
            toggle.observe(self._extra_view_changed, names="value")
        self.auto_update.observe(self._auto_update_changed, names="value")
        self.refresh_button.on_click(self.render)
        self.reset_button.on_click(self.reset)

    def _refresh_dynamic_controls(self) -> None:
        source_children: list[widgets.Widget] = [
            self.wavelength_box,
            self.refractive_index_box,
            self.illumination,
        ]
        if self.illumination.value == "point":
            source_children.append(self.source_distance_box)
        self.source_controls.children = tuple(source_children)

        if self.case.value == "slit":
            self.case_controls.children = (
                self.orientation,
                self.slit_width_box,
            )
        else:
            self.case_controls.children = (
                self.orientation,
                self.shadow_side,
            )
        self.cornu_probe_box.layout.display = (
            "" if self.extra_view_buttons["cornu"].value else "none"
        )
        edge_selected = self.case.value == "edge"
        for control in self.near_field_limit_box.children:
            control.disabled = edge_selected

    def _update_spectrum_indicator(self) -> None:
        marker = shared_visuals.wavelength_to_spectrum_position(
            self.wavelength.value
        )
        self.spectrum_indicator.value = f"""
        <div>
          <div><b>Vacuum wavelength:</b> {self.wavelength.value:.2f} nm</div>
          <div class="spectrum-track"
               style="background:{shared_config.WAVELENGTH_GRADIENT_CSS};">
            <span class="spectrum-marker"
                  style="left:{marker:.2f}%;"></span>
          </div>
          <div class="spectrum-labels">
            <span>{shared_config.VISIBLE_WAVELENGTH_MIN_NM:g} nm</span>
            <span>{shared_config.VISIBLE_WAVELENGTH_MAX_NM:g}+ nm</span>
          </div>
        </div>
        """

    def _set_status(
        self,
        report: engine.FresnelReport | None = None,
        message: str | None = None,
    ) -> None:
        if message is not None:
            color = shared_config.STATUS_FAIL_COLOR
            text = message
        elif report is None:
            color = shared_config.STATUS_NEUTRAL_COLOR
            text = "Waiting for input"
        elif report.fresnel_number is None:
            color = shared_config.STATUS_PASS_COLOR
            text = (
                "FRESNEL EDGE READY &nbsp; | &nbsp; "
                "N<sub>F</sub> criterion: N/A &nbsp; | &nbsp; "
                f"characteristic length="
                f"{report.characteristic_length * 1e3:.3g} mm"
            )
        else:
            condition_passed = bool(report.is_near_field)
            color = (
                shared_config.STATUS_PASS_COLOR
                if condition_passed
                else shared_config.STATUS_NEUTRAL_COLOR
            )
            prefix = (
                "SELECTED NEAR-FIELD CONDITION: PASS"
                if condition_passed
                else "SELECTED NEAR-FIELD CONDITION: NOT MET"
            )
            text = (
                f"{prefix} &nbsp; | &nbsp; "
                f"N<sub>F</sub>={report.fresnel_number:.3g}, "
                f"N<sub>F,min</sub>={report.near_field_limit:.3g} "
                f"&nbsp; | &nbsp; {report.regime} &nbsp; | &nbsp; "
                f"D<sub>eff</sub>={report.effective_distance:.3g} m"
            )
        self.validity_status.value = (
            f"<div class='status-badge' style='background:{color};'>"
            f"{text}</div>"
        )

    def _simulation_parameters(self) -> dict[str, object]:
        return {
            "case": self.case.value,
            "wavelength_vacuum": shared_config.WAVELENGTH_NM.to_si(
                self.wavelength.value
            ),
            "distance": float(self.distance.value),
            "slit_width": config.SLIT_WIDTH_MM.to_si(self.slit_width.value),
            "n": float(self.refractive_index.value),
            "illumination": self.illumination.value,
            "source_distance": (
                float(self.source_distance.value)
                if self.illumination.value == "point"
                else None
            ),
            "orientation": self.orientation.value,
            "shadow_side": self.shadow_side.value,
            "screen_half_width": config.SCREEN_HALF_WIDTH_MM.to_si(
                self.screen_half_width.value
            ),
            "resolution": int(self.resolution.value),
            "zoom": float(self.zoom.value),
            "near_field_limit": float(self.near_field_limit.value),
        }

    def _draw_profiles(
        self,
        result: screen.FresnelSimulationResult,
    ) -> None:
        profiles = screen.extract_central_profiles(result)
        millimetres = 1e-3
        figure, axis = plt.subplots(
            figsize=shared_config.DASHBOARD_PROFILE_FIGURE_SIZE
        )
        if self.orientation.value == "vertical":
            axis.plot(
                profiles.x / millimetres,
                profiles.horizontal,
                color=shared_config.PROFILE_COLORS[0],
                label="horizontal profile",
            )
        else:
            axis.plot(
                profiles.y / millimetres,
                profiles.vertical,
                color=shared_config.PROFILE_COLORS[1],
                label="vertical profile",
            )
        axis.axvline(0.0, color="#667085", linestyle="--", alpha=0.5)
        axis.set(
            title="Fresnel intensity profile",
            xlabel="observation-plane coordinate (mm)",
            ylabel="I / I0",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plt.show()

    @staticmethod
    def _clear_with_message(output: widgets.Output, message: str) -> None:
        with output:
            output.clear_output(wait=True)
            print(message)

    def render(self, button=None) -> None:
        """Refresh schematic, aperture, camera, status, and optional views."""
        if self._rendering:
            return
        self._rendering = True
        self._update_spectrum_indicator()
        try:
            parameters = self._simulation_parameters()
            result = screen.simulate_pattern(**parameters)
            report = result.report
            self._set_status(report=report)

            with self.schematic_output:
                self.schematic_output.clear_output(wait=True)
                figure, axis = plt.subplots(
                    figsize=shared_config.SCHEMATIC_FIGURE_SIZE
                )
                visuals.draw_optical_schematic(
                    axis,
                    self.case.value,
                    self.illumination.value,
                    float(self.distance.value),
                    parameters["source_distance"],
                    float(self.wavelength.value),
                    report,
                )
                figure.tight_layout(pad=0.4)
                plt.show()

            with self.aperture_preview_output:
                self.aperture_preview_output.clear_output(wait=True)
                figure, axis = plt.subplots(
                    figsize=shared_config.DASHBOARD_PREVIEW_FIGURE_SIZE,
                    facecolor=shared_config.DASHBOARD_BACKGROUND,
                )
                visuals.draw_aperture_preview(
                    axis,
                    self.case.value,
                    slit_width=parameters["slit_width"],
                    orientation=self.orientation.value,
                    shadow_side=self.shadow_side.value,
                )
                figure.tight_layout(pad=0.5)
                plt.show()

            with self.diffraction_preview_output:
                self.diffraction_preview_output.clear_output(wait=True)
                figure, axis = plt.subplots(
                    figsize=shared_config.DASHBOARD_PREVIEW_FIGURE_SIZE,
                    facecolor=shared_config.DASHBOARD_BACKGROUND,
                )
                visuals.draw_fresnel_preview(
                    axis,
                    result.X,
                    result.Y,
                    result.intensity,
                    float(self.wavelength.value),
                    gamma=1.0 / float(self.secondary_boost.value),
                )
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
                    self._draw_profiles(result)
                if "cornu" in enabled:
                    figure, axis = plt.subplots(figsize=(6.0, 5.0))
                    visuals.draw_cornu_construction(
                        axis,
                        self.case.value,
                        config.CORNU_PROBE_MM.to_si(self.cornu_probe.value),
                        parameters["wavelength_vacuum"],
                        parameters["distance"],
                        slit_width=parameters["slit_width"],
                        n=parameters["n"],
                        illumination=parameters["illumination"],
                        source_distance=parameters["source_distance"],
                        shadow_side=parameters["shadow_side"],
                    )
                    figure.tight_layout()
                    plt.show()
        except (TypeError, ValueError) as error:
            self._set_status(message=f"Input error: {error}")
            self._clear_with_message(self.schematic_output, str(error))
            self._clear_with_message(self.aperture_preview_output, str(error))
            self._clear_with_message(
                self.diffraction_preview_output, str(error)
            )
        finally:
            self._rendering = False

    def reset(self, button=None) -> None:
        """Restore configured defaults and redraw the dashboard."""
        self._suspend_updates = True
        try:
            self.case.value = config.DEFAULT_CASE
            self.illumination.value = config.DEFAULT_ILLUMINATION
            self.orientation.value = config.DEFAULT_ORIENTATION
            self.shadow_side.value = config.DEFAULT_SHADOW_SIDE
            self.wavelength.value = shared_config.WAVELENGTH_NM.default
            self.refractive_index.value = (
                shared_config.REFRACTIVE_INDEX.default
            )
            self.source_distance.value = config.SOURCE_DISTANCE_M.default
            self.distance.value = config.DISTANCE_M.default
            self.slit_width.value = config.SLIT_WIDTH_MM.default
            self.screen_half_width.value = config.SCREEN_HALF_WIDTH_MM.default
            self.zoom.value = config.ZOOM.default
            self.resolution.value = int(config.RESOLUTION.default)
            self.secondary_boost.value = (
                config.SECONDARY_MAXIMA_BOOST.default
            )
            self.near_field_limit.value = config.NEAR_FIELD_LIMIT.default
            self.cornu_probe.value = config.CORNU_PROBE_MM.default
            for toggle in self.extra_view_buttons.values():
                toggle.value = False
            self._refresh_dynamic_controls()
        finally:
            self._suspend_updates = False
        self.render()

    def show(self) -> None:
        """Display and initially render the Fresnel dashboard."""
        display(self.widget)
        self.render()

    def _request_update(self, change=None) -> None:
        if self._suspend_updates or not self.auto_update.value:
            return
        self.render()

    def _dynamic_control_changed(self, change=None) -> None:
        self._refresh_dynamic_controls()
        self._request_update()

    def _extra_view_changed(self, change=None) -> None:
        self._refresh_dynamic_controls()
        self._request_update()

    def _auto_update_changed(self, change) -> None:
        if change["new"]:
            self.render()
