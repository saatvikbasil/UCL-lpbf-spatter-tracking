"""Manuscript figures for the LPBF spatter study.

Reads the per-track metrics that spatter_analysis.py wrote for each dataset
(through run_folder_pipeline.py), joins them with the process parameters in the
experiment logbook, applies the track filters configured below and saves the
four figures used in the manuscript, as PNG and PDF, to comprehensive_analysis/:

It also saves the CW vs PWM statistics at 400 W and 400 mm/s as a CSV. The data
behind each figure are written to comprehensive_analysis/figure_data/ by
ExportMixin (figure_data_export.py).

Run with:

    python new_visuals.py
"""

import os
import warnings
from pathlib import Path as PathlibPath

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from figure_data_export import ExportMixin

warnings.filterwarnings("ignore")

# Dataset prefixes to include, or "all"
DATASET_SELECTION = "0111,0112,0113,0114,0115,0116,0117"

# Counts are reported per mm of track (4.4 mm field of view along the track)
TRACK_LENGTH = 4.4  # mm

# Unstable-keyhole threshold on each enthalpy scale, shaded on the enthalpy plots
UNSTABLE_KEYHOLE_THRESHOLDS = {
    "normalized_enthalpy_threshold": 11.5,
    "product_threshold": 11.5,
    "enable_threshold_shading": True,
    "threshold_color": "red",
    "threshold_alpha": 0.15,
}

# Normalised enthalpy on the x-axis of the enthalpy plots:
#   "plain"   -> dH/h_s          (Hann et al. 2011; King et al. 2014)
#   "product" -> dH/h_m * L*_th  (Ye et al. 2019) = A*P / (sqrt(pi) * h_m * v * r0^2)
ENTHALPY_SCALE = "product"

if ENTHALPY_SCALE == "product":
    ENTHALPY_COL = "NORMALIZED_ENTHALPY_PRODUCT"
    ENTHALPY_LABEL = "Normalised enthalpy product (ΔH/h$_m$·L*$_{th}$)"
    ENTHALPY_NAME = "Normalised enthalpy product"
    ENTHALPY_THRESHOLD = UNSTABLE_KEYHOLE_THRESHOLDS["product_threshold"]
    ENTHALPY_SUFFIX = "_product"
    ENTHALPY_AXIS_STEP = 1.0
else:
    ENTHALPY_COL = "NORMALIZED_ENTHALPY"
    ENTHALPY_LABEL = "Normalised enthalpy (ΔH/h$_s$)"
    ENTHALPY_NAME = "Normalised enthalpy"
    ENTHALPY_THRESHOLD = UNSTABLE_KEYHOLE_THRESHOLDS["normalized_enthalpy_threshold"]
    ENTHALPY_SUFFIX = ""
    ENTHALPY_AXIS_STEP = 2.0

# Track filters (None disables a bound)
CIRCULARITY_FILTER = {"min": 0.2, "max": None}  # 0 = irregular, 1 = perfect circle
SIZE_FILTER = {"min": 5, "max": None}  # mean radius, microns
VELOCITY_FILTER = {"min": None, "max": 5000}  # microns/ms
DURATION_FILTER = {"min": 3, "max": None}  # applied to TRACK_DURATION_MS
ANGLE_FILTER = {"min": -85, "max": 85}  # degrees from vertical
STRAIGHTNESS_FILTER = {"min": 0.5, "max": None}  # 0 = very curved, 1 = straight
TOTAL_PATH_LENGTH_FILTER = {"min": 200, "max": None}  # microns

# Vertical direction from START_Y - END_Y (positive = moving up the image)
Y_DIRECTION_FILTER = {
    "upward_only": True,
    "downward_only": False,
    "min_y_displacement": None,  # absolute, microns
    "max_y_displacement": None,
}

# Batch output of run_folder_pipeline.py and the experiment logbook
BASE_PATH = "Results/FinalUpdatedBatch"
EXCEL_FILE_PATH = "CP1 datalogbook.xlsx"

MATERIAL_PROPERTIES = {
    "density": 2730,  # kg/m^3
    "specific_heat": 900,  # J/(kg K)
    "boiling_point": 2743,  # K
    "melting_point": 933,  # K
    "room_temperature": 298,  # K
    "beam_radius": 40e-6,  # m, r0 in the normalised enthalpy
    "spot_size_sigma": 40e-6,  # m, Gaussian sigma in the recoil-pressure model
    "absorptivity": 0.18,
    "latent_heat_vaporization": 291000,  # J/mol
    "enthalpy_of_melting": 397000,  # J/kg
    "gas_constant": 8.314,  # J/(mol K)
    "atmospheric_pressure": 101325,  # Pa
    "thermal_diffusivity": 3.3e-5,  # m^2/s
    "thermal_conductivity": 89.3,  # W/(m K)
}

# add_best_fit_line only draws fits with R^2 at or above this value
R_SQUARED_THRESHOLD = 0.9

FIGURE_SIZE = (18, 12)
DPI = 600
FONT_SIZE = 22
LINE_WIDTH = 2.5
MARKER_SIZE = 22
ERROR_BAR_CAPSIZE = 6

# Line colours for the scan-speed groups in the velocity and trajectory plots
DATASET_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

LASER_MODE_COLORS = {"PWM": "#1f77b4", "CW": "#ff7f0e"}

MELTING_MODE_MARKERS = {
    "unstable keyhole": "o",
    "quasi-stable keyhole": "^",
    "stable keyhole": "^",
    "keyhole flickering": "s",
    "conduction": "v",
    "vapour depression": "D",
    "discontinuous melting": "D",
    "low porosity": "h",
    "keyhole porosity": "<",
}

MELTING_MODE_ORDER = [
    "conduction",
    "vapour depression",
    "keyhole flickering",
    "quasi-stable keyhole",
    "unstable keyhole",
]


def format_decimal(x, pos):
    """Tick formatter: scientific below 0.01, integers from 10 up, else 2 decimals."""
    if abs(x) < 1e-2 and x != 0:
        return f"{x:.2e}"
    elif abs(x) >= 10:
        return f"{x:.0f}"
    else:
        return f"{x:.2f}"


plt.style.use("default")
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FONT_SIZE,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE,
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.5,
        "xtick.minor.width": 1.0,
        "ytick.major.width": 1.5,
        "ytick.minor.width": 1.0,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.shadow": False,
        "legend.framealpha": 0.9,
        "legend.fontsize": FONT_SIZE - 2,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

print(
    f"ENTHALPY SCALE: {ENTHALPY_SCALE}  ->  column '{ENTHALPY_COL}', "
    f"threshold {ENTHALPY_THRESHOLD}"
)


class ComprehensiveLPBFAnalyzer(ExportMixin):
    """Load per-dataset spatter metrics, apply the track filters and draw the figures."""

    def __init__(self):
        self.all_data = {}
        self.laser_params = {}
        self.filters = {}
        self.output_dir = "comprehensive_analysis"

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory created/verified: {os.path.abspath(self.output_dir)}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            self.output_dir = "."

    def get_parameter_color(self, param_value, unique_params):
        """Return the palette colour for the position of ``param_value`` in ``unique_params``."""
        if param_value in unique_params:
            index = list(unique_params).index(param_value)
            return DATASET_COLORS[index % len(DATASET_COLORS)]
        return DATASET_COLORS[0]

    def calculate_recoil_pressure(self, power_W, scan_speed_mm_s):
        """Recoil pressure at the peak surface temperature, in atmospheres.

        The peak temperature is estimated analytically from the absorbed
        Gaussian intensity (sigma = spot_size_sigma) and the scan speed. The
        vapour pressure follows the Clausius-Clapeyron relation (exponent
        clipped to +/-50) and the recoil pressure is 0.56 times the vapour
        pressure.
        """
        sigma = MATERIAL_PROPERTIES["spot_size_sigma"]
        intensity = power_W / (2 * np.pi * sigma**2)
        diffusivity = MATERIAL_PROPERTIES["thermal_diffusivity"]
        conductivity = MATERIAL_PROPERTIES["thermal_conductivity"]
        absorptivity = MATERIAL_PROPERTIES["absorptivity"]

        term1 = (np.sqrt(2) * absorptivity * intensity * sigma) / (conductivity * np.sqrt(np.pi))
        # The factor 1000 converts the scan speed from mm/s to m/s.
        term2_inside = np.sqrt((1000 * 2 * diffusivity) / (scan_speed_mm_s * sigma))
        term2 = np.arctan(term2_inside)
        peak_temperature = term1 * term2

        P0 = MATERIAL_PROPERTIES["atmospheric_pressure"]
        delta_Hv = MATERIAL_PROPERTIES["latent_heat_vaporization"]
        Tb = MATERIAL_PROPERTIES["boiling_point"]
        R = MATERIAL_PROPERTIES["gas_constant"]

        exponent = delta_Hv * (peak_temperature - Tb) / (R * peak_temperature * Tb)
        exponent = np.clip(exponent, -50, 50)

        vapor_pressure = P0 * np.exp(exponent)

        recoil_pressure = 0.56 * vapor_pressure
        return recoil_pressure / MATERIAL_PROPERTIES["atmospheric_pressure"]

    def _enthalpy_of_melting_volumetric(self):
        """h_m [J/m^3] = rho * (C * (T_m - T_0) + h_sf). Shared by both scales."""
        mp = MATERIAL_PROPERTIES
        return mp["density"] * (
            mp["specific_heat"] * (mp["melting_point"] - mp["room_temperature"])
            + mp["enthalpy_of_melting"]
        )

    def calculate_normalized_enthalpy(self, power_W, scan_speed_mm_s):
        """Plain normalised enthalpy (Hann 2011 / King 2014):

        dH/h_s = A*P / (h_m * sqrt(pi * D * v * r0^3))      ~ P * v^-1/2
        """
        mp = MATERIAL_PROPERTIES
        v = scan_speed_mm_s / 1000.0
        r0 = mp["beam_radius"]
        D = mp["thermal_diffusivity"]
        h_m = self._enthalpy_of_melting_volumetric()
        return (mp["absorptivity"] * power_W) / (h_m * np.sqrt(np.pi * D * v * r0**3))

    def calculate_normalized_thermal_diffusion_length(self, scan_speed_mm_s):
        """Normalised thermal diffusion length (Ye et al. 2019):

        L*_th = delta / r0 = 1 / sqrt(Pe) = sqrt(D / (v * r0))
        """
        mp = MATERIAL_PROPERTIES
        v = scan_speed_mm_s / 1000.0
        return np.sqrt(mp["thermal_diffusivity"] / (v * mp["beam_radius"]))

    def calculate_normalized_enthalpy_product(self, power_W, scan_speed_mm_s):
        """Normalised enthalpy product, dH/h_m * L*_th (Ye et al. 2019,
        Adv. Eng. Mater. 21, 1900185; used by Huang et al. Nat Commun 2022):

            beta * L*_th = A*P / (sqrt(pi) * h_m * v * r0^2)          ~ P * v^-1

        The thermal diffusivity cancels exactly, so this is a rescaled linear
        energy density, not an independent parameter. Equivalent identity:

            product = (dH/h_s) * L*_th
        """
        mp = MATERIAL_PROPERTIES
        v = scan_speed_mm_s / 1000.0
        h_m = self._enthalpy_of_melting_volumetric()
        return (mp["absorptivity"] * power_W) / (np.sqrt(np.pi) * h_m * v * mp["beam_radius"] ** 2)

    def setup_filters(self):
        """Register the active filters from the configuration block and print them."""
        print("\n=== DATA FILTERING SETUP ===")

        print(
            f"PER-MM NORMALISATION ENABLED: All counts divided by TRACK_LENGTH = {TRACK_LENGTH} mm"
        )
        print(f"ENTHALPY SCALE: {ENTHALPY_SCALE} (plotting column '{ENTHALPY_COL}')")
        if ENTHALPY_THRESHOLD is None:
            print("  No unstable-keyhole threshold defined for this scale - shading disabled.")
        else:
            print(f"  Unstable-keyhole threshold: {ENTHALPY_THRESHOLD}")

        if CIRCULARITY_FILTER["min"] is not None or CIRCULARITY_FILTER["max"] is not None:
            self.filters["circularity"] = CIRCULARITY_FILTER
            print(f"Circularity filter: {CIRCULARITY_FILTER['min']} - {CIRCULARITY_FILTER['max']}")

        if SIZE_FILTER["min"] is not None or SIZE_FILTER["max"] is not None:
            self.filters["size"] = SIZE_FILTER
            print(f"Size filter: {SIZE_FILTER['min']} - {SIZE_FILTER['max']} μm")

        if VELOCITY_FILTER["min"] is not None or VELOCITY_FILTER["max"] is not None:
            self.filters["velocity"] = VELOCITY_FILTER
            print(f"Velocity filter: {VELOCITY_FILTER['min']} - {VELOCITY_FILTER['max']} μm/ms")

        if DURATION_FILTER["min"] is not None or DURATION_FILTER["max"] is not None:
            self.filters["duration"] = DURATION_FILTER
            print(f"Duration filter: {DURATION_FILTER['min']} - {DURATION_FILTER['max']} ms")

        if ANGLE_FILTER["min"] is not None or ANGLE_FILTER["max"] is not None:
            self.filters["angle"] = ANGLE_FILTER
            print(f"Angle filter: {ANGLE_FILTER['min']} - {ANGLE_FILTER['max']}°")

        if STRAIGHTNESS_FILTER["min"] is not None or STRAIGHTNESS_FILTER["max"] is not None:
            self.filters["straightness"] = STRAIGHTNESS_FILTER
            print(
                f"Straightness filter: {STRAIGHTNESS_FILTER['min']} - {STRAIGHTNESS_FILTER['max']}"
            )

        if (
            TOTAL_PATH_LENGTH_FILTER["min"] is not None
            or TOTAL_PATH_LENGTH_FILTER["max"] is not None
        ):
            self.filters["total_path_length"] = TOTAL_PATH_LENGTH_FILTER
            print(
                f"Total path length filter: {TOTAL_PATH_LENGTH_FILTER['min']} - {TOTAL_PATH_LENGTH_FILTER['max']} μm"
            )

        if (
            Y_DIRECTION_FILTER["upward_only"]
            or Y_DIRECTION_FILTER["downward_only"]
            or Y_DIRECTION_FILTER["min_y_displacement"] is not None
            or Y_DIRECTION_FILTER["max_y_displacement"] is not None
        ):
            self.filters["y_direction"] = Y_DIRECTION_FILTER
            direction_str = []
            if Y_DIRECTION_FILTER["upward_only"]:
                direction_str.append("upward only")
            if Y_DIRECTION_FILTER["downward_only"]:
                direction_str.append("downward only")
            if Y_DIRECTION_FILTER["min_y_displacement"] is not None:
                direction_str.append(
                    f"min displacement: {Y_DIRECTION_FILTER['min_y_displacement']} μm"
                )
            if Y_DIRECTION_FILTER["max_y_displacement"] is not None:
                direction_str.append(
                    f"max displacement: {Y_DIRECTION_FILTER['max_y_displacement']} μm"
                )
            print(f"Y-direction filter: {', '.join(direction_str)}")

    def load_laser_parameters(self):
        """Read each dataset's process parameters from the logbook.

        Stores power, scan speed, melting mode, laser mode and linear energy
        density per dataset. Laser mode is CW when the point-jump delay is zero
        or missing, and PWM otherwise. Returns False if the logbook cannot be
        read or lacks the required columns.
        """
        try:
            df = pd.read_excel(EXCEL_FILE_PATH)
            print(f"Loaded laser parameters from {EXCEL_FILE_PATH}")
            print(f"Available columns: {list(df.columns)}")

            required_cols = ["DATASET", "Averaged power", "SCAN_SPEED"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Warning: Missing columns in Excel file: {missing_cols}")
                return False

            melting_mode_col = None
            possible_names = [
                "MELTING_MODE",
                "Melting mode",
                "Melting Mode",
                "MELTING MODE",
                "melting_mode",
            ]
            for col_name in possible_names:
                if col_name in df.columns:
                    melting_mode_col = col_name
                    break

            if melting_mode_col is None:
                print("Warning: No melting mode column found. Will use default.")
            else:
                print(f"Found melting mode column: {melting_mode_col}")

            point_jump_delay_col = None
            possible_delay_names = [
                "POINT_JUMP_DELAY",
                "Point jump delay",
                "Point Jump Delay",
                "POINT JUMP DELAY",
                "point_jump_delay",
            ]
            for col_name in possible_delay_names:
                if col_name in df.columns:
                    point_jump_delay_col = col_name
                    break

            if point_jump_delay_col is None:
                print("Warning: No point jump delay column found. Will assume all CW.")
            else:
                print(f"Found point jump delay column: {point_jump_delay_col}")

            linear_energy_density_col = None
            possible_led_names = [
                "Linear energy density",
                "LINEAR_ENERGY_DENSITY",
                "Linear Energy Density",
                "LINEAR ENERGY DENSITY",
                "linear_energy_density",
            ]
            for col_name in possible_led_names:
                if col_name in df.columns:
                    linear_energy_density_col = col_name
                    break

            if linear_energy_density_col is None:
                print("Warning: No linear energy density column found.")
            else:
                print(f"Found linear energy density column: {linear_energy_density_col}")

            for _, row in df.iterrows():
                dataset_name = str(row["DATASET"])

                if melting_mode_col and pd.notna(row[melting_mode_col]):
                    melting_mode = str(row[melting_mode_col]).lower().strip()
                else:
                    melting_mode = "unknown"

                if point_jump_delay_col and pd.notna(row[point_jump_delay_col]):
                    point_jump_delay = float(row[point_jump_delay_col])
                    laser_mode = "CW" if point_jump_delay == 0 else "PWM"
                else:
                    point_jump_delay = 0
                    laser_mode = "CW"

                if linear_energy_density_col and pd.notna(row[linear_energy_density_col]):
                    linear_energy_density = float(row[linear_energy_density_col])
                else:
                    linear_energy_density = None

                self.laser_params[dataset_name] = {
                    "power": row["Averaged power"],
                    "scan_speed": row["SCAN_SPEED"],
                    "melting_mode": melting_mode,
                    "point_jump_delay": point_jump_delay,
                    "laser_mode": laser_mode,
                    "linear_energy_density": linear_energy_density,
                }

            print(f"Loaded parameters for {len(self.laser_params)} datasets")

            melting_modes = [params["melting_mode"] for params in self.laser_params.values()]
            unique_modes = sorted(list(set(melting_modes)))
            print(f"Unique melting modes found: {unique_modes}")

            laser_modes = [params["laser_mode"] for params in self.laser_params.values()]
            unique_laser_modes = sorted(list(set(laser_modes)))
            print(f"Unique laser modes found: {unique_laser_modes}")

            led_count = sum(
                1
                for params in self.laser_params.values()
                if params["linear_energy_density"] is not None
            )
            print(
                f"Linear energy density available for {led_count}/{len(self.laser_params)} datasets"
            )

            return True

        except Exception as e:
            print(f"Error loading laser parameters: {e}")
            return False

    def find_dataset_folders(self):
        """Map each dataset folder under BASE_PATH to its spatter_metrics.csv."""
        base_path = PathlibPath(BASE_PATH)

        if not base_path.exists():
            print(f"Error: Directory {base_path} does not exist")
            return {}

        dataset_folders = {}
        for folder in base_path.iterdir():
            if folder.is_dir() and folder.name != "logger":
                metrics_file = folder / "spatter_metrics.csv"
                analysis_metrics_file = folder / "analysis" / "spatter_metrics.csv"

                if metrics_file.exists():
                    dataset_folders[folder.name] = metrics_file
                elif analysis_metrics_file.exists():
                    dataset_folders[folder.name] = analysis_metrics_file

        print(f"Found {len(dataset_folders)} datasets with spatter_metrics.csv")
        return dataset_folders

    def get_dataset_selection(self, available_datasets):
        """Return the datasets matching DATASET_SELECTION (all of them if none match)."""
        selection = DATASET_SELECTION.strip()

        if selection.lower() == "all":
            return list(available_datasets.keys())

        prefixes = [prefix.strip() for prefix in selection.split(",")]
        selected_datasets = []

        for prefix in prefixes:
            matching = [name for name in available_datasets.keys() if name.startswith(prefix)]
            selected_datasets.extend(matching)

        selected_datasets = list(dict.fromkeys(selected_datasets))

        if not selected_datasets:
            print("No matching datasets found. Processing all datasets.")
            return list(available_datasets.keys())

        print(f"Selected datasets: {selected_datasets}")
        return selected_datasets

    def apply_filters(self, df):
        """Apply the configured track filters and return the retained rows."""
        filtered_df = df.copy()
        original_count = len(filtered_df)

        if "circularity" in self.filters:
            min_circ, max_circ = (
                self.filters["circularity"]["min"],
                self.filters["circularity"]["max"],
            )
            if min_circ is not None:
                filtered_df = filtered_df[filtered_df["AVG_CIRCULARITY"] >= min_circ]
            if max_circ is not None:
                filtered_df = filtered_df[filtered_df["AVG_CIRCULARITY"] <= max_circ]

        if "size" in self.filters:
            min_size, max_size = (
                self.filters["size"]["min"],
                self.filters["size"]["max"],
            )
            if min_size is not None:
                filtered_df = filtered_df[filtered_df["AVG_RADIUS"] >= min_size]
            if max_size is not None:
                filtered_df = filtered_df[filtered_df["AVG_RADIUS"] <= max_size]

        if "velocity" in self.filters:
            min_vel, max_vel = (
                self.filters["velocity"]["min"],
                self.filters["velocity"]["max"],
            )
            if min_vel is not None:
                filtered_df = filtered_df[filtered_df["AVG_VELOCITY"] >= min_vel]
            if max_vel is not None:
                filtered_df = filtered_df[filtered_df["AVG_VELOCITY"] <= max_vel]

        if "duration" in self.filters:
            min_dur, max_dur = (
                self.filters["duration"]["min"],
                self.filters["duration"]["max"],
            )
            if min_dur is not None:
                filtered_df = filtered_df[filtered_df["TRACK_DURATION_MS"] >= min_dur]
            if max_dur is not None:
                filtered_df = filtered_df[filtered_df["TRACK_DURATION_MS"] <= max_dur]

        if "angle" in self.filters:
            min_angle, max_angle = (
                self.filters["angle"]["min"],
                self.filters["angle"]["max"],
            )
            if min_angle is not None:
                filtered_df = filtered_df[filtered_df["TRAJECTORY_ANGLE"] >= min_angle]
            if max_angle is not None:
                filtered_df = filtered_df[filtered_df["TRAJECTORY_ANGLE"] <= max_angle]

        if "straightness" in self.filters:
            min_straight, max_straight = (
                self.filters["straightness"]["min"],
                self.filters["straightness"]["max"],
            )
            if min_straight is not None:
                filtered_df = filtered_df[filtered_df["STRAIGHTNESS"] >= min_straight]
            if max_straight is not None:
                filtered_df = filtered_df[filtered_df["STRAIGHTNESS"] <= max_straight]

        if "total_path_length" in self.filters:
            min_path, max_path = (
                self.filters["total_path_length"]["min"],
                self.filters["total_path_length"]["max"],
            )
            if min_path is not None:
                filtered_df = filtered_df[filtered_df["TOTAL_PATH_LENGTH"] >= min_path]
            if max_path is not None:
                filtered_df = filtered_df[filtered_df["TOTAL_PATH_LENGTH"] <= max_path]

        if "y_direction" in self.filters:
            y_filter = self.filters["y_direction"]
            required_y_cols = ["START_Y", "END_Y"]
            missing_y_cols = [col for col in required_y_cols if col not in filtered_df.columns]

            if missing_y_cols:
                print(
                    f"Warning: Y-direction filter cannot be applied. Missing columns: {missing_y_cols}"
                )
            else:
                filtered_df = filtered_df.copy()
                filtered_df["Y_DISPLACEMENT"] = filtered_df["START_Y"] - filtered_df["END_Y"]

                if y_filter["upward_only"]:
                    before_upward = len(filtered_df)
                    filtered_df = filtered_df[filtered_df["Y_DISPLACEMENT"] > 0]
                    after_upward = len(filtered_df)
                    print(f"  Upward only filter: {before_upward} -> {after_upward} tracks")

                if y_filter["downward_only"]:
                    before_downward = len(filtered_df)
                    filtered_df = filtered_df[filtered_df["Y_DISPLACEMENT"] < 0]
                    after_downward = len(filtered_df)
                    print(f"  Downward only filter: {before_downward} -> {after_downward} tracks")

                if y_filter["min_y_displacement"] is not None:
                    before_min = len(filtered_df)
                    filtered_df = filtered_df[
                        np.abs(filtered_df["Y_DISPLACEMENT"]) >= y_filter["min_y_displacement"]
                    ]
                    after_min = len(filtered_df)
                    print(f"  Min Y displacement filter: {before_min} -> {after_min} tracks")

                if y_filter["max_y_displacement"] is not None:
                    before_max = len(filtered_df)
                    filtered_df = filtered_df[
                        np.abs(filtered_df["Y_DISPLACEMENT"]) <= y_filter["max_y_displacement"]
                    ]
                    after_max = len(filtered_df)
                    print(f"  Max Y displacement filter: {before_max} -> {after_max} tracks")

        filtered_count = len(filtered_df)
        print(
            f"Filters applied: {original_count} -> {filtered_count} tracks "
            f"({filtered_count/original_count*100:.1f}% retained)"
        )

        return filtered_df

    def load_metrics_files(self, selected_datasets, dataset_folders):
        """Load, filter and annotate the metrics of each selected dataset.

        Adds the process parameters from the logbook, converts AVG_VELOCITY
        from microns/ms to m/s, and computes the recoil pressure, both
        normalised-enthalpy scales and the particle diameter.
        """
        for dataset_name in selected_datasets:
            if dataset_name not in self.laser_params:
                print(f"Warning: No laser parameters found for {dataset_name}")
                continue

            if dataset_name in dataset_folders:
                file_path = dataset_folders[dataset_name]
                try:
                    df = pd.read_csv(file_path)

                    required_columns = [
                        "AVG_VELOCITY",
                        "TRAJECTORY_ANGLE",
                        "AVG_RADIUS",
                    ]
                    missing_columns = [col for col in required_columns if col not in df.columns]

                    if missing_columns:
                        print(f"Warning: {dataset_name} missing columns: {missing_columns}")
                        continue

                    params = self.laser_params[dataset_name]
                    df_filtered = self.apply_filters(df)

                    if len(df_filtered) == 0:
                        print(f"Warning: All tracks filtered out for {dataset_name}")
                        continue

                    df_filtered = df_filtered.copy()
                    df_filtered["DATASET_NAME"] = dataset_name
                    df_filtered["Averaged power"] = params["power"]
                    df_filtered["SCAN_SPEED"] = params["scan_speed"]
                    df_filtered["MELTING_MODE"] = params["melting_mode"]
                    df_filtered["POINT_JUMP_DELAY"] = params["point_jump_delay"]
                    df_filtered["LASER_MODE"] = params["laser_mode"]
                    df_filtered["LINEAR_ENERGY_DENSITY"] = params["linear_energy_density"]

                    df_filtered["AVG_VELOCITY"] = df_filtered["AVG_VELOCITY"] / 1000

                    df_filtered["RECOIL_PRESSURE"] = df_filtered.apply(
                        lambda row: self.calculate_recoil_pressure(
                            row["Averaged power"], row["SCAN_SPEED"]
                        ),
                        axis=1,
                    )

                    df_filtered["NORMALIZED_ENTHALPY"] = df_filtered.apply(
                        lambda row: self.calculate_normalized_enthalpy(
                            row["Averaged power"], row["SCAN_SPEED"]
                        ),
                        axis=1,
                    )
                    df_filtered["NORM_THERMAL_DIFF_LENGTH"] = df_filtered.apply(
                        lambda row: self.calculate_normalized_thermal_diffusion_length(
                            row["SCAN_SPEED"]
                        ),
                        axis=1,
                    )
                    df_filtered["NORMALIZED_ENTHALPY_PRODUCT"] = df_filtered.apply(
                        lambda row: self.calculate_normalized_enthalpy_product(
                            row["Averaged power"], row["SCAN_SPEED"]
                        ),
                        axis=1,
                    )

                    # Consistency check: product == (dH/h_s) * L*_th
                    _lhs = df_filtered["NORMALIZED_ENTHALPY_PRODUCT"]
                    _rhs = (
                        df_filtered["NORMALIZED_ENTHALPY"] * df_filtered["NORM_THERMAL_DIFF_LENGTH"]
                    )
                    if not np.allclose(_lhs, _rhs, rtol=1e-9):
                        print(f"  WARNING: enthalpy identity violated for {dataset_name}")

                    df_filtered["DIAMETER"] = 2 * df_filtered["AVG_RADIUS"]

                    self.all_data[dataset_name] = df_filtered
                    led_str = (
                        f"LED: {params['linear_energy_density']}"
                        if params["linear_energy_density"] is not None
                        else "LED: N/A"
                    )
                    print(
                        f"Loaded {dataset_name}: {len(df_filtered)} tracks "
                        f"(Power: {params['power']}W, Speed: {params['scan_speed']}mm/s, "
                        f"Mode: {params['melting_mode']}, Laser: {params['laser_mode']}, {led_str}, "
                        f"dH/h_s: {df_filtered['NORMALIZED_ENTHALPY'].iloc[0]:.2f}, "
                        f"L*_th: {df_filtered['NORM_THERMAL_DIFF_LENGTH'].iloc[0]:.3f}, "
                        f"product: {df_filtered['NORMALIZED_ENTHALPY_PRODUCT'].iloc[0]:.2f})"
                    )

                except Exception as e:
                    print(f"Error loading {dataset_name}: {e}")

    def add_best_fit_line(self, ax, x_data, y_data, min_r_squared=R_SQUARED_THRESHOLD):
        """Draw a log-log power-law fit if its R^2 reaches ``min_r_squared``.

        If the power-law fit raises, a linear fit is tried instead.
        """
        if len(x_data) < 3:
            return

        mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
        x_clean = np.array(x_data)[mask]
        y_clean = np.array(y_data)[mask]

        if len(x_clean) < 3:
            return

        try:
            log_x = np.log10(x_clean)
            log_y = np.log10(y_clean)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            a_power = 10**intercept
            b_power = slope
            r2_power = r_value**2

            if r2_power >= min_r_squared:
                x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_smooth = a_power * np.power(x_smooth, b_power)

                ax.plot(
                    x_smooth,
                    y_smooth,
                    "k--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Power law fit (R² = {r2_power:.2f})",
                )

                equation = f"y = {a_power:.2e} × x^{b_power:.2f}"

                if abs(b_power - 1.0) < 0.1:
                    scaling_text = "(Linear scaling)"
                elif b_power > 1.2:
                    scaling_text = "(Superlinear scaling)"
                elif b_power < 0.8:
                    scaling_text = "(Sublinear scaling)"
                else:
                    scaling_text = "(Near-linear scaling)"

                text_x = 0.05 if b_power > 0 else 0.95
                text_ha = "left" if b_power > 0 else "right"

                full_text = f"{equation}\n{scaling_text}"
                ax.text(
                    text_x,
                    0.95,
                    full_text,
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE - 2,
                    ha=text_ha,
                    va="top",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="lightblue",
                        alpha=0.8,
                        edgecolor="darkblue",
                    ),
                )

                print(
                    f"    Power law fit: y = {a_power:.2e} × x^{b_power:.2f}, R² = {r2_power:.2f}"
                )

        except Exception as e:
            print(f"    Power law fitting failed, falling back to linear: {e}")
            try:
                X = x_clean.reshape(-1, 1)
                reg = LinearRegression().fit(X, y_clean)
                y_pred = reg.predict(X)
                r2_linear = r2_score(y_clean, y_pred)

                if r2_linear >= min_r_squared:
                    x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_smooth = reg.predict(x_smooth.reshape(-1, 1))

                    ax.plot(
                        x_smooth,
                        y_smooth,
                        "k--",
                        linewidth=2,
                        alpha=0.8,
                        label=f"Linear fit (R² = {r2_linear:.2f})",
                    )

                    slope = reg.coef_[0]
                    intercept = reg.intercept_
                    equation = f"y = {slope:.2e}x + {intercept:.2e}"

                    text_x = 0.05 if slope > 0 else 0.95
                    text_ha = "left" if slope > 0 else "right"

                    ax.text(
                        text_x,
                        0.95,
                        equation,
                        transform=ax.transAxes,
                        fontsize=FONT_SIZE - 2,
                        ha=text_ha,
                        va="top",
                        bbox=dict(
                            boxstyle="round",
                            facecolor="lightyellow",
                            alpha=0.8,
                            edgecolor="orange",
                        ),
                    )
            except Exception as e2:
                print(f"    Both power law and linear fitting failed: {e2}")

    def _apply_uniform_enthalpy_xaxis(self, ax, step=None):
        """Give a normalised-enthalpy axis evenly spaced ticks.

        Rounds the x-limits outward to multiples of ``step`` and puts a major
        tick at every step. The default step is ENTHALPY_AXIS_STEP, since the
        product scale spans a narrower range than the plain scale. Returns the
        (x_start, x_end) limits applied.
        """
        if step is None:
            step = ENTHALPY_AXIS_STEP
        x_lo, x_hi = ax.get_xlim()
        x_start = np.floor(x_lo / step) * step
        x_end = np.ceil(x_hi / step) * step
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
        return x_start, x_end

    def _add_unstable_keyhole_label(self, ax, threshold, color, x_offset=0.25, y_frac=0.98):
        """Label the shaded unstable-keyhole region just right of the threshold line.

        x is in data coordinates (threshold + x_offset) and y in axes
        coordinates, so the label sits near the top of the shaded region.
        """
        ax.text(
            threshold + x_offset,
            y_frac,
            "Unstable Keyhole\nRegion",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_SIZE - 4,
            ha="left",
            va="top",
            zorder=6,
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.3, edgecolor=color),
        )

    def create_professional_plot(
        self, df, x_col, y_col, title, xlabel, ylabel, filename, legend_title="Parameters"
    ):
        """Mean of ``y_col`` against ``x_col`` with standard-error bars.

        One line per scan speed and laser mode (dashed for CW); each marker shows
        the most common melting mode at that point.
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        export_rows = []

        group_col = "SCAN_SPEED"
        group_unit = "mm/s"

        group_combinations = df.groupby([group_col, "LASER_MODE"]).first().reset_index()

        if len(group_combinations) == 0:
            print(f"No data for {filename}")
            return

        unique_params = sorted(df[group_col].unique())
        x_data = []
        y_data = []

        for _, combo in group_combinations.iterrows():
            group_value = combo[group_col]
            laser_mode = combo["LASER_MODE"]

            combo_data = df[
                (df[group_col] == group_value) & (df["LASER_MODE"] == laser_mode)
            ].copy()
            if len(combo_data) == 0:
                continue

            grouped = combo_data.groupby([x_col])[y_col].agg(["mean", "std", "count"]).reset_index()
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            grouped = grouped.sort_values(x_col)

            if len(grouped) == 0:
                continue

            line_color = self.get_parameter_color(group_value, unique_params)
            line_style = "--" if laser_mode == "CW" else "-"

            x_vals = grouped[x_col].values
            y_vals = grouped["mean"].values
            y_err = np.nan_to_num(grouped["se"].values, nan=0.0)

            markers = []
            for x_val in x_vals:
                point_data = combo_data[combo_data[x_col] == x_val]
                if len(point_data) > 0:
                    mode_counts = point_data["MELTING_MODE"].value_counts()
                    most_common_mode = mode_counts.index[0]
                    marker = MELTING_MODE_MARKERS.get(most_common_mode, "o")
                    markers.append(marker)
                else:
                    markers.append("o")

            export_rows.extend(
                self.series_point_rows(
                    combo_data,
                    x_col,
                    y_col,
                    grouped,
                    group_col,
                    group_value,
                    group_unit,
                    laser_mode,
                    line_color,
                    line_style,
                )
            )

            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                color=line_color,
                linestyle=line_style,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                capsize=ERROR_BAR_CAPSIZE,
                capthick=1.5,
                alpha=0.8,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

            for x_val, y_val, marker in zip(x_vals, y_vals, markers):
                ax.scatter(
                    x_val,
                    y_val,
                    marker=marker,
                    color=line_color,
                    s=MARKER_SIZE**2,
                    edgecolors="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=5,
                )

            label = f"{group_value:.0f} {group_unit} ({laser_mode})"

            if line_style == "--":
                ax.plot(
                    [],
                    [],
                    color=line_color,
                    linestyle="--",
                    linewidth=LINE_WIDTH * 1.2,
                    dashes=(2, 1),
                    alpha=0.9,
                    label=label,
                )
            else:
                ax.plot(
                    [],
                    [],
                    color=line_color,
                    linestyle="-",
                    linewidth=LINE_WIDTH,
                    alpha=0.9,
                    label=label,
                )

            x_data.extend(x_vals)
            y_data.extend(y_vals)

        if len(x_data) > 2:
            self.add_best_fit_line(ax, np.array(x_data), np.array(y_data))

        ax.xaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax.yaxis.set_major_formatter(FuncFormatter(format_decimal))

        ax.set_xlabel(xlabel, fontweight="bold", fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=FONT_SIZE)
        ax.set_title(title, fontweight="bold", fontsize=FONT_SIZE + 2, pad=20)

        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        legend = ax.legend(
            title=legend_title,
            loc="best",
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=FONT_SIZE - 3,
            title_fontsize=FONT_SIZE - 2,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor("black")

        for legend_line in legend.get_lines():
            if legend_line.get_linestyle() == "--":
                legend_line.set_dashes([2, 1])
                legend_line.set_linewidth(LINE_WIDTH * 1.2)

        self.export_plot_data(
            filename,
            export_rows,
            x_col=x_col,
            y_col=y_col,
            y_label=ylabel,
            errorbar="y_sem",
            sort_by=["series_label", x_col],
        )

        plt.tight_layout()

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            png_path = os.path.join(self.output_dir, f"{filename}.png")
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            fig.savefig(png_path, dpi=DPI, bbox_inches="tight", format="png", facecolor="white")
            fig.savefig(pdf_path, bbox_inches="tight", format="pdf", facecolor="white")
            print(f"Saved: {png_path} and {pdf_path}")
        except Exception as e:
            print(f"Error saving plot {filename}: {e}")

        plt.close()

    def compare_cw_pwm_single_condition(
        self, df, power=400, scan_speed=400, power_tol=10, speed_tol=10, save_csv=True
    ):
        """Compare CW and PWM at one (power, scan speed) condition.

        Velocities are compared per track (Mann-Whitney U, Welch's t-test,
        Cohen's d) and the total counts with a conditional binomial test for
        two Poisson counts. Returns a one-row summary DataFrame, or None if
        either mode has fewer than two tracks.
        """
        mask = (np.abs(df["Averaged power"] - power) <= power_tol) & (
            np.abs(df["SCAN_SPEED"] - scan_speed) <= speed_tol
        )
        sub = df[mask]
        p = sub[sub["LASER_MODE"] == "PWM"]
        c = sub[sub["LASER_MODE"] == "CW"]
        if len(p) < 2 or len(c) < 2:
            print(
                f"Single-condition compare: insufficient data at "
                f"P~{power} W, v~{scan_speed} mm/s (PWM n={len(p)}, CW n={len(c)})."
            )
            return None

        vp = p["AVG_VELOCITY"].dropna().values  # m/s (converted at load)
        vc = c["AVG_VELOCITY"].dropna().values
        s_p, s_c = vp.std(ddof=1), vc.std(ddof=1)
        n_p, n_c = len(vp), len(vc)
        pooled = np.sqrt(((n_p - 1) * s_p**2 + (n_c - 1) * s_c**2) / (n_p + n_c - 2))
        d = (vc.mean() - vp.mean()) / pooled if pooled > 0 else np.nan
        vel_pct = (vc.mean() - vp.mean()) / vp.mean() * 100 if vp.mean() else np.nan

        u_stat, p_mwu = stats.mannwhitneyu(vc, vp, alternative="two-sided")
        t_stat, p_welch = stats.ttest_ind(vc, vp, equal_var=False)

        print(f"\n=== CW vs PWM at P~{power} W, v~{scan_speed} mm/s ===")
        print("VELOCITY (m/s):")
        print(f"  PWM: {vp.mean():.3f} +/- {s_p:.3f} SD  (n={n_p} tracks)")
        print(f"  CW : {vc.mean():.3f} +/- {s_c:.3f} SD  (n={n_c} tracks)")
        print(f"  CW vs PWM: {vel_pct:+.1f}%")
        print(f"  Mann-Whitney U p = {p_mwu:.2e}")
        print(f"  Welch t-test    p = {p_welch:.2e}")
        print(f"  Cohen's d = {d:.2f}")

        # Count: one total per mode, treated as Poisson
        nc_p, nc_c = len(p), len(c)
        permm_p, permm_c = nc_p / TRACK_LENGTH, nc_c / TRACK_LENGTH
        count_pct = (nc_c - nc_p) / nc_p * 100 if nc_p else np.nan
        try:
            p_count = stats.binomtest(nc_c, nc_c + nc_p, 0.5).pvalue
        except AttributeError:
            p_count = stats.binom_test(nc_c, nc_c + nc_p, 0.5)

        print("COUNT:")
        print(f"  PWM: {nc_p} tracks ({permm_p:.1f}/mm), Poisson SD +/-{np.sqrt(nc_p):.1f}")
        print(f"  CW : {nc_c} tracks ({permm_c:.1f}/mm), Poisson SD +/-{np.sqrt(nc_c):.1f}")
        print(f"  CW vs PWM: {count_pct:+.1f}%")
        print(f"  Poisson conditional (binomial) test p = {p_count:.2e}")
        print(
            "  NOTE: count test assumes equal exposure (same number of frames / "
            "observation window) for both datasets. Per-mm normalisation handles "
            "field of view, not temporal exposure - verify frame counts match."
        )

        summary = pd.DataFrame(
            [
                {
                    "power": power,
                    "scan_speed": scan_speed,
                    "vel_PWM_mean": vp.mean(),
                    "vel_PWM_sd": s_p,
                    "n_PWM_tracks": n_p,
                    "vel_CW_mean": vc.mean(),
                    "vel_CW_sd": s_c,
                    "n_CW_tracks": n_c,
                    "vel_pct_CW_vs_PWM": vel_pct,
                    "vel_p_mannwhitney": p_mwu,
                    "vel_p_welch": p_welch,
                    "vel_cohens_d": d,
                    "count_PWM": nc_p,
                    "count_CW": nc_c,
                    "count_pct_CW_vs_PWM": count_pct,
                    "count_p_poisson": p_count,
                }
            ]
        )
        if save_csv:
            try:
                path = os.path.join(self.output_dir, "cw_pwm_single_condition_400W_400mms.csv")
                summary.to_csv(path, index=False)
                print(f"Saved: {path}")
            except Exception as e:
                print(f"Error saving single-condition comparison: {e}")
        return summary

    def create_all_plots(self):
        """Draw the manuscript figures from the loaded datasets."""
        if len(self.all_data) == 0:
            print("No data available for plotting")
            return

        combined_df = pd.concat(list(self.all_data.values()), ignore_index=True)
        print("\n=== CREATING PLOTS ===")
        print(f"Total data points: {len(combined_df)}")
        print(f"Per-mm normalisation: all counts / {TRACK_LENGTH} mm")
        print(f"Enthalpy scale: {ENTHALPY_SCALE} (column '{ENTHALPY_COL}')")

        self._create_velocity_vs_power_plots(combined_df)
        self.compare_cw_pwm_single_condition(combined_df, power=400, scan_speed=400)
        self._create_normalized_enthalpy_vs_count_plots(combined_df)
        self._create_enthalpy_vs_count_bubble_plot(combined_df)
        self._create_trajectory_angle_vs_power_plots(combined_df)

        print("\n=== Analysis Complete ===")
        print(f"All plots saved to: {os.path.abspath(self.output_dir)}/")

    def _create_velocity_vs_power_plots(self, df):
        self.create_professional_plot(
            df,
            "Averaged power",
            "AVG_VELOCITY",
            "Maximum spatter velocity vs laser power",
            "Laser power (W)",
            "Maximum spatter velocity (m/s)",
            "01_velocity_vs_power",
            "Scan Speed",
        )

    def _create_enthalpy_vs_count_bubble_plot(self, df):
        """Spatter count per mm vs normalised enthalpy, one bubble per dataset.

        Bubble area scales with mean diameter, marker shape shows the
        circularity class and colour the laser mode.
        """
        print("Creating enthalpy vs count bubble plot (sized by diameter)...")

        dataset_summary = (
            df.groupby(["DATASET_NAME", "LASER_MODE"])
            .agg(
                {
                    ENTHALPY_COL: "mean",
                    "DIAMETER": "mean",
                    "AVG_CIRCULARITY": "mean",
                }
            )
            .reset_index()
        )

        dataset_counts = df.groupby(["DATASET_NAME"]).size().reset_index(name="COUNT")
        dataset_counts["COUNT"] = dataset_counts["COUNT"] / TRACK_LENGTH
        dataset_summary = dataset_summary.merge(dataset_counts, on="DATASET_NAME")

        if len(dataset_summary) == 0:
            return

        dataset_summary["CIRC_CLASS"] = dataset_summary["AVG_CIRCULARITY"].apply(
            lambda x: "high" if x >= 0.6 else "low"
        )

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        d_min = dataset_summary["DIAMETER"].min()
        d_max = dataset_summary["DIAMETER"].max()
        size_min, size_max = 150, 2000

        def diameter_to_size(d):
            if d_max == d_min:
                return (size_min + size_max) / 2
            return size_min + (d - d_min) / (d_max - d_min) * (size_max - size_min)

        dataset_summary["BUBBLE_SIZE"] = dataset_summary["DIAMETER"].apply(diameter_to_size)

        laser_colors = {"PWM": LASER_MODE_COLORS["PWM"], "CW": LASER_MODE_COLORS["CW"]}
        circ_markers = {"high": "o", "low": "*"}

        for laser_mode in ["PWM", "CW"]:
            for circ_class in ["high", "low"]:
                subset = dataset_summary[
                    (dataset_summary["LASER_MODE"] == laser_mode)
                    & (dataset_summary["CIRC_CLASS"] == circ_class)
                ]
                if len(subset) == 0:
                    continue

                marker = circ_markers[circ_class]
                color = laser_colors[laser_mode]
                edge_lw = 1.2 if marker == "o" else 0.8

                ax.scatter(
                    subset[ENTHALPY_COL],
                    subset["COUNT"],
                    s=subset["BUBBLE_SIZE"],
                    marker=marker,
                    c=color,
                    alpha=0.75,
                    edgecolors="black",
                    linewidths=edge_lw,
                    zorder=4,
                )

        ax.xaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax.yaxis.set_major_formatter(FuncFormatter(format_decimal))
        x_start, x_end = self._apply_uniform_enthalpy_xaxis(ax)

        if (
            UNSTABLE_KEYHOLE_THRESHOLDS["enable_threshold_shading"]
            and ENTHALPY_THRESHOLD is not None
        ):
            threshold = ENTHALPY_THRESHOLD
            th_color = UNSTABLE_KEYHOLE_THRESHOLDS["threshold_color"]
            th_alpha = UNSTABLE_KEYHOLE_THRESHOLDS["threshold_alpha"]
            ax.axvspan(threshold, x_end, color=th_color, alpha=th_alpha, zorder=1)
            ax.axvline(
                x=threshold,
                color=th_color,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                zorder=2,
            )
            self._add_unstable_keyhole_label(ax, threshold, th_color)

        ax.set_xlabel(ENTHALPY_LABEL, fontweight="bold", fontsize=FONT_SIZE)
        ax.set_ylabel("Gross spatter count per mm", fontweight="bold", fontsize=FONT_SIZE)
        ax.set_title(
            f"Spatter count per mm vs {ENTHALPY_NAME.lower()}",
            fontweight="bold",
            fontsize=FONT_SIZE + 2,
            pad=20,
        )
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        colour_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LASER_MODE_COLORS["PWM"],
                markersize=14,
                markeredgecolor="black",
                markeredgewidth=1.2,
                linestyle="None",
                label="PWM",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LASER_MODE_COLORS["CW"],
                markersize=14,
                markeredgecolor="black",
                markeredgewidth=1.2,
                linestyle="None",
                label="CW",
            ),
        ]
        colour_legend = ax.legend(
            handles=colour_handles,
            title="Laser Mode",
            loc="lower right",
            fontsize=FONT_SIZE - 3,
            title_fontsize=FONT_SIZE - 2,
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor="black",
            facecolor="white",
            framealpha=0.95,
        )
        ax.add_artist(colour_legend)

        shape_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=13,
                markeredgecolor="black",
                markeredgewidth=1.2,
                linestyle="None",
                label="Circularity ≥ 0.6",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                markersize=16,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="None",
                label="Circularity < 0.6",
            ),
        ]
        shape_legend = ax.legend(
            handles=shape_handles,
            title="Circularity",
            loc="lower left",
            fontsize=FONT_SIZE - 3,
            title_fontsize=FONT_SIZE - 2,
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor="black",
            facecolor="white",
            framealpha=0.95,
        )
        ax.add_artist(shape_legend)

        d_low = np.ceil(d_min / 5) * 5
        d_high = np.floor(d_max / 5) * 5
        d_mid = np.round((d_low + d_high) / 2 / 5) * 5
        size_legend_diameters = sorted(set([d_low, d_mid, d_high]))

        size_handles = []
        for d in size_legend_diameters:
            s = diameter_to_size(np.clip(d, d_min, d_max))
            handle = ax.scatter(
                [],
                [],
                s=s,
                marker="o",
                c="lightgray",
                edgecolors="black",
                linewidths=1.0,
                alpha=0.9,
            )
            handle.set_label(f"{d:.0f} μm")
            size_handles.append(handle)

        ax.legend(
            handles=size_handles,
            title="Mean Diameter",
            loc="upper right",
            fontsize=FONT_SIZE - 3,
            title_fontsize=FONT_SIZE - 2,
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor="black",
            facecolor="white",
            framealpha=0.95,
            labelspacing=2.0,
            handletextpad=1.8,
            borderpad=1.2,
            scatterpoints=1,
        )

        self.export_plot_data(
            f"35_enthalpy_vs_count_bubble{ENTHALPY_SUFFIX}",
            self.bubble_rows(df, dataset_summary, ENTHALPY_COL),
            x_col="enthalpy_x",
            y_col="gross_spatter_count_per_mm",
            y_label="Gross spatter count per mm",
            sort_by=["laser_mode", "enthalpy_x"],
            extra_meta={
                "bubble area": (
                    f"size_min + (d - d_min)/(d_max - d_min) * (size_max - size_min), "
                    f"with d_min={d_min:.2f} um, d_max={d_max:.2f} um, "
                    f"size_min={size_min}, size_max={size_max} "
                    f"(matplotlib s=, in points squared)"
                )
            },
        )

        plt.tight_layout()

        try:
            filename = f"35_enthalpy_vs_count_bubble{ENTHALPY_SUFFIX}"
            png_path = os.path.join(self.output_dir, f"{filename}.png")
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            fig.savefig(png_path, dpi=DPI, bbox_inches="tight", format="png", facecolor="white")
            fig.savefig(pdf_path, bbox_inches="tight", format="pdf", facecolor="white")
            print(f"Saved: {png_path} and {pdf_path}")
        except Exception as e:
            print(f"Error saving bubble plot: {e}")
        plt.close()

    def _create_trajectory_angle_vs_power_plots(self, df):
        self.create_professional_plot(
            df,
            "Averaged power",
            "TRAJECTORY_ANGLE",
            "Average trajectory angle vs laser power",
            "Laser power (W)",
            "Average trajectory angle (degrees)",
            "21_trajectory_angle_vs_power",
            "Scan Speed",
        )

    def _create_normalized_enthalpy_vs_count_plots(self, df):
        """Mean spatter count per mm vs normalised enthalpy, one line per laser mode.

        Repeat runs (identical power and scan speed) are averaged into one point;
        conditions that only share a P/v ratio stay separate. Markers show the
        melting mode.
        """
        print("Creating normalized enthalpy vs spatter count plot...")

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        laser_modes = df["LASER_MODE"].unique()
        if len(laser_modes) == 0:
            return

        mode_colors = LASER_MODE_COLORS
        mode_styles = {"PWM": "-", "CW": "--"}

        x_data = []
        y_data = []
        export_rows = []

        for laser_mode in sorted(laser_modes):
            mode_data = df[df["LASER_MODE"] == laser_mode]
            if len(mode_data) == 0:
                continue

            group_keys = ["DATASET_NAME", "Averaged power", "SCAN_SPEED", ENTHALPY_COL]

            # One row per dataset
            per_dataset = (
                mode_data.groupby(group_keys)
                .agg(
                    {
                        "MELTING_MODE": lambda x: (
                            x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"
                        )
                    }
                )
                .reset_index()
            )

            counts = mode_data.groupby(group_keys).size().reset_index(name="COUNT")
            counts["COUNT"] = counts["COUNT"] / TRACK_LENGTH
            per_dataset = per_dataset.merge(counts, on=group_keys)

            if len(per_dataset) == 0:
                continue

            # Average repeat runs only (identical power and scan speed); conditions
            # that merely share a P/v ratio stay separate.
            condition = (
                per_dataset.groupby(["Averaged power", "SCAN_SPEED"])
                .agg(
                    COUNT=("COUNT", "mean"),
                    N_RUNS=("DATASET_NAME", "nunique"),
                    MELTING_MODE=(
                        "MELTING_MODE",
                        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown",
                    ),
                    **{ENTHALPY_COL: (ENTHALPY_COL, "first")},
                )
                .reset_index()
            )

            condition = condition.sort_values([ENTHALPY_COL, "COUNT"]).reset_index(drop=True)
            export_rows.extend(
                self.enthalpy_count_rows(per_dataset, condition, laser_mode, ENTHALPY_COL)
            )

            enthalpy_points = condition[ENTHALPY_COL].values
            avg_counts = condition["COUNT"].values
            melting_modes_arr = condition["MELTING_MODE"].values

            n_merged = int((condition["N_RUNS"] > 1).sum())
            print(
                f"  {laser_mode}: {len(per_dataset)} datasets -> {len(condition)} points "
                f"({n_merged} condition(s) averaged over repeats)"
            )
            for _, r in condition[condition["N_RUNS"] > 1].iterrows():
                print(
                    f"    averaged {int(r['N_RUNS'])} runs at P={r['Averaged power']:.0f} W, "
                    f"v={r['SCAN_SPEED']:.0f} mm/s -> {r['COUNT']:.2f} /mm"
                )

            line_color = mode_colors.get(laser_mode, "#1f77b4")
            line_style = mode_styles.get(laser_mode, "-")

            ax.plot(
                enthalpy_points,
                avg_counts,
                color=line_color,
                linestyle=line_style,
                linewidth=LINE_WIDTH,
                alpha=0.9,
                label=f"{laser_mode} Mode",
                zorder=3,
            )

            for enthalpy, count, melting_mode in zip(
                enthalpy_points, avg_counts, melting_modes_arr
            ):
                marker = MELTING_MODE_MARKERS.get(melting_mode, "o")
                ax.scatter(
                    enthalpy,
                    count,
                    marker=marker,
                    color=line_color,
                    s=MARKER_SIZE**2,
                    edgecolors="black",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=4,
                )

            x_data.extend(enthalpy_points)
            y_data.extend(avg_counts)

        if len(x_data) > 2:
            self.add_best_fit_line(ax, np.array(x_data), np.array(y_data))

        ax.xaxis.set_major_formatter(FuncFormatter(format_decimal))
        ax.yaxis.set_major_formatter(FuncFormatter(format_decimal))
        x_start, x_end = self._apply_uniform_enthalpy_xaxis(ax)

        if (
            UNSTABLE_KEYHOLE_THRESHOLDS["enable_threshold_shading"]
            and ENTHALPY_THRESHOLD is not None
        ):
            threshold = ENTHALPY_THRESHOLD
            color = UNSTABLE_KEYHOLE_THRESHOLDS["threshold_color"]
            alpha = UNSTABLE_KEYHOLE_THRESHOLDS["threshold_alpha"]
            ax.axvspan(threshold, x_end, color=color, alpha=alpha, zorder=1)
            ax.axvline(
                x=threshold,
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                zorder=2,
            )
            self._add_unstable_keyhole_label(ax, threshold, color)

        ax.set_xlabel(ENTHALPY_LABEL, fontweight="bold", fontsize=FONT_SIZE)
        ax.set_ylabel("Average spatter count per mm", fontweight="bold", fontsize=FONT_SIZE)
        ax.set_title(
            f"Average spatter count per mm vs {ENTHALPY_NAME.lower()}\n(Lines: Laser Mode, Markers: Melting Mode)",
            fontweight="bold",
            fontsize=FONT_SIZE + 2,
            pad=20,
        )
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        laser_mode_lines = []
        for lm in sorted(laser_modes):
            lc = mode_colors.get(lm, "#1f77b4")
            ls = mode_styles.get(lm, "-")
            line = ax.plot(
                [], [], color=lc, linestyle=ls, linewidth=LINE_WIDTH, label=f"{lm} Mode"
            )[0]
            laser_mode_lines.append(line)

        all_melting_modes = set()
        for lm in sorted(laser_modes):
            md = df[df["LASER_MODE"] == lm]
            if len(md) > 0:
                all_melting_modes.update(md["MELTING_MODE"].unique())
        all_melting_modes.discard("unknown")
        all_melting_modes = [m for m in MELTING_MODE_ORDER if m in all_melting_modes] + [
            m for m in sorted(all_melting_modes) if m not in MELTING_MODE_ORDER
        ]

        melting_mode_points = []
        for mm in all_melting_modes:
            marker = MELTING_MODE_MARKERS.get(mm, "o")
            display_name = mm.replace("_", " ").title()
            point = ax.scatter(
                [],
                [],
                marker=marker,
                c="gray",
                s=MARKER_SIZE**2,
                edgecolors="black",
                linewidth=0.8,
                label=display_name,
            )
            melting_mode_points.append(point)

        laser_legend = ax.legend(
            handles=laser_mode_lines,
            title="Laser Mode",
            loc="upper left",
            fontsize=FONT_SIZE - 2,
            title_fontsize=FONT_SIZE - 1,
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor="black",
            facecolor="white",
            framealpha=0.9,
        )
        if melting_mode_points:
            ax.legend(
                handles=melting_mode_points,
                title="Melting Mode (Markers)",
                loc="lower right",
                fontsize=FONT_SIZE - 3,
                title_fontsize=FONT_SIZE - 2,
                frameon=True,
                fancybox=False,
                shadow=False,
                edgecolor="black",
                facecolor="white",
                framealpha=0.9,
            )
            ax.add_artist(laser_legend)

        self.export_plot_data(
            f"13_normalized_enthalpy_vs_count{ENTHALPY_SUFFIX}",
            export_rows,
            x_col="enthalpy_x",
            y_col="spatter_count_per_mm",
            y_label="Average spatter count per mm",
            sort_by=["laser_mode", "enthalpy_x"],
        )

        plt.tight_layout()
        try:
            filename = f"13_normalized_enthalpy_vs_count{ENTHALPY_SUFFIX}"
            png_path = os.path.join(self.output_dir, f"{filename}.png")
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            fig.savefig(png_path, dpi=DPI, bbox_inches="tight", format="png", facecolor="white")
            fig.savefig(pdf_path, bbox_inches="tight", format="pdf", facecolor="white")
            print(f"Saved: {png_path} and {pdf_path}")
        except Exception as e:
            print(f"Error saving enthalpy vs count plot: {e}")
        plt.close()


def main():
    """Load the data, apply the filters and draw the manuscript figures."""
    print("=== LPBF spatter analysis (counts per mm) ===")
    print(f"Track length: {TRACK_LENGTH} mm (all counts reported as per mm)")
    print(f"Enthalpy scale: {ENTHALPY_SCALE}")
    if ENTHALPY_SCALE == "product":
        print("  Plotting dH/h_m * L*_th = A*P / (sqrt(pi) * h_m * v * r0^2)   (Ye et al. 2019)")
        print("  Note: this is proportional to P/v, i.e. a rescaled linear energy density.")
    else:
        print("  Plotting dH/h_s = A*P / (h_m * sqrt(pi * D * v * r0^3))  (Hann/King)")
    print("")

    analyzer = ComprehensiveLPBFAnalyzer()
    analyzer.setup_filters()

    if not analyzer.load_laser_parameters():
        print("Failed to load laser parameters. Exiting.")
        return

    dataset_folders = analyzer.find_dataset_folders()
    if not dataset_folders:
        print("No datasets found. Exiting.")
        return

    selected_datasets = analyzer.get_dataset_selection(dataset_folders)
    analyzer.load_metrics_files(selected_datasets, dataset_folders)

    if not analyzer.all_data:
        print("No data could be loaded. Exiting.")
        return

    analyzer.create_all_plots()


if __name__ == "__main__":
    main()
