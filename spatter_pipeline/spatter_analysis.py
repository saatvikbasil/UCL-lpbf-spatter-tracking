#!/usr/bin/env python3
"""Per-track spatter metrics from TrackMate output.

Reads the spots and tracks CSVs written by spatter_tracker.py and writes:

- spatter_metrics.csv: one row per track (size, circularity, speed, ejection
  angle, displacement, straightness, ...)
- spatter_summary_report.txt
- visualizations/: diagnostic plots for the dataset

Positions and radii are in microns and TrackMate speeds in microns/ms, as
calibrated in spatter_tracker.py. The ejection angle is taken between the 1st
and 5th spot of each track (1st and last for shorter tracks), with 0 degrees
pointing straight up in the image.

Usage:
    python spatter_pipeline/spatter_analysis.py --spots spots.csv --tracks tracks.csv --output results_dir
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


def calculate_angle(x1, y1, x2, y2):
    """Direction of travel from (x1, y1) to (x2, y2), in degrees.

    0 is straight up in the image (image y increases downward); positive
    angles lean right and negative angles lean left. Downward directions are
    reflected about the horizontal, so the result lies in [-90, 90].
    """
    dx = x2 - x1
    dy = y1 - y2  # image y axis points down

    # atan2(dx, dy) rather than atan2(dy, dx) so that 0 degrees points up
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    elif angle_deg < -90:
        angle_deg = -180 - angle_deg

    return angle_deg


def calculate_track_metrics(spots_df, tracks_df, output_dir):
    """Compute one row of metrics per track and save them to spatter_metrics.csv.

    Parameters:
        spots_df: TrackMate spots table
        tracks_df: TrackMate tracks table
        output_dir: directory for spatter_metrics.csv

    Returns:
        DataFrame of track metrics
    """
    track_metrics = []

    track_ids = spots_df["TRACK_ID"].dropna().unique()

    print(f"Processing {len(track_ids)} tracks...")

    for track_id in track_ids:
        track_spots = spots_df[spots_df["TRACK_ID"] == track_id]
        track_data = tracks_df[tracks_df["TRACK_ID"] == track_id] if not tracks_df.empty else None

        track_spots = track_spots.sort_values("FRAME")

        first_spot = track_spots.iloc[0]
        last_spot = track_spots.iloc[-1]
        num_spots = len(track_spots)

        avg_radius = track_spots["RADIUS"].mean()

        # Prefer TrackMate's AREA; otherwise average the per-spot circle areas.
        if "AREA" in track_spots.columns:
            avg_area = track_spots["AREA"].mean()
        else:
            individual_areas = np.pi * (track_spots["RADIUS"] ** 2)
            avg_area = individual_areas.mean()

        if "CIRCULARITY" in track_spots.columns:
            avg_circularity = track_spots["CIRCULARITY"].mean()
        else:
            # Some TrackMate versions export shape under a different name.
            circularity_columns = ["CIRCULARITY", "SHAPE_INDEX", "ROUNDNESS"]
            avg_circularity = None
            for col in circularity_columns:
                if col in track_spots.columns:
                    avg_circularity = track_spots[col].mean()
                    break
            if avg_circularity is None:
                avg_circularity = np.nan

        if (
            track_data is not None
            and len(track_data) > 0
            and "TRACK_MEAN_SPEED" in track_data.columns
        ):
            avg_velocity = track_data["TRACK_MEAN_SPEED"].values[0]
            max_velocity = (
                track_data["TRACK_MAX_SPEED"].values[0]
                if "TRACK_MAX_SPEED" in track_data.columns
                else 0
            )
        else:
            # Fallback when TrackMate speeds are missing: displacement per
            # frame between consecutive spots.
            velocities = []
            prev_spot = None

            for _, spot in track_spots.iterrows():
                if prev_spot is not None:
                    x1, y1 = prev_spot["POSITION_X"], prev_spot["POSITION_Y"]
                    x2, y2 = spot["POSITION_X"], spot["POSITION_Y"]

                    t1, t2 = prev_spot["FRAME"], spot["FRAME"]
                    frame_diff = t2 - t1

                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    if frame_diff > 0:
                        time_diff = frame_diff
                        velocity = distance / time_diff
                        velocities.append(velocity)

                prev_spot = spot

            avg_velocity = np.mean(velocities) if velocities else 0
            max_velocity = max(velocities) if velocities else 0

        # Ejection angle from the 1st to the 5th spot (1st to last for shorter tracks)
        if num_spots >= 5:
            fifth_spot = track_spots.iloc[4]
            trajectory_angle = calculate_angle(
                first_spot["POSITION_X"],
                first_spot["POSITION_Y"],
                fifth_spot["POSITION_X"],
                fifth_spot["POSITION_Y"],
            )
            trajectory_displacement = np.sqrt(
                (fifth_spot["POSITION_X"] - first_spot["POSITION_X"]) ** 2
                + (fifth_spot["POSITION_Y"] - first_spot["POSITION_Y"]) ** 2
            )
            trajectory_frames_used = 5
        else:
            trajectory_angle = calculate_angle(
                first_spot["POSITION_X"],
                first_spot["POSITION_Y"],
                last_spot["POSITION_X"],
                last_spot["POSITION_Y"],
            )
            trajectory_displacement = np.sqrt(
                (last_spot["POSITION_X"] - first_spot["POSITION_X"]) ** 2
                + (last_spot["POSITION_Y"] - first_spot["POSITION_Y"]) ** 2
            )
            trajectory_frames_used = num_spots

        total_displacement = np.sqrt(
            (last_spot["POSITION_X"] - first_spot["POSITION_X"]) ** 2
            + (last_spot["POSITION_Y"] - first_spot["POSITION_Y"]) ** 2
        )

        track_duration_frames = last_spot["FRAME"] - first_spot["FRAME"]
        track_duration_ms = track_duration_frames

        # Straightness = end-to-end displacement / path length
        path_segments = np.sqrt(
            (track_spots["POSITION_X"].diff()) ** 2 + (track_spots["POSITION_Y"].diff()) ** 2
        )
        total_path_length = path_segments.sum()
        straightness = total_displacement / total_path_length if total_path_length > 0 else 1.0

        track_metrics.append(
            {
                "TRACK_ID": track_id,
                "NUM_SPOTS": num_spots,
                "AVG_RADIUS": avg_radius,  # microns
                "AVG_AREA": avg_area,  # microns^2
                "AVG_CIRCULARITY": avg_circularity,  # 0-1, 1 = perfect circle
                "AVG_VELOCITY": avg_velocity,  # microns/ms
                "MAX_VELOCITY": max_velocity,  # microns/ms
                "TRAJECTORY_ANGLE": trajectory_angle,  # degrees, 1st to 5th spot
                "TRAJECTORY_DISPLACEMENT": trajectory_displacement,  # microns, same spots
                "TRAJECTORY_FRAMES_USED": trajectory_frames_used,
                "TOTAL_DISPLACEMENT": total_displacement,  # microns, first to last spot
                "TRACK_DURATION_FRAMES": track_duration_frames,
                "TRACK_DURATION_MS": track_duration_ms,
                "START_FRAME": first_spot["FRAME"],
                "END_FRAME": last_spot["FRAME"],
                "START_X": first_spot["POSITION_X"],  # microns
                "START_Y": first_spot["POSITION_Y"],
                "END_X": last_spot["POSITION_X"],
                "END_Y": last_spot["POSITION_Y"],
                "STRAIGHTNESS": straightness,
                "TOTAL_PATH_LENGTH": total_path_length,  # microns
            }
        )

    metrics_df = pd.DataFrame(track_metrics)

    output_path = os.path.join(output_dir, "spatter_metrics.csv")
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved spatter metrics to {output_path}")

    print("\nSummary Statistics:")
    print(f"Total number of tracks: {len(metrics_df)}")
    print(
        f"Tracks with >=5 frames (used for ejection angle): {len(metrics_df[metrics_df['TRAJECTORY_FRAMES_USED'] == 5])}"
    )
    print(
        f"Tracks with <5 frames (fallback to last frame): {len(metrics_df[metrics_df['TRAJECTORY_FRAMES_USED'] < 5])}"
    )
    print(f"Average particle radius: {metrics_df['AVG_RADIUS'].mean():.2f} microns")
    print(f"Average particle area: {metrics_df['AVG_AREA'].mean():.1f} microns²")
    print(
        f"Average circularity: {metrics_df['AVG_CIRCULARITY'].mean():.3f}"
        if not metrics_df["AVG_CIRCULARITY"].isna().all()
        else "Average circularity: N/A (not available in data)"
    )
    print(f"Average velocity: {metrics_df['AVG_VELOCITY'].mean():.2f} microns/ms")
    print(f"Average track duration: {metrics_df['TRACK_DURATION_MS'].mean():.4f} ms")

    return metrics_df


def create_visualizations(metrics_df, output_dir, spots_df=None):
    """Save diagnostic plots of the track metrics to <output_dir>/visualizations.

    Parameters:
        metrics_df: track metrics from calculate_track_metrics
        output_dir: analysis output directory
        spots_df: optional spots table, used to draw individual track paths
    """
    print("Creating visualizations...")

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df["TRACK_ID"], metrics_df["AVG_VELOCITY"])
    plt.xlabel("Track ID")
    plt.ylabel("Average Velocity (microns/ms)")
    plt.title("Average Velocity by Track ID")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "avg_velocity_by_track.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df["TRACK_ID"], metrics_df["AVG_RADIUS"])
    plt.xlabel("Track ID")
    plt.ylabel("Average Radius (microns)")
    plt.title("Average Particle Radius by Track ID")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "avg_radius_by_track.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df["TRACK_ID"], metrics_df["AVG_AREA"])
    plt.xlabel("Track ID")
    plt.ylabel("Average Area (microns²)")
    plt.title("Average Particle Area by Track ID")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "avg_area_by_track.png"), dpi=300)
    plt.close()

    if not metrics_df["AVG_CIRCULARITY"].isna().all():
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df["TRACK_ID"], metrics_df["AVG_CIRCULARITY"])
        plt.xlabel("Track ID")
        plt.ylabel("Average Circularity")
        plt.title("Average Particle Circularity by Track ID")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "avg_circularity_by_track.png"), dpi=300)
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df["TRAJECTORY_ANGLE"], bins=36, range=(-180, 180))
    plt.xlabel("Ejection Angle (degrees)")
    plt.ylabel("Number of Tracks")
    plt.title("Distribution of Spatter Ejection Angles (1st to 5th Frame)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "ejection_angle_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        metrics_df["AVG_RADIUS"],
        metrics_df["AVG_VELOCITY"],
        alpha=0.7,
        c=metrics_df["TRACK_DURATION_MS"],
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Track Duration (ms)")
    plt.xlabel("Average Radius (microns)")
    plt.ylabel("Average Velocity (microns/ms)")
    plt.title("Relationship Between Particle Radius and Velocity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "radius_vs_velocity.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        metrics_df["AVG_AREA"],
        metrics_df["AVG_VELOCITY"],
        alpha=0.7,
        c=metrics_df["TRACK_DURATION_MS"],
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Track Duration (ms)")
    plt.xlabel("Average Area (microns²)")
    plt.ylabel("Average Velocity (microns/ms)")
    plt.title("Relationship Between Particle Area and Velocity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "area_vs_velocity.png"), dpi=300)
    plt.close()

    if not metrics_df["AVG_CIRCULARITY"].isna().all():
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            metrics_df["AVG_CIRCULARITY"],
            metrics_df["AVG_VELOCITY"],
            alpha=0.7,
            c=metrics_df["AVG_RADIUS"],
            cmap="plasma",
        )
        plt.colorbar(scatter, label="Average Radius (microns)")
        plt.xlabel("Average Circularity")
        plt.ylabel("Average Velocity (microns/ms)")
        plt.title("Relationship Between Particle Circularity and Velocity")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "circularity_vs_velocity.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df["AVG_CIRCULARITY"].dropna(), bins=20, range=(0, 1))
        plt.xlabel("Average Circularity")
        plt.ylabel("Number of Tracks")
        plt.title("Distribution of Particle Circularity")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "circularity_distribution.png"), dpi=300)
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    angles = np.radians(metrics_df["TRAJECTORY_ANGLE"] + 90)  # 0 deg (up) drawn at the top
    weights = metrics_df["TRAJECTORY_DISPLACEMENT"]
    bins = np.linspace(0, 2 * np.pi, 37)  # 36 bins of 10 degrees
    hist, bin_edges = np.histogram(angles, bins=bins, weights=weights)

    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_edges[:-1], hist, width=width, align="edge", alpha=0.7)

    plt.title(
        "Polar Distribution of Spatter Ejection Angles (1st to 5th Frame)\n(Weighted by Ejection Displacement)"
    )
    plt.savefig(os.path.join(vis_dir, "ejection_angle_polar.png"), dpi=300)
    plt.close(fig)

    plt.figure(figsize=(10, 10))
    plt.scatter(
        metrics_df["START_X"],
        metrics_df["START_Y"],
        c="blue",
        alpha=0.5,
        label="Start Position",
    )

    for _, row in metrics_df.iterrows():
        if row["TRAJECTORY_FRAMES_USED"] == 5:
            arrow_length = min(50, row["TRAJECTORY_DISPLACEMENT"])  # capped for readability
            angle_rad = np.radians(row["TRAJECTORY_ANGLE"])

            end_x = row["START_X"] + arrow_length * np.sin(angle_rad)
            end_y = row["START_Y"] - arrow_length * np.cos(angle_rad)  # image y points down

            arrow = FancyArrowPatch(
                (row["START_X"], row["START_Y"]),
                (end_x, end_y),
                arrowstyle="->",
                mutation_scale=15,
                color="red",
                alpha=0.7,
            )
            plt.gca().add_patch(arrow)
        else:
            plt.scatter(row["START_X"], row["START_Y"], c="orange", s=50, alpha=0.7, marker="x")

    plt.xlabel("X Position (microns)")
    plt.ylabel("Y Position (microns)")
    plt.title(
        "Spatter Ejection Directions (1st to 5th Frame)\nRed arrows: >=5 frames, Orange X: <5 frames"
    )
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "ejection_directions.png"), dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(metrics_df["AVG_RADIUS"], bins=20)
    ax1.set_xlabel("Average Radius (microns)")
    ax1.set_ylabel("Number of Tracks")
    ax1.set_title("Distribution of Particle Radii")
    ax1.grid(alpha=0.3)

    ax2.hist(metrics_df["AVG_AREA"], bins=20)
    ax2.set_xlabel("Average Area (microns²)")
    ax2.set_ylabel("Number of Tracks")
    ax2.set_title("Distribution of Particle Areas")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "size_distributions.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df["AVG_VELOCITY"], bins=20)
    plt.xlabel("Average Velocity (microns/ms)")
    plt.ylabel("Number of Tracks")
    plt.title("Distribution of Particle Velocities")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "velocity_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(
        metrics_df["TRAJECTORY_DISPLACEMENT"],
        metrics_df["TOTAL_DISPLACEMENT"],
        alpha=0.7,
        c=metrics_df["NUM_SPOTS"],
        cmap="viridis",
    )
    plt.colorbar(label="Number of Spots in Track")
    plt.xlabel("Ejection Displacement (1st to 5th frame, microns)")
    plt.ylabel("Total Displacement (1st to last frame, microns)")
    plt.title("Ejection vs Total Displacement")

    max_val = max(
        metrics_df["TRAJECTORY_DISPLACEMENT"].max(),
        metrics_df["TOTAL_DISPLACEMENT"].max(),
    )
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal displacement line")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "ejection_vs_total_displacement.png"), dpi=300)
    plt.close()

    if spots_df is not None and len(metrics_df) <= 20:
        plt.figure(figsize=(12, 12))

        cmap = plt.cm.rainbow
        colors = cmap(np.linspace(0, 1, len(metrics_df)))

        pixel_to_micron = 4.3  # microns per pixel

        for i, track_id in enumerate(metrics_df["TRACK_ID"]):
            track_spots = spots_df[spots_df["TRACK_ID"] == track_id].sort_values("FRAME")

            pos_x = track_spots["POSITION_X"] * pixel_to_micron
            pos_y = track_spots["POSITION_Y"] * pixel_to_micron

            plt.plot(
                pos_x,
                pos_y,
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.7,
                color=colors[i],
                label=f"Track {track_id}",
            )

            # Highlight the spots used for the ejection angle
            if len(pos_x) >= 5:
                plt.plot(
                    pos_x[:5],
                    pos_y[:5],
                    marker="s",
                    markersize=5,
                    linewidth=2,
                    alpha=0.9,
                    color=colors[i],
                )

        plt.xlabel("X Position (microns)")
        plt.ylabel("Y Position (microns)")
        plt.title("Detailed Track Paths (First 5 points highlighted for ejection analysis)")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "detailed_track_paths.png"), dpi=300)
        plt.close()

    print(f"Saved visualizations to {vis_dir}")


def create_summary_report(metrics_df, output_dir):
    """Write spatter_summary_report.txt with dataset-level statistics.

    Parameters:
        metrics_df: track metrics from calculate_track_metrics
        output_dir: directory for the report
    """
    report_path = os.path.join(output_dir, "spatter_summary_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== SPATTER TRACKING ANALYSIS SUMMARY =====\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"Total number of tracks: {len(metrics_df)}\n")
        f.write(f"Total number of tracked spots: {metrics_df['NUM_SPOTS'].sum()}\n")
        f.write(f"Average number of spots per track: {metrics_df['NUM_SPOTS'].mean():.2f}\n")
        f.write(
            f"Tracks with >=5 frames (used for ejection angle): {len(metrics_df[metrics_df['TRAJECTORY_FRAMES_USED'] == 5])}\n"
        )
        f.write(
            f"Tracks with <5 frames (fallback method): {len(metrics_df[metrics_df['TRAJECTORY_FRAMES_USED'] < 5])}\n"
        )
        f.write(f"Average particle radius: {metrics_df['AVG_RADIUS'].mean():.2f} microns\n")
        f.write(f"Average particle area: {metrics_df['AVG_AREA'].mean():.1f} microns²\n")
        f.write(
            f"Average circularity: {metrics_df['AVG_CIRCULARITY'].mean():.3f}\n"
            if not metrics_df["AVG_CIRCULARITY"].isna().all()
            else "Average circularity: N/A (not available in data)\n"
        )
        f.write(f"Average velocity: {metrics_df['AVG_VELOCITY'].mean():.2f} microns/ms\n")
        f.write(f"Average track duration: {metrics_df['TRACK_DURATION_MS'].mean():.4f} ms\n")
        f.write(
            f"Average total displacement: {metrics_df['TOTAL_DISPLACEMENT'].mean():.2f} microns\n"
        )
        f.write(
            f"Average ejection displacement (1st-5th frame): {metrics_df['TRAJECTORY_DISPLACEMENT'].mean():.2f} microns\n"
        )
        f.write(f"Average path straightness: {metrics_df['STRAIGHTNESS'].mean():.2f}\n\n")

        f.write("SIZE ANALYSIS:\n")
        f.write(f"Minimum average radius: {metrics_df['AVG_RADIUS'].min():.2f} microns\n")
        f.write(f"Maximum average radius: {metrics_df['AVG_RADIUS'].max():.2f} microns\n")
        f.write(f"Radius standard deviation: {metrics_df['AVG_RADIUS'].std():.2f} microns\n")
        f.write(f"Minimum average area: {metrics_df['AVG_AREA'].min():.1f} microns²\n")
        f.write(f"Maximum average area: {metrics_df['AVG_AREA'].max():.1f} microns²\n")
        f.write(f"Area standard deviation: {metrics_df['AVG_AREA'].std():.1f} microns²\n\n")

        if not metrics_df["AVG_CIRCULARITY"].isna().all():
            f.write("CIRCULARITY ANALYSIS:\n")
            f.write(f"Average circularity: {metrics_df['AVG_CIRCULARITY'].mean():.3f}\n")
            f.write(f"Minimum circularity: {metrics_df['AVG_CIRCULARITY'].min():.3f}\n")
            f.write(f"Maximum circularity: {metrics_df['AVG_CIRCULARITY'].max():.3f}\n")
            f.write(f"Circularity standard deviation: {metrics_df['AVG_CIRCULARITY'].std():.3f}\n")

            high_circularity = metrics_df[metrics_df["AVG_CIRCULARITY"] >= 0.8]
            medium_circularity = metrics_df[
                (metrics_df["AVG_CIRCULARITY"] >= 0.6) & (metrics_df["AVG_CIRCULARITY"] < 0.8)
            ]
            low_circularity = metrics_df[metrics_df["AVG_CIRCULARITY"] < 0.6]

            f.write(
                f"High circularity particles (>=0.8): {len(high_circularity)} ({len(high_circularity)/len(metrics_df)*100:.1f}%)\n"
            )
            f.write(
                f"Medium circularity particles (0.6-0.8): {len(medium_circularity)} ({len(medium_circularity)/len(metrics_df)*100:.1f}%)\n"
            )
            f.write(
                f"Low circularity particles (<0.6): {len(low_circularity)} ({len(low_circularity)/len(metrics_df)*100:.1f}%)\n\n"
            )
        else:
            f.write("CIRCULARITY ANALYSIS:\n")
            f.write("Circularity data not available in the input dataset.\n\n")

        f.write("VELOCITY ANALYSIS:\n")
        f.write(f"Minimum average velocity: {metrics_df['AVG_VELOCITY'].min():.2f} microns/ms\n")
        f.write(f"Maximum average velocity: {metrics_df['AVG_VELOCITY'].max():.2f} microns/ms\n")
        f.write(f"Velocity standard deviation: {metrics_df['AVG_VELOCITY'].std():.2f} microns/ms\n")
        f.write(
            f"Maximum instantaneous velocity: {metrics_df['MAX_VELOCITY'].max():.2f} microns/ms\n\n"
        )

        # Most common ejection direction in 45-degree bins, tracks with >= 5 spots
        tracks_with_5_frames = metrics_df[metrics_df["TRAJECTORY_FRAMES_USED"] == 5]
        if len(tracks_with_5_frames) > 0:
            bins = np.arange(-180, 181, 45)
            bin_labels = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
            bin_indices = np.digitize(tracks_with_5_frames["TRAJECTORY_ANGLE"], bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
            direction_counts = pd.Series(bin_indices).map(lambda x: bin_labels[x]).value_counts()
            most_common_direction = direction_counts.index[0]
        else:
            direction_counts = pd.Series()
            most_common_direction = "N/A"

        f.write("SPATTER EJECTION ANALYSIS (1st to 5th frame):\n")
        f.write(f"Tracks analyzed for ejection angle: {len(tracks_with_5_frames)}\n")
        if len(tracks_with_5_frames) > 0:
            f.write(
                f"Most common ejection direction: {most_common_direction} ({direction_counts[most_common_direction]} tracks)\n"
            )
            f.write(
                f"Average ejection displacement: {tracks_with_5_frames['TRAJECTORY_DISPLACEMENT'].mean():.2f} microns\n"
            )
            f.write(
                f"Maximum ejection displacement: {tracks_with_5_frames['TRAJECTORY_DISPLACEMENT'].max():.2f} microns\n\n"
            )

            f.write("EJECTION DIRECTION DISTRIBUTION (>=5 frame tracks):\n")
            for direction, count in direction_counts.items():
                percentage = count / len(tracks_with_5_frames) * 100
                f.write(f"{direction}: {count} tracks ({percentage:.1f}%)\n")
        else:
            f.write("No tracks with >=5 frames available for ejection angle analysis.\n")

        f.write("\nOverall displacement comparison:\n")
        f.write(
            f"Average total displacement (1st-last): {metrics_df['TOTAL_DISPLACEMENT'].mean():.2f} microns\n"
        )
        f.write(
            f"Maximum total displacement: {metrics_df['TOTAL_DISPLACEMENT'].max():.2f} microns\n"
        )

    print(f"Saved summary report to {report_path}")


def process_tracking_data(spots_csv, tracks_csv, output_dir):
    """Load the TrackMate CSVs, then write the metrics, plots and summary report.

    Parameters:
        spots_csv: path to the spots CSV
        tracks_csv: path to the tracks CSV
        output_dir: output directory

    Returns:
        True on success, False if the CSVs could not be read
    """
    print("Processing tracking data...")
    print(f"Spots CSV: {spots_csv}")
    print(f"Tracks CSV: {tracks_csv}")
    print("Using spatial calibration: 4.3 microns per pixel")
    print("Using temporal calibration: 0.025 ms per frame")
    print("Velocity unit: microns/ms")
    print("Ejection angle analysis: 1st frame to 5th frame (or last frame if <5 frames)")

    try:
        spots_df = pd.read_csv(spots_csv)
        tracks_df = pd.read_csv(tracks_csv)

        print(f"Loaded {len(spots_df)} spots and {len(tracks_df)} tracks")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return False

    metrics_df = calculate_track_metrics(spots_df, tracks_df, output_dir)
    create_visualizations(metrics_df, output_dir, spots_df)
    create_summary_report(metrics_df, output_dir)

    print("Processing completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process TrackMate tracking data and generate spatter metrics"
    )
    parser.add_argument("--spots", required=True, help="Path to spots CSV file")
    parser.add_argument("--tracks", required=True, help="Path to tracks CSV file")
    parser.add_argument("--output", required=True, help="Directory to save results")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    process_tracking_data(args.spots, args.tracks, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
