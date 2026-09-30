#!/usr/bin/env python3
"""Detect and track spatter in predicted mask stacks with Fiji's TrackMate.

The mask stack is calibrated (4.3 microns/pixel, 0.025 ms/frame), spots are
detected with TrackMate's mask detector, filtered on radius and circularity,
linked with the Kalman tracker and filtered on mean track speed. Spots and
tracks are exported to CSV for spatter_analysis.py, and the TrackMate session
can optionally be saved as XML.

Usage:
    python spatter_tracker.py --input /path/to/mask.tif --output /path/to/output_dir --memory 8G

Requires pyimagej and scyjava (Fiji 2.13.0 is fetched on first run), pandas.
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# TrackMate settings, in calibrated units (microns)
CIRCULARITY_THRESHOLD = 0.01  # minimum spot circularity
RADIUS_THRESHOLD = 4  # minimum spot radius
SEARCH_RADIUS = 100  # Kalman linking and search radius
MAX_FRAME_GAP = 3  # frames a track may skip


class TrackMateAnalyzer:
    """Run TrackMate on mask stacks through pyimagej."""

    def __init__(self, memory="16G", headless=True):
        """Start ImageJ with the given Java heap size and load the TrackMate classes."""
        self.ij = None
        self.memory = memory
        self.headless = headless
        self._initialize_imagej()
        self._import_java_classes()

    def _initialize_imagej(self):
        """Start a Fiji 2.13.0 instance; exits the process if this fails."""
        print(f"Initializing ImageJ with {self.memory} of memory...")
        os.environ["JAVA_MEM"] = f"-Xmx{self.memory}"

        try:
            # Imported here so this module can be imported without pyimagej;
            # declared global because the other methods use sj.
            global imagej, sj
            import imagej
            import scyjava as sj

            mode = "headless" if self.headless else "interactive"
            self.ij = imagej.init("sc.fiji:fiji:2.13.0", mode=mode)
            print(f"ImageJ initialized successfully in {mode} mode!")
        except Exception as e:
            print(f"Error initializing ImageJ: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _import_java_classes(self):
        """Load the TrackMate and Java classes used below; exits the process if this fails."""
        try:
            self.TrackMate = sj.jimport("fiji.plugin.trackmate.TrackMate")
            self.Model = sj.jimport("fiji.plugin.trackmate.Model")
            self.Settings = sj.jimport("fiji.plugin.trackmate.Settings")
            self.Logger = sj.jimport("fiji.plugin.trackmate.Logger")
            self.SelectionModel = sj.jimport("fiji.plugin.trackmate.SelectionModel")

            self.MaskDetectorFactory = sj.jimport(
                "fiji.plugin.trackmate.detection.MaskDetectorFactory"
            )
            self.KalmanTrackerFactory = sj.jimport(
                "fiji.plugin.trackmate.tracking.kalman.KalmanTrackerFactory"
            )

            self.FeatureFilter = sj.jimport("fiji.plugin.trackmate.features.FeatureFilter")

            self.Boolean = sj.jimport("java.lang.Boolean")
            self.Double = sj.jimport("java.lang.Double")
            self.Integer = sj.jimport("java.lang.Integer")
            self.HashMap = sj.jimport("java.util.HashMap")

            self.SpotAnalyzerProvider = sj.jimport(
                "fiji.plugin.trackmate.providers.SpotAnalyzerProvider"
            )
            self.EdgeAnalyzerProvider = sj.jimport(
                "fiji.plugin.trackmate.providers.EdgeAnalyzerProvider"
            )
            self.TrackAnalyzerProvider = sj.jimport(
                "fiji.plugin.trackmate.providers.TrackAnalyzerProvider"
            )

            self.TmXmlWriter = sj.jimport("fiji.plugin.trackmate.io.TmXmlWriter")

            print("All Java classes imported successfully!")
        except Exception as e:
            print(f"Error importing Java classes: {e}")
            traceback.print_exc()
            sys.exit(1)

    def open_image(self, image_path, pixel_size_microns=4.3, frame_interval_ms=0.025):
        """Open a TIFF stack, calibrate it and return it as an ImagePlus.

        A stack read as many slices and a single frame is reinterpreted as a
        time series. Returns None if the image cannot be opened.
        """
        print(f"Opening image: {image_path}")
        try:
            ImagePlus = sj.jimport("ij.ImagePlus")
            Calibration = sj.jimport("ij.measure.Calibration")

            dataset = self.ij.io().open(str(image_path))
            if dataset is None:
                raise ValueError(f"Failed to open image: {image_path}")

            imp = self.ij.py.to_imageplus(dataset)

            calibration = Calibration()
            calibration.setUnit("micron")
            calibration.pixelWidth = pixel_size_microns
            calibration.pixelHeight = pixel_size_microns
            calibration.frameInterval = frame_interval_ms
            calibration.setTimeUnit("ms")
            imp.setCalibration(calibration)

            print(f"Set spatial calibration: {pixel_size_microns} microns per pixel")
            print(f"Set temporal calibration: {frame_interval_ms} ms per frame")

            width = imp.getWidth()
            height = imp.getHeight()
            slices = imp.getNSlices()
            frames = imp.getNFrames()
            channels = imp.getNChannels()

            print(
                f"Image dimensions: {width}x{height}, {channels} channels, {slices} slices, {frames} frames"
            )
            print(
                f"Physical dimensions: {width*pixel_size_microns:.1f}x{height*pixel_size_microns:.1f} microns"
            )

            if slices > 1 and frames == 1:
                print("WARNING: Image appears to be a Z-stack (many slices, 1 frame)")
                print("Converting Z-stack to time series for tracking...")

                original_stack = imp.getStack()
                new_title = imp.getTitle() + " (as time series)"
                new_imp = ImagePlus(new_title, original_stack)

                # Dimensions are (C, Z, T): the slices become frames.
                new_imp.setDimensions(channels, 1, slices)

                if new_imp.getProperty("frames"):
                    new_imp.setProperty("frames", str(slices))
                if new_imp.getProperty("slices"):
                    new_imp.setProperty("slices", "1")

                new_imp.setCalibration(imp.getCalibration())

                print(
                    f"Converted to time series: {new_imp.getNFrames()} frames, {new_imp.getNSlices()} slices"
                )
                print(f"Spatial calibration preserved: {pixel_size_microns} microns per pixel")
                print(f"Temporal calibration preserved: {frame_interval_ms} ms per frame")
                return new_imp
            else:
                print(f"Using original image format: {frames} frames, {slices} slices")
                return imp

        except Exception as e:
            print(f"Error opening image: {e}")
            traceback.print_exc()
            return None

    def setup_trackmate(self, imp):
        """Create the TrackMate model, settings and runner for ``imp``."""
        print("Setting up TrackMate...")

        channels = imp.getNChannels()
        slices = imp.getNSlices()
        frames = imp.getNFrames()

        if frames <= 1:
            print("WARNING: Image has only 1 frame. Tracking requires multiple frames.")
            print(f"Current dimensions: {channels} channels, {slices} slices, {frames} frames")

            # Second attempt at turning slices into frames
            if slices > 1:
                print("Attempting to reinterpret dimensions...")

                dimensions_array = sj.jimport("java.lang.String[]")(3)
                dimensions_array[0] = "channels"
                dimensions_array[1] = "frames"
                dimensions_array[2] = "slices"

                self.ij.IJ.run(
                    imp,
                    "Stack to Hyperstack...",
                    f"order=xyczt channels={channels} slices=1 frames={slices} display=Color",
                )

                frames = imp.getNFrames()
                slices = imp.getNSlices()
                print(f"After reinterpretation: {frames} frames, {slices} slices")

        calibration = imp.getCalibration()
        spatial_units = (
            "micron" if calibration is not None and calibration.getUnit() == "micron" else "pixel"
        )
        time_units = "frame"

        model = self.Model()
        model.setPhysicalUnits(spatial_units, time_units)

        if calibration is not None:
            print(f"Using calibration: {calibration.pixelWidth} {spatial_units}/pixel")

        settings = self.Settings(imp)

        trackmate = self.TrackMate(model, settings)

        logger = self.Logger.IJ_LOGGER
        model.setLogger(logger)

        return trackmate, model, settings

    def configure_detector(self, settings):
        """Use the mask detector on channel 1 and enable every feature analyzer."""
        print("Configuring mask detector...")

        detector_factory = self.MaskDetectorFactory()
        settings.detectorFactory = detector_factory

        detector_settings = settings.detectorSettings
        detector_settings.put("SIMPLIFY_CONTOURS", self.Boolean(True))
        detector_settings.put("TARGET_CHANNEL", self.Integer(1))

        print("Adding all available spot analyzers...")
        settings.addAllAnalyzers()

        print("Detector configured with simplify contours enabled")

    def configure_spot_filters(self, settings):
        """Keep spots with radius >= RADIUS_THRESHOLD and circularity >= CIRCULARITY_THRESHOLD."""
        print("Configuring spot filters...")

        settings.addSpotFilter(
            self.FeatureFilter("RADIUS", self.Double(RADIUS_THRESHOLD), self.Boolean(True))
        )
        print(f"Spot filter added: Radius ≥ {RADIUS_THRESHOLD}")

        settings.addSpotFilter(
            self.FeatureFilter(
                "CIRCULARITY", self.Double(CIRCULARITY_THRESHOLD), self.Boolean(True)
            )
        )
        print(f"Spot filter added: Circularity ≥ {CIRCULARITY_THRESHOLD}")

        print(
            f"Using BOTH filters: Radius ≥ {RADIUS_THRESHOLD} AND Circularity ≥ {CIRCULARITY_THRESHOLD}"
        )

    def configure_tracker(self, settings):
        """Use the Kalman tracker with SEARCH_RADIUS and MAX_FRAME_GAP."""
        print("Configuring Kalman tracker...")

        tracker_factory = self.KalmanTrackerFactory()
        settings.trackerFactory = tracker_factory

        tracker_settings = settings.trackerSettings

        search_radius = SEARCH_RADIUS
        max_gap = MAX_FRAME_GAP

        tracker_settings.put("LINKING_MAX_DISTANCE", self.Double(search_radius))
        tracker_settings.put("GAP_CLOSING_MAX_DISTANCE", self.Double(search_radius))
        tracker_settings.put("MAX_FRAME_GAP", self.Integer(max_gap))
        tracker_settings.put("KALMAN_SEARCH_RADIUS", self.Double(search_radius))

        tracker_settings.put("KALMAN_POSITION_WEIGHT", self.Double(0.8))
        tracker_settings.put("KALMAN_VELOCITY_WEIGHT", self.Double(0.2))
        tracker_settings.put("INITIAL_SIGMA", self.Double(5.0))
        tracker_settings.put("USE_POSITION_WEIGHT", self.Boolean(True))
        tracker_settings.put("USE_VELOCITY_WEIGHT", self.Boolean(True))
        tracker_settings.put("POSITION_WEIGHT", self.Double(0.8))
        tracker_settings.put("VELOCITY_WEIGHT", self.Double(0.2))

        print(f"Tracker configured with search radius={search_radius}, max frame gap={max_gap}")

    def configure_track_filters(self, settings, min_mean_speed=50):
        """Keep tracks with TRACK_MEAN_SPEED >= ``min_mean_speed`` (microns/ms)."""
        print("Configuring track filters...")

        settings.addTrackFilter(
            self.FeatureFilter(
                "TRACK_MEAN_SPEED",
                self.Double(min_mean_speed),
                self.Boolean(True),  # keep values above the threshold
            )
        )

        print(f"Track filter added: Track mean speed ≥ {min_mean_speed}")

    def run_tracking(self, trackmate):
        """Run detection, spot filtering, tracking and track filtering.

        Also prints the radius and circularity distributions before filtering.
        Returns False if no spots survive the spot filters.
        """
        print("Running detection...")
        if not trackmate.execDetection():
            raise RuntimeError(f"Detection failed: {trackmate.getErrorMessage()}")

        print("Computing spot features...")
        if not trackmate.computeSpotFeatures(True):
            raise RuntimeError(f"Feature calculation failed: {trackmate.getErrorMessage()}")

        # Feature distributions before filtering, as a guide for the thresholds
        model = trackmate.getModel()
        spots = model.getSpots()
        all_spots = spots.iterable(False)

        circularity_values = []
        radius_values = []

        for spot in all_spots:
            if spot.getFeature("CIRCULARITY") is not None:
                circularity_values.append(spot.getFeature("CIRCULARITY"))
            if spot.getFeature("RADIUS") is not None:
                radius_values.append(spot.getFeature("RADIUS"))

        if circularity_values:
            circularity_values.sort()
            n_values = len(circularity_values)
            print("\n=== CIRCULARITY STATISTICS ===")
            print(f"Min: {min(circularity_values):.4f}")
            print(f"Max: {max(circularity_values):.4f}")
            print(f"Median: {circularity_values[n_values//2]:.4f}")
            print(f"Mean: {sum(circularity_values)/n_values:.4f}")
            print(
                f"# of spots with circularity ≥ {CIRCULARITY_THRESHOLD}: {sum(1 for c in circularity_values if c >= CIRCULARITY_THRESHOLD)}"
            )
            print(
                f"Percentage that pass circularity filter: {sum(1 for c in circularity_values if c >= CIRCULARITY_THRESHOLD)/n_values*100:.1f}%"
            )

            suggested_percentile = 25
            suggested_circularity = circularity_values[int(n_values * suggested_percentile / 100)]
            print(
                f"Suggested circularity threshold (keeps top {100-suggested_percentile}%): {suggested_circularity:.2f}"
            )

        if radius_values:
            radius_values.sort()
            n_values = len(radius_values)
            print("\n=== RADIUS STATISTICS ===")
            print(f"Min: {min(radius_values):.4f}")
            print(f"Max: {max(radius_values):.4f}")
            print(f"Median: {radius_values[n_values//2]:.4f}")
            print(f"Mean: {sum(radius_values)/n_values:.4f}")
            print(
                f"# of spots with radius ≥ {RADIUS_THRESHOLD}: {sum(1 for r in radius_values if r >= RADIUS_THRESHOLD)}"
            )
            print(
                f"Percentage that pass radius filter: {sum(1 for r in radius_values if r >= RADIUS_THRESHOLD)/n_values*100:.1f}%"
            )

            suggested_percentile = 25
            suggested_radius = radius_values[int(n_values * suggested_percentile / 100)]
            print(
                f"Suggested radius threshold (keeps top {100-suggested_percentile}%): {suggested_radius:.2f}"
            )

        print("\nApplying spot filters...")

        print("Filtering spots...")
        if not trackmate.execSpotFiltering(True):
            raise RuntimeError(f"Spot filtering failed: {trackmate.getErrorMessage()}")

        filtered_spots = trackmate.getModel().getSpots().getNSpots(True)
        print(f"Number of spots after filtering: {filtered_spots}")
        if filtered_spots == 0:
            print("WARNING: No spots remain after filtering! Adjust your thresholds.")
            return False

        print("Running tracking...")
        if not trackmate.execTracking():
            raise RuntimeError(f"Tracking failed: {trackmate.getErrorMessage()}")

        print("Computing track features...")
        if not trackmate.computeTrackFeatures(True):
            raise RuntimeError(f"Track feature calculation failed: {trackmate.getErrorMessage()}")

        print("Applying track filters...")
        if not trackmate.execTrackFiltering(True):
            raise RuntimeError(f"Track filtering failed: {trackmate.getErrorMessage()}")

        n_tracks_before = len(model.getTrackModel().trackIDs(False))
        n_tracks_after = len(model.getTrackModel().trackIDs(True))
        print(f"Number of tracks before filtering: {n_tracks_before}")
        print(f"Number of tracks after filtering: {n_tracks_after}")
        print(f"Filtered out {n_tracks_before - n_tracks_after} tracks")

        print("Tracking completed successfully!")
        return True

    def save_xml(self, trackmate, output_path):
        """Save the TrackMate model and settings as XML. Returns True on success."""
        print(f"Saving TrackMate session to {output_path}")
        try:
            file = sj.jimport("java.io.File")(str(output_path))
            writer = self.TmXmlWriter(file)
            writer.appendModel(trackmate.getModel())
            writer.appendSettings(trackmate.getSettings())
            writer.writeToFile()
            print(f"TrackMate session saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving TrackMate session: {e}")
            traceback.print_exc()
            return False

    def export_spots_to_csv(self, model, output_path, filtered_only=True):
        """Write one row per spot, with its track ID and every computed feature.

        Returns the output path, or None on failure.
        """
        try:
            print(f"Exporting spots to {output_path}")

            spots = model.getSpots()
            track_model = model.getTrackModel()

            total_spots = spots.getNSpots(filtered_only)
            print(f"Total spots to export: {total_spots}")

            print("Building spot-to-track lookup map...")
            spot_to_track = {}
            for track_id in track_model.trackIDs(True):
                for spot in track_model.trackSpots(track_id):
                    spot_to_track[spot.ID()] = track_id

            basic_columns = [
                "ID",
                "TRACK_ID",
                "FRAME",
                "POSITION_X",
                "POSITION_Y",
                "RADIUS",
            ]

            # The remaining feature columns are taken from the first spot.
            feature_columns = []
            spot_iterator = spots.iterator(filtered_only)
            if spot_iterator.hasNext():
                sample_spot = spot_iterator.next()
                features = sample_spot.getFeatures()
                feature_columns = [
                    str(key) for key in features.keySet() if str(key) not in basic_columns
                ]

            all_columns = basic_columns + feature_columns

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_columns)
                writer.writeheader()

                processed = 0
                for spot in spots.iterable(filtered_only):
                    spot_id = spot.ID()

                    spot_record = {
                        "ID": spot_id,
                        "TRACK_ID": spot_to_track.get(spot_id, None),
                        "FRAME": int(spot.getFeature("FRAME")),
                        "POSITION_X": spot.getFeature("POSITION_X"),
                        "POSITION_Y": spot.getFeature("POSITION_Y"),
                        "RADIUS": spot.getFeature("RADIUS"),
                    }

                    for feature in feature_columns:
                        spot_record[feature] = spot.getFeature(feature)

                    writer.writerow(spot_record)

                    processed += 1
                    if processed % 5000 == 0:
                        print(f"Processed {processed} spots of {total_spots}...")

            print(f"Successfully exported {processed} spots to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error exporting spots: {e}")
            traceback.print_exc()
            return None

    def export_tracks_to_csv(self, model, output_dir, image_name=None):
        """Write one row per filtered track, with every track feature.

        The file is named <image_name>_tracks_<timestamp>.csv. Returns its
        path, or None on failure.
        """
        try:
            # Timestamped so repeated runs do not overwrite each other
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if image_name:
                base_name = f"{image_name}_tracks_{timestamp}"
            else:
                base_name = f"tracks_{timestamp}"

            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{base_name}.csv")
            print(f"Exporting tracks to {output_path}")

            track_model = model.getTrackModel()
            feature_model = model.getFeatureModel()

            track_ids = list(track_model.trackIDs(True))
            print(f"Total track IDs (filtered): {len(track_ids)}")

            track_features = []
            if feature_model is not None:
                java_features = feature_model.getTrackFeatures()
                if java_features:
                    for feature in java_features:
                        track_features.append(str(feature))

            print(
                f"Available track features: {', '.join(track_features) if track_features else 'None'}"
            )

            tracks_data = []

            for i, track_id in enumerate(track_ids):
                track_spots = track_model.trackSpots(track_id)

                if not track_spots or len(track_spots) == 0:
                    continue

                spot_list = list(track_spots)
                sorted_spots = sorted(spot_list, key=lambda s: s.getFeature("FRAME"))

                start_frame = sorted_spots[0].getFeature("FRAME")
                end_frame = sorted_spots[-1].getFeature("FRAME")
                duration = end_frame - start_frame + 1

                track_record = {
                    "TRACK_ID": track_id,
                    "NUMBER_SPOTS": len(spot_list),
                    "START_FRAME": int(start_frame),
                    "END_FRAME": int(end_frame),
                    "TRACK_DURATION": duration,
                }

                for feature in track_features:
                    feature_value = feature_model.getTrackFeature(track_id, feature)
                    track_record[feature] = feature_value

                tracks_data.append(track_record)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1} tracks of {len(track_ids)}...")

            if tracks_data:
                df = pd.DataFrame(tracks_data)
                df.to_csv(output_path, index=False)
                print(f"Exported {len(tracks_data)} tracks to {output_path}")
            else:
                columns = [
                    "TRACK_ID",
                    "NUMBER_SPOTS",
                    "START_FRAME",
                    "END_FRAME",
                    "TRACK_DURATION",
                ] + track_features
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_path, index=False)
                print(f"Created empty tracks file: {output_path}")

            return output_path

        except Exception as e:
            print(f"Error exporting tracks: {e}")
            traceback.print_exc()

            return None

    def process_image(
        self,
        image_path,
        output_dir,
        save_xml=True,
        pixel_size_microns=4.3,
        frame_interval_ms=0.025,
        min_mean_speed=50,
    ):
        """Track one mask stack and export the results.

        Returns (success, spots_csv, tracks_csv).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem
        print(f"Processing image: {image_name}")
        print(f"Output directory: {output_dir}")
        print(f"Using scale: {pixel_size_microns} microns per pixel")
        print(f"Using frame interval: {frame_interval_ms} ms per frame")

        spots_csv_path = output_dir / f"{image_name}_spots.csv"
        tracks_csv_path = output_dir / f"{image_name}_tracks.csv"
        xml_path = output_dir / f"{image_name}_trackmate.xml"

        print("Results will be saved to:")
        print(f"  - Spots CSV: {spots_csv_path}")
        print(f"  - Tracks CSV: {tracks_csv_path}")
        if save_xml:
            print(f"  - TrackMate XML: {xml_path}")

        imp = self.open_image(
            image_path,
            pixel_size_microns=pixel_size_microns,
            frame_interval_ms=frame_interval_ms,
        )
        if imp is None:
            print(f"ERROR: Failed to open image: {image_path}")
            return False, None, None

        try:
            trackmate, model, settings = self.setup_trackmate(imp)

            self.configure_detector(settings)
            self.configure_spot_filters(settings)
            self.configure_track_filters(settings, min_mean_speed)
            self.configure_tracker(settings)

            success = self.run_tracking(trackmate)
            if not success:
                print(f"ERROR: Tracking failed for {image_path}")
                return False, None, None

            if save_xml:
                self.save_xml(trackmate, xml_path)

            spots_csv = self.export_spots_to_csv(model, spots_csv_path)
            tracks_csv = self.export_tracks_to_csv(model, output_dir, image_name)

            print("\nResults saved to:")
            print(f"  - Spots CSV: {spots_csv}")
            print(f"  - Tracks CSV: {tracks_csv}")
            if save_xml:
                print(f"  - TrackMate XML: {xml_path}")

            imp.close()

            print(f"Processing completed for {image_name}")

            # The tracks file name carries a timestamp, so return the paths
            # that were actually written.
            return True, spots_csv, tracks_csv

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            traceback.print_exc()
            if imp is not None:
                imp.close()
            return False, None, None

    def batch_process(self, input_files, output_dir):
        """Run process_image on each file; return per-file results and timings."""
        if isinstance(input_files, str):
            input_files = [input_files]

        results = {}
        for input_file in input_files:
            print(f"\nProcessing {input_file}...")
            start_time = time.time()
            success, spots_csv, tracks_csv = self.process_image(input_file, output_dir)
            elapsed_time = time.time() - start_time
            results[input_file] = {
                "success": success,
                "time": elapsed_time,
                "spots_csv": spots_csv,
                "tracks_csv": tracks_csv,
            }
            print(f"Finished processing {input_file} in {elapsed_time:.2f} seconds")

        return results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TrackMate Analyzer for LPBF data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Path to input mask file (TIFF stack)")

    parser.add_argument(
        "--output",
        default="trackmate_output",
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--memory", default="8G", help="Java memory allocation for ImageJ/TrackMate"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--no-save-xml",
        dest="save_xml",
        action="store_false",
        help="Skip saving TrackMate XML files (saves disk space)",
    )
    parser.add_argument(
        "--save-xml", action="store_true", default=True, help="Save TrackMate XML files"
    )

    return parser.parse_args()


def main():
    """Track one mask stack given on the command line."""
    args = parse_arguments()

    print("=" * 80)
    print("TRACKMATE ANALYZER FOR LPBF DATA")
    print("=" * 80)
    print(f"Input mask file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Memory allocation: {args.memory}")
    print(f"Save XML files: {args.save_xml}")
    print("=" * 80)

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    analyzer = TrackMateAnalyzer(memory=args.memory, headless=args.headless)

    success, spots_csv, tracks_csv = analyzer.process_image(
        input_path,
        output_dir,
        save_xml=args.save_xml,
        pixel_size_microns=4.3,
        frame_interval_ms=0.025,
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"ANALYSIS {'COMPLETED SUCCESSFULLY' if success else 'FAILED'}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    if success:
        print(f"Results saved to: {output_dir}")
        print(f"Spots CSV: {spots_csv}")
        print(f"Tracks CSV: {tracks_csv}")
    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
