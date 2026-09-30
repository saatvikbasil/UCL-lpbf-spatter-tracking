#!/usr/bin/env python3
"""Run the spatter pipeline on every dataset folder in a batch directory.

Each sub-folder of the batch input that contains images goes through three
stages, each run as a subprocess:

1. predict_tiff_mask.py  - U-Net segmentation, saved as a TIFF mask stack
2. spatter_tracker.py    - TrackMate detection and Kalman tracking
3. spatter_analysis.py   - per-track metrics, figures and a summary report

Set the paths at the top of main(), then run:

    python spatter_pipeline/run_folder_pipeline.py

"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent


def run_direct_cmd(cmd, description):
    """Run a command with its output going straight to the console.

    Returns True if the command exits with code 0.
    """
    print(f"Starting: {description} (DIRECT MODE)")
    print(f"Command: {' '.join(cmd)}", flush=True)

    try:
        start_time = time.time()
        result = subprocess.call(cmd)
        elapsed_time = time.time() - start_time

        if result == 0:
            print(f"Completed: {description} in {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"Failed: {description} (return code: {result})")
            return False

    except Exception as e:
        print(f"Error running {description}: {e}")
        print(traceback.format_exc())
        return False


def run_command(command, description):
    """Run a command and print its stdout/stderr once it has finished.

    Returns True if the command exits with code 0.
    """
    print(f"Starting: {description}")
    print(f"Command: {' '.join(command)}", flush=True)

    try:
        start_time = time.time()

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if process.stdout:
            for line in process.stdout.splitlines():
                if line.strip():
                    print(f"> {line.strip()}")

        if process.stderr:
            for line in process.stderr.splitlines():
                if line.strip():
                    print(f"! {line.strip()}")

        elapsed_time = time.time() - start_time

        if process.returncode == 0:
            print(f"Completed: {description} in {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"Failed: {description} (return code: {process.returncode})")
            return False

    except Exception as e:
        print(f"Error running {description}: {e}")
        print(traceback.format_exc())
        return False


def find_latest_file(directory, pattern):
    """Return the most recently modified file in ``directory`` matching ``pattern``.

    Falls back to a case-insensitive match, since Path.glob is case-sensitive.
    If nothing matches, lists the directory contents and returns None.
    """
    dir_path = Path(directory)
    print(f"Looking for files matching '{pattern}' in {dir_path}")

    matching_files = list(dir_path.glob(pattern))

    if not matching_files:
        print("No matches found with direct pattern. Trying case-insensitive search...")
        all_files = list(dir_path.glob("*"))
        pattern_regex = re.compile(pattern.replace("*", ".*"), re.IGNORECASE)
        matching_files = [f for f in all_files if pattern_regex.search(f.name)]

    if not matching_files:
        print(f"No matches found. Contents of directory {dir_path}:")
        all_files = list(dir_path.glob("*"))
        for f in all_files:
            print(f"  - {f.name}")
        return None

    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    print(f"Found latest file: {latest_file}")
    return latest_file


def find_dataset_folders(batch_input_dir):
    """Return the sub-folders of ``batch_input_dir`` that contain images, sorted."""
    batch_path = Path(batch_input_dir)
    dataset_folders = []

    if not batch_path.exists():
        print(f"Error: Batch input directory does not exist: {batch_input_dir}")
        return []

    for item in batch_path.iterdir():
        if item.is_dir():
            image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
            has_images = any(list(item.glob(ext)) for ext in image_extensions)

            if has_images:
                dataset_folders.append(item)
                print(f"Found dataset folder: {item.name}")
            else:
                print(f"Skipping folder (no images found): {item.name}")

    if not dataset_folders:
        print(f"No dataset folders with images found in: {batch_input_dir}")

    return sorted(dataset_folders)


def create_output_dirs(base_output_dir, dataset_name, timestamp):
    """Create <base>/<dataset>/{masks,tracking,analysis} and return the four paths."""
    output_dir = Path(base_output_dir) / dataset_name
    masks_dir = output_dir / "masks"
    tracking_dir = output_dir / "tracking"
    analysis_dir = output_dir / "analysis"

    for directory in [output_dir, masks_dir, tracking_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    return output_dir, masks_dir, tracking_dir, analysis_dir


def process_single_dataset(input_folder, output_dir, args, dataset_name):
    """Run segmentation, tracking and analysis for one dataset folder.

    Returns True if all three stages succeed.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir, masks_dir, tracking_dir, analysis_dir = create_output_dirs(
            args.batch_output, dataset_name, timestamp
        )

        print(f"\n{'=' * 80}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'=' * 80}\n")

        print(f"Input folder: {input_folder}")
        print(f"Output directory: {output_dir}")
        print(f"Model path: {args.model}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        input_basename = Path(input_folder).name.replace(" ", "_")
        mask_file = masks_dir / f"{input_basename}_predicted.tif"

        print("=" * 50)
        print("STEP 1/3: GENERATING SEGMENTATION MASKS")
        print("=" * 50)

        # Windows paths longer than 260 characters need the short (8.3) form.
        if platform.system() == "Windows":
            mask_file_str = str(mask_file)
            if len(mask_file_str) > 260:
                mask_file_str = os.path.abspath(mask_file_str)
                try:
                    import win32api

                    mask_file_str = win32api.GetShortPathName(mask_file_str)
                    print(f"Using short path for mask file: {mask_file_str}")
                except ImportError:
                    print("win32api not available, using long path (may cause issues)")
        else:
            mask_file_str = str(mask_file)

        predict_cmd = [
            sys.executable,
            str(PIPELINE_DIR / "predict_tiff_mask.py"),
            "--input",
            str(input_folder),
            "--output",
            mask_file_str,
            "--model",
            str(args.model),
            "--threshold",
            str(args.threshold),
            "--min-area",
            str(args.min_area),
            "--batch-size",
            str(args.batch_size),
        ]

        if args.fast:
            success = run_direct_cmd(predict_cmd, "Mask prediction (U-Net)")
        else:
            success = run_command(predict_cmd, "Mask prediction (U-Net)")

        if not success:
            print(f"Mask prediction failed for {dataset_name}. Skipping this dataset.")
            return False

        print(f"Mask prediction completed. Output saved to: {mask_file}")

        print(f"\n{'=' * 50}")
        print("STEP 2/3: TRACKING PARTICLES WITH TRACKMATE")
        print("=" * 50)

        tracker_cmd = [
            sys.executable,
            str(PIPELINE_DIR / "spatter_tracker.py"),
            "--input",
            str(mask_file),
            "--output",
            str(tracking_dir),
            "--memory",
            args.memory,
        ]

        if not args.no_save_xml:
            tracker_cmd.append("--save-xml")

        if args.fast:
            success = run_direct_cmd(tracker_cmd, "Particle tracking (TrackMate)")
        else:
            success = run_command(tracker_cmd, "Particle tracking (TrackMate)")

        if not success:
            print(f"Particle tracking failed for {dataset_name}. Skipping this dataset.")
            return False

        # The tracker's CSVs are looked for in the tracking folder, then in its
        # sub-folders, then in its parent.
        print(f"\nSearching for tracking results in: {tracking_dir}")

        spots_csv = find_latest_file(tracking_dir, "*spots*.csv")
        tracks_csv = find_latest_file(tracking_dir, "*tracks*.csv")

        if not spots_csv or not tracks_csv:
            print("Searching subdirectories...")
            for subdir in Path(tracking_dir).glob("**/"):
                if not spots_csv:
                    spots_csv = find_latest_file(subdir, "*spots*.csv")
                if not tracks_csv:
                    tracks_csv = find_latest_file(subdir, "*tracks*.csv")
                if spots_csv and tracks_csv:
                    break

        if not spots_csv or not tracks_csv:
            parent_dir = Path(tracking_dir).parent
            print(f"Searching parent directory: {parent_dir}")
            if not spots_csv:
                spots_csv = find_latest_file(parent_dir, "*spots*.csv")
            if not tracks_csv:
                tracks_csv = find_latest_file(parent_dir, "*tracks*.csv")

        if not spots_csv or not tracks_csv:
            print(f"Could not find tracking CSV files for {dataset_name}. Skipping this dataset.")
            print(f"Please check the output directory: {tracking_dir}")
            return False

        print(f"Tracking data saved to: {spots_csv} and {tracks_csv}")

        print(f"\n{'=' * 50}")
        print("STEP 3/3: ANALYZING SPATTER DATA")
        print("=" * 50)

        analysis_cmd = [
            sys.executable,
            str(PIPELINE_DIR / "spatter_analysis.py"),
            "--spots",
            str(spots_csv),
            "--tracks",
            str(tracks_csv),
            "--output",
            str(analysis_dir),
        ]

        if args.fast:
            success = run_direct_cmd(analysis_cmd, "Spatter analysis")
        else:
            success = run_command(analysis_cmd, "Spatter analysis")

        if not success:
            print(f"Spatter analysis failed for {dataset_name}.")
            return False

        summary_report = os.path.join(analysis_dir, "spatter_summary_report.txt")
        metrics_csv = os.path.join(analysis_dir, "spatter_metrics.csv")

        if os.path.exists(summary_report):
            shutil.copy2(summary_report, os.path.join(output_dir, "spatter_summary_report.txt"))

        if os.path.exists(metrics_csv):
            shutil.copy2(metrics_csv, os.path.join(output_dir, "spatter_metrics.csv"))

        print(f"\n{'=' * 80}")
        print(f"DATASET {dataset_name} COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Input folder: {input_folder}")
        print(f"Output directory: {output_dir}")
        print(f"Segmentation mask: {mask_file}")
        print(f"Tracking data: {tracking_dir}")
        print(f"Analysis results: {analysis_dir}")
        print(f"Summary report: {os.path.join(output_dir, 'spatter_summary_report.txt')}")
        print(f"Metrics CSV: {os.path.join(output_dir, 'spatter_metrics.csv')}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"Pipeline error for {dataset_name}: {e}")
        print(traceback.format_exc())
        return False


def parse_arguments():
    """Parse the batch pipeline's command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LPBF Spatter Analysis Pipeline - Batch Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--batch-input",
        required=True,
        help="Path to batch folder containing multiple dataset folders",
    )

    parser.add_argument(
        "--batch-output", default="BatchResults", help="Path to save all batch analysis results"
    )
    parser.add_argument("--model", default="unet_CP1_final.pth", help="Path to trained U-Net model")
    parser.add_argument("--memory", default="8G", help="Java memory allocation for TrackMate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Sigmoid threshold passed to predict_tiff_mask.py (above = background)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Mask regions smaller than this (pixels) are removed",
    )
    parser.add_argument(
        "--no-save-xml",
        action="store_true",
        help="Skip saving TrackMate XML (saves disk space)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Let each stage print straight to the console instead of capturing its output",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other datasets if one fails",
    )

    return parser.parse_args()


def run_batch_pipeline():
    """Process every dataset folder and write batch_summary.txt.

    Returns True if no dataset failed.
    """
    args = parse_arguments()

    dataset_folders = find_dataset_folders(args.batch_input)

    if not dataset_folders:
        print(f"No dataset folders found in {args.batch_input}")
        return False

    Path(args.batch_output).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 100}")
    print("LPBF SPATTER ANALYSIS PIPELINE - BATCH PROCESSING v1.0")
    print(f"{'=' * 100}\n")

    print(f"Batch input folder: {args.batch_input}")
    print(f"Batch output directory: {args.batch_output}")
    print(f"Model path: {args.model}")
    print(f"Found {len(dataset_folders)} datasets to process")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    successful_datasets = []
    failed_datasets = []

    for i, dataset_folder in enumerate(dataset_folders, 1):
        dataset_name = dataset_folder.name

        print(f"\n{'=' * 100}")
        print(f"PROCESSING DATASET {i}/{len(dataset_folders)}: {dataset_name}")
        print("=" * 100)

        try:
            success = process_single_dataset(dataset_folder, args.batch_output, args, dataset_name)

            if success:
                successful_datasets.append(dataset_name)
                print(f"+ Successfully processed: {dataset_name}")
            else:
                failed_datasets.append(dataset_name)
                print(f"x Failed to process: {dataset_name}")

                if not args.continue_on_error:
                    print(
                        "Stopping batch processing due to error. "
                        "Use --continue-on-error to continue processing other datasets."
                    )
                    break

        except Exception as e:
            failed_datasets.append(dataset_name)
            print(f"x Error processing {dataset_name}: {e}")

            if not args.continue_on_error:
                print(
                    "Stopping batch processing due to error. "
                    "Use --continue-on-error to continue processing other datasets."
                )
                break

    print(f"\n{'=' * 100}")
    print("BATCH PROCESSING COMPLETED")
    print("=" * 100)
    print(f"Total datasets: {len(dataset_folders)}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"Batch output directory: {args.batch_output}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if successful_datasets:
        print("\nSuccessfully processed datasets:")
        for dataset in successful_datasets:
            print(f"  + {dataset}")

    if failed_datasets:
        print("\nFailed datasets:")
        for dataset in failed_datasets:
            print(f"  x {dataset}")

    print("=" * 100)

    summary_file = Path(args.batch_output) / "batch_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("LPBF Spatter Analysis Pipeline - Batch Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Batch input folder: {args.batch_input}\n")
        f.write(f"Batch output directory: {args.batch_output}\n")
        f.write(f"Model path: {args.model}\n\n")
        f.write(f"Total datasets: {len(dataset_folders)}\n")
        f.write(f"Successful: {len(successful_datasets)}\n")
        f.write(f"Failed: {len(failed_datasets)}\n\n")

        if successful_datasets:
            f.write("Successfully processed datasets:\n")
            for dataset in successful_datasets:
                f.write(f"  + {dataset}\n")
            f.write("\n")

        if failed_datasets:
            f.write("Failed datasets:\n")
            for dataset in failed_datasets:
                f.write(f"  x {dataset}\n")

    print(f"Batch summary saved to: {summary_file}")

    return len(failed_datasets) == 0


def main():
    """Run the batch pipeline on the paths below.

    These values are handed to the argument parser in place of any
    command-line arguments.
    """
    BATCH_INPUT_FOLDER = "Data/test"  # one sub-folder per dataset
    BATCH_OUTPUT_DIR = "Results/FinalUpdatedBatch"
    MODEL_PATH = "model_weights/Modified_Unet.pth"
    MEMORY = "16G"  # Java heap for TrackMate

    sys.argv = [
        sys.argv[0],
        "--batch-input",
        BATCH_INPUT_FOLDER,
        "--batch-output",
        BATCH_OUTPUT_DIR,
        "--model",
        MODEL_PATH,
        "--memory",
        MEMORY,
        "--fast",
        "--continue-on-error",
    ]

    start_time = time.time()
    success = run_batch_pipeline()
    elapsed_time = time.time() - start_time

    print(
        f"\nTotal batch processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)"
    )
    print(f"Batch pipeline {'succeeded' if success else 'had some failures'}")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()  # os._exit skips the normal flush
    os._exit(exit_code)
