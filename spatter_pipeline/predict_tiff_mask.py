#!/usr/bin/env python3
"""Segment spatter in a folder of X-ray frames with the trained U-Net.

Every image in the input folder (searched recursively, in sorted order) is
resized to 1024 x 512 and passed through the model, whose sigmoid output is the
probability that a pixel is background. Pixels above --threshold form a
background mask, which is cleaned with a median filter, a 2 x 2 morphological
opening (skipped with --no-morphology) and removal of regions smaller than
--min-area pixels. The masks are written to a single ImageJ-compatible TIFF
stack, which by default is then inverted so that spatter is white (255) on a
black background.

Usage:
    python spatter_pipeline/predict_tiff_mask.py --input input_folder --output output_stack.tif \
        --model model_weights/Modified_Unet.pth
"""

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import cv2
import tifffile
from tqdm import tqdm

# modified_unet/ sits in the repository root, one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modified_unet.model  # noqa: E402
from modified_unet import config  # noqa: E402

# The saved model was pickled when modified_unet was still called pyimagesearch,
# so register the old name for torch.load to find the classes.
sys.modules.setdefault("pyimagesearch", modified_unet)
sys.modules.setdefault("pyimagesearch.model", modified_unet.model)

DEFAULT_BATCH_SIZE = 16
TARGET_WIDTH = 1024
TARGET_HEIGHT = 512


def load_model(model_path, device):
    """Load the saved model in evaluation mode; return None on failure.

    The checkpoint holds the whole pickled UNet rather than a state dict, so
    the modified_unet classes must be importable (see the alias above).
    """
    print(f"Loading model from {model_path}...")
    try:
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Trying CPU loading with explicit map_location...")

            model = torch.load(model_path, map_location="cpu", weights_only=False)
            model = model.to(device)
            print("Successfully loaded model using CPU loading")

        model.eval()
        print("Model loaded successfully and set to evaluation mode")
        return model

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return None


def preprocess_batch(image_paths, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """Load images as float32 CHW arrays scaled to [0, 1] and resized.

    Grayscale images are replicated to three channels. Returns the processed
    images and the paths that could not be read.
    """
    batch_data = []
    failed_paths = []

    for image_path in image_paths:
        try:
            image_path_str = str(image_path).lower()

            if image_path_str.endswith((".tif", ".tiff")):
                image = tifffile.imread(image_path)
            else:
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image is None:
                failed_paths.append(image_path)
                continue

            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=2)
            elif image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)

            image = image.astype(np.float32) / 255.0
            image = cv2.resize(image, (target_width, target_height))
            image = np.transpose(image, (2, 0, 1))

            batch_data.append(image)

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            failed_paths.append(image_path)

    return batch_data, failed_paths


def process_batch(batch_data, model, device, threshold=0.2):
    """Predict background masks (sigmoid > threshold) at TARGET_WIDTH x TARGET_HEIGHT."""
    if not batch_data:
        return []

    batch_tensor = torch.from_numpy(np.array(batch_data)).to(device)

    with torch.no_grad():
        batch_predictions = model(batch_tensor)
        batch_predictions = torch.sigmoid(batch_predictions)
        batch_masks = (batch_predictions > threshold).cpu().numpy()

        if len(batch_masks.shape) == 3:
            batch_masks = batch_masks[:, None, :, :]

        batch_masks = batch_masks.squeeze(1)

    upscaled_masks = []
    for mask in batch_masks:
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Nearest-neighbour keeps the mask binary
        upscaled_mask = cv2.resize(
            mask_uint8, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        upscaled_mask = (upscaled_mask > 127).astype(np.uint8)

        upscaled_masks.append(upscaled_mask)

    return upscaled_masks


def postprocess_masks(
    masks, min_area=10, apply_morphology=False, invert=True, median_kernel_size=3
):
    """Clean binary masks and return them as uint8 (0/255) images.

    Applies a median filter, an optional 2 x 2 morphological opening and
    removes connected components smaller than ``min_area`` pixels. With
    ``invert`` set, the result is inverted.
    """
    processed_masks = []

    for mask in masks:
        binary_mask = (mask > 0).astype(np.uint8)
        binary_mask = binary_mask * 255

        binary_mask = cv2.medianBlur(binary_mask, median_kernel_size)

        if apply_morphology:
            kernel = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        if min_area > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            filtered_mask = np.zeros_like(binary_mask)
            for i in range(1, num_labels):  # label 0 is the background
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered_mask[labels == i] = 255

            binary_mask = filtered_mask

        if invert:
            binary_mask = 255 - binary_mask

        processed_masks.append(binary_mask)

    return processed_masks


def write_tiff_stack(mask_stack, output_path):
    """Write masks to an ImageJ TIFF with the frames along the time axis.

    Returns True on success.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if len(mask_stack.shape) == 3:
            total_frames, height, width = mask_stack.shape
        else:
            total_frames = 1
            height, width = mask_stack.shape
            mask_stack = mask_stack.reshape(1, height, width)

        print(f"Writing {total_frames} frames ({width}x{height}) to {output_path}...")

        metadata = {
            "axes": "TYX",
            "frames": total_frames,
            "slices": 1,
            "channels": 1,
            "hyperstack": True,
            "mode": "grayscale",
            "loop": False,
        }

        tifffile.imwrite(
            output_path,
            mask_stack,
            imagej=True,
            metadata=metadata,
            photometric="minisblack",
            resolution=(1.0, 1.0),
            resolutionunit="NONE",
        )

        print(f"Successfully saved {total_frames} frames to {output_path}")
        return True

    except Exception as e:
        print(f"Error writing TIFF file: {e}")
        traceback.print_exc()
        return False


def invert_tiff_stack(tiff_path):
    """Invert an 8-bit TIFF stack in place (255 - value). Returns True on success."""
    try:
        print(f"Inverting TIFF stack: {tiff_path}")

        mask_stack = tifffile.imread(tiff_path)

        print(f"Stack shape: {mask_stack.shape}")
        print(f"Original values - min: {mask_stack.min()}, max: {mask_stack.max()}")

        inverted_stack = 255 - mask_stack

        tifffile.imwrite(
            tiff_path,
            inverted_stack,
            imagej=True,
            metadata={"axes": "TYX"} if len(mask_stack.shape) > 2 else None,
            photometric="minisblack",
        )

        print(f"Inverted TIFF stack saved to: {tiff_path}")
        print(f"Inverted values - min: {inverted_stack.min()}, max: {inverted_stack.max()}")
        return True

    except Exception as e:
        print(f"Error inverting TIFF stack: {e}")
        traceback.print_exc()
        return False


def process_folder(
    input_folder,
    output_path,
    model_path,
    threshold=0.35,
    min_area=10,
    apply_morphology=False,
    invert_masks=True,
    extensions=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    batch_size=DEFAULT_BATCH_SIZE,
    force_final_invert=True,
):
    """Segment every image in ``input_folder`` and save the masks as one TIFF stack.

    Images go through the model in batches of ``batch_size``; all masks are
    kept in memory and written once at the end. If ``force_final_invert`` is set
    and the masks were not inverted during processing, the saved stack is
    inverted. Returns True if at least one image was processed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(model_path, device)
    if model is None:
        return False

    input_path = Path(input_folder)

    extensions = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    )

    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(f"**/*{ext}")))

    if not image_files:
        print(f"ERROR: No image files found in {input_folder} with extensions {extensions}")
        return False

    image_files = sorted(image_files)
    total_files = len(image_files)
    print(f"Found {total_files} image files to process")

    # One byte per pixel per mask
    estimated_memory_mb = (total_files * TARGET_WIDTH * TARGET_HEIGHT) / (1024 * 1024)
    print(f"Estimated memory usage for all masks: ~{estimated_memory_mb:.1f} MB")

    if estimated_memory_mb > 1000:
        print("WARNING: Processing all images at once may require significant memory!")

    start_time = time.time()
    all_masks = []
    processed_count = 0
    failed_count = 0

    progress_bar = tqdm(total=total_files, desc="Processing images")

    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        current_batch_files = image_files[start_idx:end_idx]

        batch_data, failed_paths = preprocess_batch(current_batch_files)
        failed_count += len(failed_paths)

        if not batch_data:
            progress_bar.update(len(current_batch_files))
            continue

        batch_masks = process_batch(batch_data, model, device, threshold)

        processed_masks = postprocess_masks(
            batch_masks,
            min_area=min_area,
            apply_morphology=apply_morphology,
            invert=invert_masks,
        )

        if processed_masks:
            all_masks.extend(processed_masks)

        del batch_data, batch_masks, processed_masks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        processed_count += len(current_batch_files) - len(failed_paths)
        progress_bar.update(len(current_batch_files))

    progress_bar.close()

    if all_masks:
        print("Converting all masks to a single array...")
        mask_stack = np.array(all_masks)

        del all_masks
        gc.collect()

        print("Writing all masks to disk...")
        success = write_tiff_stack(mask_stack, output_path)

        del mask_stack
        gc.collect()

        if success and force_final_invert and not invert_masks:
            print("Performing final mask inversion...")
            invert_tiff_stack(output_path)
    else:
        print("ERROR: No masks were successfully processed")
        return False

    elapsed_time = time.time() - start_time
    processing_speed = processed_count / elapsed_time if elapsed_time > 0 else 0

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print(f"Processed {processed_count} images in {elapsed_time:.2f} seconds")
    print(f"Processing speed: {processing_speed:.2f} images/second")

    if failed_count > 0:
        print(f"Warning: Failed to process {failed_count} images")

    print(f"Output saved to: {output_path}")
    print("=" * 80)

    return processed_count > 0


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LPBF Spatter Mask Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Path to input folder containing images")
    parser.add_argument("--output", required=True, help="Path to save output TIFF stack file")

    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to trained model")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Pixels with a sigmoid output above this are background (0.0-1.0)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Mask regions smaller than this (pixels) are removed",
    )
    parser.add_argument(
        "--no-morphology", action="store_true", help="Disable morphological operations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing"
    )

    parser.add_argument(
        "--extensions",
        default=".tif,.tiff,.png,.jpg,.jpeg",
        help="Comma-separated list of file extensions",
    )

    parser.add_argument("--invert", action="store_true", help="Invert masks during processing")
    parser.add_argument(
        "--no-final-invert", action="store_true", help="Disable final mask inversion"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    output_path = args.output
    if not str(output_path).lower().endswith((".tif", ".tiff")):
        output_path = f"{output_path}.tif"
        print(f"Added .tif extension to output path: {output_path}")

    extensions = tuple(args.extensions.split(","))

    success = process_folder(
        input_folder=args.input,
        output_path=output_path,
        model_path=args.model,
        threshold=args.threshold,
        min_area=args.min_area,
        apply_morphology=not args.no_morphology,
        invert_masks=args.invert,
        extensions=extensions,
        batch_size=args.batch_size,
        force_final_invert=not args.no_final_invert,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
