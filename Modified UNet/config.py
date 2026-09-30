"""Paths and hyperparameters for U-Net training and inference."""

import os

import torch

# Training data: <DATASET_PATH>/images and <DATASET_PATH>/masks
DATASET_PATH = os.path.join("dataset", "train")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

# Fraction of the training images held out for validation
TEST_SPLIT = 0.10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

INIT_LR = 0.0001
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Network input size in pixels
INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_HEIGHT = 512

# Threshold on the sigmoid output when binarising predicted masks
THRESHOLD = 0.35

BASE_OUTPUT = "output"
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_CP1_final.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")
