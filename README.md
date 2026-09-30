# UCL LPBF spatter tracking

In this work, we use high-speed synchrotron X-ray imaging at beamline ID19 of the
ESRF to study spatter in laser powder bed fusion (LPBF) of an Al-Fe-Zr alloy
across 42 processing conditions. A modified U-Net segments the spatter in every
frame, TrackMate links the detections into tracks, and each track is analysed
for size, velocity and ejection angle. The results are then related to the laser
power, scan speed, laser mode (continuous wave or pulsed) and melting regime
through the normalised enthalpy. This repository contains the full pipeline,
from the raw radiographs to the figures in the paper.

![alt text](https://github.com/saatvikbasil/UCL-lpbf-spatter-tracking/blob/main/modified_unet/Architecture.png)

## Modified U-Net

The network follows the U-Net encoder-decoder design, with two encoder stages
(32 and 64 channels), a 128-channel bottleneck and two decoder stages. Each stage
is a block of 3×3, 1×1 and 3×3 convolutions with batch normalisation and ReLU,
with 2D dropout after the 1×1 convolution. The decoder upsamples with 2×2
transposed convolutions and concatenates the encoder features at the same
resolution, and a final 1×1 convolution gives a single output channel. The model
has 495,393 trainable parameters and takes 1024×512 radiographs. Its sigmoid
output is the probability that a pixel is background, so pixels at or below the
threshold are labelled as spatter. The masks are cleaned with a median filter
and a morphological opening, and small holes inside spatter are filled before
tracking. The implementation started from the PyImageSearch U-Net tutorial.

## Tracking and analysis

The mask stacks are calibrated to 4.3 µm per pixel and 0.025 ms per frame and
passed to TrackMate, which detects each spatter particle with its mask detector
and links the detections with a Kalman filter. Spots are filtered by size and
circularity and tracks by mean speed. Each track is then described by its size,
circularity, velocity, displacement, straightness and the ejection angle between
its first and fifth detections.

## Running the code

Set the data and model paths at the top of main() in
spatter_pipeline/run_folder_pipeline.py and run it from the repository root. It
segments, tracks and analyses every dataset folder.

```
python spatter_pipeline/run_folder_pipeline.py
python new_visuals.py
```

## Version information

- Python: 3.11
- torch: 1.13 or newer
