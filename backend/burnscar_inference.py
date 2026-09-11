"""
Real BurnScars inference using the reconstructed Prithvi-EO-2.0-300M
BurnScars model.

Usage from the SatQuery AI project root:

    .burnenv\Scripts\python.exe backend\burnscar_inference.py

Optional:

    .burnenv\Scripts\python.exe backend\burnscar_inference.py path\to\image.tif

Outputs:

    backend/burnscar_mask.png
    backend/burnscar_result.png
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from PIL import Image

from burnscar_model import BurnScarModel, load_checkpoint


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE = ROOT / "backend" / "Mexico_HLS_example.tif"

OUTPUT_MASK = ROOT / "backend" / "burnscar_mask.png"
OUTPUT_RESULT = ROOT / "backend" / "burnscar_result.png"

# Official BurnScars normalization statistics.
MEANS = np.array(
    [
        0.033349706741586264,
        0.05701185520536179,
        0.05889748132001316,
        0.2323245113436119,
        0.19728548537606537,
        0.11944914225186566,
    ],
    dtype=np.float32,
)

STDS = np.array(
    [
        0.02269135568823774,
        0.026807560223070237,
        0.04004109844362779,
        0.07791732423672637,
        0.08708738838140137,
        0.07241979477437814,
    ],
    dtype=np.float32,
)

# Prithvi spatial input used by the current backbone.
MODEL_SIZE = 224

BAND_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]


# ============================================================
# DATA LOADING
# ============================================================

def read_hls_image(image_path):
    """
    Read the six required HLS bands.

    Expected raster order:
        Blue, Green, Red, NIR_Narrow, SWIR1, SWIR2
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    print(f"\nReading HLS image:")
    print(f"  {image_path}")

    with rasterio.open(image_path) as src:

        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Bands: {src.count}")
        print(f"  Dtype: {src.dtypes}")

        if src.count < 6:
            raise ValueError(
                f"Expected at least 6 bands, found {src.count}."
            )

        data = src.read(
            indexes=[1, 2, 3, 4, 5, 6]
        ).astype(np.float32)

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": src.transform,
        }

    print(f"  Loaded bands: {BAND_NAMES}")

    return data, metadata


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image):
    """
    Prepare HLS image for BurnScars inference.

    HLS reflectance is commonly stored scaled by 10000.
    The BurnScars model expects reflectance in approximately
    [0,1], followed by the official mean/std normalization.
    """

    image = image.copy()

    finite_values = image[np.isfinite(image)]

    if finite_values.size == 0:
        raise ValueError("Input contains no finite pixel values.")

    maximum = float(np.max(finite_values))

    print(f"\nRaw maximum value: {maximum:.4f}")

    # HLS integer reflectance scaling.
    if maximum > 1.0:
        print("Detected scaled HLS reflectance -> dividing by 10000.")
        image = image / 10000.0
    else:
        print("Input already appears to be reflectance in [0,1].")

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Official BurnScars normalization.
    image = (
        image - MEANS[:, None, None]
    ) / STDS[:, None, None]

    tensor = torch.from_numpy(image)

    # [6,H,W] -> [1,6,1,H,W]
    tensor = tensor.unsqueeze(0).unsqueeze(2)

    original_height = tensor.shape[-2]
    original_width = tensor.shape[-1]

    # Current reconstructed Prithvi backbone uses 224x224.
    tensor = F.interpolate(
        tensor.squeeze(2),
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    tensor = tensor.unsqueeze(2)

    print(f"Model input shape: {tuple(tensor.shape)}")

    return tensor, original_height, original_width


# ============================================================
# RGB VISUALIZATION
# ============================================================

def make_rgb(image):
    """
    Create a display RGB image from:
        Red   = band 3
        Green = band 2
        Blue  = band 1
    """

    rgb = image[[2, 1, 0]].copy()

    # Robust percentile stretch for visualization.
    for channel in range(3):
        band = rgb[channel]

        valid = band[np.isfinite(band)]

        if valid.size == 0:
            rgb[channel] = 0
            continue

        low = np.percentile(valid, 2)
        high = np.percentile(valid, 98)

        if high <= low:
            rgb[channel] = 0
        else:
            rgb[channel] = (
                (band - low) / (high - low)
            )

    rgb = np.clip(
        rgb,
        0,
        1,
    )

    return np.transpose(
        rgb,
        (1, 2, 0),
    )


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, tensor):

    print("\nRunning Prithvi BurnScars inference...")

    model.eval()

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

    burn_probability = probabilities[
        0, 1
    ].cpu().numpy()

    mask = prediction[
        0
    ].cpu().numpy().astype(np.uint8)

    return mask, burn_probability


# ============================================================
# POSTPROCESSING
# ============================================================

def resize_prediction(mask, probability, height, width):

    mask_tensor = torch.from_numpy(
        mask.astype(np.float32)
    )[None, None]

    probability_tensor = torch.from_numpy(
        probability.astype(np.float32)
    )[None, None]

    mask = F.interpolate(
        mask_tensor,
        size=(height, width),
        mode="nearest",
    )[0, 0].numpy().astype(np.uint8)

    probability = F.interpolate(
        probability_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    return mask, probability


def create_overlay(rgb, mask):

    overlay = rgb.copy()

    # Burned pixels are highlighted.
    burned = mask == 1

    # Blend burned pixels with a red highlight.
    highlight = np.zeros_like(overlay)
    highlight[:, :, 0] = 1.0

    alpha = 0.55

    overlay[burned] = (
        (1 - alpha) * overlay[burned]
        + alpha * highlight[burned]
    )

    return np.clip(
        overlay,
        0,
        1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SatQuery AI - Prithvi BurnScars Inference")
    print("=" * 70)

    # --------------------------------------------------------
    # Input image
    # --------------------------------------------------------

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = DEFAULT_IMAGE

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image, metadata = read_hls_image(
        image_path
    )

    original_height = image.shape[1]
    original_width = image.shape[2]

    # --------------------------------------------------------
    # RGB before model preprocessing
    # --------------------------------------------------------

    reflectance = image.copy()

    if np.max(
        reflectance[
            np.isfinite(reflectance)
        ]
    ) > 1.0:
        reflectance = reflectance / 10000.0

    reflectance = np.nan_to_num(
        reflectance,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    rgb = make_rgb(
        reflectance
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    tensor, _, _ = preprocess(
        image
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = BurnScarModel()

    # --------------------------------------------------------
    # Load official checkpoint
    # --------------------------------------------------------

    load_checkpoint(
        model
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    mask, probability = run_inference(
        model,
        tensor,
    )

    # --------------------------------------------------------
    # Restore original image resolution
    # --------------------------------------------------------

    mask, probability = resize_prediction(
        mask,
        probability,
        original_height,
        original_width,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_pixels = mask.size

    burned_pixels = int(
        np.sum(mask == 1)
    )

    burned_percentage = (
        burned_pixels / total_pixels
    ) * 100.0

    mean_burn_probability = float(
        np.mean(probability)
    )

    mean_predicted_confidence = float(
        np.mean(
            np.maximum(
                probability,
                1.0 - probability,
            )
        )
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Image size           : "
        f"{original_width} x {original_height}"
    )

    print(
        f"Burned pixels        : "
        f"{burned_pixels:,}"
    )

    print(
        f"Total pixels         : "
        f"{total_pixels:,}"
    )

    print(
        f"Estimated burned area: "
        f"{burned_percentage:.2f}%"
    )

    print(
        f"Mean burn probability: "
        f"{mean_burn_probability:.4f}"
    )

    print(
        f"Mean confidence      : "
        f"{mean_predicted_confidence:.4f}"
    )

    # --------------------------------------------------------
    # Save binary mask
    # --------------------------------------------------------

    mask_image = (
        mask * 255
    ).astype(np.uint8)

    Image.fromarray(
        mask_image,
        mode="L",
    ).save(
        OUTPUT_MASK
    )

    # --------------------------------------------------------
    # Save RGB + burn overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        rgb,
        mask,
    )

    overlay_image = (
        overlay * 255
    ).astype(np.uint8)

    Image.fromarray(
        overlay_image,
        mode="RGB",
    ).save(
        OUTPUT_RESULT
    )

    print("\nSaved:")
    print(f"  Mask   : {OUTPUT_MASK}")
    print(f"  Result : {OUTPUT_RESULT}")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
