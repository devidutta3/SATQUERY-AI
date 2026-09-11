import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from PIL import Image

from burnscar_model import BurnScarModel, load_checkpoint


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE = ROOT / "backend" / "Mexico_HLS_example.tif"

OUTPUT_MASK = ROOT / "backend" / "burnscar_tiled_mask.png"
OUTPUT_OVERLAY = ROOT / "backend" / "burnscar_tiled_result.png"
OUTPUT_GEOTIFF = ROOT / "backend" / "burnscar_tiled_mask.tif"


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_SIZE = 224

BAND_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]


# Official BurnScars normalization
MEANS = np.array(
    [
        0.033349706741586264,
        0.05701185520536179,
        0.05889748132001316,
        0.2323245113436119,
        0.1972854853760658,
        0.11944914225186566,
    ],
    dtype=np.float32,
)

STDS = np.array(
    [
        0.02269135568823774,
        0.026807560223070237,
        0.04004109844362779,
        0.07791732423672691,
        0.08708738838140137,
        0.07241979477437814,
    ],
    dtype=np.float32,
)


# ============================================================
# READ HLS
# ============================================================

def read_hls(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    print("\nReading HLS image...")
    print(path)

    with rasterio.open(path) as src:

        if src.count < 6:
            raise ValueError(
                f"Expected 6 bands, found {src.count}"
            )

        image = src.read(
            [1, 2, 3, 4, 5, 6]
        ).astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

        width = src.width
        height = src.height

    print(f"Size  : {width} x {height}")
    print(f"Bands : {image.shape[0]}")
    print(f"CRS   : {crs}")

    return image, profile, transform, crs


# ============================================================
# REFLECTANCE
# ============================================================

def convert_to_reflectance(image):

    image = image.copy()

    valid = image[np.isfinite(image)]

    maximum = float(np.max(valid))

    if maximum > 1.0:

        print(
            f"Scaled reflectance detected "
            f"(max={maximum:.2f})"
        )

        image /= 10000.0

    else:

        print("Input already appears to be reflectance.")

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return image


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

    return (
        image - MEANS[:, None, None]
    ) / STDS[:, None, None]


# ============================================================
# RGB
# ============================================================

def create_rgb(image):

    rgb = image[
        [2, 1, 0]
    ].copy()

    for i in range(3):

        band = rgb[i]

        low = np.percentile(
            band,
            2
        )

        high = np.percentile(
            band,
            98
        )

        if high > low:

            rgb[i] = (
                band - low
            ) / (
                high - low
            )

    rgb = np.clip(
        rgb,
        0,
        1
    )

    return np.transpose(
        rgb,
        (1, 2, 0)
    )


# ============================================================
# TILE PADDING
# ============================================================

def prepare_tile(
    image,
    y,
    x,
):
    """
    Extract a MODEL_SIZE x MODEL_SIZE tile.

    Edge tiles are padded using reflection.
    """

    h, w = image.shape[1:]

    y2 = min(
        y + MODEL_SIZE,
        h
    )

    x2 = min(
        x + MODEL_SIZE,
        w
    )

    tile = image[
        :,
        y:y2,
        x:x2
    ]

    pad_h = MODEL_SIZE - tile.shape[1]
    pad_w = MODEL_SIZE - tile.shape[2]

    if pad_h > 0 or pad_w > 0:

        tile = np.pad(
            tile,
            (
                (0, 0),
                (0, pad_h),
                (0, pad_w),
            ),
            mode="reflect",
        )

    return tile


# ============================================================
# TILE INFERENCE
# ============================================================

def infer_tile(
    model,
    tile,
):

    # [6,H,W]
    tensor = torch.from_numpy(
        tile.astype(np.float32)
    )

    # [1,6,1,H,W]
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(2)

    with torch.inference_mode():

        logits = model(tensor)

        # The reconstructed decoder returns a lower-resolution
        # segmentation map (112x112 for a 224x224 input).
        # Resize logits back to tile resolution before stitching.
        if logits.shape[-2:] != (
            MODEL_SIZE,
            MODEL_SIZE,
        ):
            logits = F.interpolate(
                logits,
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        burn_probability = probabilities[
            0,
            1,
        ]

    return burn_probability.cpu().numpy()


# ============================================================
# TILED INFERENCE
# ============================================================

def tiled_inference(
    model,
    image,
):

    _, height, width = image.shape

    print("\nStarting tiled inference...")
    print(
        f"Full image: {width} x {height}"
    )

    # Normalize entire reflectance image first.
    normalized = normalize(
        image
    )

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    tile_number = 0

    for y in range(
        0,
        height,
        MODEL_SIZE
    ):

        for x in range(
            0,
            width,
            MODEL_SIZE
        ):

            tile_number += 1

            print(
                f"Processing tile "
                f"{tile_number} "
                f"at x={x}, y={y}"
            )

            tile = prepare_tile(
                normalized,
                y,
                x,
            )

            probability = infer_tile(
                model,
                tile,
            )

            valid_h = min(
                MODEL_SIZE,
                height - y
            )

            valid_w = min(
                MODEL_SIZE,
                width - x
            )

            probability = probability[
                :valid_h,
                :valid_w
            ]

            probability_map[
                y:y + valid_h,
                x:x + valid_w
            ] += probability

            count_map[
                y:y + valid_h,
                x:x + valid_w
            ] += 1.0

    count_map[
        count_map == 0
    ] = 1.0

    probability_map /= count_map

    mask = (
        probability_map >= 0.5
    ).astype(np.uint8)

    return mask, probability_map


# ============================================================
# SAVE PNG MASK
# ============================================================

def save_mask(mask):

    Image.fromarray(
        mask * 255,
        mode="L"
    ).save(
        OUTPUT_MASK
    )

    print(
        f"\nMask saved:\n{OUTPUT_MASK}"
    )


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_geotiff(
    mask,
    profile,
):

    profile = profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
        }
    )

    with rasterio.open(
        OUTPUT_GEOTIFF,
        "w",
        **profile
    ) as dst:

        dst.write(
            mask,
            1
        )

    print(
        f"GeoTIFF saved:\n{OUTPUT_GEOTIFF}"
    )


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_overlay(
    rgb,
    mask,
):

    overlay = rgb.copy()

    burned = (
        mask == 1
    )

    # Red highlight
    highlight = np.zeros_like(
        overlay
    )

    highlight[:, :, 0] = 1.0

    alpha = 0.55

    overlay[burned] = (
        (1 - alpha)
        * overlay[burned]
        +
        alpha
        * highlight[burned]
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    Image.fromarray(
        (overlay * 255).astype(
            np.uint8
        ),
        mode="RGB"
    ).save(
        OUTPUT_OVERLAY
    )

    print(
        f"Overlay saved:\n{OUTPUT_OVERLAY}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SatQuery AI - Tiled Prithvi BurnScars Inference"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        image_path = Path(
            sys.argv[1]
        )

    else:

        image_path = DEFAULT_IMAGE

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    raw_image, profile, transform, crs = read_hls(
        image_path
    )

    # --------------------------------------------------------
    # REFLECTANCE
    # --------------------------------------------------------

    image = convert_to_reflectance(
        raw_image
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = create_rgb(
        image
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nCreating BurnScars model...")

    model = BurnScarModel()

    load_checkpoint(
        model
    )

    model.eval()

    # --------------------------------------------------------
    # TILED INFERENCE
    # --------------------------------------------------------

    mask, probability = tiled_inference(
        model,
        image
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_pixels = mask.size

    burned_pixels = int(
        np.sum(mask == 1)
    )

    burned_percentage = (
        burned_pixels
        /
        total_pixels
    ) * 100.0

    mean_probability = float(
        probability.mean()
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TILED INFERENCE RESULT")
    print("=" * 70)

    print(
        f"Image size           : "
        f"{mask.shape[1]} x {mask.shape[0]}"
    )

    print(
        f"Tiles processed      : "
        f"{int(np.ceil(mask.shape[0] / MODEL_SIZE)) * int(np.ceil(mask.shape[1] / MODEL_SIZE))}"
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
        f"{mean_probability:.4f}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_mask(
        mask
    )

    save_overlay(
        rgb,
        mask
    )

    save_geotiff(
        mask,
        profile
    )

    print("\n")
    print("=" * 70)
    print("TILED INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()