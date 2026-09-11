import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from PIL import Image

from .burnscar_model import BurnScarModel, load_checkpoint


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE = ROOT / "backend" / "Mexico_HLS_example.tif"

OUTPUT_MASK = ROOT / "backend" / "burnscar_overlap_mask.png"
OUTPUT_OVERLAY = ROOT / "backend" / "burnscar_overlap_result.png"
OUTPUT_GEOTIFF = ROOT / "backend" / "burnscar_overlap_mask.tif"


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_SIZE = 224

# 50% overlap
STRIDE = MODEL_SIZE // 2

BAND_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]


# ============================================================
# OFFICIAL BURNSCARS NORMALIZATION
# ============================================================

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
# READ HLS IMAGE
# ============================================================

def read_hls(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{path}"
        )

    print("\nReading HLS image...")
    print(path)

    with rasterio.open(path) as src:

        if src.count < 6:
            raise ValueError(
                f"Expected at least 6 bands, found {src.count}"
            )

        image = src.read(
            [1, 2, 3, 4, 5, 6]
        ).astype(np.float32)

        profile = src.profile.copy()

        width = src.width
        height = src.height

        crs = src.crs

    print(f"Size  : {width} x {height}")
    print(f"Bands : {image.shape[0]}")
    print(f"CRS   : {crs}")

    return image, profile


# ============================================================
# CONVERT TO REFLECTANCE
# ============================================================

def convert_to_reflectance(image):

    image = image.copy()

    valid = image[np.isfinite(image)]

    if valid.size == 0:
        raise ValueError(
            "Image contains no valid pixels."
        )

    maximum = float(
        np.max(valid)
    )

    if maximum > 1.0:

        print(
            f"Scaled reflectance detected "
            f"(max={maximum:.2f})"
        )

        image /= 10000.0

    else:

        print(
            "Input already appears to be reflectance."
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return image


# ============================================================
# NORMALIZE
# ============================================================

def normalize(image):

    return (
        image - MEANS[:, None, None]
    ) / STDS[:, None, None]


# ============================================================
# RGB VISUALIZATION
# ============================================================

def create_rgb(image):

    # HLS:
    # 0 = Blue
    # 1 = Green
    # 2 = Red

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
# TILE POSITIONS
# ============================================================

def get_positions(length):

    """
    Generate tile start positions.

    Guarantees that the final tile reaches
    the image boundary.
    """

    if length <= MODEL_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - MODEL_SIZE + 1,
            STRIDE,
        )
    )

    last_position = (
        length - MODEL_SIZE
    )

    if positions[-1] != last_position:
        positions.append(
            last_position
        )

    return positions


# ============================================================
# PREPARE TILE
# ============================================================

def prepare_tile(
    image,
    y,
    x,
):

    _, height, width = image.shape

    y2 = min(
        y + MODEL_SIZE,
        height
    )

    x2 = min(
        x + MODEL_SIZE,
        width
    )

    tile = image[
        :,
        y:y2,
        x:x2
    ]

    pad_h = (
        MODEL_SIZE
        - tile.shape[1]
    )

    pad_w = (
        MODEL_SIZE
        - tile.shape[2]
    )

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
# MODEL INFERENCE ON ONE TILE
# ============================================================

def infer_tile(
    model,
    tile,
):

    tensor = torch.from_numpy(
        tile.astype(np.float32)
    )

    # [6,H,W]
    #     ↓
    # [1,6,1,H,W]

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(2)

    with torch.inference_mode():

        logits = model(
            tensor
        )

        # Current decoder output is 112x112.
        # Restore to 224x224 before stitching.
        if logits.shape[-2:] != (
            MODEL_SIZE,
            MODEL_SIZE,
        ):

            logits = F.interpolate(
                logits,
                size=(
                    MODEL_SIZE,
                    MODEL_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        burn_probability = probabilities[
            0,
            1
        ]

    return burn_probability.cpu().numpy()


# ============================================================
# OVERLAPPING TILED INFERENCE
# ============================================================

def tiled_inference(
    model,
    image,
):

    _, height, width = image.shape

    normalized = normalize(
        image
    )

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    y_positions = get_positions(
        height
    )

    x_positions = get_positions(
        width
    )

    total_tiles = (
        len(y_positions)
        *
        len(x_positions)
    )

    print("\nStarting overlapping tiled inference...")
    print(
        f"Tile size : {MODEL_SIZE} x {MODEL_SIZE}"
    )
    print(
        f"Stride    : {STRIDE}"
    )
    print(
        f"Overlap   : 50%"
    )
    print(
        f"Tiles     : {total_tiles}"
    )

    tile_number = 0

    for y in y_positions:

        for x in x_positions:

            tile_number += 1

            print(
                f"Processing tile "
                f"{tile_number}/{total_tiles} "
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

            probability_sum[
                y:y + valid_h,
                x:x + valid_w
            ] += probability

            weight_sum[
                y:y + valid_h,
                x:x + valid_w
            ] += 1.0

    weight_sum[
        weight_sum == 0
    ] = 1.0

    probability_map = (
        probability_sum
        /
        weight_sum
    )

    # Binary burn mask
    mask = (
        probability_map >= 0.5
    ).astype(np.uint8)

    return (
        mask,
        probability_map,
        total_tiles,
    )


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(mask):

    Image.fromarray(
        mask * 255,
        mode="L"
    ).save(
        OUTPUT_MASK
    )

    print(
        f"\nMask saved:"
        f"\n{OUTPUT_MASK}"
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
        f"GeoTIFF saved:"
        f"\n{OUTPUT_GEOTIFF}"
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
        (
            overlay * 255
        ).astype(np.uint8),
        mode="RGB"
    ).save(
        OUTPUT_OVERLAY
    )

    print(
        f"Overlay saved:"
        f"\n{OUTPUT_OVERLAY}"
    )

# ============================================================
# FASTAPI / REUSABLE INFERENCE
# ============================================================

def run_burnscar_inference(image_path, output_dir=None):

    image_path = Path(image_path)

    if output_dir is None:
        output_dir = ROOT / "backend" / "outputs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # Output paths
    # ----------------------------------------------

    output_mask = output_dir / "burnscar_mask.png"
    output_overlay = output_dir / "burnscar_result.png"
    output_geotiff = output_dir / "burnscar_mask.tif"

    # ----------------------------------------------
    # Read image
    # ----------------------------------------------

    raw_image, profile = read_hls(
        image_path
    )

    # ----------------------------------------------
    # Convert to reflectance
    # ----------------------------------------------

    image = convert_to_reflectance(
        raw_image
    )

    # ----------------------------------------------
    # RGB visualization
    # ----------------------------------------------

    rgb = create_rgb(
        image
    )

    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    model = BurnScarModel()

    load_checkpoint(
        model
    )

    model.eval()

    # ----------------------------------------------
    # Inference
    # ----------------------------------------------

    mask, probability, total_tiles = tiled_inference(
        model,
        image
    )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    total_pixels = mask.size

    burned_pixels = int(
        np.sum(mask == 1)
    )

    burned_percentage = (
        burned_pixels / total_pixels
    ) * 100.0

    mean_probability = float(
        probability.mean()
    )

    # ----------------------------------------------
    # Save mask
    # ----------------------------------------------

    Image.fromarray(
        mask * 255,
        mode="L"
    ).save(
        output_mask
    )

    # ----------------------------------------------
    # Save overlay
    # ----------------------------------------------

    overlay = rgb.copy()

    burned = mask == 1

    highlight = np.zeros_like(
        overlay
    )

    highlight[:, :, 0] = 1.0

    alpha = 0.55

    overlay[burned] = (
        (1 - alpha) * overlay[burned]
        +
        alpha * highlight[burned]
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    Image.fromarray(
        (overlay * 255).astype(np.uint8),
        mode="RGB"
    ).save(
        output_overlay
    )

    # ----------------------------------------------
    # Save GeoTIFF
    # ----------------------------------------------

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
        output_geotiff,
        "w",
        **profile
    ) as dst:

        dst.write(
            mask,
            1
        )

    # ----------------------------------------------
    # Return API-friendly result
    # ----------------------------------------------

    return {
        "model": "Prithvi-EO-2.0-300M-BurnScars",
        "task": "burn_scar_segmentation",
        "image": str(image_path),
        "output_mask": str(output_mask),
        "output_overlay": str(output_overlay),
        "output_geotiff": str(output_geotiff),
        "image_width": int(mask.shape[1]),
        "image_height": int(mask.shape[0]),
        "tiles_processed": int(total_tiles),
        "burned_pixels": burned_pixels,
        "total_pixels": int(total_pixels),
        "burned_area_percentage": round(
            burned_percentage,
            2
        ),
        "mean_burn_probability": round(
            mean_probability,
            4
        )
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SatQuery AI - Overlapping Prithvi BurnScars Inference"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # IMAGE
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

    raw_image, profile = read_hls(
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

    print(
        "\nCreating BurnScars model..."
    )

    model = BurnScarModel()

    load_checkpoint(
        model
    )

    model.eval()

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    (
        mask,
        probability,
        total_tiles,
    ) = tiled_inference(
        model,
        image
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_pixels = mask.size

    burned_pixels = int(
        np.sum(
            mask == 1
        )
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
    # RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("OVERLAPPING TILED RESULT")
    print("=" * 70)

    print(
        f"Image size           : "
        f"{mask.shape[1]} x {mask.shape[0]}"
    )

    print(
        f"Tile size            : "
        f"{MODEL_SIZE} x {MODEL_SIZE}"
    )

    print(
        f"Stride               : "
        f"{STRIDE}"
    )

    print(
        f"Overlap              : "
        f"50%"
    )

    print(
        f"Tiles processed      : "
        f"{total_tiles}"
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
    # SAVE OUTPUTS
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
    print("OVERLAPPING TILED INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()