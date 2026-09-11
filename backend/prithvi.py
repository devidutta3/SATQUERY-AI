from .prithvi_mae import PrithviMAE
import torch
import rasterio
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "Prithvi_EO_V2_300M.pt"
IMAGE_PATH = BASE_DIR / "backend" / "Mexico_HLS_example.tif"
FEATURE_PATH = BASE_DIR / "backend" / "prithvi_features.pt"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Create Prithvi model
# --------------------------------------------------

model = PrithviMAE(
    img_size=224,
    num_frames=1,
    in_chans=6,
    patch_size=(1, 16, 16),
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    coords_encoding=None,
    coords_scale_learn=False,
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

# Positional embeddings are generated according
# to the current model configuration.
for key in list(checkpoint.keys()):
    if "pos_embed" in key:
        del checkpoint[key]


missing, unexpected = model.load_state_dict(
    checkpoint,
    strict=False
)

model.to(device)
model.eval()

print("Prithvi model loaded!")
print("Missing:", len(missing))
print("Unexpected:", len(unexpected))


# --------------------------------------------------
# Read HLS GeoTIFF
# --------------------------------------------------

print("\nReading satellite image...")

with rasterio.open(IMAGE_PATH) as src:

    image = src.read()

    print("Original shape:", image.shape)
    print("Bands:", src.count)
    print("Width:", src.width)
    print("Height:", src.height)


# --------------------------------------------------
# Convert to float32
# --------------------------------------------------

image = image.astype(np.float32)


# --------------------------------------------------
# Select six Prithvi bands
# --------------------------------------------------

image = image[:6]

print("Six-band shape:", image.shape)


# --------------------------------------------------
# Resize to 224 × 224
# --------------------------------------------------

tensor = torch.from_numpy(image)

tensor = torch.nn.functional.interpolate(
    tensor.unsqueeze(0),
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

# Shape:
# (1, 6, 224, 224)

tensor = tensor.unsqueeze(2)

# Shape:
# (1, 6, 1, 224, 224)

print("Model input shape:", tuple(tensor.shape))


# --------------------------------------------------
# Prithvi normalization
# --------------------------------------------------

mean = torch.tensor(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
    dtype=torch.float32
).view(1, 6, 1, 1, 1)

std = torch.tensor(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    dtype=torch.float32
).view(1, 6, 1, 1, 1)

tensor = (tensor - mean) / std

tensor = tensor.to(device)


# --------------------------------------------------
# Extract encoder features
# --------------------------------------------------

print("\nExtracting Prithvi encoder features...")

temporal_coords = torch.zeros(
    1, 1, 4,
    device=device
)

location_coords = torch.zeros(
    1, 2,
    device=device
)

with torch.no_grad():

    encoder_output = model.encoder(
        tensor,
        temporal_coords,
        location_coords
    )


# --------------------------------------------------
# Inspect encoder output
# --------------------------------------------------

print("\n========== ENCODER OUTPUT ==========")

print("Type:", type(encoder_output))
print("Number of returned values:", len(encoder_output))

for i, item in enumerate(encoder_output):

    print(f"\nOutput {i}:")
    print("Type:", type(item))

    if torch.is_tensor(item):

        print("Shape:", tuple(item.shape))
        print("Dtype:", item.dtype)

        if item.is_floating_point():

            print("Mean:", item.mean().item())
            print("Std:", item.std().item())

print("====================================")


# --------------------------------------------------
# Extract feature tokens
# --------------------------------------------------

features = encoder_output[0]

print("\nFeature tokens:", tuple(features.shape))


# --------------------------------------------------
# Global feature pooling
# --------------------------------------------------

pooled_features = features.mean(dim=1)

print("\n========== POOLED FEATURES ==========")

print("Shape:", tuple(pooled_features.shape))
print("Dtype:", pooled_features.dtype)
print("Mean:", pooled_features.mean().item())
print("Std:", pooled_features.std().item())

print("=====================================")


# --------------------------------------------------
# Save Prithvi feature vector
# --------------------------------------------------

torch.save(
    pooled_features.cpu(),
    FEATURE_PATH
)

print("\nPrithvi features saved!")
print("Feature file:", FEATURE_PATH)