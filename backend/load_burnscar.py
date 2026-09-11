import os
import torch

from huggingface_hub import hf_hub_download
from burnscar_model import BurnScarModel


# ============================================================
# DOWNLOAD / LOCATE OFFICIAL CHECKPOINT
# ============================================================

REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"

print("=" * 60)
print("Locating official BurnScars checkpoint...")
print("=" * 60)

CHECKPOINT = hf_hub_download(
    repo_id=REPO_ID,
    filename="Prithvi_EO_V2_300M_BurnScars.pt",
    repo_type="model",
)

print("Checkpoint found:")
print(CHECKPOINT)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT = r"models\burnscar_model_converted.pt"


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 60)
print("Loading BurnScars checkpoint...")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded!")
print("Checkpoint type:", type(checkpoint))


# ============================================================
# GET STATE DICT
# ============================================================

state_dict = checkpoint["state_dict"]

print("Original state_dict keys:", len(state_dict))


# ============================================================
# CREATE LOCAL MODEL
# ============================================================

print("\nCreating local BurnScarModel...")

model = BurnScarModel()

print("Local model created.")


# ============================================================
# CONVERT TERRATORCH CHECKPOINT KEYS
# ============================================================

converted = {}

print("\nConverting checkpoint keys...")

for key, value in state_dict.items():

    if not key.startswith("model."):
        continue

    key = key[len("model."):]

    # Backbone
    if key.startswith("encoder."):
        new_key = "backbone." + key[len("encoder."):]

    # Neck
    elif key.startswith("neck."):
        new_key = "necks." + key[len("neck."):]

    # Decoder
    elif key.startswith("decoder.decoder."):
        new_key = "decoder." + key[len("decoder.decoder."):]

    # Segmentation head
    elif key == "head.head.2.weight":
        new_key = "head.weight"

    elif key == "head.head.2.bias":
        new_key = "head.bias"

    else:
        continue

    converted[new_key] = value


print("Converted keys:", len(converted))


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nChecking parameter shapes...")

model_state = model.state_dict()

shape_mismatches = []

for key, value in converted.items():

    if key not in model_state:
        continue

    if tuple(value.shape) != tuple(model_state[key].shape):

        shape_mismatches.append(
            (
                key,
                tuple(value.shape),
                tuple(model_state[key].shape)
            )
        )


if shape_mismatches:

    print("\n" + "=" * 60)
    print("SHAPE MISMATCHES FOUND")
    print("=" * 60)

    for key, checkpoint_shape, model_shape in shape_mismatches:
        print(f"\nKEY: {key}")
        print(f"Checkpoint: {checkpoint_shape}")
        print(f"Model:      {model_shape}")

else:

    print("No parameter shape mismatches found.")


# ============================================================
# LOAD WEIGHTS
# ============================================================

print("\nLoading converted weights...")

missing, unexpected = model.load_state_dict(
    converted,
    strict=False
)


# ============================================================
# MISSING
# ============================================================

print("\n" + "=" * 60)
print("MISSING KEYS")
print("=" * 60)

if missing:
    for key in missing:
        print(key)
else:
    print("None")


# ============================================================
# UNEXPECTED
# ============================================================

print("\n" + "=" * 60)
print("UNEXPECTED KEYS")
print("=" * 60)

if unexpected:
    for key in unexpected:
        print(key)
else:
    print("None")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LOAD SUMMARY")
print("=" * 60)

print("Converted parameters :", len(converted))
print("Missing parameters   :", len(missing))
print("Unexpected parameters:", len(unexpected))
print("Shape mismatches     :", len(shape_mismatches))


# ============================================================
# SAVE
# ============================================================

if not shape_mismatches:

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        OUTPUT
    )

    print("\nConverted model saved successfully:")
    print(OUTPUT)

else:

    print("\nModel NOT saved because shape mismatches exist.")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)