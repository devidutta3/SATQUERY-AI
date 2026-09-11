import torch
from huggingface_hub import hf_hub_download

from burnscar_model import BurnScarModel


# ============================================================
# LOCATE OFFICIAL CHECKPOINT
# ============================================================

REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"

CHECKPOINT = hf_hub_download(
    repo_id=REPO_ID,
    filename="Prithvi_EO_V2_300M_BurnScars.pt",
    repo_type="model",
)

print("Checkpoint:")
print(CHECKPOINT)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint["state_dict"]


# ============================================================
# CREATE LOCAL MODEL
# ============================================================

print("\nCreating local model...")

model = BurnScarModel()

model_state = model.state_dict()


# ============================================================
# CHECKPOINT DECODER
# ============================================================

print("\n" + "=" * 70)
print("CHECKPOINT DECODER KEYS")
print("=" * 70)

for key, value in state_dict.items():

    if key.startswith("model.decoder."):
        print(key, tuple(value.shape))


# ============================================================
# LOCAL DECODER
# ============================================================

print("\n" + "=" * 70)
print("LOCAL DECODER KEYS")
print("=" * 70)

for key, value in model_state.items():

    if key.startswith("decoder."):
        print(key, tuple(value.shape))


# ============================================================
# CHECKPOINT NECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKPOINT NECK KEYS")
print("=" * 70)

for key, value in state_dict.items():

    if key.startswith("model.neck."):
        print(key, tuple(value.shape))


# ============================================================
# LOCAL NECK
# ============================================================

print("\n" + "=" * 70)
print("LOCAL NECK KEYS")
print("=" * 70)

for key, value in model_state.items():

    if key.startswith("necks."):
        print(key, tuple(value.shape))


# ============================================================
# POSITION EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("POSITION EMBEDDING COMPARISON")
print("=" * 70)

print("\nCHECKPOINT:")

for key, value in state_dict.items():

    if "pos_embed" in key:
        print(
            key,
            tuple(value.shape)
        )


print("\nLOCAL MODEL:")

for key, value in model_state.items():

    if "pos_embed" in key:
        print(
            key,
            tuple(value.shape)
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL KEY COUNT")
print("=" * 70)

print("Checkpoint state_dict:", len(state_dict))
print("Local model:", len(model_state))

print("\nDiagnostic completed.")