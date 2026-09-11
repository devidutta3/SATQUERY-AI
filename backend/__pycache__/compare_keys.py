import torch
from huggingface_hub import hf_hub_download

from burnscar_model import BurnScarModel


# ============================================================
# OFFICIAL BURNSCARS CHECKPOINT
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
    weights_only=False,
)

state_dict = checkpoint["state_dict"]

print("\nCheckpoint state_dict keys:", len(state_dict))


# ============================================================
# CREATE LOCAL MODEL
# ============================================================

print("\nCreating local BurnScarModel...")

model = BurnScarModel()

model_state = model.state_dict()

print("Local model parameters:", len(model_state))


# ============================================================
# CHECKPOINT DECODER
# ============================================================

print("\n" + "=" * 70)
print("CHECKPOINT DECODER KEYS")
print("=" * 70)

for key, value in state_dict.items():

    if key.startswith("model.decoder."):
        print(
            key,
            "->",
            tuple(value.shape)
        )


# ============================================================
# LOCAL DECODER
# ============================================================

print("\n" + "=" * 70)
print("LOCAL DECODER KEYS")
print("=" * 70)

for key, value in model_state.items():

    if key.startswith("decoder."):
        print(
            key,
            "->",
            tuple(value.shape)
        )


# ============================================================
# CHECKPOINT NECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKPOINT NECK KEYS")
print("=" * 70)

for key, value in state_dict.items():

    if key.startswith("model.neck."):
        print(
            key,
            "->",
            tuple(value.shape)
        )


# ============================================================
# LOCAL NECK
# ============================================================

print("\n" + "=" * 70)
print("LOCAL NECK KEYS")
print("=" * 70)

for key, value in model_state.items():

    if key.startswith("necks."):
        print(
            key,
            "->",
            tuple(value.shape)
        )


# ============================================================
# POSITION EMBEDDING COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("POSITION EMBEDDING COMPARISON")
print("=" * 70)


print("\nCHECKPOINT:")

for key, value in state_dict.items():

    if "pos_embed" in key:

        print(
            key,
            "->",
            tuple(value.shape)
        )


print("\nLOCAL MODEL:")

for key, value in model_state.items():

    if "pos_embed" in key:

        print(
            key,
            "->",
            tuple(value.shape)
        )


# ============================================================
# EXACT KEY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("EXACT KEY COMPARISON")
print("=" * 70)


checkpoint_model_keys = set()

for key in state_dict:

    if key.startswith("model."):

        checkpoint_model_keys.add(
            key[len("model."):]
        )


local_keys = set(model_state.keys())


common_keys = checkpoint_model_keys.intersection(local_keys)

checkpoint_only = checkpoint_model_keys - local_keys

local_only = local_keys - checkpoint_model_keys


print("\nCheckpoint model keys:", len(checkpoint_model_keys))
print("Local model keys:     ", len(local_keys))
print("Common keys:          ", len(common_keys))
print("Checkpoint only:      ", len(checkpoint_only))
print("Local only:           ", len(local_only))


# ============================================================
# CHECKPOINT-ONLY KEYS
# ============================================================

print("\n" + "=" * 70)
print("CHECKPOINT-ONLY KEYS")
print("=" * 70)

for key in sorted(checkpoint_only):

    print(key)


# ============================================================
# LOCAL-ONLY KEYS
# ============================================================

print("\n" + "=" * 70)
print("LOCAL-ONLY KEYS")
print("=" * 70)

for key in sorted(local_only):

    print(key)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)