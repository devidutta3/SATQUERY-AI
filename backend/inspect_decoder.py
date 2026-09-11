import torch

CHECKPOINT = r"C:\Users\dasde\.cache\huggingface\hub\models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-BurnScars\snapshots\a3f2c410e45b8ac7417976614528a872f024d831\Prithvi_EO_V2_300M_BurnScars.pt"

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("=" * 70)
print("BURN SCARS DECODER CHECKPOINT")
print("=" * 70)

for key, value in checkpoint["state_dict"].items():

    if key.startswith("model.decoder."):

        print(
            f"{key:60s} {tuple(value.shape)}"
        )

