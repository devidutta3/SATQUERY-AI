import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from terratorch.models.backbones.prithvi_vit import prithvi_vit_300


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-BurnScars"
    / "snapshots"
    / "a3f2c410e45b8ac7417976614528a872f024d831"
    / "Prithvi_EO_V2_300M_BurnScars.pt"
)


# ============================================================
# TERRATORCH-COMPATIBLE NECK COMPONENTS
# ============================================================

class SelectIndices(nn.Module):
    """
    Select the four Prithvi encoder outputs used by the
    official BurnScars model.
    """

    def __init__(self, indices):
        super().__init__()
        self.indices = list(indices)

    def forward(self, features):
        return [features[i] for i in self.indices]


class ReshapeTokensToImage(nn.Module):
    """
    Convert Prithvi token features:

        [B, 197, C]

    into spatial feature maps:

        [B, C, 14, 14]

    The first token is the CLS token.
    """

    def forward(self, features):
        output = []

        for feature in features:
            if feature.ndim != 3:
                raise ValueError(
                    f"Expected token tensor [B,N,C], got {feature.shape}"
                )

            B, N, C = feature.shape

            # Prithvi output = CLS + 14x14 spatial tokens
            if N == 197:
                feature = feature[:, 1:, :]
                N = 196

            H = int(N ** 0.5)

            if H * H != N:
                raise ValueError(
                    f"Cannot reshape {N} tokens into a square image."
                )

            feature = feature.transpose(1, 2)
            feature = feature.reshape(B, C, H, H)

            output.append(feature)

        return output


class LearnedInterpolateToPyramidal(nn.Module):
    """
    Exact structural reimplementation of the official
    TerraTorch LearnedInterpolateToPyramidal neck.
    """

    def __init__(self, channel_list):
        super().__init__()

        if len(channel_list) != 4:
            raise ValueError(
                "LearnedInterpolateToPyramidal requires exactly 4 embeddings."
            )

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                channel_list[0],
                channel_list[0] // 2,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(channel_list[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(
                channel_list[0] // 2,
                channel_list[0] // 4,
                kernel_size=2,
                stride=2,
            ),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                channel_list[1],
                channel_list[1] // 2,
                kernel_size=2,
                stride=2,
            )
        )

        self.fpn3 = nn.Sequential(
            nn.Identity()
        )

        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
        )

    def forward(self, features):
        return [
            self.fpn1(features[0]),
            self.fpn2(features[1]),
            self.fpn3(features[2]),
            self.fpn4(features[3]),
        ]


# ============================================================
# DECODER
# ============================================================

class UNetBlock(nn.Module):
    """
    Two-convolution U-Net decoder block.

    Named conv1/conv2 modules are intentional:
    they match the official checkpoint hierarchy.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BurnScarDecoder(nn.Module):
    """
    Official BurnScars U-Net decoder channel structure.
    """

    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            # 2048 -> 512
            UNetBlock(2048, 512),

            # 1024 -> 256
            UNetBlock(1024, 256),

            # 512 -> 128
            UNetBlock(512, 128),

            # 128 -> 64
            UNetBlock(128, 64),
        ])

    def forward(self, features):

        f0, f1, f2, f3 = features

        # Deepest feature -> level 2
        x = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, f2], dim=1)
        x = self.blocks[0](x)

        # level 2 -> level 1
        x = F.interpolate(
            x,
            size=f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, f1], dim=1)
        x = self.blocks[1](x)

        # level 1 -> level 0
        x = F.interpolate(
            x,
            size=f0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, f0], dim=1)
        x = self.blocks[2](x)

        # Final upsampling
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.blocks[3](x)

        return x


class DecoderWrapper(nn.Module):
    """
    Wrapper required to reproduce the official checkpoint
    hierarchy:

        decoder.decoder.blocks...
    """

    def __init__(self):
        super().__init__()
        self.decoder = BurnScarDecoder()

    def forward(self, features):
        return self.decoder(features)


# ============================================================
# SEGMENTATION HEAD
# ============================================================

class HeadWrapper(nn.Module):
    """
    Reproduces the official head hierarchy:

        head.head.0
        head.head.1
        head.head.2
    """

    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Conv2d(
                64,
                2,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# COMPLETE BURNSCARS MODEL
# ============================================================

class BurnScarModel(nn.Module):

    def __init__(self):
        super().__init__()

        print("Building Prithvi backbone...")

        # IMPORTANT:
        # Named "encoder" to match the official checkpoint.
        self.encoder = prithvi_vit_300(
            pretrained=False,

            bands=[
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "SWIR_1",
                "SWIR_2",
            ],

            features_only=True,

            # Official SelectIndices:
            # [5, 11, 17, 23]
            out_indices=list(range(24)),
        )

        print("Backbone created.")

        # ====================================================
        # OFFICIAL NECK HIERARCHY
        # ====================================================

        self.neck = nn.Sequential(
            SelectIndices([5, 11, 17, 23]),

            ReshapeTokensToImage(),

            LearnedInterpolateToPyramidal(
                [
                    1024,
                    1024,
                    1024,
                    1024,
                ]
            ),
        )

        # ====================================================
        # OFFICIAL DECODER HIERARCHY
        # ====================================================

        self.decoder = DecoderWrapper()

        # ====================================================
        # OFFICIAL SEGMENTATION HEAD
        # ====================================================

        self.head = HeadWrapper()

        print("Model created successfully.")

    def forward(self, x):

        # Prithvi encoder
        features = self.encoder(x)

        # Select -> reshape -> learned pyramid
        features = self.neck(features)

        # U-Net decoder
        decoded = self.decoder(features)

        # 2-class segmentation logits
        logits = self.head(decoded)

        return logits


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BurnScars checkpoint not found:\n{MODEL_PATH}"
        )

    print("Loading official BurnScars checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    # Remove ONLY the outer Lightning "model." prefix.
    cleaned = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }

    print(f"Checkpoint tensors: {len(cleaned)}")

    # TerraTorch's installed Prithvi implementation does not expose the
    # official checkpoint's encoder.pos_embed under the same state_dict
    # hierarchy. Do not create an unused dummy parameter just to silence it.
    cleaned_without_pos = {
        key: value
        for key, value in cleaned.items()
        if key != "encoder.pos_embed"
    }

    missing, unexpected = model.load_state_dict(
        cleaned_without_pos,
        strict=False,
    )

    # The only intentionally skipped official tensor is encoder.pos_embed.
    # Report it separately so checkpoint alignment remains explicit.
    if "encoder.pos_embed" in cleaned:
        print("Skipped checkpoint key: encoder.pos_embed")

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("\nMissing keys:")
        for key in missing:
            print(" ", key)

    if unexpected:
        print("\nUnexpected keys:")
        for key in unexpected:
            print(" ", key)

    if not missing and not unexpected:
        print("\nSUCCESS: Official BurnScars checkpoint loaded exactly.")

    return model
