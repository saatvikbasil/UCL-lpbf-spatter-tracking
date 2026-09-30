"""U-Net used to segment spatter in synchrotron X-ray frames.

With the default channel settings the network has three encoder stages
(3 -> 32 -> 64 -> 128 channels), two decoder stages and a 1x1 convolution head.
"""

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    Module,
    ModuleList,
    ReLU,
)
from torch.nn import functional as F
from torchvision.transforms import CenterCrop

from . import config


class Block(Module):
    """Conv3x3-BN-ReLU, Conv1x1-BN-ReLU, Dropout2d(0.1), Conv3x3-BN-ReLU."""

    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(outChannels)
        self.conv_middle = Conv2d(outChannels, outChannels, kernel_size=1)
        self.bn_middle = BatchNorm2d(outChannels)
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(outChannels)
        self.relu = ReLU(inplace=True)
        self.dropout = torch.nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_middle(x)
        x = self.bn_middle(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)


class Encoder(Module):
    """Stack of Blocks separated by 2x2 max pooling."""

    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = MaxPool2d(2)

    def forward(self, x):
        """Return the output of every block (before pooling), shallowest first."""
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs


class Decoder(Module):
    """Transposed-conv upsampling, skip concatenation and a Block per stage."""

    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]
        )
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        """Center-crop encoder features to the spatial size of ``x``.

        All convolutions are same-padded, so for the 512 x 1024 input used
        here the sizes already match and the crop has no effect.
        """
        _, _, H, W = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures


class UNet(Module):
    """U-Net returning raw logits (apply a sigmoid to get probabilities).

    If ``retainDim`` is set, the output is resized to ``outSize`` with
    nearest-neighbour interpolation.
    """

    def __init__(
        self,
        encChannels=(3, 32, 64, 128),
        decChannels=(128, 64, 32),
        nbClasses=1,
        retainDim=True,
        outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    ):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encFeatures = self.encoder(x)
        # Deepest features seed the decoder; the rest are skip connections.
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        logits = self.head(decFeatures)
        if self.retainDim:
            logits = F.interpolate(logits, self.outSize)
        return logits
