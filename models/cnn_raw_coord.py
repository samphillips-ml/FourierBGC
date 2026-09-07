"""
CNN-RawCoord: the four coordinates supplied raw, fused late.

day-of-year, year, latitude and longitude are z-scored with fixed statistics
from the training split (helpers/scalar_norm.py) and broadcast along depth,
then concatenated after the full convolutional stack, immediately before the
final projection, which widens from 32 to 36 input channels.

Late fusion was adopted after concatenating raw-scale values at the input
produced training instability: unlike the Fourier features, raw coordinates
are not bounded, so every layer from conv1 onward saw inputs on wildly
different scales. This model isolates whether making the coordinates available
at all, untransformed, is sufficient. It is not -- see Sect. 6.2 in the paper
for more details.

Broadcasting and concatenation are done by the caller (train.py / evaluate.py).
"""
import torch
import torch.nn as nn

from models.backbone import Conv1dMed


class CNNRawCoord(Conv1dMed):
    def __init__(self, in_channels=3, n_scalars=4, dp_rate=0.2):
        super().__init__(in_channels=in_channels, dp_rate=dp_rate)
        self.conv17 = nn.Conv1d(32 + n_scalars, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, profile, depth_levels=None, scalars=None):
        # profile: (B, D, 3), scalars: (B, D, 4) already broadcast and z-scored
        x = profile.transpose(1, 2)
        x = self.bn1(self.af1(self.conv1(x))); x = self.do1(x)
        x = self.bn2(self.af2(self.conv2(x))); x = self.do2(x)
        x = self.bn3(self.af3(self.conv3(x))); x = self.do3(x)
        x = self.bn12(self.af12(self.conv12(x))); x = self.do12(x)
        x = self.bn13(self.af13(self.deconv13(x))); x = self.do13(x)
        x = self.bn14(self.af14(self.conv14(x))); x = self.do14(x)
        x = self.bn15(self.af15(self.deconv15(x))); x = self.do15(x)
        x = self.bn16(self.af16(self.conv16(x))); x = self.do16(x)  # (B, 32, 200)
        x = torch.cat([x, scalars.transpose(1, 2)], dim=1)          # (B, 36, 200)
        x = self.conv17(x)
        return x.transpose(1, 2)  # back to (B, D, 1)
