"""
Conv1dMed: the shared convolutional backbone.

Adopted unchanged from PPCon (Pietropolli et al., 2024;
github.com/gpietrop/PPCon, MIT) and held fixed across every model in the
ablation, so that differences in reconstruction accuracy reflect the
coordinate encoding under test and not the architecture. Nine layers of 1D
convolutions and transposed convolutions, each followed by SELU then batch
normalization then dropout, except the final projection (conv17), which maps
the last hidden state directly to the single output channel.

Every model in models/ subclasses this. Only `in_channels` and, for
CNN-RawCoord, the width of conv17 differ.

Note that the attribute names below (conv1, bn1, deconv13, ...) are the keys
of every saved state_dict. Renaming these will invalidate the checkpoints.
 Class and file names are free to change; these are not.
"""
import torch.nn as nn


class Conv1dMed(nn.Module):
    """253,249 params at in_channels=7 (PPCon's own figure), 252,737 at
    in_channels=3. Dropping four of seven input channels costs only 512
    params, all in conv1."""

    def __init__(self, in_channels=3, dp_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=2, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.af1 = nn.SELU()
        self.do1 = nn.Dropout(p=dp_rate)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.af2 = nn.SELU()
        self.do2 = nn.Dropout(p=dp_rate)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.af3 = nn.SELU()
        self.do3 = nn.Dropout(p=dp_rate)

        self.conv12 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=2)
        self.bn12 = nn.BatchNorm1d(128)
        self.af12 = nn.SELU()
        self.do12 = nn.Dropout(p=dp_rate)

        self.deconv13 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, padding=2)
        self.bn13 = nn.BatchNorm1d(128)
        self.af13 = nn.SELU()
        self.do13 = nn.Dropout(p=dp_rate)

        self.conv14 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm1d(128)
        self.af14 = nn.SELU()
        self.do14 = nn.Dropout(p=dp_rate)

        self.deconv15 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, padding=1)
        self.bn15 = nn.BatchNorm1d(64)
        self.af15 = nn.SELU()
        self.do15 = nn.Dropout(p=dp_rate)

        self.conv16 = nn.Conv1d(64, 32, kernel_size=2, stride=2, padding=1)
        self.bn16 = nn.BatchNorm1d(32)
        self.af16 = nn.SELU()
        self.do16 = nn.Dropout(p=dp_rate)

        self.conv17 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, profile, depth_levels=None, scalars=None):
        # profile: (B, D, C) -> conv1d wants (B, C, D). depth_levels/scalars are
        # unused here, kept in the signature so callers can treat every model
        # in the spine identically.
        x = profile.transpose(1, 2)
        x = self.bn1(self.af1(self.conv1(x))); x = self.do1(x)
        x = self.bn2(self.af2(self.conv2(x))); x = self.do2(x)
        x = self.bn3(self.af3(self.conv3(x))); x = self.do3(x)
        x = self.bn12(self.af12(self.conv12(x))); x = self.do12(x)
        x = self.bn13(self.af13(self.deconv13(x))); x = self.do13(x)
        x = self.bn14(self.af14(self.conv14(x))); x = self.do14(x)
        x = self.bn15(self.af15(self.deconv15(x))); x = self.do15(x)
        x = self.bn16(self.af16(self.conv16(x))); x = self.do16(x)
        x = self.conv17(x)
        return x.transpose(1, 2)  # back to (B, D, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
