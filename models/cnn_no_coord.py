"""
CNN-NoCoord: the shared backbone with no coordinate representation at all.

C = 3 (temperature, salinity, dissolved oxygen). Trained under the common
recipe. Two roles in the ablation: compared against PPCon-NoCoord it isolates
the training procedure with the architecture held fixed, and it is the model
that tests whether the spatiotemporal coordinates are needed at all.

252,737 parameters, 0 of them in a coordinate encoder.
"""
from models.backbone import Conv1dMed


class CNNNoCoord(Conv1dMed):
    def __init__(self, in_channels=3, dp_rate=0.2):
        super().__init__(in_channels=in_channels, dp_rate=dp_rate)
