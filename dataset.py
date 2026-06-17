"""
PPCon's own FloatDataset, adapted from github.com/gpietrop/PPCon (MIT license,
see LICENSE). Kept faithful to their training behavior: the validity label is
computed but discarded, exactly as in their own train.py, which never used it
either. The only change is renaming the hardcoded "nitrate" variable, since
this loads CHLA and BBP700 too, that's a pure naming fix with no effect on
values or computation.

NOTE: their __len__ is off by one (counts the index/label column as a sample),
which makes the last index silently return a duplicate of the second-to-last
profile (confirmed directly against their actual NITRATE train CSV). Left in
here for 1:1 fidelity, since PPCon's own training ran with this duplicate too.
One repeated real sample out of ~2500 profiles, not worth losing sleep over,
but worth a one-line note in the methods section either way.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset


def from_string_to_tensor(string):
    threshold = 99999
    label = 1  # 1 means the sample is good
    string = string[8:-2].split(",")
    out = torch.zeros(200)
    for ind in range(len(string)):
        if float(string[ind]) >= threshold:
            label = 0
        out[ind] = torch.tensor(float(string[ind]))
    return out, label


class FloatDataset(Dataset):

    def __init__(self, path_df=None):
        super().__init__()
        if path_df is None:
            raise Exception("Paths should be given as input to initialize the Float class.")
        self.path_df = path_df
        self.df = pd.read_csv(self.path_df)

    def __len__(self):
        return len(self.df.iloc[0, :])

    def __getitem__(self, index):
        try:
            self.samples = self.df.iloc[:, index + 1].tolist()
        except IndexError:
            pass  # off-by-one in __len__, last index reuses previous self.samples.
            # this matches PPCon's own get_reconstruction (same FloatDataset),
            # so their published RMSE numbers carry this exact quirk too, not
            # just their training.

        year = torch.tensor(float(self.samples[0]))
        day_rad = torch.tensor(float(self.samples[1]))
        lat = torch.tensor(float(self.samples[2]))
        lon = torch.tensor(float(self.samples[3]))
        temp, label_temp = from_string_to_tensor(self.samples[4])
        psal, label_psal = from_string_to_tensor(self.samples[5])
        doxy, label_doxy = from_string_to_tensor(self.samples[6])
        target, label_target = from_string_to_tensor(self.samples[7])

        # label computed, matches PPCon's own code, also never used downstream
        label = label_doxy * label_psal * label_temp

        return year, day_rad, lat, lon, temp, psal, doxy, target