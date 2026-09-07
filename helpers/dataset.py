"""
PPCon's FloatDataset, from github.com/gpietrop/PPCon (MIT). I maintained the 
behavior of the code, in order to preserve the validity of our comparison

Two specfic characteristics are deliberately preserved from PPcon. First, the validity 
label is computed and then thrown away, so the label plumbing below is dead code. 
Second, __len__ counts the index column as a profile, so the last index returns a duplicate of
the second-to-last -- which is why the paper's denominators are 626/945/948 and
not 625/944/947. Their own get_reconstruction has the same bug, so their
published RMSE carries the duplicate too.
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