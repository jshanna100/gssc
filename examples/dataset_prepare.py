import torch
from os.path import join
import glob
from gssc.train.DataManage import PSGDataSet

### basic example of how to make a PSG Dataset object for use in training, etc.

# define where the MNE -raw.fif files are, and also where the resulting output
# files will be saved.
proc_dir = "/your/directory/path" # where the MNE -raw.fif files are, and

files = glob.glob(proc_dir + "*-raw.fif") # get all the MNE filenames

# we define here both an EEG and EOG signal, with lists of possible channels.
# By defining perms as 0 and 1 for both signals, we allow permutations with no
# EEG/EOG to exist. Flip set to true allows permutations where polarity is
# flipped
sigs = {"eeg":{"chans":["C3", "C4"], "perms":[0,1], "flip":True},
        "eog":{"chans":["EOG(L)"], "perms":[0,1], "flip":True}}

# now actually define and build the PSGDataSet
template = files[0] # assume the first file is like the rest
dset = PSGDataSet(template, sigs, 2560, proc_dir, "example_PSG.hdf5",
                  "example_PSG.pickle")
# add the raw eeg files. the bulk of these data will be stored in an
# accompanying .h5 file
dset.append(files)

# now save the instance
outfile = join(proc_dir, "example_PSG.pt")
torch.save(dset, outfile)
