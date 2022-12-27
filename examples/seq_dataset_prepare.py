import torch
import numpy as np
import glob
from gssc.train.DataManage import SeqDataSet
from os.path import join

### basic example of how to make a Sequential Dataset. There wrap around a list
### of the more fundamental PSG DataSets, allowing one to grab a random PSG
### out of the entire dataset for training

# we make a SeqDataSet out of two PSG Datasets, but any # is possible
dset_files = ["/path/to/dataset1/dataset1.pt",
              "/path/to/dataset2/dataset2.pt"]
# specify the names these datasets will have internally to SeqDataSet
dset_names = ["DSet1", "DSet2"]

seq_dset = SeqDataSet(dset_files, dset_names)
torch.save(seq_dset, "/path/to/seq_dataset/seq.pt"))
