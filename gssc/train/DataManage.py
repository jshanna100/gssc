from torch.utils.data import Dataset
import numpy as np
import torch
import mne
from scipy.stats import zscore
from collections import Counter
import h5py
from os.path import exists
import pickle
from time import perf_counter
from gssc.utils import *

class SeqDataSet(Dataset):
    """
    This organises multiple datasets into one overarching dataset. During
    training,
    Parameters
    ----------
    template : str
        Full path to MNE raw file that contains the channels you expect the
        other files to contain.
    """
    def __init__(self, dset_filenames, dset_names):
        super().__init__()

        self.dsets = [torch.load(ds_f) for ds_f in dset_filenames]
        self.dset_names = dset_names
        for ds in self.dsets:
            ds.out_mode ="sequential"
        dset_lens = [len(ds) for ds in self.dsets]
        self.length = sum(dset_lens)
        cumsum = np.cumsum([0] + dset_lens)
        self.dset_borders = np.zeros((len(self.dsets), 2), dtype=int)
        for ds_idx in range(len(self.dsets)):
            self.dset_borders[ds_idx,] = cumsum[ds_idx:ds_idx+2]

    def __len__(self):
        return self.length

    def get_dset_idx(self, idx):
        dset_idx = np.where((np.any(self.dset_borders > idx, axis=1)) &
                            (np.any(self.dset_borders <= idx, axis=1)))[0][0]
        series_idx = idx - self.dset_borders[dset_idx, 0]
        return dset_idx, int(series_idx)

    def __getitem__(self, idx):
        dset_idx, series_idx = self.get_dset_idx(idx)
        sigs, stage = self.dsets[dset_idx].__getitem__(series_idx)

        return sigs, stage, idx

    def get_counts(self, return_stages=False):
        if return_stages:
            all_stages = []
            for ds in self.dsets:
                all_stages.extend(ds.get_counts(return_stages=True))
            return all_stages
        else:
            all_counts = Counter()
            for ds in self.dsets:
                 all_counts += ds.get_counts()
            return dict(all_counts)

class PSGDataSet(Dataset):
    def __init__(self, template, signals, target_length, path, hdffile,
                 file_record, files=None, val_dict=None, resamp_jobs=1):
        """Use this for putting EEG data into pytorch format.
        Parameters
        ----------
        template : str
            Full path to MNE raw file that contains the channels you expect the
            other files to contain.
        signals : dict
            Dictionary where the keys are the names of the signal types that
            can potentially be analysed (e.g. EEG, EOG). The values of this
            dictionary are themselves dictionaries which each define the
            signal type and permutational properties of the signal during
            training. For example, the dictionary entry:
            "eog":{"chans":"eog", "perms":[0,1], "flip":True}
            indicates that it will grab all eog channels for potential use
            in training, that with each iteration it will use either 0 or 1 eog
            channels, and the polarity of the signal may also be flipped.
        target_length : int
            The number of samples that the data must be resampled to in order
            to be fed into the network
        path : str
            Directory where the hdf and file records are stored.
        file_record:
            name of file containing list of already assimilated/failed
            files. Files in this list will be skipped when running
            append()
        files : list
            List of full paths to MNE raw files that will constitute your
            dataset
        val_dict : dict
            Dictionary which maps the sleep stage annotations found in the file
            with the internal ones of the GSSC.
        resamp_jobs : int or "cuda"
            This will be passed to method raw.resample(n_jobs=resamp_jobs)
            when the data are resampled to reach target_length
        """
        super().__init__()

        self.resamp_jobs = resamp_jobs
        self.target_length = target_length
        self.hdffile = hdffile
        self.all_chans = None
        self.path = path

        if exists(self.path+file_record):
            self.file_record = file_record
        else:
            # starting anew here
            file_record_dict = {"files":[], "failed_files":[]}
            with open(self.path+file_record, "wb") as f:
                pickle.dump(file_record_dict, f)
            self.file_record = file_record

        if val_dict is None:
            # default: annotations in MNE file already match native system:
            # "0"=Wake, "1"=N1, "2"=N2, "3"=N3, "4"=REM
            self.val_dict = {str(x):x for x in np.arange(5)}
        else:
            self.val_dict = val_dict

        temp_raw = mne.io.Raw(template)
        sig_combs, perm_matrix, all_chans, eeg_chans = permute_sigs(temp_raw,
                                                                    signals,
                                                                    all_upper=True)
        self.sig_combs = sig_combs
        self.sig_n = len(sig_combs)
        self.perm_mat = perm_matrix
        self.perm_n = len(perm_matrix)
        self.all_chans = all_chans
        self.eeg_chans = eeg_chans
        self.out_mode = "single"
        self.seq_perm = perm_matrix

        # initialise the hdf5 file where actual data are stored
        if not exists(self.path+self.hdffile):  # don't overwrite if something's there
            with h5py.File(self.path+self.hdffile, "w") as f:
                f.create_dataset("phys", (0, len(self.all_chans), target_length),
                                 maxshape=(None, len(self.all_chans), target_length),
                                 dtype=np.float32, chunks=True)
                f.create_dataset("stage", (0, 1), maxshape=(None,1),
                                 dtype=np.int8, chunks=True)
                dt = h5py.special_dtype(vlen=bytes)
                f.create_dataset("ch_names", len(self.all_chans), dtype=dt)
                f["ch_names"][:] = self.all_chans

        # append the data
        if files:
            self.append(files)

    def append(self, files, cap=None):
        # adds eeg datasets. Files should be a list of paths to MNE raw files
        file_idx_key = {}
        with open(self.path+self.file_record, "rb") as f:
            file_record = pickle.load(f)
        for file_idx, file in enumerate(files):
            print("\n\nProcessing file {} of {}...\n\n".format(file_idx+1,
                                                               len(files)))
            if (file in file_record["files"] or
                file in file_record["failed_files"]):
                print("File already integrated. Skipping.")
                continue

            try:
                epo, dur, dur_idx = raw_to_epo(file, self.target_length,
                                               all_chans=self.all_chans,
                                               val_dict=self.val_dict)
            except:
                print("Could not assimilate {}".format(file))
                continue
            data = epo.get_data(picks=self.all_chans) * 1e+6
            data = epo_arr_zscore(data, cap=cap) # normalise
            stages = epo.events[:, -1] # sleep stages
            events = epo.events

            # write to disk
            with h5py.File(self.path+self.hdffile, "a") as f:
                file_row_beg = len(f["phys"])
                data_n = len(data)
                file_row_end = file_row_beg + data_n
                f["phys"].resize((file_row_end, *f["phys"].shape[1:]))
                f["phys"][file_row_beg:,] = data[..., :self.target_length]
                f["stage"].resize((file_row_end, 1))
                stages = np.expand_dims(np.array(stages), 1).astype(np.int8)
                f["stage"][file_row_beg:] = stages
            file_idx_key[file] = (file_row_beg, file_row_beg+len(stages))
            file_record["files"].append(file)
            with open(self.path+self.file_record, "wb") as f:
                pickle.dump(file_record, f)

        self.file_idx_key = file_idx_key
        self.stage_n = self.get_counts()


    def __len__(self):
        if self.out_mode == "single":
            with h5py.File(self.path+self.hdffile, "r") as f:
                data_n = len(f["phys"])
            return data_n * self.perm_n
        elif self.out_mode == "sequential":
            return len(self.file_idx_key)

    def get_counts(self, inds=None, return_stages=False):
        # return counts of each class
        # NOTE inds must be in data_n form, not perm_n*data_n
        with h5py.File(self.path+self.hdffile, "r") as f:
            stages = f["stage"][:]
        if inds is not None:
            stages = stages[inds]

        unqs, counts = np.unique(stages, return_counts=True)
        count_dict = {unq:count for unq, count in zip(unqs, counts)}
        if return_stages:
            return stages
        else:
            return count_dict

    def __getitem__(self, idx):
        if self.out_mode == "single":
            ## single epoch
            # first convert idx into respective data idx and permutation idx
            data_idx = idx // self.perm_n
            perm_idx = idx % self.perm_n
            perm = self.perm_mat[perm_idx,]
            if "ram_loaded" in dir(self) and self.ram_loaded:
                data = self.data[np.where(self.data_inds==data_idx)[0][0],]
                stage = self.stage[np.where(self.data_inds==data_idx)[0][0],]
            else:
                with h5py.File(self.path+self.hdffile, "r") as f:
                    data = f["phys"][data_idx,]
                    stage = f["stage"][data_idx,]
                    all_chans = [x.decode() for x in f["ch_names"][:]]
            sigs = {}
            for sig_idx, (sig_name, chans) in enumerate(self.sig_combs.items()):
                # build the data array for each signal, convert to tensor
                max_chan_n = np.array([len(c) for c in chans]).max()
                signal = np.zeros((max_chan_n, data.shape[-1]),
                                  dtype=np.float32)
                chan = chans[perm[sig_idx]]
                for ch_idx, ch in enumerate(chan):
                    ch, coef = check_flip_chan(ch)
                    all_chans_idx = all_chans.index(ch)
                    signal[ch_idx,] = coef * data[all_chans_idx]
                sigs[sig_name] = torch.tensor(signal, dtype=torch.float32)
            # convert to pytorch tensors
            stage = torch.tensor(stage, dtype=torch.long)
            return sigs, stage
        elif self.out_mode == "sequential":
            ## whole PSG
            file_list = list(self.file_idx_key.keys())
            file_list.sort()
            d_inds = self.file_idx_key[file_list[idx]]
            with h5py.File(self.path+self.hdffile, "r") as f:
                data = f["phys"][d_inds[0]:d_inds[1]]
                stage = f["stage"][d_inds[0]:d_inds[1]]
                all_chans = [x.decode() for x in f["ch_names"][:]]
            all_sigs = []
            for perm_idx in range(len(self.seq_perm)):
                perm = self.seq_perm[perm_idx,]
                sigs = {}
                for sig_idx, (sig_name, chans) in enumerate(self.sig_combs.items()):
                    # build the data array for each signal, convert to tensor
                    if not len(chans[perm[sig_idx]]):
                        continue
                    chan = chans[perm[sig_idx]][0]
                    ch, coef = check_flip_chan(chan)
                    all_chans_idx = all_chans.index(ch)
                    signal = coef * data[:, all_chans_idx, :]
                    sigs[sig_name] = torch.tensor(signal, dtype=torch.float32)
                all_sigs.append(sigs)
            return all_sigs, stage


    """
    The methods load_to_ram, to_ and from_numpy, and clear_ram are for use
    on e.g. HPC systems, where RAM is usually plentiful, but access to storage
    media may be very slow. They are otherwise not recommended.
    """

    def load_to_ram(self, inds):
        print("Loading to RAM...")
        data_inds = inds // self.perm_n
        data_inds = np.sort(np.unique(data_inds))
        start_time = perf_counter()
        with h5py.File(self.path+self.hdffile, "r") as f:
            temp = f["phys"][0,]
            chunk_n = len(data_inds)//500
            self.data = np.zeros((len(data_inds), *temp.shape), dtype=np.float32)
            self.stage = np.zeros((len(data_inds), 1), dtype=np.int8)
            ind_start = 0
            for chunk, di in enumerate(np.array_split(data_inds, chunk_n)):
                temp_inds = (ind_start, ind_start + len(di))
                self.data[temp_inds[0]:temp_inds[1],] = f["phys"][di,]
                self.stage[temp_inds[0]:temp_inds[1],] = f["stage"][di,]
                ind_start += len(di)
        load_dur = perf_counter() - start_time
        print("Done in {}m and {}s.".format(load_dur//60, load_dur%60))
        self.ram_loaded = True
        self.data_inds = data_inds
        print("Done.")

    def to_numpy(self, filename):
        np.save("{}_datainds.npy".format(filename), self.data_inds)
        np.save("{}_data.npy".format(filename), self.data)
        np.save("{}_stage.npy".format(filename), self.stage)

    def from_numpy(self, filename):
        self.data_inds = np.load("{}_datainds.npy".format(filename))
        self.data = np.load("{}_data.npy".format(filename))
        self.stage = np.load("{}_stage.npy".format(filename))
        self.ram_loaded = True

    def clear_ram(self):
        del (self.data, self.stage, self.data_inds)
        self.ram_loaded = False
