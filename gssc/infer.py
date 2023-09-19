import mne
from .utils import (epo_arr_zscore, permute_sigs, check_flip_chan,
                   loudest_vote, prepare_inst)
import torch
import numpy as np
import warnings
from importlib_resources import files

class ArrayInfer():
    def __init__(self, net, con_net, sig_combs, perm_matrix, all_chans,
                 sig_len=2560):
        if net is None:
            net_name = files('gssc.nets').joinpath("sig_net_v1.pt")
            net = torch.load(net_name)
        if con_net is None:
            con_net_name = files('gssc.nets').joinpath("gru_net_v1.pt")
            con_net = torch.load(con_net_name)
        self.sig_combs = sig_combs
        self.perm_matrix = perm_matrix
        self.perm_inds = np.arange(len(perm_matrix))
        self.all_chans = all_chans
        self.net = net
        self.con_net = con_net
        self.sig_len = sig_len

    def infer(self, arr, hiddens, perm_inds=None):
        sig_combs = self.sig_combs
        perm_matrix = self.perm_matrix
        all_chans = self.all_chans
        net = self.net
        con_net = self.con_net
        perm_inds = perm_inds if perm_inds else self.perm_inds

        all_sigs = {}
        logits = []
        res_logits = []
        for perm_idx in perm_inds:
            these_chans = {"eeg":None, "eog":None}
            perm = perm_matrix[perm_idx,]
            sigs = {}
            for sig_idx, (sig_name, chans) in enumerate(sig_combs.items()):
                # build the data array for each signal, convert to tensor
                if not len(chans[perm[sig_idx]]):
                    continue
                chan = chans[perm[sig_idx]][0]
                ch, coef = check_flip_chan(chan)
                all_chans_idx = all_chans.index(ch)
                signal = coef * arr[:, all_chans_idx, :self.sig_len]
                signal = signal.reshape(-1, 1, signal.shape[-1])
                sigs[sig_name] = signal
                these_chans[sig_name] = chan
            perm_str = "eeg: {}, eog: {}".format(*these_chans.values())
            all_sigs[perm_str] = sigs

        with torch.no_grad():
            for sig_idx, (sig_k, sigs) in enumerate(all_sigs.items()):
                reps, dec = net(sigs, rep_output=True)
                reps = reps.swapaxes(-1, 1)
                y, hidden = con_net(reps, hiddens[sig_idx,])
                hiddens[sig_idx,] = hidden
                del reps

                logits.append(y[:,0,].cpu().numpy())
                dec = dec[...,0]
                res_logits.append(dec.cpu().numpy())

        return np.array(logits), np.array(res_logits), hiddens


class EEGInfer():
    """Class for sleep stage inference on a set of EEG data.

    Parameters
    ----------
    net : PyTorch Module (default None)
        Neural network which processes the signals. If None defaults to built-in nets.
    con_net : PyTorch Module (default None)
        Neural network which takes context into account and infers the sleep stage.
        If None defaults to built-in nets.
    sig_len : int (default 2560)
        Length of signal in samples.
    cut : str (default 'back')
        Either 'back' or 'front'. When the signal is cut into epochs of 2560 samples,
        any remaining time is cut, either from the front or back.
    use_cuda : bool (default True)
        Use CUDA acceleration for inference.
    chunk_n : int (default 0)
        Limit inference to chunk_n epochs at a time - useful in the case of CUDA memory
        constraints. When set at 0 (default), the whole file is done in one step.
    gpu_idx : int or None (default)
        In the case of multiple GPUs on a single system, select with an integer which
        GPU to use.

    """
    def __init__(self, net=None, con_net=None, sig_len=2560, cut="back",
                 use_cuda=True, chunk_n=0, gpu_idx=None):
        if net is None:
            net_name = files('gssc.nets').joinpath("sig_net_v1.pt")
            net = torch.load(net_name)
        if con_net is None:
            con_net_name = files('gssc.nets').joinpath("gru_net_v1.pt")
            con_net = torch.load(con_net_name)
        self.net = net
        self.con_net = con_net
        self.sig_len = sig_len
        self.cut = cut
        self.chunk_n = chunk_n

        # transfer to cuda if possible
        is_cuda = torch.cuda.is_available()
        if use_cuda and is_cuda:
            print("CUDA detected and selected.")
            self.net.cuda(device=gpu_idx)
            self.con_net.cuda(device=gpu_idx)
            self.use_cuda = True
        elif use_cuda and not is_cuda:
            warnings.warn("CUDA was selected but cannot be found on this "
                          "machine. If you believe this is in error, check "
                          "your CUDA installation. Otherwise set use_cuda "
                          "to False.")
            self.use_cuda = False
        elif not use_cuda and is_cuda:
            warnings.warn("WARNING: CUDA is available on this machine, but "
                          "use_cuda is marked as False. Running on CPU "
                          "instead of CUDA is significantly slower!")
            self.use_cuda = False
        else:
            print("Running without CUDA on CPU. Speed will be suboptimal.")
            self.use_cuda = False
        # make sure they're in evaluation mode for inference
        self.net.eval()
        self.con_net.eval()

    def mne_infer(self, inst, chunk_n=0, eeg="eeg", eog="eog", eeg_drop=True,
                  eog_drop=True, filter=True):
        """
        Performs inference on a EEG recording.

        Parameters
        ----------
        inst : MNE-Python Raw or Epochs instance.
            The EEG instance to be inferred, in MNE Python format.
        chunk_n : int (default 0)
            Limit inference to chunk_n epochs at a time - useful in the case of CUDA memory
            constraints. When set at 0 (default), the whole file is done in one step. 
            Overrides self.chunk_n
        eeg : "eeg" or List of str (default 'eeg')
            EEG channels to use for inference. If 'eeg' then all available EEG channels are
            used (generally not recommended).
        eog : "eog" or List of str (default 'eog')
            EOG channels to use for inference. If 'eog' then all available EOG channels are
            used (generally not recommended).
        eeg_drop : bool (default True)
            Allow null permutations for EEG
        eog_drop : bool (default True)
            Allow null permutations for EOG
        filter : bool (default True)
            Filter the data to a bandpass of 0.3-30Hz, where possible, if this has not
            already been done.
        """
        
        net = self.net
        con_net = self.con_net
        sig_len = self.sig_len
        cut = self.cut
        chunk_n = self.chunk_n if not chunk_n else chunk_n
        use_cuda = self.use_cuda

        # check for filtering
        if filter:
            filter_band = [None, None]
            if round(inst.info["highpass"], 2) < 0.3:
                filter_band[0] = 0.3
            if round(inst.info["lowpass"], 2) > 30.:
                filter_band[1] = 30.
            inst.filter(*filter_band)
        if round(inst.info["highpass"], 2) != 0.3:
            warnings.warn("WARNING: GSSC was trained on data with a highpass "
                         "filter of 0.3Hz. These data have a highpass filter "
                         f"of {inst.info['highpass']}Hz")
        if round(inst.info["lowpass"], 2) != 30.:
            warnings.warn("WARNING: GSSC was trained on data with a lowpass "
                         "filter of 30Hz. These data have a lowpass filter "
                         f"of {inst.info['lowpass']}Hz")

        signals = {"eeg":{"chans":eeg, "drop":eeg_drop, "flip":False},
                   "eog":{"chans":eog, "drop":eog_drop, "flip":False}}

        # get the MNE inst into correct form, convert to z-scored array
        epo, start_time = prepare_inst(inst, sig_len, cut)
        sig_combs, perm_matrix, all_chans, _ = permute_sigs(epo, signals)
        data = epo.get_data(picks=all_chans) * 1e+6
        data = epo_arr_zscore(data)

        all_sigs = {}
        logits = []
        infs = []
        res_logits = []
        res_infs = []

        for perm_idx in range(len(perm_matrix)):
            these_chans = {"eeg":None, "eog":None}
            perm = perm_matrix[perm_idx,]
            sigs = {}
            for sig_idx, (sig_name, chans) in enumerate(sig_combs.items()):
                # build the data array for each signal, convert to tensor
                if not len(chans[perm[sig_idx]]):
                    continue
                chan = chans[perm[sig_idx]][0]
                ch, coef = check_flip_chan(chan)
                all_chans_idx = all_chans.index(ch)
                signal = coef * data[:, all_chans_idx, :]
                sigs[sig_name] = torch.tensor(signal, dtype=torch.float32)
                these_chans[sig_name] = chan
            perm_str = "eeg: {}, eog: {}".format(*these_chans.values())
            all_sigs[perm_str] = sigs

        for sig_idx, (sig_k, sigs) in enumerate(all_sigs.items()):
            print(f"Inferring permutation {sig_idx+1} of {len(all_sigs)}")
            for k in sigs.keys():
                sigs[k] = sigs[k].reshape(-1, 1, sigs[k].shape[-1])
                sigs[k] = torch.FloatTensor(sigs[k][..., :sig_len])
            with torch.no_grad():
                if use_cuda:
                    for k in sigs.keys():
                        sigs[k] = sigs[k].to("cuda")
                # prepare chunk indices
                hypno_len = len(sigs[k])
                chunk_nn = hypno_len if not chunk_n else chunk_n
                chunks = np.arange(0, hypno_len, chunk_nn)
                chunks = np.append(chunks, hypno_len)
                all_reps = []
                # signal pass
                for c_idx in range(len(chunks[:-1])):
                    these_sigs = {}
                    for k in sigs.keys():
                        these_sigs[k] = sigs[k][chunks[c_idx]:chunks[c_idx+1],]

                    reps = net(these_sigs, rep_output="rep_only")

                    reps = reps.swapaxes(-1, 1)
                    all_reps.append(reps)
                reps = torch.cat(all_reps)

                # RNN pass on abstract representations
                hidden = torch.zeros(10, 1, 256)
                if use_cuda:
                    hidden = hidden.to("cuda")

                y, hidden = con_net(reps, hidden)

                del reps

                inf = torch.argmax(y, dim=-1)
                y = y.float()
                logits.append(y[:,0,].cpu().numpy())

        # calculate consensus
        out_infs = loudest_vote(np.array(logits))

        times = np.arange(start_time, len(out_infs)*30, 30)
        return out_infs, times
