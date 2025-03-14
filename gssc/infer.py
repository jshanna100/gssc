import mne
from .utils import (epo_arr_zscore, permute_sigs, check_flip_chan,
                   loudest_vote, prepare_inst)
import torch
import numpy as np
import warnings
from importlib_resources import files

class ArrayInfer():
    """Class for sleep stage inference on a set of EEG data.

    Parameters
    ----------
    net : PyTorch Module (default None)
        Neural network which processes the signals. If None defaults to built-in nets.
    con_net : PyTorch Module (default None)
        Neural network which takes context into account and infers the sleep stage.
        If None defaults to built-in nets.
    use_cuda : bool
        Use CUDA acceleration
    gpu_idx : int
        Index of which GPU to use. Defaults to None (the first GPU).
    """
    def __init__(self, net=None, con_net=None, use_cuda=False, gpu_idx=None):
        if net is None:
            net_name = files('gssc.nets').joinpath("sig_net_v1.pt")
            net = torch.load(net_name, weights_only=False)
        if con_net is None:
            con_net_name = files('gssc.nets').joinpath("gru_net_v1.pt")
            con_net = torch.load(con_net_name, weights_only=False)
        self.net = net
        self.con_net = con_net
        self.use_cuda = use_cuda
        if use_cuda:
            self.net.cuda(device=gpu_idx)
            self.con_net.cuda(device=gpu_idx)
        self.net.eval()
        self.con_net.eval()

    @torch.no_grad()
    def infer(self, sigs, hidden=None):
        """
        Performs inference on a single EEG and/or EOG channel(s)

        Parameters
        ----------
        sigs : dictionary
            The keys to this dictionary should match the keys of the signal
            processing networks - on the built-in networks these are "eeg" 
            and "eog". It is allowed to omit keys, so long as at least one
            is specified. The values the dictionary should be the actual
            signals, as PyTorch torch.float32 tensors. The dimensionality
            of the tensor should be E*1*T where E is the number of epochs
            and T is the number of samples. For the built-in networks, the
            number of samples must be 2560, and these samples must represent
            a time period of 30s, i.e. a sampling rate of 85.33333 Hz.
        hidden : Pytorch float32 tensor
            These represent the hidden state, or "context" of the GRU network.
            For the built-in networks, this tensor should have a shape of
            (10, 1, 256). If not specified, a tensor of zeros will be
            initialised.
            
        Returns
        --------
        logits : Pytorch float32 tensor
            These are the logits, which indicate the probability that the
            classifier assigns to each sleep stage. For each 1x5 row, the 
            column with the highest number indicates the sleep stage, e.g. if
            column 0 has the highest logit, then the classifier infers Wake.
            Columns 0,1,2,3,4 indicate Wake, N1, N2, N3, and REM respectively.
            These can be converted into probabilities by e.g. using the 
            torch.nn.functional softmax function on the final dimension. 
            Tensors have shape of E * 5 with the built-in networks, where E is
            the number of epochs.
        nocontext_logits : Pytorch float32 tensor
            These are exactly the same as the logits, but are inferred without
            taking surrounding context into consideration
        hidden : Pytorch float32 tensor
            This tensor (10, 1, 256) encodes the context. When doing inference
            sequentially, feed this into the next inference call.

        """
        if hidden is None:
            hidden = torch.zeros(10, 1, 256)
        if self.use_cuda:
            for k in sigs.keys():
                sigs[k] = sigs[k].to("cuda")
            hidden = hidden.to("cuda")
        reps, nocontext_logits = self.net(sigs, rep_output=True)
        reps = reps.swapaxes(-1, 1)
        logits, hidden = self.con_net(reps, hidden)
        del reps
        return logits[:,0,], nocontext_logits[...,0], hidden


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
            net = torch.load(net_name, weights_only=False)
        if con_net is None:
            con_net_name = files('gssc.nets').joinpath("gru_net_v1.pt")
            con_net = torch.load(con_net_name, weights_only=False)
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

        Returns
        --------
        out_infs : numpy int array
            Sleep stages for each epoch, integer-coded as follows: 
            0:Wake, 1:N1, 2:N2, 3:N3, 4:REM
        times : numpy int array
            Time in seconds when sleep stages begin
        probs : numpy float array
            Inference as a hypnodensity
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

        for sig_idx, sigs in enumerate(all_sigs.values()):
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

                y = y.float()
                logits.append(y[:,0,].cpu().numpy())

        logits = np.array(logits)
        # calculate consensus 
        out_infs, probs = loudest_vote(logits, return_probs=True)

        times = np.arange(start_time, len(out_infs)*30, 30)
        return out_infs, times, probs
