import mne
import numpy as np
from itertools import combinations, product
import re
import torch
from torch.nn import Softmax, NLLLoss
#from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, f1_score


def make_blueprint(start_feat_exp, end_feat_exp, feat_steps, bn_factor, depth):
    f = [int(np.round(2**x)) for x in np.linspace(start_feat_exp,
                                                  end_feat_exp,
                                                  feat_steps)]
    f = [fn_aprx(ff) for ff in f]
    blueprint = {}
    block_idx = 0

    for fi in [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]:
        for d in range(depth):
            blueprint["Block {}".format(block_idx)] = {"in_feat":f[fi[0]],
                                                       "out_feat":f[fi[0]],
                                                       "downsamp":False}
            block_idx += 1
        blueprint["Block {}".format(block_idx)] = {"in_feat":f[fi[0]],
                                                   "out_feat":f[fi[1]],
                                                   "downsamp":True}
        block_idx += 1
    if bn_factor:
        blueprint["Bottleneck"] = {"feat":f[6],
                                   "mid_feat":int(np.round(f[6]/bn_factor))}
    for fi in [(6,7), (7,8), (8,9)]:
        for d in range(depth):
            blueprint["Block {}".format(block_idx)] = {"in_feat":f[fi[0]],
                                                       "out_feat":f[fi[0]],
                                                       "downsamp":False}
            block_idx += 1
        blueprint["Block {}".format(block_idx)] = {"in_feat":f[fi[0]],
                                                   "out_feat":f[fi[1]],
                                                   "downsamp":True}
        block_idx += 1
    for d in range(depth):
        blueprint["Block {}".format(block_idx)] = {"in_feat":f[fi[1]],
                                                   "out_feat":f[fi[1]],
                                                   "downsamp":False}
        block_idx += 1
    return blueprint

def calc_accuracy(input, target, dim=1):
    preds = torch.argmax(input, dim)
    acc_percent = (sum(preds == target) / len(preds)) * 100
    return acc_percent

def check_flip_chan(chan):
    match = re.match("FLIP_(.*)", chan)
    if match:
        chan_name = match.groups(1)[0]
        coef = -1
    else:
        chan_name = chan
        coef = 1
    return chan_name, coef

def pick_chan_names(inst, chans):
    if chans == "eeg" or chans == "eeg_mix":
        chans = [inst.ch_names[c] for c in
                 mne.pick_types(inst.info, eeg=True)]
    elif chans == "eog":
        chans = [inst.ch_names[c] for c in
                 mne.pick_types(inst.info, eog=True)]
    elif chans == "ecg":
        chans = [inst.ch_names[c] for c in
                 mne.pick_types(inst.info, ecg=True)]
    elif chans == "emg":
        chans = [inst.ch_names[c] for c in
                 mne.pick_types(inst.info, emg=True)]
    return chans

def check_null_perm(sig_combs, perm_mat):
    keys = list(sig_combs.keys())
    bad_row_inds = []
    for perm_row_idx in range(len(perm_mat)):
        atleast_one = False
        for perm_col, perm_val in enumerate(perm_mat[perm_row_idx]):
            chan = sig_combs[keys[perm_col]][perm_val]
            if chan:
                atleast_one = True
        if not atleast_one:
            bad_row_inds.append(perm_row_idx)
    perm_mat = np.delete(perm_mat, bad_row_inds, axis=0)
    #print("Deleted {} null permutations.".format(len(bad_row_inds)))
    return perm_mat

def check_annot_lengths(raw, dur):
    # check that all annotations are of equal duration, output that
    # duration, as well as its length in samples
    rems = []
    for idx, annot in enumerate(raw.annotations):
        if annot["duration"] > dur+1 or annot["duration"] < dur-1:
            print("Removed annotations of unequal duration.")
            rems.append(idx)
    dur_idx = raw.time_as_index(dur)
    raw.annotations.delete(rems)
    return dur, dur_idx

def raw_to_epo(rawfile, target_length, val_dict=None, all_chans=None):
    if val_dict is None:
        val_dict = {"0":0, "1":1, "2":2, "3":3, "4":4}
    if isinstance(rawfile, str):
        raw = mne.io.Raw(rawfile)
    else:
        raw = rawfile
    # convert all channels to upper-case, against inconsitent naming
    up_dict = {ch:ch.upper() for ch in raw.ch_names}
    raw.rename_channels(up_dict)
    # check if channel names match
    if all_chans:
        for ch in all_chans:
            if ch.upper() not in raw.ch_names:
                raise ValueError("Channels do not match template.")
    dur, dur_idx = check_annot_lengths(raw, 30.)
    sfreq = target_length / dur
    print("Resampling to {} to achieve target length of {}"
          " samples.".format(sfreq, target_length))
    raw.resample(sfreq)
    dur, dur_idx = check_annot_lengths(raw, 30.)
    events = mne.events_from_annotations(raw, event_id=val_dict)
    epo = mne.Epochs(raw, events[0], tmin=0, tmax=dur, baseline=None,
                     preload=True)
    return epo, dur, dur_idx

def permute_sigs(temp_inst, signals, all_upper=False):
    # build up the sig_combs dictionary which for each signal
    # specifies the combinations of channels that can go into it
    all_chans = []
    sig_combs = {}
    # build up the different combos
    for sig_name, sig_dict in signals.items():
        this_combo = []
        # what channels can comprise each signal
        chans = sig_dict["chans"]
        if isinstance(chans, str):
            chans = pick_chan_names(temp_inst, chans)
        if all_upper:
            chans = [ch.upper() for ch in chans]
        all_chans.extend(chans)
        if sig_name == "eeg":
            eeg_chans = chans.copy()
        if "flip" in sig_dict.keys() and sig_dict["flip"] == True:
            chans.extend(["FLIP_"+c for c in chans])
        perms = [0, 1] if sig_dict["drop"] else [1]
        for perm_n in perms:
            this_combo.extend(combinations(chans, perm_n))
        sig_combs[sig_name] = this_combo
    del temp_inst

    # use sig_combs to build a grid of all possible signal permutations
    perm_matrix = np.meshgrid(*[np.arange(len(x))
                              for x in sig_combs.values()])
    perm_matrix = np.array(perm_matrix).T.reshape(-1, len(signals))
    # disallow permutions with no signal input at all
    perm_matrix = check_null_perm(sig_combs, perm_matrix)

    # all uppercase for names
    if all_upper:
        all_chans = [ac.upper() for ac in all_chans]

    return sig_combs, perm_matrix, all_chans, eeg_chans

def check_times(inst, times):
    if not np.array_equal(inst.times, times):
        raise ValueError("Times do not match.")

def epo_arr_zscore(epo_arr, cap=None):
    for ch_idx in range(epo_arr.shape[1]):
        e_flat = epo_arr[:,ch_idx].flatten()
        mean = e_flat.mean()
        std = e_flat.std()
        epo_arr[:, ch_idx] = (epo_arr[:, ch_idx] - mean) / std
        if cap:
            epo_arr[epo_arr > cap] = cap
            epo_arr[epo_arr < -cap] = -cap
    return epo_arr

def fn_aprx(feat_num, stepsize=16, max=512):
    # find number nearest to feat_num which is divisible by stepsize
    feat_range = np.arange(stepsize, max+1, stepsize)
    feat_out = feat_range[np.argmin(np.abs(feat_range - feat_num))]
    return feat_out

def score_infs(true, pred):
    mc = matthews_corrcoef(true, pred)
    ck = cohen_kappa_score(true, pred)
    f1 = f1_score(true, pred, average=None)
    f1_macro = f1_score(true, pred, average="macro")
    acc = np.sum(true==pred) / len(true)

    return mc, ck, f1, f1_macro, acc

def loudest_vote(logits):
    loss_func = NLLLoss(reduction="none")
    logits = torch.FloatTensor(np.array(logits))
    entrs = torch.FloatTensor(logits.shape[:2])
    # calculate entropy loss
    for idx in range(len(logits)):
        targs = torch.LongTensor(np.argmax(logits[idx], axis=-1))
        entrs[idx] = loss_func(logits[idx], targs)
    # assemble logits of minimal entropy across the PSG
    min_inds = np.argmin(entrs, axis=0)
    min_logits = logits[min_inds, np.arange(logits.shape[1])]
    out_infs  = np.array(np.argmax(min_logits, axis=1))
    return out_infs

def prepare_inst(inst, sig_len, cut):
    sfreq = sig_len / 30.
    if isinstance(inst, mne.io.base.BaseRaw):
        raw = inst.copy()
        dur = raw.times[-1]
        if dur % 30.:
            overshoot = dur - (dur//30. * 30)
            if cut == "back":
                print("Cutting {} seconds from the back.".format(overshoot))
                raw.crop(tmin=0, tmax=dur-overshoot)
            if cut == "front":
                print("Cutting {} seconds from the front.".format(overshoot))
                raw.crop(tmin=overshoot)
        events = mne.make_fixed_length_events(raw, duration=30.)
        epo = mne.Epochs(raw, events, tmin=0, tmax=30, picks=raw.ch_names,
                         baseline=None)
    elif isinstance(inst, mne.epochs.BaseEpochs):
        if epo.times[-1] != 30.:
            raise ValueError("Epoch lengths must be exactly 30 seconds. "
                             f"Length of these epochs is {epo.times[-1]}.")
        epo = inst.copy()
        overshoot = 0

    epo.load_data()
    epo.resample(sfreq)

    start_time = 0 if cut == "back" else overshoot

    return epo, start_time
