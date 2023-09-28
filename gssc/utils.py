import mne
import numpy as np
from itertools import combinations, product
import pandas as pd
import re
import torch
from torch.nn import Softmax, NLLLoss
import csv
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from datetime import timedelta

# flatten tensor, but leave batch dimension intact
def _batch_flat(x):
    return x.flatten(start_dim=1).unsqueeze(1)

def param_combos(param_dict):
    combos = list(product(*param_dict.values()))
    comb_dict_list = []
    for comb in combos:
        comb_dict_list.append({k:v for k,v in zip(param_dict.keys(), comb)})
    return comb_dict_list

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
        if sig_dict["chans"] == "eeg_mix":
            chans = ["eeg_mix"]
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
    from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, f1_score
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

def consensus(infs, certs, class_n=5, span=0.2):
    cons_infs = []
    infs = infs[:, np.newaxis] if len(infs.shape) == 1 else infs
    certs = certs[:, np.newaxis] if len(certs.shape) == 1 else certs
    votes = np.zeros((class_n, infs.shape[1]))
    for idx, (inf, cert) in enumerate(zip(infs.T, certs.T)):
        cmin, cmax = cert.min(), cert.max()
        cert_reg = (1 - (cert - cmin) / (cmax - cmin)) * span + (1 - span)
        for i, c in zip(inf, cert_reg):
            votes[i, idx] += c
    opt_infs = np.argmax(votes, axis=0)
    return opt_infs

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
        start_time = inst.times[events[0, 0] - raw.first_samp]
    elif isinstance(inst, mne.epochs.BaseEpochs):
        epo = inst.copy()
        start_time = 0

    epo.load_data()
    epo.resample(sfreq)

    return epo, start_time

def inst_load(filename):
    ## detect type of input and load as MNE instance
    # raw MNE
    if filename[-8:] == "-raw.fif":
        inst = mne.io.Raw(filename, preload=True)
    # epoched MNE
    elif filename[-8:] == "-epo.fif":
        inst = mne.Epochs(filename, preload=True)
    # edf
    elif filename[-4:] == ".edf":
        inst = mne.io.read_raw_edf(filename, preload=True)
    # brainvision
    elif filename[-5:] == ".vhdr":
        inst = mne.io.read_raw_brainvision(filename, preload=True)
    # eeglab
    elif filename[-4:] == ".set":
        inst = mne.io.read_raw_edf(filename, preload=True)
    else:
        raise ValueError("Format of input does not appear to be recognised. "
                         "Try manually converting to MNE-Python format first.")
    return inst

def output_stages(stages, times, out_form, out_dir, fileroot):
    if out_form == "csv":
        with open(f"{join(out_dir, fileroot)}.csv", "wt") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([f"Hypnogram of {fileroot}"])
            csv_writer.writerow(["Epoch", "Time", "Stage"])
            for epo_idx, time, stage in zip(np.arange(len(stages)), times, stages):
                csv_writer.writerow([epo_idx, time, stage])
    elif out_form == "mne":
        annot = mne.Annotations(times, 30., stages.astype("str"))
        annot.save(f"{join(out_dir, fileroot)}-annot.fif")


def graph_summary(stages, times, inst, eegs, outdir, fileroot):
    matplotlib.rc("font", weight="bold")
    stage_col = {0:"blue", 1:"orange", 2:"green", 3:"purple", 4:"cyan"}
    stage_names = {0:"Wake", 1:"N1", 2:"N2", 3:"N3", 4:"REM"}
    stage_proportions = {s:0 for s in stage_names.keys()}

    annot = mne.Annotations(times, 30., stages.astype("str"))
    mos_str = '''
                    AAAAAAAAAACCCC
                    AAAAAAAAAACCCC
                    AAAAAAAAAACCCC
                    BBBBBBBBBBCCCC
                    '''
    fig, axes = plt.subplot_mosaic(mos_str, figsize=(19.2, 8.))
    if len(eegs):
        inst.set_annotations(annot)
        events = mne.events_from_annotations(inst)
        epo = mne.Epochs(inst, events[0], event_id=events[1], tmin=0., tmax=30.,
                            baseline=None, preload=True)
        n_fft = int(np.round(epo.info["sfreq"] * 4))
        psd = epo.compute_psd(fmax=30., method="welch", n_fft=4096,
                              picks=eegs)
        psd_dat = np.log10(np.squeeze(psd.get_data().mean(axis=1)) * 1e12)
        
        axes["A"].imshow(psd_dat.T, aspect="auto", vmin=-1, vmax=3., cmap="jet")
        yticks = np.arange(0, len(psd.freqs), 30)
        axes["A"].set_yticks(yticks)
        axes["A"].set_yticklabels([f"{psd.freqs[y]:.1f}" for y in yticks])
        axes["A"].invert_yaxis()
        axes["A"].set_ylabel("Hz", weight="bold")
        axes["A"].set_xticks([])
    else:
        axes["A"].axis("off")

    stages = [int(x["description"]) for idx, x in enumerate(annot)]
    axes["B"].set_xlim(0, len(stages))
    bar_w = 1
    for idx, stage in enumerate(stages):
        axes["B"].add_patch(Rectangle((idx, 0), bar_w, 1, color=stage_col[stage]))
    axes["B"].set_xlabel("Time", weight="bold")
    # figure out the xticks, convert to hh:mm format
    xticks = np.arange(0, len(stages), 60) # xticks every 30m
    xticklabels = [str(timedelta(seconds=int(xt*30)))[:-3] for xt in xticks]
    axes["B"].set_xticks(xticks, labels=xticklabels)
    axes["B"].set_yticks([])

    unqs, counts = np.unique(stages, return_counts=True)
    proportions = counts / len(stages)
    for unq, proportion in zip(unqs, proportions):
        stage_proportions[unq] = proportion
    axes["C"].bar(list(stage_names.values()), list(stage_proportions.values()), 
                  color=list(stage_col.values()))
    axes["C"].set_title("Proportions", weight="bold")

    plt.suptitle(fileroot, weight="bold")
    plt.tight_layout()
    plt.savefig(join(outdir, f"{fileroot}.png"))
    plt.close("all")