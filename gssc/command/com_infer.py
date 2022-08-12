import argparse
import mne
import numpy as np
from os import path
import csv
from ..infer import EEGInfer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--out_form", type=str, default="csv")
    parser.add_argument("--cut_from", type=str, default="back")
    parser.add_argument("--use_cuda", default=True)
    parser.add_argument("--use_cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--eeg", action="append", default=["eeg"])
    parser.add_argument("--eog", action="append", default=["eog"])
    parser.add_argument("--drop_eeg", default=True)
    parser.add_argument("--drop_eog", default=True)
    parser.add_argument("--no_drop_eeg", dest="drop_eeg", action="store_false")
    parser.add_argument("--no_drop_eog", dest="drop_eog", action="store_false")
    opt = vars(parser.parse_args())


    ## detect type of input and load as MNE instance
    # raw MNE
    filepath, filename = path.split(opt["file"])
    fileroot, fileext = path.splitext(filename)
    if filename[-8:] == "-raw.fif":
        inst = mne.io.Raw(opt["file"])
    # epoched MNE
    elif filename[-8:] == "-epo.fif":
        inst = mne.Epochs(opt["file"])
    # edf
    elif filename[-4:] == ".edf":
        inst = mne.io.read_raw_edf(opt["file"])
    # brainvision
    elif filename[-5:] == ".vhdr":
        inst = mne.io.read_raw_brainvision(opt["file"])
    # eeglab
    elif filename[-4:] == ".set":
        inst = mne.io.read_raw_edf(opt["file"])
    else:
        raise ValueError("Format of input does not appear to be recognised. "
                         "Try manually converting to MNE-Python format first.")

    eeginfer = EEGInfer(cut=opt["cut_from"], use_cuda=opt["use_cuda"],
                        chunk_n=opt["chunk_size"])
    if len(opt["eeg"])==1:
        eeg = "eeg"
    elif opt["eeg"][1] == "none":
        eeg = []
    else:
        eeg = opt["eeg"][1:]
    if len(opt["eog"])==1:
        eog = "eog"
    elif opt["eog"][1] == "none":
        eog = []
    else:
        eog = opt["eog"][1:]
    stages, times = eeginfer.mne_infer(inst, eeg=eeg, eog=eog,
                                       eeg_drop=opt["drop_eeg"],
                                       eog_drop=opt["drop_eog"])

    if opt["out_form"] == "csv":
        with open(f"{path.join(filepath, fileroot)}.csv", "wt") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([f"Hypnogram of {filename}"])
            csv_writer.writerow(["Epoch", "Time", "Stage"])
            for epo_idx, time, stage in zip(np.arange(len(stages)), times, stages):
                csv_writer.writerow([epo_idx, time, stage])
    elif opt["out_form"] == "mne":
        annot = mne.Annotations(times, 30., stages.astype("str"))
        annot.save(f"{path.join(filepath, fileroot)}-annot.fif")
