import argparse
import mne
import numpy as np
from os import path
from ..infer import EEGInfer
from ..utils import inst_load, output_stages, pick_chan_names, graph_summary

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
    parser.add_argument("--no_filter", dest="filter", action="store_false")
    parser.add_argument("--graph", dest="graph", action="store_true")
    opt = vars(parser.parse_args())

    inst = inst_load(opt["file"])

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
                                       eog_drop=opt["drop_eog"],
                                       filter=opt["filter"])

    filepath, filename = path.split(opt["file"])
    fileroot, fileext = path.splitext(filename)
    output_stages(stages, times, opt["out_form"], filepath, fileroot)
    
    if opt["graph"]:
        eegs = pick_chan_names(inst, "eeg") if isinstance(eeg, str) else eeg
        graph_summary(stages, times, inst, eegs, filepath, fileroot)

