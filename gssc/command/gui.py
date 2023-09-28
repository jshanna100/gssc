import tkinter as tk
from tkinter import font
from tkinter import filedialog
import torch
from os.path import join, splitext, split
import glob
import mne
from gssc.infer import EEGInfer
from gssc.utils import output_stages, inst_load, graph_summary

def add_files():
    files = filedialog.askopenfilenames()
    if len(files):
        filelist.insert(tk.END, *files)

def add_dirs():
    extensions = [".vhdr", "-raw.fif", ".edf", ".set"]
    directory = filedialog.askdirectory()
    if not len(directory):
        return
    files = []
    for ext in extensions:
        files.extend(glob.glob(join(directory, f"*{ext}")))
    files = [join(directory, file) for file in files]
    filelist.insert(tk.END, *files)

def remove_files():
    file_idx = filelist.curselection()
    if not len(file_idx):
        return
    filelist.delete(file_idx)

def clear_files():
    filelist.delete(0, tk.END)

def add_eeg():
    ch_inds = channellist.curselection()
    if not len(ch_inds):
        return
    chs_to_add = [channellist.get(x) for x in ch_inds]
    eeglist.insert(tk.END, *chs_to_add)
    channellist.selection_clear(0, channellist.size())
    
def remove_eeg():
    eeg_inds = eeglist.curselection()
    if not len(eeg_inds):
        return
    del_idx = 0
    for eeg_idx in eeg_inds:
        eeglist.delete(eeg_idx-del_idx)
        del_idx += 1

def add_eog():
    ch_inds = channellist.curselection()
    if not len(ch_inds):
        return
    chs_to_add = [channellist.get(x) for x in ch_inds]
    eoglist.insert(tk.END, *chs_to_add)
    channellist.selection_clear(0, channellist.size())

def remove_eog():
    eog_inds = eoglist.curselection()
    if not len(eog_inds):
        return
    del_idx = 0
    for eog_idx in eog_inds:
        eoglist.delete(eog_idx-del_idx)
        del_idx += 1

def get_chan():
    file_idx = filelist.curselection()
    if not len(file_idx):
        return
    file = filelist.get(file_idx[0])
    if ".vhdr" in file:
        raw = mne.io.read_raw_brainvision(file)
    elif ".edf" in file:
        raw = mne.io.read_raw_edf(file)
    elif "-raw.fif" in file:
        raw = mne.io.Raw(file)
    else:
        raise ValueError("Not a recognised EEG format.")
    channellist.delete(0, tk.END)
    channellist.insert(tk.END, *raw.ch_names)

def score():
    # do some checks
    if not eeglist.size() and not eoglist.size():
        tk.messagebox.showerror("No electrodes selected",
                                "Error: you have not selected any EEG or EOG electrodes to use for inference.")
        return
    if outformatVar.get() == "Output format":
        tk.messagebox.showerror("No output format selected",
                                "Error: you have not selected an output format.")
        return
    if epo_chunknVar.get().casefold() == "Max".casefold():
        chunk_n = 0
    elif epo_chunknVar.get().isnumeric():
        chunk_n = int(epo_chunknVar.get())
    else:
        tk.messagebox.showerror("Invalid chunk size",
                                "Error: Chunk size must either be 'Max' or a positive integer.")
        return
        
    outform_dict = {"MNE-Python":"mne", "Text":"csv"}
    outform = outform_dict[outformatVar.get()]
    eeginfer = EEGInfer(use_cuda=use_cudaVar.get(), chunk_n=chunk_n)
    files = filelist.get(0, filelist.size())
    eegs = list(eeglist.get(0, eeglist.size()))
    eogs = list(eoglist.get(0, eoglist.size()))
    for idx, file in enumerate(files):
        if filelist.itemcget(idx, "foreground") == "blue":
            continue # already succesfully done at an earlier pass
        try:
            inst = inst_load(file)
            stages, times = eeginfer.mne_infer(inst, eeg=eegs, eog=eogs, 
                                               eeg_drop=eeg_dropVar.get(), 
                                               eog_drop=eog_dropVar.get(),
                                               filter=filterVar.get())
            filepath, filename = split(file)
            fileroot, fileext = splitext(filename)
            outdir = filepath if outdirVar.get() == "" else outdirVar.get()
            output_stages(stages, times, outform, outdir, fileroot)
            print("Sleep stages written.")
            if graphVar:
                print("Producing graphic summary...")
                graph_summary(stages, times, inst, eegs, outdir, fileroot)
        except Exception as e:
            print(f"\nError: could not infer {file}.\n{e}")
            filelist.itemconfig(idx, foreground="red")
            continue
        filelist.itemconfig(idx, foreground="blue")
    print("\nDone!\n")

def outdir_dialog():
    out_dir = filedialog.askdirectory()
    if not len(out_dir):
        return
    outdirVar.set(out_dir)

root = tk.Tk()
frame = tk.Frame(root)
root.title("Greifswald Sleep Stage Classifier")


# vars
avail_cudaVar = tk.BooleanVar()
use_cudaVar = tk.BooleanVar()
eeg_dropVar = tk.BooleanVar()
eog_dropVar = tk.BooleanVar()
epo_chunknVar = tk.StringVar()
outdirVar = tk.StringVar()
outformatVar = tk.StringVar()
filterVar = tk.BooleanVar()
graphVar = tk.BooleanVar()

if torch.cuda.is_available():
    avail_cudaVar.set(True)

# fonts
butt_font = tk.font.Font(size=14, weight="bold")
list_font = tk.font.Font(size=10, weight="bold")

# filenames
file_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
filelist = tk.Listbox(frame, font=list_font, yscrollcommand=file_scroll.set)
file_scroll["command"] = filelist.yview
# file buttons
add_file = tk.Button(frame, text="Add File", font=butt_font, command=add_files)
add_dir = tk.Button(frame, text="Add Directory", font=butt_font, command=add_dirs)
remove_file = tk.Button(frame, text="Remove File", font=butt_font,
                        command=remove_files)
clear = tk.Button(frame, text="Clear all", font=butt_font, command=clear_files)
filelabel = tk.Label(frame, text="PSG Files", font=butt_font)

# channel lists
channel_label = tk.Label(frame, text="Available electrodes", font=butt_font)
channel_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
channellist = tk.Listbox(frame, font=list_font, yscrollcommand=channel_scroll.set,
                        selectmode="multiple")
channel_scroll["command"] = channellist.yview
get_chan_butt = tk.Button(frame, text="Get EEG channels", font=butt_font,
                        command=get_chan)
eeg_label = tk.Label(frame, text="Use EEG chanels", font=butt_font)
eeg_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
eeglist = tk.Listbox(frame, font=list_font, yscrollcommand=eeg_scroll.set,
                    selectmode="multiple")
eeg_scroll["command"] = eeglist.yview

eog_label = tk.Label(frame, text="Use EOG channels", font=butt_font)
eog_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
eoglist = tk.Listbox(frame, font=list_font, yscrollcommand=eog_scroll.set,
                        selectmode="multiple")
eog_scroll["command"] = eoglist.yview

# channel buttons
eeg_add = tk.Button(frame, text="Add Elec", font=butt_font,
                    command=add_eeg)
eeg_remove = tk.Button(frame, text="Remove Elec", font=butt_font,
                    command=remove_eeg)
eog_add = tk.Button(frame, text="Add Elec", font=butt_font,
                    command=add_eog)
eog_remove = tk.Button(frame, text="Remove Elec", font=butt_font,
                    command=remove_eog)

# inference parameters
inf_label = tk.Label(frame, text="Sleep staging", font=butt_font)
outdir_butt = tk.Button(frame, text="Output directory", font=butt_font,
                        command=outdir_dialog)
outformat_menu = tk.OptionMenu(frame, outformatVar,
                            "MNE-Python", "Text")
outformatVar.set("Output format")
outformat_menu.config(font=butt_font)

epo_chunkn_label = tk.Label(frame, text="Chunk size", font=butt_font, anchor="w")
epo_chunkn_entry = tk.Entry(frame, textvariable=epo_chunknVar, font=butt_font,
                            width=5)
epo_chunkn_entry.insert(0, "Max")
drop_label = tk.Label(frame, text="Allow Null Permutation", font=butt_font, anchor="w")
drop_eeg = tk.Checkbutton(frame, text="EEG", font=butt_font,
                        variable=eeg_dropVar, anchor="w")
drop_eeg.select()
drop_eog = tk.Checkbutton(frame, text="EOG", font=butt_font,
                        variable=eog_dropVar, anchor="w")
drop_eog.select()
score_butt = tk.Button(frame, text="Score", command=score, font=butt_font)
do_filter = tk.Checkbutton(frame, text="Filter", font=butt_font,
                        variable=filterVar, anchor="w")
do_filter.select()
do_graph = tk.Checkbutton(frame, text="Graphic summary", font=butt_font,
                        variable=graphVar, anchor="w")

# cuda
use_cuda = tk.Checkbutton(frame, text="Use CUDA", variable=use_cudaVar,
                        font=butt_font, anchor="w")
if avail_cudaVar.get():
    use_cuda.select()
else:
    use_cuda.state = tk.DISABLED

## grids
frame.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
# files
filelabel.grid(column=0, row=0, columnspan=4)
filelist.grid(column=0, row=2, columnspan=4, rowspan=15,
            sticky=(tk.N, tk.S, tk.E, tk.W))
add_file.grid(column=0, row=1, sticky=(tk.E, tk.W))
add_dir.grid(column=1, row=1, sticky=(tk.E, tk.W))
remove_file.grid(column=2, row=1, sticky=(tk.E, tk.W))
clear.grid(column=3, row=1, sticky=(tk.E, tk.W))
file_scroll.grid(column=4, row=2, rowspan=15, sticky=(tk.N, tk.S))

# channels
channel_label.grid(column=5, row=0, sticky=(tk.E, tk.W))
channellist.grid(column=5, row=2, columnspan=3, rowspan=15,
                sticky=(tk.N, tk.S, tk.E, tk.W))
get_chan_butt.grid(column=5, row=1, columnspan=3, sticky=(tk.E, tk.W))
channel_scroll.grid(column=8, row=2, rowspan=15, sticky=(tk.N, tk.S))

eeg_label.grid(column=9, row=0, columnspan=4, sticky=(tk.E, tk.W))
eeglist.grid(column=9, row=2, columnspan=4, rowspan=3,
                sticky=(tk.N, tk.S, tk.E, tk.W))
eeg_add.grid(column=9, row=1, columnspan=2, sticky=(tk.E, tk.W))
eeg_remove.grid(column=11, row=1, columnspan=2, sticky=(tk.E, tk.W))
eeg_scroll.grid(column=13, row=2, rowspan=3, sticky=(tk.N, tk.S))

eog_label.grid(column=9, row=5, columnspan=4, sticky=(tk.E, tk.W))
eoglist.grid(column=9, row=7, columnspan=4, rowspan=3,
                sticky=(tk.N, tk.S, tk.E, tk.W))
eog_add.grid(column=9, row=6, columnspan=2, sticky=(tk.E, tk.W))
eog_remove.grid(column=11, row=6, columnspan=2, sticky=(tk.E, tk.W))
eog_scroll.grid(column=13, row=7, rowspan=3, sticky=(tk.N, tk.S))

# inference
inf_label.grid(column=9, row=10, columnspan=4, sticky=(tk.E, tk.W))
outdir_butt.grid(column=12, row=11, columnspan=1, sticky=(tk.E, tk.W))
outformat_menu.grid(column=12, row=12, columnspan=1, sticky=(tk.E, tk.W))
use_cuda.grid(column=9, row=11, columnspan=1, sticky=(tk.E, tk.W))
do_filter.grid(column=10, row=11, columnspan=1, sticky=(tk.E, tk.W))
do_graph.grid(column=12, row=13, columnspan=1, sticky=(tk.E, tk.W))
epo_chunkn_label.grid(column=9, row=12, sticky=(tk.E, tk.W))
epo_chunkn_entry.grid(column=10, row=12, sticky=(tk.E, tk.W))
drop_label.grid(column=9, row=13, columnspan=2, sticky=(tk.E, tk.W))
drop_eeg.grid(column=9, row=14, sticky=(tk.E, tk.W))
drop_eog.grid(column=10, row=14, sticky=(tk.E, tk.W))
score_butt.grid(column=12, row=14, rowspan=2, sticky=(tk.E, tk.W))

tk.Grid.columnconfigure(root, 0, weight=1)
tk.Grid.rowconfigure(root, 0, weight=1)
tk.Grid.columnconfigure(frame, 0, weight=1)
tk.Grid.columnconfigure(frame, 1, weight=1)
tk.Grid.columnconfigure(frame, 2, weight=1)
tk.Grid.columnconfigure(frame, 3, weight=1)
tk.Grid.columnconfigure(frame, 4, weight=0)
tk.Grid.columnconfigure(frame, 5, weight=1)
tk.Grid.columnconfigure(frame, 6, weight=1)
tk.Grid.columnconfigure(frame, 7, weight=1)
tk.Grid.columnconfigure(frame, 8, weight=0)
tk.Grid.columnconfigure(frame, 9, weight=1)
tk.Grid.columnconfigure(frame, 10, weight=1)
tk.Grid.columnconfigure(frame, 11, weight=1)
tk.Grid.columnconfigure(frame, 12, weight=1)
tk.Grid.columnconfigure(frame, 13, weight=0)
tk.Grid.rowconfigure(frame, 0, weight=0)
tk.Grid.rowconfigure(frame, 1, weight=0)
tk.Grid.rowconfigure(frame, 2, weight=1)
tk.Grid.rowconfigure(frame, 3, weight=1)
tk.Grid.rowconfigure(frame, 4, weight=1)
tk.Grid.rowconfigure(frame, 5, weight=0)
tk.Grid.rowconfigure(frame, 6, weight=0)
tk.Grid.rowconfigure(frame, 7, weight=1)
tk.Grid.rowconfigure(frame, 8, weight=1)
tk.Grid.rowconfigure(frame, 9, weight=1)
tk.Grid.rowconfigure(frame, 10, weight=0)
tk.Grid.rowconfigure(frame, 11, weight=0)
tk.Grid.rowconfigure(frame, 12, weight=0)
tk.Grid.rowconfigure(frame, 13, weight=0)
tk.Grid.rowconfigure(frame, 14, weight=0)
tk.Grid.rowconfigure(frame, 15, weight=0)

def main():
    # start
    root.mainloop()
