import tkinter as tk
from tkinter import font
from tkinter import filedialog
import torch

def add_files():
    files = filedialog.askopenfilenames()
    filelist.insert(tk.END, *files)

def remove_files():
    filelist.delete(filelist.curselection())

def clear_files():
    filelist.delete(0, tk.END)

def add_eeg():
    pass

def remove_eeg():
    pass

def add_eog():
    pass

def remove_eog():
    pass

def get_chan():
    pass

def score():
    pass

def outdir_dialog():
    pass

root = tk.Tk()
frame = tk.Frame(root)

# vars
avail_cudaVar = tk.BooleanVar()
use_cudaVar = tk.BooleanVar()
eeg_dropVar = tk.BooleanVar()
eog_dropVar = tk.BooleanVar()
epo_chunknVar = tk.StringVar()
outdirVar = tk.StringVar()
outformatVar = tk.StringVar()

if torch.cuda.is_available():
    avail_cudaVar = True

# fonts
butt_font = tk.font.Font(size=14, weight="bold")
list_font = tk.font.Font(size=10, weight="bold")

# filenames
file_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
filelist = tk.Listbox(frame, font=list_font, yscrollcommand=file_scroll.set)
file_scroll["command"] = filelist.yview
# file buttons
add_file = tk.Button(frame, text="Add Files", font=butt_font, command=add_files)
remove_file = tk.Button(frame, text="Remove File", font=butt_font,
                        command=remove_files)
clear = tk.Button(frame, text="Clear all", font=butt_font, command=clear_files)
filelabel = tk.Label(frame, text="PSG Files", font=butt_font)

# channel lists
channel_label = tk.Label(frame, text="Available electrodes", font=butt_font)
channel_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
channellist = tk.Listbox(frame, font=list_font, yscrollcommand=channel_scroll)
channel_scroll["command"] = channellist.yview
get_chan_butt = tk.Button(frame, text="Get EEG file channels", font=butt_font,
                          command=get_chan)
eeg_label = tk.Label(frame, text="Use EEG electrodes", font=butt_font)
eeg_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
eeglist = tk.Listbox(frame, font=list_font, yscrollcommand=eeg_scroll)
eeg_scroll["command"] = eeglist.yview

eog_label = tk.Label(frame, text="Use EOG electrodes", font=butt_font)
eog_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
eoglist = tk.Listbox(frame, font=list_font, yscrollcommand=eog_scroll)
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
outdir_butt = tk.Button(frame, text="Select output directory", font=butt_font,
                        command="outdir_dialog")
outformat_menu = tk.OptionMenu(frame, outformatVar,
                               "MNE-Python", "Text", "Brainvision")
outformatVar.set("Output format")
outformat_menu.config(font=butt_font)

use_cuda = tk.Checkbutton(frame, text="Use CUDA", variable=use_cudaVar,
                          font=butt_font, anchor="w")
if avail_cudaVar:
    use_cuda.select()
else:
    use_cuda.state = tk.DISABLED
epo_chunkn_label = tk.Label(frame, text="Chunk size", font=butt_font, anchor="w")
epo_chunkn_entry = tk.Entry(frame, textvariable=epo_chunknVar, font=butt_font,
                            width=5)
epo_chunkn_entry.insert(0, "Max")
drop_label = tk.Label(frame, text="Null allowed", font=butt_font, anchor="w")
drop_eeg = tk.Checkbutton(frame, text="EEG", font=butt_font,
                          variable=eeg_dropVar, anchor="w")
drop_eeg.select()
drop_eog = tk.Checkbutton(frame, text="EOG", font=butt_font,
                          variable=eog_dropVar, anchor="w")
drop_eog.select()
score_butt = tk.Button(frame, text="Score", command="score", font=butt_font)


## grids
frame.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
# files
filelabel.grid(column=0, row=0, columnspan=3)
filelist.grid(column=0, row=2, columnspan=3, rowspan=15,
              sticky=(tk.N, tk.S, tk.E, tk.W))
file_scroll.grid(column=3, row=2, rowspan=15, sticky=(tk.N, tk.S))
add_file.grid(column=0, row=1, sticky=(tk.E, tk.W))
remove_file.grid(column=1, row=1, sticky=(tk.E, tk.W))
clear.grid(column=2, row=1, sticky=(tk.E, tk.W))
# channels
channel_label.grid(column=4, row=0, sticky=(tk.E, tk.W))
channellist.grid(column=4, row=2, columnspan=2, rowspan=15,
                 sticky=(tk.N, tk.S, tk.E, tk.W))
channel_scroll.grid(column=7, row=2, rowspan=15,
                    sticky=(tk.N, tk.S))
get_chan_butt.grid(column=4, row=1, columnspan=2, sticky=(tk.E, tk.W))

eeg_label.grid(column=8, row=0, columnspan=2, sticky=(tk.E, tk.W))
eeglist.grid(column=8, row=2, columnspan=4, rowspan=3,
                 sticky=(tk.N, tk.S, tk.E, tk.W))
eeg_scroll.grid(column=12, row=2, rowspan=4, sticky=(tk.N, tk.S))
eeg_add.grid(column=8, row=1, columnspan=2, sticky=(tk.E, tk.W))
eeg_remove.grid(column=10, row=1, columnspan=2, sticky=(tk.E, tk.W))

eog_label.grid(column=8, row=5, columnspan=2, sticky=(tk.E, tk.W))
eoglist.grid(column=8, row=7, columnspan=4, rowspan=3,
                 sticky=(tk.N, tk.S, tk.E, tk.W))
eog_scroll.grid(column=12, row=7, rowspan=3, sticky=(tk.N, tk.S))
eog_add.grid(column=8, row=6, columnspan=2, sticky=(tk.E, tk.W))
eog_remove.grid(column=10, row=6, columnspan=2, sticky=(tk.E, tk.W))
# inference
inf_label.grid(column=8, row=10, columnspan=4, sticky=(tk.E, tk.W))
outdir_butt.grid(column=8, row=11, columnspan=2, sticky=(tk.E, tk.W))
outformat_menu.grid(column=10, row=11, columnspan=2, sticky=(tk.E, tk.W))
use_cuda.grid(column=10, row=13, columnspan=2, sticky=(tk.E, tk.W))
epo_chunkn_label.grid(column=10, row=14, sticky=(tk.E, tk.W))
epo_chunkn_entry.grid(column=11, row=14, sticky=(tk.E, tk.W))
drop_label.grid(column=8, row=13, columnspan=2, sticky=(tk.E, tk.W))
drop_eeg.grid(column=8, row=14, columnspan=2, sticky=(tk.E, tk.W))
drop_eog.grid(column=8, row=15, columnspan=2, sticky=(tk.E, tk.W))
score_butt.grid(column=10, row=15, columnspan=2, sticky=(tk.E, tk.W))

tk.Grid.columnconfigure(root, 0, weight=1)
tk.Grid.rowconfigure(root, 0, weight=1)
tk.Grid.columnconfigure(frame, 0, weight=1)
tk.Grid.columnconfigure(frame, 1, weight=1)
tk.Grid.columnconfigure(frame, 2, weight=1)
tk.Grid.columnconfigure(frame, 3, weight=0)
tk.Grid.columnconfigure(frame, 4, weight=1)
tk.Grid.columnconfigure(frame, 5, weight=1)
tk.Grid.columnconfigure(frame, 6, weight=0)
tk.Grid.columnconfigure(frame, 7, weight=0)
tk.Grid.columnconfigure(frame, 8, weight=1)
tk.Grid.columnconfigure(frame, 9, weight=1)
tk.Grid.columnconfigure(frame, 10, weight=1)
tk.Grid.columnconfigure(frame, 11, weight=1)
tk.Grid.columnconfigure(frame, 12, weight=0)
tk.Grid.rowconfigure(frame, 0, weight=0)
tk.Grid.rowconfigure(frame, 1, weight=0)
tk.Grid.rowconfigure(frame, 2, weight=3)
tk.Grid.rowconfigure(frame, 3, weight=3)
tk.Grid.rowconfigure(frame, 4, weight=3)
tk.Grid.rowconfigure(frame, 5, weight=0)
tk.Grid.rowconfigure(frame, 6, weight=0)
tk.Grid.rowconfigure(frame, 7, weight=3)
tk.Grid.rowconfigure(frame, 8, weight=3)
tk.Grid.rowconfigure(frame, 9, weight=3)
tk.Grid.rowconfigure(frame, 10, weight=0)
tk.Grid.rowconfigure(frame, 11, weight=0)
tk.Grid.rowconfigure(frame, 12, weight=0)
tk.Grid.rowconfigure(frame, 13, weight=0)
tk.Grid.rowconfigure(frame, 14, weight=0)
tk.Grid.rowconfigure(frame, 15, weight=0)

# start
root.mainloop()
