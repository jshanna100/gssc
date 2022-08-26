Greifswald Sleep Stage Classifier
=================================

Version 0.0.3

## What is this?

The Greifswald Sleep Stage Classifier (GSSC) is an automatic sleep stage
classifer that takes an EEG or Polysomnography (PSG) recording as input
and outputs a hypnogram. It is deep learning based, and therefore runs best
on a CUDA-capable GPU, i.e. with an NVIDIA graphics card, preferably not
more than five years old. It will also run without a CUDA GPU, though
significantly slower.

## Installation

### 1. Install Python
The GSSC runs within Python, so you will need to have that installed on your
computer in one form or another. We recommend
[Anaconda Distribution](https://www.anaconda.com/products/distribution)
for better management of environments for different projects.
In the case that you use Anaconda for Python, you should create an environment
for gssc. Input the following in the Anaconda terminal:

    conda create --name gssc

Before moving on with the installation, be sure to activate the environment:

    conda activate gssc

Also activate the environment anytime you wish to use gssc in the future.

### 2. Install PyTorch
The GSSC uses PyTorch for deep learning implementation. All of the software
the GSSC needs to run will be installed automatically along with the GSSC, but
because the appropriate PyTorch version varies depending on operating system,
Python version, CUDA availability, etc. it is recommended that you install it
yourself by going to the front page of [PyTorch]("https://pytorch.org") and
following the instructions for your particular system.

### 3. Install GSCC
If you have sucessfully completed the first two steps, you can now install
the GSSC with:

    pip install gssc

or:

    pip3 install gssc

## Use

### GUI

Coming soon!

### Command line

The GSSC can be used from the command line with gssc_infer, and by specifying
the filename e.g.:

    gssc_infer myeegfile-raw.fif

Internally, the GSSC works with EEG data in MNE Python -raw.fif or -epo.fif
format, and it is recommended that you convert your data to that format first.
Nevertheless, gssc_infer will also accept files in .edf, .vhdr (Brainvision),
or .set (EEGLAB) format and attempt to convert them. In the case that you input
epoched data, make sure that all epochs are exactly 30 seconds long. Note that
in the case where you do not explicitly specify channels, the GSSC will use
the EEG/EOG identifications that it finds. There is no guarantee that these
are correct, especially when you convert from another format, so either convert
to MNE by hand and make sure the channel IDs are correct, or specify the
channels to use with parameters (see "channels and combinations" below).

The GSSC was trained on data that were bandpass filtered at 0.3-30Hz, and
we strongly recommend you also filter your data the same way for inference.

### Integration into Python code

You can also integrate the GSSC into your Python code, see the API
(coming soon!)

### Channels and combinations

By default, the GSSC will perform a separate inference for each possible
combination of EEG and EOG channels, and, for each 30s period, will use the
inference which is most likely to be correct. These combinations include the
possibility of using no EEG or no EOG channel. For example, if there are two
EEG channels (C3 and C4), and one EOG channel (HEOG), this results in 5
combinations: C3 and HEOG, C4 and HEOG, C3 alone, C4 alone, and HEOG alone.
With more channels, it is easy to see that the number of combinations can
quickly become very high. It is sensible then to limit the number of channels
used for inference.

You can do this manually by removing all the irrelevant channels from the file
yourself, or by specifying the channels to GSSC. At the command line this is
accomplished with the use of the --eeg and --eog parameters. For example:

    gssc_infer myeegfile-raw.fif --eeg C3 --eeg C4 --eog "EOG L"

will use C3 and C4 as possible EEG channels and EOG L as the EOG channel - note
we have used quotes around EOG L in the command line to deal with the space
in the channel name. This configuration will also make combinations with no
EEG or EOG. If you want turn this off, e.g.

    gssc_infer myeegfile-raw.fif --eeg C3 --eog "EOG L" --no_drop_eeg --no_drop_eog

will perform inference on one, single combination: C3 and EOG L. This may be
desirable to e.g. speed up performance on systems without CUDA (see below).

### Output
By default the GSSC will output the sleep stages as comma separated values
(CSV) file with the same root name as the input. You can alternatively output
as an MNE-Python Annotation object, which could then be recombined with the
original MNE-Python raw file if desired. Use the out_form parameter:

    gssc_infer myeegfile-raw.fif --out_form mne

### CUDA Memory
By default the GSSC will infer the entire PSG recording in one step, which is
the most computationally efficient way to do it. If you are using CUDA, you
might get an error reporting that you are out of memory, particularly if your
GPU is on the lower end. This concerns the memory on your GPU, not your main
CPU memory. You can get around this problem by limiting the number of epochs
the GSSC infers in one step. This is specified on the command line with the
chunk_size parameter:

    gssc_infer myeegfile-raw.fif --chunk_size 500

which will limit the maximum epochs to 500 at a time. You might try 500 first,
and reduce the number further if you still run out of memory.

### Speed issues
With CUDA, the GSSC will finish very quickly under most conditions, but if you
are using the CPU, each combination can take up a lot of time. Therefore it is
recommended that you only use one or two EEG channels and only one EOG channel,
also using the --no_drop parameters (see channels and combinations above).
Whether you use CUDA or CPU, there is not likely to be any noticeable accuracy
benefit from using more than 3-4 EEG channels, as long as those channels are
the preferred ones used the most during training of the GSSC: C3, C4, F3,
and F4.

## Real-time inference

Coming soon!

## Training

Coming soon!

## License

This software is copyright of the University Clinic Greifswald, Germany
under a GNU Affero General Public License v3.
