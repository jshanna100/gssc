import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.nn import (Module, Sequential, Conv1d, LeakyReLU, Linear,
                      ModuleDict, ModuleList, GroupNorm, AvgPool1d, Identity,
                      LogSoftmax, GRU, Dropout)

class ResBlock(Module):
    def __init__(self, in_feat, out_feat, downsamp=False, relu_leak=0.2,
                 norm=False, dropout=None):
        super().__init__()
        ''' Resnet Block
        Parameters
        ----------
        in_feat : int
            Number of features for input
        out_feat : int
            Number of featurs for output
        downsamp : bool
            Whether to downsample output or not
        relu_leak : float
            relu leak parameter
        '''

        stride = 2 if downsamp else 1
        net = Sequential()
        if norm:
            if in_feat == 16:
                net.add_module("Norm 1", GroupNorm(2, in_feat))
            else:
                net.add_module("Norm 1", GroupNorm(in_feat//16, in_feat))
        net.add_module("RELU 1", LeakyReLU(relu_leak))
        if dropout is not None:
            net.add_module("Dropout 1", Dropout(dropout))
        net.add_module("Conv 1", Conv1d(in_feat, out_feat, 3, stride=stride,
                        padding=1, bias=False))
        if norm:
            if out_feat == 16:
                net.add_module("Norm 2", GroupNorm(2, out_feat))
            else:
                net.add_module("Norm 2", GroupNorm(out_feat//16, out_feat))
        net.add_module("RELU 2", LeakyReLU(relu_leak))
        if dropout is not None:
            net.add_module("Dropout 2", Dropout(dropout))
        net.add_module("Conv 2", Conv1d(out_feat, out_feat, 3, stride=1,
                        padding=1, bias=False))

        self.net = net

        # define the shortcut connection
        shortcut = Sequential()
        if downsamp:
            shortcut.add_module("Avg Pool", AvgPool1d(3, stride=2, padding=1))
        if in_feat == out_feat:
            shortcut.add_module("Identity", Identity())
        else:
            shortcut.add_module("Conv MapUp", Conv1d(in_feat, out_feat, 1))
        self.shortcut = shortcut

    def forward(self, x):
        resid = self.shortcut(x)
        y = self.net(x)
        return resid + y

class ResBottleneck(Module):
    def __init__(self, feat, mid_feat, relu_leak=0.2, norm=False, dropout=None):
        ''' ResNet Bottleneck
        Parameters
        ----------
        feat : int
            Number of features for input and output
        mid_feat : int
            number of bottleneck features
        relu_leak : float
            relu leak parameter
        '''
        super().__init__()

        net = Sequential()
        if norm:
            if feat == 16:
                net.add_module("Norm 1", GroupNorm(2, feat))
            else:
                net.add_module("Norm 1", GroupNorm(feat//16, feat))
        net.add_module("RELU 1", LeakyReLU(relu_leak))
        if dropout is not None:
            net.add_module("Dropout 1", Dropout(dropout))
        net.add_module("Conv 1", Conv1d(feat, mid_feat, 3, padding=1,
                        bias=False))
        if norm:
            if mid_feat == 16:
                net.add_module("Norm 2", GroupNorm(2, mid_feat))
            else:
                net.add_module("Norm 2", GroupNorm(mid_feat//16, mid_feat))
        net.add_module("RELU 2", LeakyReLU(relu_leak))
        if dropout is not None:
            net.add_module("Dropout 2", Dropout(dropout))
        net.add_module("Conv 2", Conv1d(mid_feat, feat, 3, padding=1,
                        bias=False))
        self.net = net

    def forward(self, x):
        return self.net(x)

class ResSignal(Module):
    def __init__(self, blueprint, relu_leak=0.2, norm=False, dropout=None):
        ''' Signal network; infers a representation from a single signal
        Parameters
        ----------
        blueprint : dict
            Dictionary where each entry contains a NN layer. Keys should
            contain either "Block" or "Bottleneck." Values are themselves
            dictionaries which specify the ResBlock or ResBottleneck parameters.
        downsamp : bool
            Whether to downsample output or not
        relu_leak : float
            relu leak parameter
        '''
        super().__init__()

        net = Sequential()
        first_feat = blueprint["Block 0"]["in_feat"]
        net.add_module("Conv 0", Conv1d(1, first_feat, 3, padding=1))
        # initial conv outputs with features nums that resnet begins with
        for idx, (k, v) in enumerate(blueprint.items()):
            if "Block" in k:
                net.add_module("ResBlock {}".format(idx),
                                ResBlock(**v, relu_leak=relu_leak,
                                         norm=norm, dropout=dropout))
            elif "Bottleneck" in k:
                net.add_module("ResBottleneck {}".format(idx),
                                ResBottleneck(**v, relu_leak=relu_leak,
                                norm=norm, dropout=dropout))

        self.net = net

    def forward(self, x):
        return self.net(x)

class ResMixer(Module):
    def __init__(self, fcs, sig_num, last_length, norm=False, relu_leak=0.2,
                 dropout=None):
        '''Mixes the reps of the various signals before they go to Decision
        fcs : list
            a list of fc out feature numbers e.g. [256, 256, 256] would be
            three fcs outputting 256 features. Use -1 for Identity.
        last_length : int
            final length of signal representation output by ResSignals
        '''
        super().__init__()
        net = Sequential()
        sig_len = last_length * sig_num
        for fc_idx, fc in enumerate(fcs):
            if fc == -1:
                net.add_module("Identity", Identity())
            else:
                if norm:
                    net.add_module("Norm {}".format(fc_idx),
                                   GroupNorm(sig_len//16, sig_len))
                net.add_module("ReLU FC {}".format(fc_idx),
                               LeakyReLU(relu_leak))
                if dropout is not None:
                    net.add_module("Dropout {}".format(fc_idx),
                                   Dropout(dropout))
                net.add_module("Conv FC {}".format(fc_idx),
                               Conv1d(sig_len, fc, 1))
            sig_len = fc
        self.net = net

    def forward(self, x):
        mix = self.net(x)
        return mix

class ResDecision(Module):
    def __init__(self, sig_len, class_n, fc=None, relu_leak=0.2,
                 norm=False, dropout=None):
        '''Makes decision on sleep stage on basis of representations output
        by ResSignals, and possibly by sleep stage information of adjacent
        epochs

        Parameters
        ----------
        sig_len : int
            Length of input layer, i.e. the number of signals * the length
            of the representations * the number of features + whatever other
            information you include (i.e. context stages)
        class_n : int
            Number of classes
        fc : list
            Out features of fully connected layers
        relu_leak : float
            relu leak parameter
        '''
        super().__init__()

        net = Sequential()
        latest_feat = sig_len
        if fc:
            for fc_idx, fc in enumerate(fc):
                if norm:
                    net.add_module("Norm {}".format(fc_idx),
                                   GroupNorm(latest_feat//16, latest_feat))
                net.add_module("ReLU FC {}".format(fc_idx),
                               LeakyReLU(relu_leak))
                if dropout is not None:
                    net.add_module("Dropout {}".format(fc_idx),
                                   Dropout(dropout))
                net.add_module("Conv FC {}".format(fc_idx),
                               Conv1d(latest_feat, fc, 1))
                latest_feat = fc

        if norm:
            net.add_module("Norm 1", GroupNorm(latest_feat//16, latest_feat))
        net.add_module("ReLU Final", LeakyReLU(relu_leak))
        net.add_module("Conv Decoder", Conv1d(latest_feat, class_n, 1))
        net.add_module("LogSoftmax", LogSoftmax(dim=1))
        self.net = net

    def forward(self, x):
        return self.net(x)


class ResSleep(Module):
    def __init__(self, signals, blueprint, last_length, class_n, mix_specs,
                 fc_dec=None, relu_leak=0.2, norm=False, dropout=None):
        '''Overarching class which encapsulates all the networks

        Parameters
        ----------
        signals : list
            names of signals
        blueprint : dict
            passed onto ResSignals by initialisation - see that class's
            documentation
        last_length : int
            final length of signal representation output by ResSignals
        class_n : int
            number of classes to predict
        do_context : bool
            Take the surrounding epochs into account when training
        fc_dec : list
            list of widths for full connected decision layers
        relu_leak : float
            relu leak parameter

        '''
        super().__init__()
        sig_n = len(signals)
        # need to know final feature num from resnet for FC layers
        last_feat_out = list(blueprint["eeg"].values())[-1]["out_feat"] # hackish
        flat_feat = last_feat_out * last_length
        # check that all mix networks output the same
        all_outs = np.array([v[-1] for v in mix_specs.values()])
        if not all(all_outs==all_outs[0]):
            raise ValueError("Mix nets output different feature lengths.")
        # build the signal networks
        sig_net_dict = {}
        for sig in signals:
            sig_net_dict[sig] = ResSignal(blueprint[sig], relu_leak=relu_leak,
                                          norm=norm, dropout=dropout)
        self.sig_net = ModuleDict(sig_net_dict)
        # build the mix networks
        mix_net_dict = {}
        for k,v in mix_specs.items():
            sig_num = len(k.split())
            mix_net_dict[k] = ResMixer(v, sig_num, flat_feat, norm=norm,
                                       relu_leak=relu_leak, dropout=dropout)
        self.mix_net = ModuleDict(mix_net_dict)
        # build the decision networks
        self.dec_net = ResDecision(all_outs[0], class_n, fc=fc_dec,
                                   relu_leak=relu_leak, norm=norm,
                                   dropout=dropout)

        self.sig_names = signals
        self.class_n = class_n

    def _sig_consolidate(self, x):
        reps = []
        for sig in x.keys():
            reps.append(self.sig_net[sig](x[sig]))
        reps = torch.stack(reps).mean(axis=0)
        reps = reps.reshape(reps.shape[0], -1, 1)
        return reps

    def forward(self, x, rep_output=False):
        sig_list = list(x.keys())
        sig_list.sort()
        sigs = {}
        # calculate each signal
        for sig in sig_list:
            sigs[sig] = self.sig_net[sig](x[sig])
            sigs[sig] = sigs[sig].reshape(len(sigs[sig]), -1, 1)
        # mix the signals
        net_name = " ".join(sig_list)
        all_sigs = torch.hstack([sigs[sig] for sig in sig_list])
        mixes = self.mix_net[net_name](all_sigs)
        if rep_output != "rep_only":
            dec = self.dec_net(mixes)
            if rep_output:
                return mixes, dec
            else:
                return dec
        else:
            return mixes

class SleepGRU(Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,
                 bidirectional=False, norm=False, relu_leak=0.2,
                 dropout=None):
        super().__init__()

        _hidden_dim = hidden_dim * (int(bidirectional)+1)
        self.gru = GRU(input_dim, hidden_dim, num_layers,
                       bidirectional=bidirectional, dropout=dropout)
        self.fc = Linear(_hidden_dim, output_dim)
        self.softmax = LogSoftmax(dim=2)
        self.relu = LeakyReLU(relu_leak)
        if norm:
            self.norm = GroupNorm(_hidden_dim // 16, _hidden_dim)
        self.do_norm = norm
        if dropout is None:
            self.dropout = Identity()
        else:
            self.dropout = Dropout(p=dropout)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        if self.do_norm:
            output = torch.swapaxes(output, 1, 2)
            output = self.norm(output)
            output = torch.swapaxes(output, 1, 2)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.softmax(output)

        return output, hidden
