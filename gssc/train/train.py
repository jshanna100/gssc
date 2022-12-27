from gssc.networks import ResSleep, SleepGRU
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.nn import NLLLoss
from torch.optim import AdamW
from torch.nn.init import kaiming_normal_, constant_
from gssc.utils import calc_accuracy, make_blueprint
import argparse
import datetime
import time

torch.backends.cudnn.benchmark = True

t = datetime.datetime.fromtimestamp(time.time())
time_str = t.strftime("%H:%M:%S on %d.%m.%y")
print("Started on {}.".format(time_str))

## define hyper-parameters
# defaults
"""
dataset: This is required. The full path of a SeqDataSet .pt file.
out_dir: Output directory. Where you want results saved.
load_res/opt/gru_state: If you are resuming training, full paths of the saved
                        results

feat: list of feature/filter depth parameters for make_blueprint. E.g. [4,8,10]
      means the starting feature depth is 2**4 (16), the end feature depth is
      2**8, (256), and from start to end there are 10 steps evenly spaced
      from 4 to 8.
depth_eog/eeg: How deep the Resnets are for eog and eeg networks
bn_fact_eog/eeg: How thin the bottleneck is in relation to input/output;
                 e.g. 2 is half
hidden_dim: Dimensionality of GRU layers
num_layers: Number of GRU layers
beta1/2: Beta parameters for Adam optimiser
lr: Learning rate
start/end_epo: Epochs to start and stop with. Start is only useful for resuming
num_workers: Number of CPU threads to use for the DataLoader
psg_divisor: How many parts of near equal length to divide a single PSG into
             for training steps
norm: Whether to use GroupNorm
drx_num: Bi- or unidirectional GRU
dropout: Dropout p
batch_size: How many PSGs to load at once during training
"""
arg_dict = {"feat":[4,8,10], "depth_eog":0, "depth_eeg":4, "bn_fact_eog":None,
            "bn_fact_eeg":2, "hidden_dim":256, "num_layers":5, "beta1":0.9,
            "beta2":0.999, "lr":0.00003, "start_epo":0, "end_epo":1,
            "num_workers":0, "psg_divisor":128, "norm":True, "drx_num":2,
            "dropout":0.5, "batch_size":1}
parser = argparse.ArgumentParser()
for k,v in arg_dict.items():
    parser.add_argument("--"+k, type=type(v), default=v)

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--load_res_state", type=str, default="")
parser.add_argument("--load_opt_state", type=str, default="")
parser.add_argument("--load_gru_state", type=str, default="")

opt = vars(parser.parse_args())
print(opt)

bidirect = True if opt["drx_num"] == 2 else False

# set up data loader
print("Loading dataset.")
dataset = torch.load(opt["dataset"])
out_dir = opt["out_dir"]

# initialise resnet model, optimizer
# configure signals
blueprints = {}
# make_blueprint is a helper function to create the network architecture
# described in the paper. it is not strictly necessary
blueprints["eog"] = make_blueprint(opt["feat"][0], opt["feat"][1],
                                   opt["feat"][2], opt["bn_fact_eog"],
                                   opt["depth_eog"])
blueprints["eeg"] = make_blueprint(opt["feat"][0], opt["feat"][1],
                                   opt["feat"][2], opt["bn_fact_eeg"],
                                   opt["depth_eeg"])
net = ResSleep(["eeg", "eog"], blueprints, 5, len(dataset.get_counts()),
               {"eeg":[512], "eog":[512], "eeg eog":[1024, 512, 512]},
               fc_dec=[512, 512, 512], norm=opt["norm"], dropout=opt["dropout"])
torch.save(net, "{}net.pt".format(out_dir))

# init parameters
if opt["load_res_state"]:
    net.load_state_dict(torch.load("{}/{}".format(out_dir,
                                                  opt["load_res_state"])))
else:
    for p in net.named_parameters():
        if "Conv" in p[0] and "weight" in p[0]:
            kaiming_normal_(p[1])
        elif "Conv" in p[0] and "bias" in p[0]:
            constant_(p[1], 0)
        elif "Norm" in p[0] and "weight" in p[0]:
            constant_(p[1], 1)
        elif "Norm" in p[0] and "bias" in p[0]:
            constant_(p[1], 0)
        else:
            raise ValueError("Unknown parameter {}; not initialised.".format(p[0]))
net.cuda()
gru_net = SleepGRU(512, opt["hidden_dim"], opt["num_layers"], 5,
                   bidirectional=bidirect, norm=opt["norm"],
                   dropout=opt["dropout"])
torch.save(gru_net, "{}gru_net.pt".format(out_dir))
if opt["load_gru_state"]:
    gru_net.load_state_dict(torch.load("{}/{}".format(out_dir,
                                                      opt["load_gru_state"])))
gru_net.cuda()
gru_net = gru_net.train()

optimiser = AdamW(list(net.parameters()) + list(gru_net.parameters()),
                  betas=(opt["beta1"], opt["beta2"]), lr=opt["lr"])
if opt["load_opt_state"]:
    print("Loading optimiser state: {}".format(opt["load_opt_state"]))
    optimiser.load_state_dict(torch.load("{}/{}".format(out_dir,
                                                        opt["load_opt_state"])))

# weights for Negative Log Likelihood loss
weights = np.array([1, 2.4, 1, 1.2, 1.4])
print(f"Weights: {weights}")
weights = torch.from_numpy(weights).float().to("cuda")

# loss function
loss_func = NLLLoss(weight=weights)
abs_idx = 0

net.train()
gru_net.train()
# you could uncomment and constrain your dataset to a subset, specified by a
# numpy array of indices.
#inds = np.load("path/to/train_indices.npy")
inds = np.arange(len(dataset)) # this, OTOH, just uses everything
for epo_idx in range(opt["start_epo"], opt["end_epo"]):
    sampler = SubsetRandomSampler(inds)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True,
                            sampler=sampler, num_workers=opt["num_workers"])
    # initialise training info storage
    losses = np.ones((len(dataloader)), dtype=np.float32)*np.nan
    accs = np.ones((len(dataloader)), dtype=np.float32)*np.nan
    gru_losses = np.ones((len(dataloader)), dtype=np.float32)*np.nan
    gru_accs = np.ones((len(dataloader)), dtype=np.float32)*np.nan
    seq_idx = np.ones((len(dataloader)), dtype=np.int32)*np.nan
    dset_idx = np.ones((len(dataloader)), dtype=np.int8)*np.nan
    train_info = {"Loss":losses, "Acc":accs, "GRU_Loss":gru_losses,
                  "GRU_Acc":gru_accs, "Seq_Idx":seq_idx, "Dset_Idx":dset_idx}
    # iterate through the training partition
    for itr_idx, itr in enumerate(dataloader):
        perm_data, stage, idx = itr
        idx = idx.numpy()
        train_info["Dset_Idx"][itr_idx], _ = dataset.get_dset_idx(idx)
        train_info["Seq_Idx"][itr_idx] = idx
        if stage.shape[1] < opt["psg_divisor"]:
            continue
        stage = stage.long().to("cuda")
        sta_n = stage.shape[1]
        inf_accs, gru_accs, losses, gru_losses = [], [], [], []
        for perm_idx, perm in enumerate(perm_data):
            data = perm
            if len(stage[0]) < opt["psg_divisor"]:
                continue
            for k in data.keys():
                data[k] = data[k].to("cuda")
            hidden = torch.zeros(opt["num_layers"] * opt["drx_num"], 1,
                                 opt["hidden_dim"]).to("cuda")

            # chunks are for the typical case that GPU RAM can't fit a whole PSG
            chunks = np.array_split(np.arange(sta_n),
                                    sta_n / opt["psg_divisor"])
            infs, gru_infs = [], [],
            for sta in chunks:
                int_stages = stage[0, sta, 0]
                these_data = {}
                for k in data.keys():
                    these_data[k] = data[k][:, sta,].reshape(-1, 1,
                                                             data[k].shape[-1])
                # signal processing
                reps, y = net(these_data, rep_output=True)
                loss = loss_func(y[...,0], int_stages)
                loss.backward(retain_graph=True)

                # now gru
                reps = torch.swapaxes(reps, 1, 2)
                y_gru, hidden = gru_net(reps, hidden)
                gru_loss = loss_func(y_gru.mean(axis=1), int_stages)
                gru_loss.backward()
                if opt["batch_size"] == 1:
                    optimiser.step()
                    optimiser.zero_grad()
                hidden = hidden.detach()

                infs.append(torch.argmax(y, dim=1))
                gru_infs.append(torch.argmax(y_gru, dim=-1))
                losses.append(loss.item())
                gru_losses.append(gru_loss.item())

            inf, gru_inf = torch.cat(infs)[:,0], torch.cat(gru_infs)[:,0]

            targs = stage[0,:,0]
            inf_accs.append((sum(inf == targs) / len(targs)).cpu().numpy())
            gru_accs.append((sum(gru_inf == targs) / len(targs)).cpu().numpy())

        train_info["Loss"][itr_idx] = np.mean(losses)
        train_info["Acc"][itr_idx] = np.mean(inf_accs)
        train_info["GRU_Loss"][itr_idx] = np.mean(gru_losses)
        train_info["GRU_Acc"][itr_idx] = np.mean(gru_accs)
        print("Epo {}/{}, PSG {}/{}, "
              "Loss: {:.3f}, Accuracy: {:.2f}, "
              "GRU_Loss: {:.3f}, "
              "GRU_Accuracy: {:.2f}, "
              "Dataset: {}".format(epo_idx, opt["end_epo"],
                                   itr_idx, len(dataloader),
                                   train_info["Loss"][itr_idx],
                                   train_info["Acc"][itr_idx],
                                   train_info["GRU_Loss"][itr_idx],
                                   train_info["GRU_Acc"][itr_idx],
                                   dataset.get_dset_idx(idx)),
                                   flush=True)

        if (opt["batch_size"] > 1) and ((itr_idx+1) % opt["batch_size"] == 0):
            print("Learning step.")
            optimiser.step()
            optimiser.zero_grad()

    torch.save(optimiser.state_dict(), "{}opt_state_{}.pt".format(out_dir, epo_idx))
    torch.save(net.state_dict(), "{}net_state_{}.pt".format(out_dir, epo_idx))
    torch.save(gru_net.state_dict(), "{}gru_net_state_{}.pt".format(out_dir,
                                                                    epo_idx))
    train_file_name = "{}training_info_{}.pickle".format(out_dir, epo_idx)
    with open(train_file_name, "wb") as f:
        pickle.dump(train_info, f)
