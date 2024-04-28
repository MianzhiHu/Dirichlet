import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats.stats import pearsonr
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import seaborn as sns

sns.set()
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression


MSE = mean_squared_error
lag = 1

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

## Plotting Config

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def MSE_by_time(r, p):
    err = []
    for t in np.arange(r.shape[1]):
        if len(p.shape) == 3:
            err.append(MSE(r[:, t, :], p[:, t, :]))
        else:
            err.append(MSE(r[:, t, 0], p[:, t]))
    return np.array(err)


choice_95 = pd.read_csv("./data/choice_95.csv", delimiter=",")
choice_100 = pd.read_csv("./data/choice_100.csv", delimiter=",")
choice_150 = pd.read_csv("./data/choice_150.csv", delimiter=",")


def getTS(r):
    ts = np.zeros((r.shape[0], r.shape[1], 4))
    for i in np.arange(4):
        ts[r == i + 1, i] = 1
    return ts


def revTS(r):
    ts = r
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            choice = np.argmax(ts[b, t, :])
            ts[b, t, :] = 0
            ts[b, t, choice] = 1
    for i in np.arange(4):
        ts[:, :, i] = ts[:, :, i] * (i + 1)
    ts = ts.sum(2)
    return ts


def getChR(r):  # choice rate
    ts = r
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            choice = np.argmax(ts[b, t, :])
            ts[b, t, :] = 0
            ts[b, t, choice] = 1
    cr = np.zeros(r.shape)
    for i in np.arange(ts.shape[2]):
        for t in np.arange(ts.shape[1]):
            for b in np.arange(ts.shape[0]):
                cr[b, t, i] = ts[b, : t + 1, i].sum() / (t + 1)
    return cr


def getCoR(r):  # correct rate
    ts = r
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            choice = np.argmax(ts[b, t, :])
            ts[b, t, :] = 0
            ts[b, t, choice] = 1
    ts[:, :, 2] = ts[:, :, 2] + ts[:, :, 3]
    cr = np.zeros((r.shape[0], r.shape[1]))
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            cr[b, t] = ts[b, : t + 1, 2].sum() / (t + 1)
    return cr


def valid_igt(n):
    if n < 1:
        n = int(617 * n)
    set_100 = np.array(getTS(choice_100))[:, :94, :]
    set_95 = np.array(getTS(choice_95))[:, :94, :]
    set_150 = np.array(getTS(choice_150))[:, :94, :]
    full_set = np.concatenate((set_100, set_95, set_150), axis=0)
    np.random.shuffle(full_set)
    return full_set[:n], full_set[n:]


def igt_set2arset():
    null_arset = -np.ones((94, 4))
    train_arset_igt = -np.ones((94, 4))
    for ins in train_set_igt:
        train_arset_igt = np.concatenate((train_arset_igt, null_arset, ins), axis=0)
    return train_arset_igt


class lstmIGT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        out = nn.Softmax(dim=-1)(out)
        return out


n_fold = 5
for fold in np.arange(n_fold):
    test_set_igt, train_set_igt = valid_igt(0.2)
    # lstm
    n_nodes, n_layers = 10, 2
    lstm_igt = lstmIGT(4, n_nodes, 4, n_layers)
    criterion_igt = nn.MSELoss()
    optimizer_igt = optim.Adam(lstm_igt.parameters(), lr=1e-2)
    n_epochs, window, batch_size = 10, 10, 20
    loss_set_igt = []
    for ep in np.arange(n_epochs):
        for bc in np.arange(train_set_igt.shape[0] / batch_size):
            inputs = Variable(
                torch.from_numpy(
                    train_set_igt[int(bc * batch_size) : int((bc + 1) * batch_size)]
                ).float()
            )
            target = Variable(
                torch.from_numpy(
                    train_set_igt[int(bc * batch_size) : int((bc + 1) * batch_size)]
                ).float()
            )
            output = lstm_igt(inputs)
            loss = criterion_igt(output[:, :-lag], target[:, lag:])
            optimizer_igt.zero_grad()
            loss.backward()
            optimizer_igt.step()
            print_loss = loss.item()
            loss_set_igt.append(print_loss)
            if bc % window == 0:
                print(fold)
                print(
                    "Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set_igt.shape[0] / batch_size,
                        print_loss,
                    )
                )
    lstm_igt = lstm_igt.eval()
    # ar
    train_arset_igt = igt_set2arset()
    armodel_igt = VAR(train_arset_igt)
    armodel_igt = armodel_igt.fit()
    # eval
    px2 = torch.from_numpy(test_set_igt).float()
    ry2 = torch.from_numpy(test_set_igt).float()
    pyar2 = np.zeros(px2.shape)
    for i in np.arange(px2.shape[0]):
        for t in np.arange(px2.shape[1]):
            pyar2[i, t, :] = armodel_igt.forecast(np.array(px2[i, : t + 1]), lag)
    varX = Variable(px2)
    py2 = lstm_igt(varX).data.cpu().numpy()
    if fold == 0:
        test_set_igt_full = test_set_igt
        py2_full = py2
        pyar2_full = pyar2
    else:
        test_set_igt_full = np.concatenate((test_set_igt_full, test_set_igt))
        px2 = torch.from_numpy(test_set_igt_full).float()
        ry2 = torch.from_numpy(test_set_igt_full).float()
        py2_full = np.concatenate((py2_full, py2))
        pyar2_full = np.concatenate((pyar2_full, pyar2))
        py2 = py2_full
        pyar2 = pyar2_full


x = getTS(choice_100)

ryc2 = revTS(ry2[:, lag:].cpu().numpy().copy())
ryr2 = getChR(ry2[:, lag:, :].cpu().numpy().copy())
ryo2 = getCoR(ry2[:, lag:, :].cpu().numpy().copy())
pyc2 = revTS(py2[:, :-lag].copy())
pyr2 = getChR(py2[:, :-lag, :].copy())
pyo2 = getCoR(py2[:, :-lag, :].copy())
pycar2 = revTS(pyar2[:, :-lag].copy())
pyrar2 = getChR(pyar2[:, :-lag, :].copy())
pyoar2 = getCoR(pyar2[:, :-lag, :].copy())

igt_lstm_mse = np.mean(MSE_by_time(ryr2, pyr2))
igt_ar_mse = np.mean(MSE_by_time(ryr2, pyrar2))
print(n_nodes, n_layers, "MSE:", igt_lstm_mse, igt_ar_mse)
print(np.mean(MSE_by_time(ry2, py2)))