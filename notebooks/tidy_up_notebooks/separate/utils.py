import math
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


def setup(seed=1):
    torch.manual_seed(seed)
    torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_fashion_mnist(root="data"):
    train_raw = datasets.FashionMNIST(root, train=True, download=True)
    valid_raw = datasets.FashionMNIST(root, train=False, download=True)
    x_train = train_raw.data.float().div(255)[:, None]
    x_valid = valid_raw.data.float().div(255)[:, None]
    return (x_train, train_raw.targets), (x_valid, valid_raw.targets), train_raw.classes


def get_dls(train_ds, valid_ds, bs=256):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def show(img, ax=None, title=None, vmin=None, vmax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)
    return ax


def show_batch(xb, yb, classes, n=16, vmin=0, vmax=1):
    xb, yb = xb[:n].cpu(), yb[:n].cpu()
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows), squeeze=False
    )
    for ax in axs.flat:
        ax.axis("off")
    for ax, im, label in zip(axs.flat, xb, yb):
        show(im[0], ax, classes[label], vmin, vmax)
    fig.tight_layout()
    return fig


def accuracy(logits, yb):
    return (logits.argmax(dim=1) == yb).float().mean()


class Optimizer:
    def __init__(self, params, lr=0.5):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                assert p.grad is not None
                p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def plot_history(hist):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    axs[0].plot(hist["train_loss"], label="train")
    axs[0].plot(hist["val_loss"], label="val")
    axs[0].set_title("loss")
    axs[0].legend()
    axs[1].plot(hist["val_acc"])
    axs[1].set_title("accuracy (val)")
    for ax in axs:
        ax.set_xlabel("época")
        ax.grid(True)
    fig.tight_layout()
    return fig


def to_device(*tensors, device=None):
    if device is None:
        return tensors
    return tuple(t.to(device) for t in tensors)


def evaluate(model, loss_func, valid_dl, device=None):
    model.eval()
    tot_loss, tot_acc, count = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb, yb = to_device(xb, yb, device=device)
            pred = model(xb)
            n = len(xb)
            count += n
            tot_loss += loss_func(pred, yb).item() * n
            tot_acc += accuracy(pred, yb).item() * n
    return tot_loss / count, tot_acc / count


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device=None, verbose=True):
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tot_loss, count = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = to_device(xb, yb, device=device)
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            n = len(xb)
            tot_loss += loss.item() * n
            count += n
        train_loss = tot_loss / count
        val_loss, val_acc = evaluate(model, loss_func, valid_dl, device)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        if verbose:
            dt = time.perf_counter() - t0
            print(
                f"{epoch}  train={train_loss:.3f}  val={val_loss:.3f}  acc={val_acc:.3f}  {dt:.1f}s"
            )
    return hist


def conv(ci, co, ks=3, relu=True):
    capas: list[nn.Module] = [nn.Conv2d(ci, co, ks, stride=2, padding=ks // 2)]
    if relu:
        capas.append(nn.ReLU())
    return capas


def deconv(ci, co, ks=3, relu=True):
    capas: list[nn.Module] = [
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ci, co, ks, stride=1, padding=ks // 2),
    ]
    if relu:
        capas.append(nn.ReLU())
    return capas
