# %%
import matplotlib.pyplot as plt
import torch
from torch import tensor
from torchvision import datasets

train_raw = datasets.FashionMNIST("data", train=True, download=True)
valid_raw = datasets.FashionMNIST("data", train=False, download=True)
classes = train_raw.classes

len(train_raw), len(valid_raw), classes
# %%
img, label = train_raw[0]
type(img), img.size, label

# %%

t = torch.zeros(60000, 28, 28)

t[:, None].shape, t[None].shape, t[..., None].shape,t[:, :, None].shape

# %%
# Pasar todo a tensores
train_raw.data.shape, train_raw.data.dtype, train_raw.targets.shape
# %%
im = train_raw.data[0]
im.dtype, im.min(), im.max()

im.float().div(255).max()

# %%
x_train = train_raw.data.float().div(255)[:, None]
y_train = train_raw.targets
x_valid = valid_raw.data.float().div(255)[:, None]
y_valid = valid_raw.targets

x_train.shape, x_train.dtype, x_train.min(), x_train.max(), x_train.mean()


# %%
# Función para mostrar las imágenes
def show(img, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)


show(x_train[0, 0], title=classes[y_train[0]])


# %%
class Dataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = Dataset(x_train, y_train)
valid_ds = Dataset(x_valid, y_valid)

len(train_ds), train_ds[0][0].shape, train_ds[0][1]

# %%
xb, yb = train_ds[0:5]
xb.shape, yb


# %%
class Loader:
    def __init__(self, ds, bs, shuffle=False):
        self.ds, self.bs, self.shuffle = ds, bs, shuffle

    def __len__(self):
        return (len(self.ds) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.ds)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.bs):
            yield self.ds[idx[i : i + self.bs]]


xb, yb = next(iter(Loader(train_ds, 256, shuffle=True)))
xb.shape, yb.shape, len(Loader(train_ds, 256))



# %%
%timeit for _ in Loader(train_ds, 256, shuffle=True): pass
%timeit for _ in DataLoader(train_ds, batch_size=256, shuffle=True): pass

# %%
# Visualizar un lote

xb, yb = next(iter(train_dl))
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for ax, im, l in zip(axs.flat, xb, yb):
    show(im[0], ax, classes[l])


# %%
def lotes(ds, bs, shuffle=False):
    n = len(ds)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield ds[idx[i:i + bs]]

xb, yb = next(lotes(train_ds, 8, shuffle=False))
yb

# %%
xb, yb = next(lotes(train_ds, 8, shuffle=True))
yb
# %%
sum(1 for _ in lotes(train_ds, 256))


# %%

idx = torch.randperm(10)
idx
# %%
t = torch.arange(10) * 10
t, t[idx]
# %%
sum(len(yb) for _, yb in lotes(train_ds, 256, shuffle=True))

# %%
# DataLoader
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=256)

xb, yb = next(iter(train_dl))
xb.shape, yb.shape, len(train_dl), len(valid_dl)


# %%
muestras = [train_ds[i] for i in range(4)]
len(muestras), type(muestras[0])

from torch.utils.data import default_collate
xb, yb = default_collate(muestras)
xb.shape, yb.shape
