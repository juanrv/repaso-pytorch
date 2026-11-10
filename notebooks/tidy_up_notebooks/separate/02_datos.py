# %% [markdown]
# # 02 · Datos: Fashion-MNIST, Dataset y DataLoader

# %%
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets
from utils import Dataset, get_dls, load_fashion_mnist, setup, show, show_batch

setup()

# %%
# Datos crudos: ya vienen separados en entrenamiento (60000) y validación (10000)
train_raw = datasets.FashionMNIST("data", train=True, download=True)
img, label = train_raw[0]
print(type(img), img.size, label, train_raw.classes[label])
print(train_raw.data.shape, train_raw.data.dtype)  # uint8: valores de 0 a 255

# %%
# Preprocesado: float32 en [0, 1] y un eje de canal en la posición 1 -> (N, 1, 28, 28)
(x_train, y_train), (x_valid, y_valid), classes = load_fashion_mnist()
print(x_train.shape, x_train.dtype)
print(x_train.min().item(), x_train.max().item(), round(x_train.mean().item(), 4))

# %%
# Dataset: contrato de dos métodos, __len__ y __getitem__
train_ds = Dataset(x_train, y_train)
valid_ds = Dataset(x_valid, y_valid)

x0, y0 = train_ds[0]
print(len(train_ds), x0.shape, y0.shape, classes[y0])

xb, yb = train_ds[0:5]
print(xb.shape, yb)

try:
    Dataset(x_train, y_valid)
except AssertionError:
    print("x e y con distinta longitud -> AssertionError")


# %%
# Lotes con un generador. El barajado se hace sobre los índices, una vez por recorrido
def lotes(ds, bs, shuffle=False):
    n = len(ds)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield ds[idx[i : i + bs]]


n_lotes = sum(1 for _ in lotes(train_ds, 256))
ultimo = None
for ultimo in lotes(train_ds, 256):
    pass
assert ultimo is not None
print(n_lotes, ultimo[0].shape, ultimo[1].shape)  # 235 lotes, el último de 96

print(next(lotes(train_ds, 8))[1])
print(next(lotes(train_ds, 8, shuffle=True))[1])

# %%
# Reproducibilidad: misma semilla, misma permutación
torch.manual_seed(1)
a = torch.randperm(5)
torch.manual_seed(1)
b = torch.randperm(5)
print(a, b, torch.equal(a, b))


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


mi_loader = Loader(train_ds, 256, shuffle=True)
xb, yb = next(iter(mi_loader))
print(len(mi_loader), xb.shape, yb.shape)

# %%
# DataLoader nativo: pide cada ejemplo por separado y los junta con collate
train_dl, valid_dl = get_dls(train_ds, valid_ds, bs=256)
xb, yb = next(iter(train_dl))
print(len(train_dl), len(valid_dl), xb.shape, yb.shape)

muestras = [train_ds[i] for i in range(4)]
xc, yc = default_collate(muestras)
print(len(muestras), type(muestras[0]), xc.shape, yc.shape)

# %%
# Las parejas (x, y) se conservan al barajar
idx = torch.randperm(len(train_ds))[:8]
xb, yb = train_ds[idx]
assert torch.equal(yb, train_raw.targets[idx])
print("parejas intactas")

# %%
xb, yb = next(iter(train_dl))
show_batch(xb, yb, classes)
plt.show()

# %%
# Mediciones propias:
# Loader a mano: 11.6 ms
# DataLoader de Pytorch: 138 ms
# %timeit for _ in Loader(train_ds, 256, shuffle=True): pass
# %timeit for _ in DataLoader(train_ds, batch_size=256, shuffle=True): pass
