# %%
import torch
import torch.nn.functional as F
from torch import nn
from utils import conv, deconv, load_fashion_mnist, setup, show_batch

device = setup()
(x_train, y_train), (x_valid, y_valid), classes = load_fashion_mnist()


def crear_ae():
    torch.manual_seed(1)
    return nn.Sequential(
        nn.ZeroPad2d(2),
        *conv(1, 2),
        *conv(2, 4),
        *deconv(4, 2),
        *deconv(2, 1, relu=False),
        nn.ZeroPad2d(-2),
        nn.Sigmoid(),
    ).to(device)


ae = crear_ae()

# %%
xb = x_train[:256].to(device)
print("entrada", tuple(xb.shape))
for capa in ae:
    xb = capa(xb)
    print(f"{type(capa).__name__:<20}", tuple(xb.shape))

print(sum(p.numel() for p in ae.parameters()))

# %%
x = x_train[:256].to(device)
with torch.no_grad():
    salida = ae(x)
print(F.mse_loss(salida, x).item(), salida.min().item(), salida.max().item())


# %%
m = x_train.mean()
print("predecir la media  :", ((x_train - m) ** 2).mean().item())
print("predecir 0.5       :", ((x_train - 0.5) ** 2).mean().item())
print("predecir todo negro:", (x_train**2).mean().item())


# %%
import matplotlib.pyplot as plt

show_batch(x, y_train[:256], classes, n=8)
show_batch(salida, y_train[:256], classes, n=8)
plt.show()


# %%
im = salida[0, 0].cpu()
print(im.min().item(), im.max().item())
print(im[14, 14].item(), im[0, 0].item(), im[0, 14].item())
