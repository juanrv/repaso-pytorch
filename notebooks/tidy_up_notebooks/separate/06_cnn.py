# %%
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, tensor
from utils import load_fashion_mnist, setup, show,get_dls, Dataset, Optimizer, plot_history, fit

setup()
(x_train, y_train), (x_valid, y_valid), classes = load_fashion_mnist()

# %%
kernel = tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])


def aplicar_kernel(img, k):
    h, w = img.shape
    kh, kw = k.shape
    out = torch.zeros(h - kh + 1, w - kw + 1)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (img[i : i + kh, j : j + kw] * k).sum()
    return out


img = x_train[0, 0]
res = aplicar_kernel(img, kernel)
print(img.shape, res.shape)
# %%
fig, axs = plt.subplots(1, 3, figsize=(8, 3))
show(img, axs[0], "original")
show(res, axs[1], "bordes verticales")
show(aplicar_kernel(img, kernel.T), axs[2], "bordes horizontales")
plt.show()

# %%
print(sum(p.numel() for p in nn.Linear(784, 50).parameters()))
print(sum(p.numel() for p in nn.Conv2d(1, 4, kernel_size=3).parameters()))
# %%
conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
print(conv.weight.shape)
with torch.no_grad():
    conv.weight.copy_(kernel[None, None])
    res_conv = conv(img[None, None])

print(res_conv.shape)
print(torch.allclose(res_conv[0, 0], res, atol=1e-5))
# %%
sum(p.numel() for p in nn.Conv2d(3, 16, 3).parameters())

# %%
x = torch.randn(1, 1, 28, 28)

for k, p, s in [(3, 0, 1), (3, 1, 1), (3, 1, 2), (5, 2, 1), (5, 2, 2)]:
    conv = nn.Conv2d(1, 1, kernel_size=k, padding=p, stride=s)
    print(f"k={k} p={p} s={s} ->", tuple(conv(x).shape[2:]))

# %%
# Cuántas ventanas del kernel cubren cada píxel de una imagen 5x5
unos = torch.ones(1, 1, 3, 3)
print(F.conv_transpose2d(unos, torch.ones(1, 1, 3, 3))[0, 0])

unos = torch.ones(1, 1, 5, 5)
print(F.conv_transpose2d(unos, torch.ones(1, 1, 3, 3), padding=1)[0, 0])


# %%
def cobertura(n, k, p, s):
    o = (n + 2 * p - k) // s + 1
    unos = torch.ones(1, 1, o, o)
    return F.conv_transpose2d(unos, torch.ones(1, 1, k, k), stride=s, padding=p)[0, 0]


configs = [(7, 3, 0, 1), (7, 3, 1, 1), (7, 3, 1, 2)]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, (n, k, p, s) in zip(axs, configs):
    c = cobertura(n, k, p, s)
    ax.imshow(c, cmap="viridis")
    for i in range(n):
        for j in range(n):
            color = "white" if c[i, j] < c.max() / 2 else "black"
            ax.text(j, i, int(c[i, j]), ha="center", va="center", color=color)
    ax.set_title(f"k={k} p={p} s={s}")
    ax.axis("off")
plt.show()


# %%
entrada = torch.ones(1, 1, 4, 4)
w = torch.ones(1, 1, 3, 3)

a = F.conv_transpose2d(entrada, w, stride=2, padding=1, output_padding=1)[0, 0]
b = F.conv2d(F.interpolate(entrada, scale_factor=2, mode="nearest"), w, padding=1)[0, 0]
print(a)
print(b)


# %%
def estampar(n, k, s):
    tam = (n - 1) * s + k
    acum = torch.zeros(tam, tam)
    pasos = []
    for i in range(n):
        for j in range(n):
            acum[i * s : i * s + k, j * s : j * s + k] += 1
            pasos.append(acum.clone())
    return pasos


def numeros(ax, m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m[i, j])
            color = "white" if v < m.max() / 2 else "black"
            ax.text(j, i, v, ha="center", va="center", fontsize=7, color=color)


def ver_estampas(n, k, s, p, extra=0):
    pasos = estampar(n, k, s)
    fig, axs = plt.subplots(n, n, figsize=(2 * n, 2 * n))
    for idx, (ax, m) in enumerate(zip(axs.flat, pasos)):
        ax.imshow(m, cmap="viridis", vmin=0, vmax=pasos[-1].max())
        numeros(ax, m)
        ax.set_title(f"estampa {idx + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"k={k} s={s}")

    final = pasos[-1]
    tam = final.shape[0]
    ini, fin = p, tam - p + extra
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(final, cmap="viridis")
    numeros(ax2, final)
    ax2.add_patch(
        plt.Rectangle(
            (ini - 0.5, ini - 0.5),
            fin - ini,
            fin - ini,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    ax2.set_title(f"k={k} s={s}: suma {tam}x{tam}, recuadro = lo que se queda")
    ax2.axis("off")
    return final[ini:fin, ini:fin]


# %%
res3 = ver_estampas(4, 3, 2, p=1, extra=1)
res4 = ver_estampas(4, 4, 2, p=1)
res5 = ver_estampas(4, 5, 2, p=1)
res6 = ver_estampas(6, 6, 2, p=1)
plt.show()
print(res3)
print(res4)
print(res5)
print(res6)


# %%
def conv(ci, co, ks=3, relu=True):
    capas: list[nn.Module] = [nn.Conv2d(ci, co, ks, stride=2, padding=ks // 2)]
    if relu:
        capas.append(nn.ReLU())
    return capas


cnn = nn.Sequential(
    *conv(1, 8, ks=5),
    *conv(8, 16),
    *conv(16, 32),
    *conv(32, 64),
    *conv(64, 10, relu=False),
    nn.Flatten(),
)

# %%
xb = x_train[:256]
print("entrada", tuple(xb.shape))
for capa in cnn:
    xb = capa(xb)
    print(f"{type(capa).__name__:<8}", tuple(xb.shape))

# %%
print(sum(p.numel() for p in cnn.parameters()))
# %%
logits = cnn(x_train[:256])
print(logits.shape, F.cross_entropy(logits, y_train[:256]).item())
# %%
device = setup()
print(device)

x = torch.randn(3, 3)
x_gpu = x.to(device)
print(x.device, x_gpu.device)

# %%
try:
    x + x_gpu
except RuntimeError as e:
    print(str(e)[:100])

# %%
cnn_cpu = copy.deepcopy(cnn)
cnn.to(device)
print(next(cnn.parameters()).device, next(cnn_cpu.parameters()).device)
# %%
try:
    cnn(x_train[:256])
except RuntimeError as e:
    print(str(e)[:100])

print(cnn(x_train[:256].to(device)).device)
# %%
def fwd(m,x):
    with torch.no_grad():
        m(x)
    if x.is_cuda:
        torch.cuda.synchronize()

xb = x_train[:256]
xb_g = xb.to(device)

# %%
%timeit fwd(cnn_cpu, xb)
%timeit fwd(cnn, xb_g)

# %%
# Lote de mayor size
xb4 = x_train[:4096]
xg4 = xb4.to(device)
%timeit fwd(cnn_cpu, xb4)
%timeit fwd(cnn, xg4)


# %%
train_dl, valid_dl = get_dls(Dataset(x_train, y_train), Dataset(x_valid, y_valid))


def crear_cnn():
    torch.manual_seed(1)
    return nn.Sequential(
        *conv(1, 8, ks=5),
        *conv(8, 16),
        *conv(16, 32),
        *conv(32, 64),
        *conv(64, 10, relu=False),
        nn.Flatten(),
    ).to(device)

# %%
for lr in (0.1, 0.4):
    modelo = crear_cnn()
    opt = Optimizer(modelo.parameters(), lr=lr)
    print(f"lr={lr}")
    hist = fit(20, modelo, F.cross_entropy, opt, train_dl, valid_dl, device=device)
    plot_history(hist).suptitle(f"CNN lr={lr}")
plt.show()
