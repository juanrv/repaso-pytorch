# %% [markdown]
# # 05 · fit: entrenar durante varias épocas

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from utils import (
    Dataset,
    Optimizer,
    evaluate,
    fit,
    get_dls,
    load_fashion_mnist,
    plot_history,
    setup,
)

setup()

(x_train, y_train), (x_valid, y_valid), classes = load_fashion_mnist()
train_dl, valid_dl = get_dls(Dataset(x_train, y_train), Dataset(x_valid, y_valid))


def crear_modelo():
    torch.manual_seed(1)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))


# %%
modelo = crear_modelo()
print("antes de entrenar:", evaluate(modelo, F.cross_entropy, valid_dl))

opt = Optimizer(modelo.parameters(), lr=0.5)
fit(15, modelo, F.cross_entropy, opt, train_dl, valid_dl)

# %%
# Nuestro Optimizer frente al de PyTorch: misma semilla, deberían coincidir
modelo = crear_modelo()
fit(
    3,
    modelo,
    F.cross_entropy,
    Optimizer(modelo.parameters(), lr=0.5),
    train_dl,
    valid_dl,
)

modelo = crear_modelo()
fit(
    3,
    modelo,
    F.cross_entropy,
    optim.SGD(modelo.parameters(), lr=0.5),
    train_dl,
    valid_dl,
)

# %%
# Promedio simple frente a ponderado por tamaño de lote (el último lote es de 272)
perdidas, tamanos = [], []
with torch.no_grad():
    for xb, yb in valid_dl:
        perdidas.append(F.cross_entropy(modelo(xb), yb).item())
        tamanos.append(len(xb))

simple = sum(perdidas) / len(perdidas)
ponderado = sum(l * n for l, n in zip(perdidas, tamanos)) / sum(tamanos)
print(tamanos[:2], tamanos[-1], simple, ponderado)
print(evaluate(modelo, F.cross_entropy, valid_dl)[0])

# %%
# Comparar learning rates con épocas completas
for lr in (0.05, 0.2, 0.5, 1.0):
    modelo = crear_modelo()
    res = fit(
        3,
        modelo,
        F.cross_entropy,
        Optimizer(modelo.parameters(), lr=lr),
        train_dl,
        valid_dl,
        verbose=False,
    )
    print(f"lr={lr:<5} loss={res['val_loss'][-1]:.3f}  acc={res['val_acc'][-1]:.3f}")

# %%
# Prueba de varianza inicial con varias semillas
# %%
for s in range(1, 8):
    torch.manual_seed(s)
    m = nn.Sequential(nn.Flatten(), nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
    print(s, evaluate(m, F.cross_entropy, valid_dl))
# %%


def recorrido(evaluar_antes):
    modelo = crear_modelo()
    if evaluar_antes:
        evaluate(modelo, F.cross_entropy, valid_dl)
    opt = Optimizer(modelo.parameters(), lr=0.5)
    return fit(3, modelo, F.cross_entropy, opt, train_dl, valid_dl, verbose=False)


print(recorrido(False), recorrido(False), recorrido(True))
# %%
for lr in (0.1, 0.5):
    modelo = crear_modelo()
    opt = Optimizer(modelo.parameters(), lr=lr)
    hist = fit(100, modelo, F.cross_entropy, opt, train_dl, valid_dl, verbose=False)
    plot_history(hist).suptitle(f"lr={lr}")
plt.show()
# %%
[round(t - v, 3) for t, v in zip(hist["train_loss"], hist["val_loss"])]
