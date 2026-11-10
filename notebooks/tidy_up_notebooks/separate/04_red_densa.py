# %% [markdown]
# # 04 · Red densa: capa lineal, activaciones, loss y un paso de entrenamiento

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, tensor
from utils import Dataset, accuracy, get_dls, load_fashion_mnist, setup

setup()

(x_train, y_train), (x_valid, y_valid), classes = load_fashion_mnist()
train_dl, valid_dl = get_dls(Dataset(x_train, y_train), Dataset(x_valid, y_valid))

xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)

# %% [markdown]
# ## Capa lineal: salida = x @ W + b

# %%
xb_plano = xb.flatten(1)  # (256, 1, 28, 28) -> (256, 784)
print(xb_plano.shape)

W = (torch.randn(784, 10) * 0.01).requires_grad_()
b = torch.zeros(10).requires_grad_()
logits_manual = xb_plano @ W + b
print(logits_manual.shape)

capa = nn.Linear(784, 10)
print(capa.weight.shape, capa.bias.shape, capa(xb_plano).shape)

# %%
# Dos capas lineales seguidas equivalen a una sola
W1 = torch.randn(784, 50)
W2 = torch.randn(50, 10)
x = torch.randn(5, 784)
print(torch.allclose((x @ W1) @ W2, x @ (W1 @ W2), atol=1e-3))

# %% [markdown]
# ## Activaciones

# %%
x = torch.linspace(-5, 5, 11)
print(x)
print(torch.relu(x))

xs = torch.linspace(-8, 8, 200)
fig, axs = plt.subplots(1, 2, figsize=(9, 3))
axs[0].plot(xs, torch.relu(xs))
axs[0].set_title("ReLU")
axs[1].plot(xs, torch.sigmoid(xs))
axs[1].set_title("Sigmoide")
for ax in axs:
    ax.grid(True)
plt.show()

# %%
# La sigmoide se satura: en 20 el resultado es exactamente 1.0 y el gradiente exactamente 0
s = torch.sigmoid(tensor([-20.0, 20.0]))
print(s.tolist(), s[1] == 1.0)

xg = tensor([-20.0, 0.0, 20.0], requires_grad=True)
torch.sigmoid(xg).sum().backward()
assert xg.grad is not None
print(xg.grad.tolist())

# %%
modelo = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
print(modelo(xb_plano).shape)
print(sum(p.numel() for p in modelo.parameters()))  # 784*50+50 + 50*10+10 = 39760

# %% [markdown]
# ## Softmax, cross-entropy y accuracy

# %%
torch.manual_seed(1)
modelo = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
logits = modelo(xb_plano)

probs = logits.softmax(dim=1)  # dim=1: normaliza a lo largo del eje de las clases
print(probs.shape, probs.sum(dim=1)[:5])

# probs[range(n), yb] empareja índices: un valor por fila, el de su clase correcta
p_correcta = probs[range(len(yb)), yb]
print(p_correcta.shape)

loss_manual = -p_correcta.log().mean()
loss_nativo = F.cross_entropy(logits, yb)  # recibe logits, no probabilidades
print(loss_manual.item(), loss_nativo.item())
print("esperado al inicio: ln(10) =", torch.log(tensor(10.0)).item())

print("accuracy inicial:", accuracy(logits, yb).item())  # ~0.10


# %% [markdown]
# ## Un paso de entrenamiento


# %%
def medir(m, x, y):
    with torch.no_grad():
        return F.cross_entropy(m(x.flatten(1)), y).item()


def paso(m, x, y, lr):
    loss = F.cross_entropy(m(x.flatten(1)), y)
    loss.backward()
    with torch.no_grad():
        for p in m.parameters():
            assert p.grad is not None
            p -= lr * p.grad
            p.grad.zero_()


torch.manual_seed(1)
modelo = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
xb2, yb2 = next(iter(train_dl))

antes_mismo, antes_otro = medir(modelo, xb, yb), medir(modelo, xb2, yb2)
paso(modelo, xb, yb, lr=0.5)
despues_mismo, despues_otro = medir(modelo, xb, yb), medir(modelo, xb2, yb2)

print(f"mismo lote: {antes_mismo:.4f} -> {despues_mismo:.4f}")
print(f"otro lote:  {antes_otro:.4f} -> {despues_otro:.4f}")


# %%
# Experimento: ¿qué lr diverge en esta red? Predice antes de ejecutar
def probar_lr(lr, pasos=30):
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
    it = iter(train_dl)
    for _ in range(pasos):
        x, y = next(it)
        paso(m, x, y, lr)
    xv, yv = next(iter(valid_dl))
    with torch.no_grad():
        out = m(xv.flatten(1))
    return F.cross_entropy(out, yv).item(), accuracy(out, yv).item()


for lr in (0.01, 0.5, 5.0, 50.0):
    loss_v, acc_v = probar_lr(lr)
    print(f"lr={lr:>5}  loss validación={loss_v:.3f}  accuracy={acc_v:.3f}")

# %%
y = tensor([0, 1])

seguro = tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
dudoso = tensor([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])

for nombre, lg in (("seguro", seguro), ("dudoso", dudoso)):
    print(nombre, F.cross_entropy(lg, y).item(), accuracy(lg, y).item())
