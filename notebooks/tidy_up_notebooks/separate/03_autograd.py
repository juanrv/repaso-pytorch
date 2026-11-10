# %% [markdown]
# # 03 · Autograd: forward, loss, backward, step
# loss = (2w - 10)^2  ->  gradiente = 8 (w - 5)  ->  óptimo en w = 5

# %%
import torch
from utils import setup

setup()

# %%
for w0 in (3.0, 7.0, 10.0, 5.0):
    w = torch.tensor(w0, requires_grad=True)
    loss = (w * 2 - 10) ** 2
    loss.backward()
    assert w.grad is not None
    print(f"w={w0:>4}  loss={loss.item():>5.1f}  grad={w.grad.item():>5.1f}")

# %%
# Un paso: w = w - lr * grad
w = torch.tensor(3.0, requires_grad=True)
loss = (w * 2 - 10) ** 2
loss.backward()
assert w.grad is not None
with torch.no_grad():
    w -= 0.1 * w.grad
print(w.item())  # 4.6

# %%
# backward dos veces sobre el mismo loss: el grafo ya se liberó
w = torch.tensor(3.0, requires_grad=True)
loss = (w * 2 - 10) ** 2
loss.backward()
try:
    loss.backward()
except RuntimeError as e:
    print("RuntimeError:", str(e)[:70], "...")

# %%
# Con un forward nuevo cada vez, los gradientes se acumulan: por eso existe zero_grad
w = torch.tensor(3.0, requires_grad=True)
for _ in range(2):
    loss = (w * 2 - 10) ** 2
    loss.backward()
assert w.grad is not None
print(w.grad.item())  # -32 = -16 + -16
w.grad.zero_()
print(w.grad.item())

# %%
# no_grad: sin historial de cálculo, usa menos memoria (validación)
w = torch.tensor(3.0, requires_grad=True)

loss_a = (w * 2 - 10) ** 2
print(loss_a.requires_grad, loss_a.grad_fn)

with torch.no_grad():
    loss_b = (w * 2 - 10) ** 2
print(loss_b.requires_grad, loss_b.grad_fn)


# %%
# Learning rate: cada paso multiplica la distancia al óptimo por (1 - 8*lr)
def entrenar(lr, w0=3.0, pasos=6):
    w = torch.tensor(w0, requires_grad=True)
    historial = [round(w.item(), 3)]
    for _ in range(pasos):
        loss = (w * 2 - 10) ** 2
        loss.backward()
        assert w.grad is not None
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
        historial.append(round(w.item(), 3))
    return historial


print("lr=0.100", entrenar(0.1))  # se acerca poco a poco
print("lr=0.125", entrenar(0.125))  # 1 - 8*0.125 = 0: llega en un paso
print("lr=0.300", entrenar(0.3))  # factor -1.4: se aleja oscilando
print("w0=10    ", entrenar(0.1, w0=10.0))  # 10 -> 6 -> 5.2 ...
