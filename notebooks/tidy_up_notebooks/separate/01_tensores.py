# %% [markdown]
# # 01 · Tensores, broadcasting, matmul y einsum

# %%
import torch
from torch import tensor
from utils import setup

setup()

# %%
escalar = tensor(7)
vector = tensor([3, 2.1, 42, 13])
matriz = tensor([[4, 3, 1], [4, 4, 1]])
cubo = torch.rand(5, 3, 4)

for t in (escalar, vector, matriz, cubo):
    print(t.ndim, tuple(t.shape), t.dtype)

# %% [markdown]
# ## Broadcasting
# Se alinean las formas desde la derecha. Dos dimensiones son compatibles
# si son iguales o si una de ellas es 1 (o no existe).

# %%
m = torch.arange(9.0).reshape(3, 3)
v = tensor([1.0, 2.0, 3.0])
col = v[:, None]

print("m * v   (v se estira por filas):\n", m * v)
print("m * col (col se estira por columnas):\n", m * col)

# %%
# El None inserta el eje nuevo en la posición donde lo escribes
t = torch.zeros(60000, 28, 28)
print(t[:, None].shape)
print(t[None].shape)
print(t[:, :, None].shape)
print(t[..., None].shape)

# %%
# ¿Cuáles dan error? Predice antes de ejecutar
casos = [
    ((5,), ()),
    ((3, 3), (3,)),
    ((3, 3), (3, 1)),
    ((10, 3), (3, 1)),
    ((256, 256, 3), (3,)),
    ((9, 9, 5, 5), (9, 1, 5, 5)),
    ((10, 40, 2, 54), (10, 39, 2, 54)),
]

for sa, sb in casos:
    try:
        print(sa, sb, "->", tuple((torch.ones(sa) * torch.ones(sb)).shape))
    except RuntimeError:
        print(sa, sb, "-> error")


# %% [markdown]
# ## Multiplicación de matrices
# c[i, k] = suma sobre j de a[i, j] * b[j, k]


# %%
def matmul_loops(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k] * b[k, j]
    return c


def matmul_bc(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i] = (a[i, :, None] * b).sum(dim=0)
    return c


a = torch.rand(5, 10)
b = torch.rand(10, 6)

print(a[0].shape, a[0, :, None].shape)
print((a[0, :, None] * b).shape, (a[0, :, None] * b).sum(dim=0).shape)

c_loops = matmul_loops(a, b)
c_bc = matmul_bc(a, b)
c_ein = torch.einsum("ij,jk->ik", a, b)
c_at = a @ b
print(
    torch.allclose(c_loops, c_bc),
    torch.allclose(c_loops, c_ein),
    torch.allclose(c_loops, c_at),
)

# %%
# Mediciones propias con a=(100, 784) y b=(784, 10), en Jupyter:
#   tres bucles   5.08 s
#   broadcasting  819 µs
#   einsum        32.2 µs
#   a @ b         22.9 µs
# a = torch.rand(100, 784); b = torch.rand(784, 10)
# %timeit matmul_loops(a, b)
# %timeit matmul_bc(a, b)
# %timeit torch.einsum('ij,jk->ik', a, b)
# %timeit a @ b

# %% [markdown]
# ## einsum: la letra que no aparece en la salida se suma

# %%
intermedio = torch.einsum("ij,jk->ijk", a, b)
print(intermedio.shape)
print(torch.allclose(intermedio.sum(dim=1), a @ b))
print(torch.allclose(a[:, :, None] * b[None, :, :], intermedio))

# %% [markdown]
# ## dim: el eje indicado desaparece de la forma

# %%
m = torch.arange(12).reshape(4, 3)
print(m)
print("sum(dim=0):", m.sum(dim=0), tuple(m.sum(dim=0).shape))
print("sum(dim=1):", m.sum(dim=1), tuple(m.sum(dim=1).shape))

# %%
m = torch.rand(4, 3)
assert torch.allclose(torch.einsum("ij->ji", m), m.T)
assert torch.allclose(torch.einsum("ij->", m), m.sum())
assert torch.allclose(torch.einsum("ij->i", m), m.sum(dim=1))

t = torch.rand(2, 3, 4)
assert torch.allclose(torch.einsum("ijk->jk", t), t.sum(dim=0))
assert torch.allclose(torch.einsum("ijk->ik", t), t.sum(dim=1))
assert torch.allclose(torch.einsum("ijk->ij", t), t.sum(dim=2))
assert torch.allclose(torch.einsum("ijk->j", t), t.sum(dim=(0, 2)))
print("einsum y sum coinciden")
