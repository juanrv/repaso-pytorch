# %%
import matplotlib.pyplot as plt
import torch
from torch import tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
device

# %%
escalar = tensor(7)
vector = tensor([3, 2.1, 42, 13])
matriz = tensor([[4, 3, 1], [4, 4, 1]])
cubo = torch.rand(3, 2, 1)

for t in (escalar, vector, matriz, cubo):
    print(t.ndim, t.shape, t.dtype)
# %%
a = torch.ones(5)
a

# %%
a + 1, a * 3, a > 0, a.dtype

# %%
matriz = torch.ones(3, 3)
vector = tensor([1, 2, 3])
matriz + vector, matriz * vector
# %%
columna = vector[:, None]
columna.shape, matriz * columna

# %%
vector.shape, vector[None].shape, vector[:, None].shape, vector[None, :, None].shape

# %%
vector[None, :, None]

# %%
# %%
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


# %%
def matmul_loops(a, b):
    ar, ac = a.shape
    _, bc = b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k] * b[k, j]
    return c


a = torch.rand(5, 10)
b = torch.rand(10, 6)
matmul_loops(a, b).shape


# %%
a[0, :]
# %%
a[0,:,None]

# %%
def matmul_bc(a, b):
    ar, _ = a.shape
    _, bc = b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i] = (a[i, :, None] * b).sum(dim=0)
    return c


a[0, :, None].shape, (a[0, :, None] * b).shape

# %%
c1 = matmul_loops(a, b)
c2 = matmul_bc(a, b)
c3 = torch.einsum("ij,jk->ik", a, b)
c4 = a @ b

torch.allclose(c1, c2), torch.allclose(c1, c3), torch.allclose(c1, c4)

# %%
# Speed test between different approaches
a = torch.rand(100,784)
b = torch.rand(784,10)

%timeit matmul_loops(a,b)
%timeit matmul_bc(a, b)
%timeit torch.einsum('ij,jk->ik', a, b)
%timeit a @ b
# %%
m = torch.rand(4, 3)
# %%
m.shape
# %%
m.T.shape
# %%

torch.allclose(torch.einsum('ij->ji', m), m.T)
torch.allclose(torch.einsum('ij->', m), m.sum())
torch.allclose(torch.einsum('ij->i', m), m.sum(dim=1))


# %%

m = torch.arange(12).reshape(4, 3)
m, m.sum(dim=0), m.sum(dim=1)

# %%
t = torch.rand(2, 3, 4)

torch.allclose(torch.einsum("ijk->jk",t), t.sum(dim=0))
torch.allclose(torch.einsum("ijk->ik",t), t.sum(dim=1))
torch.allclose(torch.einsum("ijk->ij",t), t.sum(dim=2))
torch.allclose(torch.einsum("ijk->j",t), t.sum(dim=(0,2)))
