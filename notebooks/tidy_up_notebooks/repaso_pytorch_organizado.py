# %% [markdown]
# # Repaso de PyTorch: Fundamentos, Broadcasting y Rendimiento
# Archivo organizado modularmente para repasar conceptos clave.

# %%
# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
import torch
from torch import tensor

# Detectar hardware para aceleración (GPU/CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fijar semilla para reproducibilidad y configurar formato de impresión limpio
torch.manual_seed(1)
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
print("Dispositivo activo:", device)

# %%
# ==========================================
# 2. CREACIÓN Y PROPIEDADES DE TENSORES
# ==========================================
escalar = tensor(7)
vector = tensor([3, 2.1, 42, 13])
matriz = tensor([[4, 3, 1], [4, 4, 1]])
cubo = torch.rand(3, 2, 1)

print("--- Propiedades de los Tensores (ndim, shape, dtype) ---")
for t in (escalar, vector, matriz, cubo):
    print(f"ndim: {t.ndim}, shape: {t.shape}, dtype: {t.dtype}")

# %%
# ==========================================
# 3. OPERACIONES BÁSICAS Y TIPOS
# ==========================================
a = torch.ones(5)
print("Tensor original:", a)
print("Operaciones elemento a elemento:")
print("a + 1:", a + 1)
print("a * 3:", a * 3)
print("a > 0:", a > 0)
print("Tipo de dato (dtype):", a.dtype)

# %%
# ==========================================
# 4. INTRODUCCIÓN AL BROADCASTING (Expansión dimensional)
# ==========================================
matriz = torch.ones(3, 3)
vector = tensor([1, 2, 3])

# El vector (3,) se expande automáticamente a (3,3) para operar con la matriz
print("Suma matriz + vector:\n", matriz + vector)
print("Multiplicación matriz * vector:\n", matriz * vector)

# %%
# Manipulación manual de dimensiones usando None (alias de torch.unsqueeze)
columna = vector[:, None]  # Convierte shape (3,) en shape (3,1)
print("Shape columna:", columna.shape)
print("Multiplicación matriz * columna:\n", matriz * columna)

print("\nDiferentes formas de agregar dimensiones ficticias:")
print(
    vector.shape, vector[None].shape, vector[:, None].shape, vector[None, :, None].shape
)

# %%
# Comprobación de compatibilidad de Broadcasting
# Regla principal: Alineados a la derecha, las dimensiones deben ser iguales o una de ellas debe ser 1.
casos = [
    ((5,), ()),
    ((3, 3), (3,)),
    ((3, 3), (3, 1)),
    ((10, 3), (3, 1)),
    ((256, 256, 3), (3,)),
    ((9, 9, 5, 5), (9, 1, 5, 5)),
    ((10, 40, 2, 54), (10, 39, 2, 54)),
]

print("--- Test de Reglas de Broadcasting ---")
for sa, sb in casos:
    try:
        # Multiplicar tensores de unos simulados con esas dimensiones
        shape_resultado = tuple((torch.ones(sa) * torch.ones(sb)).shape)
        print(f"{sa} * {sb} -> {shape_resultado}")
    except RuntimeError:
        print(f"{sa} * {sb} -> ERROR de compatibilidad")

# %%
# ==========================================
# 5. MULTIPLICACIÓN DE MATRICES (Evolución)
# ==========================================


# 5.1 Enfoque ingenuo: Bucles anidados (MUY LENTO EN PYTHON)
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
print("Shape resultado loops:", matmul_loops(a, b).shape)


# %%
# 5.2 Enfoque con Broadcasting (Vectorización parcial)
def matmul_bc(a, b):
    ar, _ = a.shape
    _, bc = b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        # Extrae la fila 'i', la vuelve columna, multiplica contra 'b' y suma verticalmente
        c[i] = (a[i, :, None] * b).sum(dim=0)
    return c


# Ejemplo de la transformación en una fila:
print("Shape a[0] extendida:", a[0, :, None].shape)
print("Shape antes del sum():", (a[0, :, None] * b).shape)

# %%
# Validación matemática: ¿Todos los métodos dan exactamente el mismo resultado?
c1 = matmul_loops(a, b)
c2 = matmul_bc(a, b)
c3 = torch.einsum("ij,jk->ik", a, b)  # Notación de Einstein
c4 = a @ b  # Operador nativo de Python para matmul

print("¿Broadcasting es igual a loops?:", torch.allclose(c1, c2))
print("¿Einsum es igual a loops?:", torch.allclose(c1, c3))
print("¿'@' nativo es igual a loops?:", torch.allclose(c1, c4))

# %%
# ==========================================
# 6. BENCHMARKING (Test de rendimiento)
# ==========================================
# Nota: IPython Magics (%timeit) funcionan al ejecutar la celda en Zed/Jupyter
a_big = torch.rand(100, 784)
b_big = torch.rand(784, 10)

print("Tiempos de ejecución (descomenta en Jupyter para probar):")
# %timeit matmul_loops(a_big, b_big)
# %timeit matmul_bc(a_big, b_big)
# %timeit torch.einsum('ij,jk->ik', a_big, b_big)
# %timeit a_big @ b_big

# %%
# ==========================================
# 7. REDUCCIONES Y NOTACIÓN EINSTEIN (EINSUM)
# ==========================================
m = torch.arange(12).reshape(4, 3)

print("Matriz original:\n", m)
print("Suma en eje 0 (por columnas):", m.sum(dim=0))
print("Suma en eje 1 (por filas):", m.sum(dim=1))

# %%
# Equivalencias entre métodos convencionales y Einsum
m_rand = torch.rand(4, 3)

print("\nValidación de notación Einsum en 2D:")
print(
    "- Transposición (ij->ji):",
    torch.allclose(torch.einsum("ij->ji", m_rand), m_rand.T),
)
print(
    "- Suma total (ij->):", torch.allclose(torch.einsum("ij->", m_rand), m_rand.sum())
)
print(
    "- Suma filas (ij->i):",
    torch.allclose(torch.einsum("ij->i", m_rand), m_rand.sum(dim=1)),
)

# %%
# Einsum en tensores 3D (Ej: Batch x Alto x Ancho)
t = torch.rand(2, 3, 4)

print("\nValidación de notación Einsum en 3D:")
# Reducir el eje 0 (i)
print("- Reducir dim=0:", torch.allclose(torch.einsum("ijk->jk", t), t.sum(dim=0)))
# Reducir el eje 1 (j)
print("- Reducir dim=1:", torch.allclose(torch.einsum("ijk->ik", t), t.sum(dim=1)))
# Reducir el eje 2 (k)
print("- Reducir dim=2:", torch.allclose(torch.einsum("ijk->ij", t), t.sum(dim=2)))
# Reducir los ejes 0 y 2 (i, k)
print(
    "- Reducir dim=0 y dim=2:",
    torch.allclose(torch.einsum("ijk->j", t), t.sum(dim=(0, 2))),
)
