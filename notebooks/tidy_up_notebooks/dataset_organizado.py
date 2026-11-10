# %% [markdown]
# # Repaso de PyTorch: Datasets, Loaders, Autograd y Redes Neuronales
# Archivo organizado progresivamente desde la ingesta de datos hasta un paso de entrenamiento.

# %%
# ==========================================
# 1. IMPORTACIONES Y DESCARGA DE DATOS
# ==========================================
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

# Descargar FashionMNIST (imágenes de ropa de 28x28 píxeles)
train_raw = datasets.FashionMNIST("data", train=True, download=True)
valid_raw = datasets.FashionMNIST("data", train=False, download=True)
classes = train_raw.classes

print("Entrenamiento:", len(train_raw), "Validación:", len(valid_raw))
print("Clases:", classes)

# %%
# Exploración inicial de un dato crudo (formato PIL Image)
img, label = train_raw[0]
print(
    f"Tipo crudo: {type(img)}, Tamaño: {img.size}, Etiqueta: {label} ({classes[label]})"
)

# %%
# ==========================================
# 2. PREPROCESAMIENTO Y NORMALIZACIÓN
# ==========================================
x_train = train_raw.data.float().div(255)[:, None]
y_train = train_raw.targets

x_valid = valid_raw.data.float().div(255)[:, None]
y_valid = valid_raw.targets

print("--- Datos procesados ---")
print(f"Shape X: {x_train.shape}, Dtype: {x_train.dtype}")
print(
    f"Mínimo: {x_train.min():.2f}, Máximo: {x_train.max():.2f}, Media: {x_train.mean():.4f}"
)


# %%
# ==========================================
# 3. VISUALIZACIÓN BÁSICA
# ==========================================
def show(img, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)


show(x_train[0, 0], title=classes[y_train[0]])
plt.show()


# %%
# ==========================================
# 4. CREACIÓN DEL DATASET PERSONALIZADO
# ==========================================
class Dataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = Dataset(x_train, y_train)
valid_ds = Dataset(x_valid, y_valid)

print("Total de datos en Dataset:", len(train_ds))
xb, yb = train_ds[0:5]
print("Slicing (primeros 5):", xb.shape, yb)


# %%
# ==========================================
# 5. EXPERIMENTANDO CON LOTES (Generador paso a paso)
# ==========================================
def lotes(ds, bs, shuffle=False):
    n = len(ds)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield ds[idx[i : i + bs]]


total_muestras = sum(
    len(yb_batch) for _, yb_batch in lotes(train_ds, 256, shuffle=True)
)
print("Total de muestras recorridas en lotes manuales:", total_muestras)


# %%
# ==========================================
# 6. ENCAPSULANDO EN LA CLASE LOADER
# ==========================================
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
print(f"Loader Custom - Shape batch X: {xb.shape}, y: {yb.shape}")

# %%
# ==========================================
# 7. EL DATALOADER NATIVO DE PYTORCH
# ==========================================
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=256)

xb, yb = next(iter(train_dl))
print("DataLoader Nativo - X shape:", xb.shape, "y shape:", yb.shape)

# %%
# ==========================================
# 8. VISUALIZACIÓN DE UN LOTE COMPLETO
# ==========================================
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for ax, im, l in zip(axs.flat, xb[:16], yb[:16]):
    show(im[0], ax, classes[l])
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 9. BENCHMARKING (Custom vs Nativo)
# ==========================================
print("Tiempos de ejecución (descomenta en Jupyter para probar):")
# %timeit for _ in Loader(train_ds, 256, shuffle=True): pass
# %timeit for _ in DataLoader(train_ds, batch_size=256, shuffle=True): pass

# %%
# =========================================
# 10. AUTOGRAD Y CÁLCULO DE GRADIENTES
# =========================================
# Definimos un peso que requiere seguimiento de gradientes
w = torch.tensor(3.0, requires_grad=True)
loss = (w * 2 - 10) ** 2
loss.backward()

print("Derivada de la pérdida respecto a w (w=3):", w.grad)

# Actualización manual del peso (Descenso de Gradiente)
lr = 0.1
with torch.no_grad():
    assert w.grad is not None
    w -= lr * w.grad
print("Nuevo valor de w tras un paso:", w.item())

# %%
# Comprobación del rastreo (Grafo Dinámico)
w = torch.tensor(3.0, requires_grad=True)

# Operación normal: PyTorch construye el grafo
loss_a = (w * 2 - 10) ** 2
print("¿loss_a requiere grad?:", loss_a.requires_grad, "| Función:", loss_a.grad_fn)

# Operación dentro de no_grad: PyTorch ignora el rastreo para ahorrar memoria
with torch.no_grad():
    loss_b = (w * 2 - 10) ** 2
print("¿loss_b requiere grad?:", loss_b.requires_grad, "| Función:", loss_b.grad_fn)


# %%
# Experimentando con diferentes Learning Rates
def entrenar(lr, w0=3.0, pasos=6):
    w = torch.tensor(w0, requires_grad=True)
    historial = [round(w.item(), 3)]
    for _ in range(pasos):
        loss = (w * 2 - 10) ** 2
        loss.backward()
        with torch.no_grad():
            assert w.grad is not None
            w -= lr * w.grad
        w.grad.zero_()  # Limpiar gradientes es vital antes de la siguiente iteración
        historial.append(round(w.item(), 3))
    return historial


print("Evolución de w (LR=0.100):", entrenar(0.1))
print("Evolución de w (LR=0.125):", entrenar(0.125))  # Convergencia exacta en 1 paso
print("Evolución de w (LR=0.300):", entrenar(0.3))  # Rebote/Divergencia

# %%
# =========================================
# 11. CAPAS LINEALES (Redes Neuronales Básicas)
# =========================================
xb, yb = next(iter(train_dl))

# Aplanado (Flatten): Convierte (256, 1, 28, 28) en (256, 784)
xb_plano = xb.flatten(1)
print("Shape de entrada aplanada:", xb_plano.shape)

# Creacion manual de pesos y sesgos (bias)
weights = torch.randn(784, 10) * 0.01
bias = torch.zeros(10)
weights.requires_grad_()
bias.requires_grad_()

logits_manual = xb_plano @ weights + bias
print("Shape de logits (manual):", logits_manual.shape)

# Implementación nativa de PyTorch (nn.Linear hace lo mismo internamente)
capa = nn.Linear(784, 10)
print("Shape pesos capa:", capa.weight.shape, "| Shape bias capa:", capa.bias.shape)
print("Shape de logits (nn.Linear):", capa(xb_plano).shape)

# %%
# =========================================
# 12. FUNCIONES DE ACTIVACIÓN
# =========================================
# ReLU: max(0, x). Apaga valores negativos, deja pasar positivos.
# Impide que múltiples capas lineales colapsen matemáticamente en una sola.
x_relu = torch.linspace(-5, 5, 11)
print("Entrada:", x_relu)
print("ReLU:   ", torch.relu(x_relu))

# Sigmoide: 1 / (1 + exp(-x)). Comprime cualquier número al rango (0, 1).
x_sig = torch.linspace(-8, 8, 200)

fig, axs = plt.subplots(1, 2, figsize=(9, 3))
axs[0].plot(x_sig, torch.relu(x_sig))
axs[0].set_title("ReLU")
axs[1].plot(x_sig, torch.sigmoid(x_sig))
axs[1].set_title("Sigmoide")
for ax in axs:
    ax.grid(True)
plt.show()

# Comprobación de gradientes en Sigmoide
x_grad = torch.tensor([-20.0, 0.0, 20.0], requires_grad=True)
torch.sigmoid(x_grad).sum().backward()
print("Gradientes de Sigmoide en -20, 0 y 20:", x_grad.grad.tolist())
# Nota: Observa cómo en -20 y 20 el gradiente es casi 0 (Vanishing Gradient)

# Creación de un modelo secuencial con activación intermedia
modelo = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
print("Shape salida del modelo secuencial:", modelo(xb_plano).shape)

# %%
# =========================================
# 13. SOFTMAX, CROSS-ENTROPY Y MÉTRICAS
# =========================================
logits = modelo(xb.flatten(1))

# Softmax: Convierte los logits en probabilidades que suman 1
probs = logits.softmax(dim=1)
print("Probabilidades (primeras 5 suman 1):", probs.sum(dim=1)[:5])

# Cross-entropy manual: Negativo del logaritmo de las probabilidades correctas
p_correcta = probs[range(len(yb)), yb]
loss_manual = -p_correcta.log().mean()

# Cross-entropy nativo: F.cross_entropy hace el softmax y el log de forma combinada y más estable
loss_nativo = F.cross_entropy(logits, yb)

print("Pérdida manual:", loss_manual.item())
print("Pérdida nativa (F.cross_entropy):", loss_nativo.item())


def accuracy(logits, yb):
    # argmax(dim=1) encuentra el índice con mayor probabilidad (la predicción)
    return (logits.argmax(dim=1) == yb).float().mean()


print("Precisión inicial (aleatoria):", accuracy(logits, yb).item())

# %%
# =========================================
# 14. UN PASO COMPLETO DE ENTRENAMIENTO
# =========================================
lr = 0.5
loss_inicial = F.cross_entropy(modelo(xb.flatten(1)), yb)

# Backward pass
loss_inicial.backward()

# Actualización de parámetros de toda la red
with torch.no_grad():
    for p in modelo.parameters():
        assert p.grad is not None
        p -= lr * p.grad
        p.grad.zero_()

loss_final = F.cross_entropy(modelo(xb.flatten(1)), yb)
print(f"Pérdida antes del paso: {loss_inicial.item():.4f}")
print(f"Pérdida después del paso: {loss_final.item():.4f}")

# %%
probs[range(len(yb)), yb]
