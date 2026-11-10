# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Guía de Laboratorio - CNN y Autoencoder

# %% [markdown]
# ## Objetivos

# %% [markdown]
# Los objetivos de esta guía de laboratorio son los siguientes:
# * Familiarizar a los estudiantes con el entorno de desarrollo de Jupyter Notebook.
# * Comprender las bases de la inteligencia artificial.
# * Comprender las bases y la creacion de las redes neuronales convolucionales.
# * Entender la creación del autoencoder.
#
# Debido a la naturaleza de jupyter notebook el papel de "ejercicio" lo cumplen las múltiples celdas de codigo presentes en esta guía de laboratorio, las cuales los estudiantes pueden modificar a su gusto e incluso romper la ejecución natural de este, por ese motivo se recomienda visualizar el funcionamiento base de todo antes de comenzar con la modificación de parámetros.

# %% [markdown]
# ## Bases para Jupyter Notebook

# %% [markdown]
# Las notebook son documentos creados con celdas, en algunas de ellas se puede escribir texto usando __markdown__ y en otras se puede ejecutar código en python.
#
# Se puede ejecutar una celda usando `Shift + Enter`, `Ctrl + Enter` o dando al boton `▶ Run`.

# %% [markdown]
# Hay dos tipos de celda:
#  - _`Markdown Cell`_: Son las celdas donde se puede escribir texto e incluso adjuntar imágenes, también aceptan LaTeX para la escritura de formulas. 
#  
#  $$a_{n} = 2^{(4-n)}$$
#  
#  
#  $$\lim_{n\to\infty}a_n = L$$
#
#   
#       Para ver como hacer esto dar doble click a esta celda para ver su estructura.
#       
#   - _`Code Cell`_: Son las celdas donde se puede ejecutar codigo en python, el orden de ejecución aparece al lado izquierdo de la celda marcado con <kbd>In [#]</kbd>, en caso de tener algún tipo de salida esta aparecerá debajo de la celda y marcada al lado izquierdo con <kbd>Out [#]</kbd>.

# %%
2/3

# %% [markdown]
# Hay que tener en cuenta que no es necesario colocar `print()` ya que jupyter notebook ya asume que se quiere ver el resultado de operaciones directas o variables

# %%
a = 37
a, a + 3, a*958394934

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Modos

# %% [markdown]
# En jupyter notebook hay dos modos, `Edit Mode` y `Command Mode`.
#
# - `Edit Mode`: Te permite editar las celdas, para editar una celda, dar doble click si es una _markdown cell_ o dar al `Enter`, la celda tendrá un borde verde o azule en este modo.
# - `Command Mode`: Te permite editar la estructura general de la notebook.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Comandos

# %% [markdown]
# Comandos importantes a tener en cuenta para un correcto uso de las notebook:
#
# - `Shift + Enter`: Ejecutar las celdas seleccionadas y seleccionar la siguiente.
# - `Ctrl + Enter`: Ejecutar las celdas seleccionadas.
# - `Shift + Tab`: Ver información de una función.
# - `D+D`: En _command mode_, eliminar las celdas seleccionadas.
# - `0+0`: Reiniciar el kernel para ejecutar todo de nuevo.
# - `A`: En _command mode_, crea una celda arriba de la seleccionada.
# - `B`: En _command mode_, crea una celda debajo de la seleccionada.
# - `M`: En _command mode_, convierte la celda seleccionada en una _markdown cell_.
# - `Y`: En _command mode_, convierte la celda seleccionada en una _code cell_.
# - `Z`: Retroceder en _command mode_, funciona igual que el `Ctrl + Z`.
#
#   
# Para ver más comandos es recomendado ver la sección <kbd>Help > Show Keyboard Shortcuts</kbd> a lo largo de esta guía de laboratorio es recomendado ejecutar todo multiples veces o en caso de encontrar un error y se requiere ejecutar todo ir a <kbd>Run > Run All Above Selected</kbd>.

# %% [markdown]
# ## Imports Necesarios

# %% [markdown]
# Imports necesarios para la guía de laboratorio

# %%
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
import fastcore.all as fc
import pandas as pd
from collections.abc import Mapping
from pathlib import Path
from operator import attrgetter,itemgetter
from functools import partial
from __future__ import annotations
from itertools import zip_longest

from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate, SequentialSampler, RandomSampler, BatchSampler
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datasets import load_dataset,load_dataset_builder

from fastprogress import progress_bar,master_bar
from typing import Mapping
from fastcore.test import test_close

# %%
import logging
logging.disable(logging.WARNING)

# %% [markdown]
# ## Bases

# %% [markdown]
# ### Tensores

# %%
from torch import tensor

# %% [markdown]
# En matemáticas los tensores son una generalización de escalares, vectores y matrices, los tensores se pueden clasificar segun su rango, teniendo n-rangos.

# %% [markdown]
# Ejemplo de un tensor de rango 0 o simplemente un escalar.

# %%
a = torch.tensor(14029309)
a

# %%
a = 9902320
b = torch.tensor(a)
b

# %% [markdown]
# Ejemplo de un tensor de rango 1 o vector.

# %%
a = [3,2.1,42,13]
b = torch.tensor(a)
b

# %% [markdown]
# Ejemplo de un tensor de rango 2 o matriz.

# %%
a = [[4,3,1], [4,4,1]]
a = torch.tensor(a)
a

# %% [markdown]
# Se pueden crear tensores de diferentes formas y tamaños.

# %%
shape = (2,3)
a = torch.rand(shape)
a

# %% [markdown]
# Tensor de rango 3

# %%
shape = (5,3,4)
a = torch.rand(shape)
a

# %% [markdown]
# Tensor de rango 4

# %%
shape = (5,3,4,2)
a = torch.rand(shape)
a

# %% [markdown]
# Suma de tensores con escalares

# %%
a = torch.rand(2)
a, a + 1

# %% [markdown]
# Multiplicación de tensores por escalar

# %%
a = torch.ones(3)
a, a * 2, a * 45

# %% [markdown]
# ### Broadcasting 

# %% [markdown]
# Internamente para las operaciones de multiplicaciones entre tensores y por consiguiente optimizar procesos como la multiplicación de matrices se usa algo conocido como __broadcasting__, inicialmente es un concepto procedente de un lenguaje de programación conocido como [APL](https://tryapl.org/) donde se podia utilizar para las operaciones como la de un tensor por un escalar.

# %%
a = torch.ones(5)
a + 1, a * 3, a > 0

# %% [markdown]
# Sin embargo la idea del broadcasting no se quedó ahí, gracias a otro lenguaje de programación algo desconocido llamado [Yorick](https://software.llnl.gov/yorick-doc/manual/yorick_50.html) se implemento una forma de realizar broadcasting con tensores de rango superior y no solo escalares.
#
# De esta manera se puede hacer cosas como un broadcast de una matriz con un vector

# %%
matrix = torch.ones(3,3);matrix

# %%
vector = torch.rand(3);vector

# %%
matrix + vector, matrix * vector, matrix / vector

# %% [markdown]
# Los tensores también permiten agregar ejes colocando `[:]` que sigifica que se toma todo de ese eje y `[None]` para agregar un eje más.

# %%
vector.shape, vector[None,:,None,None].shape, vector[None,:,None,None]

# %% [markdown]
# De esta manera se puede hacer el broadcasting con un "vector vertical" y hacer el broadcasting en columnas en vez de filas.

# %%
vertical_vector = vector[:,None]; vertical_vector

# %%
matrix + vertical_vector, matrix * vertical_vector, matrix / vertical_vector

# %% [markdown]
# La matriz no necesariamente tiene que ser cuadrada

# %%
matrix = torch.ones(10,3); matrix

# %%
matrix * vector, matrix + vector

# %% [markdown]
# Pero si se hace con el "vector vertical" arrojará un error.

# %% tags=["raises-exception"]
matrix * vertical_vector

# %% [markdown]
# Esto es debido a que el broadcasting tiene normas y no se puede usar con cualquier tamaño o rango de tensor, debido a que internamente lo que está haciendo es expandir los valores al tamaño necesario, no ocupa memoria ya que usa un truco a la hora de leer en memoria.
#
#  - La dimensión debe tener el mismo tamaño o ser 1
#  - Numpy/Pytorch empieza a verificar los tamaños de derecha a izquierda
#  
# Por ejemplo si se tiene algo como una imagen de 256x256x3 por los 3 canales RGB, se puede hacer algo así:
#         
#         Imagen      (256 x 256 x 3)
#         Escalar                  3
#         Resultado   (256 x 256 x 3)

# %%
a = torch.rand(256,256,3)
b = torch.ones(3)
(a*b).shape

# %%
a = torch.ones(9,9,5,5)
b = torch.rand(9,1,5,5)
(a*b).shape

# %% tags=["raises-exception"]
a = torch.ones(10,40,2,54)
b = torch.ones(10,39,2,54)
a*b

# %% [markdown]
# ### Multiplicación de Matrices

# %% [markdown]
# Internamente toda inteligencia artficial que use redes neuronales utilizan como base la multiplicación de matrices, por ejemplo esta red neuronal convolucional en el excel que se encuentra en este [repositorio](https://github.com/fastai/course22p2/blob/master/xl/conv-example.xlsx).
#
# ![image.png](attachment:image.png)

# %% [markdown]
# Por este motivo es importante recordar la multiplicación de matrices.
#
#  $$ \begin{bmatrix}
#          a_{11} & a_{12} & \cdots & a_{1n}\\
#          a_{21} & a_{22} & \cdots & a_{2n}\\
#          \vdots & \vdots & \ddots & \vdots\\
#          a_{m1} & a_{m2} & \cdots & a_{mn}\\ 
#      \end{bmatrix}
#      \times
#      \begin{bmatrix}
#          b_{11} & b_{12} & \cdots & b_{1p}\\
#          b_{21} & b_{22} & \cdots & b_{2p}\\
#          \vdots & \vdots & \ddots & \vdots\\
#          b_{n1} & b_{n2} & \cdots & b_{np}\\
#      \end{bmatrix}
#       =
#      \begin{bmatrix}
#          c_{11} & c_{12} & \cdots & c_{1p}\\
#          c_{21} & c_{22} & \cdots & c_{2p}\\
#          \vdots & \vdots & \ddots & \vdots\\
#          c_{m1} & c_{m2} & \cdots & c_{mp}\\
#      \end{bmatrix}\\
#   c_{ij}= a_{i1} b_{1j} + a_{i2} b_{2j} +\cdots+ a_{in} + b_{nj} = \sum_{k=1}^n a_{ik}b_{kj}  $$

# %% [markdown]
# Con _broadcasting_

# %%
a = torch.rand(5,10)
b = torch.rand(10,6)


# %%
def matmul(a,b):
    (ar,ac),(br,bc) = a.shape,b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i]   = (a[i,:,None] * b).sum(dim=0) 
    return c


# %%
c = matmul(a,b)
c.shape

# %% [markdown]
# Además de cosas como el broadcasting también se hace uso de algo llamado __Einstein Summation__ que es una forma compacta de representar la combinación de productos y sumas, lo que hay que tener en cuenta para usarlo es lo siguiente: 
#
# - Letras repetidas en la cadena de entrada signfica que los valores en esos ejes se van a multiplicar juntos
# - Ignorar una letra en l cadena de salida signfica que los valores en ese eje se van a sumar

# %%
a = torch.rand(4,3)
b = torch.rand(3,5)
a.shape, b.shape

# %% [markdown]
# $$a[i,j]$$  $$b[j,k]$$
#
# Entonces $$c[i, k] = (a[i,1] * b[1,k] + a[i,2] * b[2,k] + ... + a[i,j] * b[j,k])$$
#
# La matriz ___c[i, k]___ también se puede conseguir de la siguiente manera
#
# $$c[i, k] = (a[i,:] * b[:,k]).sum()$$
#
# Porque hay que recordar que `[:]` significa que se toma todos los valores de ese eje y la multiplicacion de matrices es el producto punto de esos dos vectores 
# ![image-5.png](attachment:image-5.png)
# Y así con todo lo demás.
#
# Ahora con __`Einsum`__

# %%
c = torch.einsum('ij,jk->ik', a,b)
c, c.shape

# %% [markdown]
# Pytorch ya tiene su propia implementación de la multiplicación de matrices y se hace con `@`

# %%
a@b,(a@b).shape


# %% [markdown]
# ### Red Neuronal

# %% [markdown]
# Una red neuronal es un conjunto de nodos o neuronas separadas en capas y conectadas entre si, cada neurona tiene una serie de valores de entrada, un peso asignado a cada valor y una o multiples salidas.
#
# ![image.png](attachment:image.png)
#
# Cuando se entrena una red neuronal lo que se hace es actualizar los pesos de cada capa haciendo uso de multiples tecnicas, una de las más comunes es el _Stochastic Gradient Descent_.

# %% [markdown]
# ### Modelo

# %% [markdown]
# Un modelo en inteligencia artificial es una estructura que agrupa al algoritmo de inteligencia artificial como tal, en este caso las diferentes redes neuronales, estas se entrenan posteriormente en base a un conjunto de datos.

# %% [markdown]
# ### _Learning Rate_

# %% [markdown]
# En inteligencia artificial existe un concepto y parámetro general llamado _learning rate_, dicta que tanto van a cambiar los pesos dentro de la cada capa de la red neuronal.
#
# Encontrar un _learning rate_ adecuado para un modelo es una tarea compleja debido a que si este es muy pequeño se tiene un aprendizaje mermado y que posiblemente se quede estancado, por lo contrario si se usa un _learning rate_ elevado el entrenamiento será descontrolado y erróneo, una recomendación general es usar valores dentro del rango __(0.0, 0.5]__.

# %% [markdown]
# ### Derivadas y Regla de la cadena

# %% [markdown]
# Otra base para crear una IA con redes neuronales son las derivadas y la regla de la cadena
#
# Para empezar una capa de una red neuronal se toma como la función ___f(x)___ y cuando vamos a realizar una actualización de la matriz de los pesos se debe realizar la derivada de esa función ___f'(x)___, debido a que por lo general los modelos de inteligencia artificial se componen de redes neuronales con multiples capas llegara el punto en el que se tendrá una cadena de funciones
#
# Por ejemplo.
#
# Se tiene una red neuronal de dos capas, la primera ___f(x)___ y la segunda ___g(x)___ debido a que la segunda capa de la red usa los resultados de la primera capa se tiene que el resultado final de la red tiene la forma ___g(f(x))___.
# Ya que para realizar una actualización de los pesos dentro de la red se tiene que hacer una derivada para obtener un gradiente y multiplicarlo por un _learning rate_ para actualizar los pesos de cada capa se debe hacer estas derivadas, para ello se hace uso de la regla de la cadena.
#
# Teniendo 
# $$ f(x) = y     $$
# $$f'(x) = \frac{dy}{dx} $$
# $$ g(y) = z  $$
# $$ g'(y) = \frac{dz}{dy}$$ 
#
# _z_ e _y_ son variables dependientes de _x_
#
# Para obtener la derivada de _g(y)_ se hace uso de la regla de la cadena que dice:
#
# $$ g'(y) =  \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} $$
#
# Afortunadamente pytorch ya hace todo este proceso automaticamente y es llamado __backpropagation__.

# %% [markdown]
# ### Épocas

# %% [markdown]
# Se le llama época a un ciclo de entrenamiento del modelo de inteligencia artificial

# %% [markdown]
# ### _Loss_

# %% [markdown]
# La _loss_ o perdida es un valor que dice que tan alejadas están las predicciones del modelo con respecto a los datos de entrenamiento, de esta manera se puede medir u observar el ritmo al que está entrenando el modelo.
#
# La perdida se puede calcular de multiples formas, pero la más común en procesamiento de imágenes es usando el _mean shift error_ o _cross entropy_.

# %%
# F.cross_entropy??

# %%
# F.mse_loss??

# %% [markdown]
# ### Conjunto de Entrenamiento y Validación

# %% [markdown]
# Para entrenar un modelo se necesitan un conjunto de datos de entrenamiento y un conjunto de datos de validación, cuando un modelo se está entrenando usa datos del conjunto de entrenamiento y compara los valores con el conjunto de validación, un modelo está entrenando cuando la perdida disminuye.
#
# Por lo general no se quiere un modelo extremadamente preciso en la fase de entrenamiento debido a que es más probable que este no tenga un buen rendimiento fuera de él, cuando esto ocurre se le llama _overfitting_ y el caso contrario cuando se entrena muy poco _underfitting_, para solucionarlo se debe encontrar un equilibrio entre el numero de epocas y también se puede hacer _data augmentation_ en caso de tener pocos datos de entrenamiento, este consiste en randomizar el orden de los datos de entrenamiento, en imágenes se puede hacer _padding_, eliminar canales de color, distorcionar las imágenes o cortalas, entre otras maneras de aumentar artificialmente los datos de entrenamiento.

# %% [markdown]
# ### Optimizer, Data Set y Data Loader

# %% [markdown]
# El _optimizer_ se encarga de actualizar los pesos de la red neuronal, a este es al que se le pasa el valor del _learning rate_

# %%
class Optimizer():
    def __init__(self, params, lr=0.5): 
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params: p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params: p.grad.data.zero_()


# %% [markdown]
# El Data set es el conjunto de datos, organizado a modo de tuplas `(x , y)`

# %%
class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    #El __getitem__ es lo que nos da los valores cuando hacemos por ejemplo 
    #Dataset[n:m]
    def __getitem__(self, i): return self.x[i],self.y[i]


# %% [markdown]
# Los Dataloaders son los conjuntos de datos con los que trabajará directamente el modelo de inteligencia artificial, estos se encargan de cargar los datos de acuerdo a un _batch size_.

# %%
class DataLoaders:
    def __init__(self, *dls): self.train,self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        f = collate_dict(dd['train'])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))


# %% [markdown]
# Una función `collate` se encarga de convertir un conjunto de datos en otro que sea de interes, en este caso se tiene una función `collate_dict` que se encarga de convertir un diccionario en un DataSet de tuplas de la forma `(x , y)`, con `x` siendo los datos de entrenamiento y `y` las labels.

# %%
def collate_dict(ds):
    get = itemgetter(*ds.features)
    def _f(b): return get(default_collate(b))
    return _f


# %% [markdown]
# La función `accuracy` o precisión se encarga de hacer una media de las _loss_ de la capa.

# %%
def accuracy(out, yb): return (out.argmax(dim=1)==yb).float().mean()


# %% [markdown]
# `Report` simplemente se encarga de imprimir en pantalla la precisión.

# %%
def report(loss, preds, yb): print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')


# %% [markdown]
# ### _Forward_ y _Backward_

# %% [markdown]
# Dentro de las redes neuronales se hace un proceso llamado _forward pass_ y _backward pass_, el primero de ellos siendo simplemente el paso normal de los valores de una capa de la red neuronal a la otra, dicho de otra forma los datos "avanzan", en este punto no hay demasiadas complicaciones.
#
# Por otra parte el _backward pass_ es el proceso de entrenamiento en sí mismo, en este paso es donde se hace una actualización de los pesos de las capas de la red neuronal, haciendo uso de los valores obtenidos al final del _forward pass_, la _loss_ y los gradientes obtenidos mediante la regla de la cadena o _backpropagation_. 

# %% [markdown]
# ### Función de Entrenamiento

# %% [markdown]
# La función `fit` es la encargada de empezar el entrenamiento de un modelo que se le pasa, internamente es la encargada de todo el proceso y la que realiza el _forward_ y el _backward_.
# * `epochs`: Número de epocas que se quiere que dure el entrenamiento.
# * `model`: Modelo que se va a entrenar.
# * `loss_func`: La función de perdida que se quiere usar.
# * `train_dl`: _Dataloader_ de entrenamiento.
# * `valid_dl`: _Dataloader_ de validación.

# %%
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss,tot_acc,count = 0.,0.,0
            for xb,yb in valid_dl:
                pred = model(xb)
                n = len(xb)
                count += n
                tot_loss += loss_func(pred,yb).item()*n
                tot_acc  += accuracy (pred,yb).item()*n
        print(epoch, tot_loss/count, tot_acc/count)
    return tot_loss/count, tot_acc/count


# %% [markdown]
# La función `get_dls` se encarga de crear los Data Loaders de los Data Sets que se le pasan.
#
# * `train_ds`: Data set de entrenamiento.
# * `valid_ds`: Data set de validación.
# * `bs`: _Batch size_.
# * `**kwargs`: Otros argumentos nombrados, como <kbd>shuffle = False</kbd>.
# * `shuffle = True`: Crea data loaders tomando datos aleatorios del data set, 
# * `shuffle = False`: Crea los data loaders en orden
#
# Ejemplo para llamar la función:
# ```
# DataLoaders = get_dls(train_ds, valid_ds, 256, shuffle = False, collate_fn = collate_dict)
# ```

# %%
def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))


# %% [markdown]
# ### Funciones de Impresion de Imágenes

# %% [markdown]
# Las siguientes 3 celdas solo son funciones para mostrar las imágenes, no es necesario prestarles demasiada atención

# %%
@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ('cpu','permute','detach')):
        im = im.detach().cpu()
        if len(im.shape)==3 and im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=np.array(im)
    if im.shape[-1]==1: im=im[...,0]
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    if noframe: ax.axis('off')
    return ax


# %%
@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
): # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None: figsize=(ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = np.array([ax])
    return fig,ax


# %%
@fc.delegates(subplots)
def get_grid(
    n:int, # Number of axes
    nrows:int=None, # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols:int=None, # Number of columns, defaulting to `ceil(n/rows)`
    title:str=None, # If passed, title set to the figure
    weight:str='bold', # Title font weight
    size:int=14, # Title font size
    **kwargs,
): # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: ncols = ncols or int(np.floor(n/nrows))
    elif ncols: nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n/nrows))
    fig,axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols): axs.flat[i].set_axis_off()
    if title is not None: fig.suptitle(title, weight=weight, size=size)
    return fig,axs


# %%
@fc.delegates(subplots)
def show_images(ims:list, # Images to show
                nrows:int|None=None, # Number of rows in grid
                ncols:int|None=None, # Number of columns in grid (auto-calculated if None)
                titles:list|None=None, # Optional list of titles for each image
                **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip_longest(ims, titles or [], axs): show_image(im, ax=ax, title=t)


# %% [markdown]
# Configuración adicional para mostrar las imágenes

# %%
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'
mpl.rcParams['figure.dpi'] = 70

# %% [markdown]
# ### Generales

# %% [markdown]
# En esta parte solo se definen unas funciones para envíar los datos al dispositivo
#
# * `mps`: Para dispositivos Apple que usen un chip M1 en adelante.
# * `cuda`: Para dispositivos que tengan tarjetas gráficas compatibles de NVIDIA
# * `cpu`: En caso de no tener alguno de estos dispositivos se envía a la CPU

# %%
def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def collate_device(b): return to_device(default_collate(b))


# %% [markdown]
# ## Presentación de los Datos

# %% [markdown]
# Se va a a hacer uso de los datos de **Fashion MNIST**, es un conjunto de datos que consiste en 60.000 ejemplares de entrenamiento y 10.000 ejemplares de validación, cada ejemplar consiste en una imagen de 28x28 pixeles en escala de grises además de una etiqueta correspondiente.

# %%
name = "fashion_mnist"
ds_builder = load_dataset_builder(name, trust_remote_code=True)
print(ds_builder.info.description)

# %% [markdown]
# Con `ds_builder.info.features` se puede ver las caracteríticas de los datos.

# %%
ds_builder.info.features

# %% [markdown]
# Con `ds_builder.info.splits` se puede ver como están separados los datos.

# %%
ds_builder.info.splits

# %%
x,y = 'image','label'
dsd = load_dataset(name, trust_remote_code=True)


# %% [markdown]
# _Inplace_ es una función que recibe otra función y la retorna.

# %%
def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f


# %% [markdown]
# En este punto es donde se hará uso de la función _inplace_
#
# Esta función simplemente se encarga de transformar los datos en tensores

# %%
@inplace
def transformi(b): b[x] = [TF.to_tensor(o) for o in b[x]]


# %% [markdown]
# Es necesario hacer esto para poder utilizar el `.with_transform` de los data sets de hugging face debido a que estos requieren de un _Callable_ o un objeto que _se pueda llamar_ para funcionar.

# %%
batch_size = 256
transformed_dataset = dsd.with_transform(transformi)

# %%
transformed_dataset

# %%
transformed_dataset['train']

# %% [markdown]
# Aquí se puede ver la primera imagen del Data set de entrenamiento.

# %%
ds = transformed_dataset['train']
img = ds[0]['image']
show_image(img, figsize=(1,1));

# %%
collate_function = collate_dict(ds)


# %% [markdown]
# Aquí se define una función para crear los data loaders en este caso

# %%
def collate_(b): return to_device(collate_function(b))
def data_loaders(dsd, bs, **kwargs): return {k:DataLoader(v, bs, **kwargs) for k,v in dsd.items()}


# %%
dls = data_loaders(transformed_dataset, batch_size, collate_fn = collate_)

# %%
dls

# %% [markdown]
# - `dt`: Data Loader Training
# - `dv`: data Loader Validation

# %%
dt = dls['train']
dv = dls['test']

# %%
x_batch, y_batch = next(iter(dt))

# %% [markdown]
# Aquí se puede ver que el `x_batch` consiste de 256 imágenes de 28x28.

# %%
x_batch.shape

# %% [markdown]
# Y aquí se puede ver que `y_batch` simplemente tiene las 256 etiquetas

# %%
y_batch.shape

# %%
y_batch[1]

# %%
x_batch

# %%
y_batch

# %%
labels = ds.features[y].names

# %%
labels

# %%
label_getter = itemgetter(*y_batch[:16])
titles = label_getter(labels)

# %%
show_images(x_batch[:16], imsize=1.7, titles=titles)

# %% [markdown]
# ## _Convolutional Neural Network_ (CNN)

# %% [markdown]
# Para el desarrollo de la teoría de las CNN se hará uso de los datos de **MNIST**, es una base de datos similar a la de Fashion MNIST y de hecho es en la que esta última se basa.

# %%
MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
path_data = Path('data')
#El mdkir crea un directorio,] exist_ok=True es para que no de error cuando vea que el directorio ya existe previamente
path_data.mkdir(exist_ok=True)
path_gz = path_data/'mnist.pkl.gz'

# %%
from urllib.request import urlopen
from shutil import copyfileobj

with urlopen(MNIST_URL) as in_stream, open(path_gz, 'wb') as out_file:
    copyfileobj(in_stream, out_file)
    
with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
x_train, y_train, x_valid, y_valid = map(torch.tensor, [x_train, y_train, x_valid, y_valid])

# %% [markdown]
# ### Convoluciones

# %% [markdown]
# Cuando se hace uso de inteligencia artificial en el procesamiento de imágenes por lo general se quiere priorizar el reconocimiento de patrones y la velocidad de procesamiento, para hacer esto se usa las convoluciones.
#
# Para empezar se tiene un input que puede ser una matriz de 3x3:
#
# <img alt="input" align="center" src = "images/MatrizCNN.png"/>
#
# Luego se dispone algo llamado filtro o kernel de un tamaño inferior, por ejemplo de 2x2:
#
#
# <img alt="kernel" align="center" src = "images/KernelCNN.png"/>
#
# Luego se multiplica y suma el kernel por un fragmento de la imagen y este se va _deslizando_ para obtener los demás resultados:
#
# <img alt="kernel" align="center" src = "images/AplicarKernel.png"/>
#
# <img alt="kernel" align="center" src = "images/EcuacionKernel.png"/>
#
# De esta manera se obtiene una imagen de menor tamaño, en este caso de 2x2 que debería tener las características importantes de la imagen original.

# %% [markdown]
# Se va a tomar una imagen del _training data_ y una imagen del _validation data_.

# %%
x_imgs = x_train.view(-1,28,28)
xv_imgs = x_valid.view(-1,28,28)

# %%
im3 = x_imgs[7]
show_image(im3);

# %% [markdown]
# Se crea un kernel de 3x3, en este caso para identificar los "bordes superiores" de una imagen.

# %%
top_edge = torch.tensor([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]]).float()

# %%
show_image(top_edge, noframe=False);

# %% [markdown]
# El kernel es como una pequeña ventana de 3x3 que se va deslizando por la imagen y si se tiene por ejemplo:
#
# $$\begin{matrix} a1 & a2 & a3 \\ a4 & a5 & a6 \\ a7 & a8 & a9 \end{matrix}$$
#
# Dará como resultado: $-a1-a2-a3+a7+a8+a9$.

# %% [markdown]
# Aquí se procede a visualizar la imagen que se sacó de los datos, en concentro `primeras 13 filas` y las `primeras 23 columnas`

# %%
df = pd.DataFrame(im3[:13,:23])
df.style.format(precision=2).set_properties(**{'font-size':'7pt'}).background_gradient('Greys')

# %% [markdown]
# Ahora se va a aplicar el kernel `top_edge` a una posición que va desde 3 a 6 (sin incluir el 6) y las columnas desde 14 a 17 (sin incluir el 17)

# %% [markdown]
# Aquí se obtiene un valor positivo porque "detectó" un borde superior en la imagen.

# %%
(im3[3:6,14:17] * top_edge).sum()

# %% [markdown]
# Y en este caso un valor negativo porque no es un borde superior.

# %%
(im3[7:10,14:17] * top_edge).sum()


# %% [markdown]
# Ahora se va a definir una función que haga esa multiplicación del kernel automáticamente.

# %%
def apply_kernel(row, col, kernel): return (im3[row-1:row+2,col-1:col+2] * kernel).sum()


# %% [markdown]
# Para usarla se le pasa el centro donde se requiere aplicar el kernel, por ejemplo para aplicar el que se tiene arriba se coloca el centro (4,15).

# %%
apply_kernel(4,15,top_edge)

# %% [markdown]
# Ahora bien, para que el kernel se "deslice" a lo largo de la imagen se puede hacer uso de una _list comprehension_, por ejemplo para una imagen de 5x5:

# %%
[[(i,j) for j in range(5)] for i in range(5)]

# %% [markdown]
# <img alt="kernel" align="center" src = "images/KernelSlide.svg"/>

# %% [markdown]
# y aquí se aplica con el kernel haciendo uso de la _list comprehension_ y un nuevo rango.

# %%
rng = range(1,27)
top_edge3 = torch.tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])
top_edge3

# %%
show_image(top_edge3);

# %% [markdown]
# Ahora se puede hacer lo mismo con los bordes izquierdos.

# %%
left_edge = torch.tensor([[-1,0,1],
                    [-1,0,1],
                    [-1,0,1]]).float()

# %%
show_image(left_edge, noframe=False);

# %%
left_edge3 = torch.tensor([[apply_kernel(i,j,left_edge) for j in rng] for i in rng])
show_image(left_edge3);

# %% [markdown]
# Con este proceso se está reduciendo el tamaño de la imagen y resaltando características de interés.

# %%
top_edge3.shape, left_edge3.shape

# %% [markdown]
# <img alt="kernel" align="center" src = "images/AplicarKernel2.png"/>

# %% [markdown]
# Para más información visitar: [CNNs From Different Viewpoints](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c).

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Convoluciones en PyTorch

# %%
import torch.nn.functional as F
import torch

# %% [markdown]
# Para hacer uso de las convoluciones en python se usa un truco llamado [im2col](https://hal.inria.fr/inria-00112631/) creado por Yangqing Jia para convertir una convolución tradicional en una multiplicación de matrices.

# %% [markdown]
# <img alt="kernel" align="center" src = "images/ConvMatriz.png"/>

# %% [markdown]
# Para más información visitar [Convolution in Caffe: a memo](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo) y otra forma de inplementarlo en [Numpy](https://github.com/3outeille/CNNumpy/blob/5394f13e7ed67a808a3e39fd381f168825d65ff5/src/fast/utils.py#L360).

# %% [markdown]
# Aquí lo que se va a hacer es convertir todo el proceso de la convolución en una multiplicación de matrices.

# %%
im3.shape 

# %% [markdown]
# Luego se le agregan dimensiones a la imagen (_batch y channel dimension_) porque es algo que espera PyTorch para poder usar el unfold.

# %%
inp = im3[None,None,:,:].float()
inp.shape

# %% [markdown]
# Se usa el `unfold` para un 3x3.

# %%
inp_unf = F.unfold(inp, (3,3))[0]
inp_unf.shape

# %% [markdown]
# Luego se convierte el kernel en un vector, `view` cambia la forma de un tensor, y con el -1 se está indicando que arroje todo el contenido del kernel a esa dimensión.

# %%
w = left_edge.view(-1)
w.shape

# %% [markdown]
# Y con esto se realiza la multiplicación de matrices

# %%
out_unf = w@inp_unf
out_unf.shape

# %% [markdown]
# Con `.view(26,26)` se ve la imagen con la reducción de tamaño que se esperaba.

# %%
out = out_unf.view(26,26)
show_image(out);

# %%
# %timeit -n 1 torch.tensor([[apply_kernel(i,j,left_edge) for j in rng] for i in rng]);

# %%
# %timeit -n 100 (w@F.unfold(inp, (3,3))[0]).view(26,26);

# %% [markdown]
# Y como se puede ver en los resultados de tiempo este proceso es mucho más rapido que la implementación _from scratch_ usando el `apply_kernel` que se definió arriba.

# %% [markdown]
# Pytorch ya tiene su implementación de las convolutions usando `F.conv2d`

# %%
# %timeit -n 100 F.conv2d(inp, left_edge[None,None])

# %%
diag1_edge = torch.tensor([[ 0,-1, 1],
                     [-1, 1, 0],
                     [ 1, 0, 0]]).float()

# %%
show_image(diag1_edge, noframe=False);

# %%
diag2_edge = torch.tensor([[ 1,-1, 0],
                     [ 0, 1,-1],
                     [ 0, 0, 1]]).float()

# %%
show_image(diag2_edge, noframe=False);

# %%
xb = x_imgs[:16][:,None]
xb.shape

# %%
edge_kernels = torch.stack([left_edge, top_edge, diag1_edge, diag2_edge])[:,None]
edge_kernels.shape

# %%
batch_features = F.conv2d(xb, edge_kernels)
batch_features.shape

# %% [markdown]
# Esta salida lo que demuestra es que se dispones de 16 imágenes, 4 kernels, y mapas de bordes de 26x26 (comienza con 28x28, pero se pierde un pixel a cada lado por lo discutido anteriormente). 
#
# Sin embargo, se puede ver que se obtienen los mismos resultados que antes.
#

# %%
img0 = xb[7,0]
show_image(img0);

# %%
show_images([batch_features[7,i] for i in range(4)])

# %% [markdown]
# ### _Strides_ y _Padding_

# %% [markdown]
# El _padding_ es el espaciado del kernel con respecto al input donde se aplicará, por ejemplo un _padding_ de 1 pixel.

# %% [markdown]
# <img alt="kernel" align="center" src = "images/PaddingKernel.png"/>

# %% [markdown]
# Cuando el padding se "sale" de la cuadricula se multiplica por valores que se pueden definir a conveniencia, pero es más común tomarlos como 0.
#
# El padding se utiliza cuando se quiere mantener el tamaño original del input o aumentarlo, por ejemplo un input de 5x5, un kernel de 4x4 y 1 de _padding_:

# %% [markdown]
# <img alt="kernel" align="center" src = "images/PaddingAugment.png"/>

# %% [markdown]
# Si se tiene un kernel de tamaño `ks`x`ks`, el _padding_ necesario para mantener el tamaño es dado por `ks//2`. Algo a tener en cuenta es que es recomendado usar kernels de tamaño impar debido a que son más faciles de aplicar.

# %% [markdown]
# No hay necesidad de mover el kernel de uno e uno, se puede hacer de diferentes numeros y eso es llamado  _stride_, ejemplo un _stride-2 convolution_:

# %% [markdown]
# <img alt="kernel" align="center" src = "images/PaddingStride2.png"/>

# %% [markdown]
# Las convoluciones con _stride_ de 2 y con un _padding_ de 1 son especialmente utiles en el procesamiento de imágenes, por ejemplo en los autoencoders y en clasificacion, debido a que reducen el tamaño del input en un factor de 2.

# %% [markdown]
# ### Creando la CNN

# %% [markdown]
# Para empezar se tienen los siguientes parámetros:
# - `n` y `m`: Son las dimensiones del set de entrenamiento.
# - `c`: El número de categorías.
# - `nh`: _Number of hidden_ es un tamaño intermedio por el que se va a pasar.

# %%
n,m = x_train.shape
c = y_train.max()+1
nh = 50
n,m,c,nh

# %% [markdown]
# Si se quisiera hacer un modelo lineal se haría de la siguiente forma:
# 1. Se crea un modelo secuencial con `nn.Sequential()` que a su vez contendrá la totalidad del modelo.
# 2. La primera capa será una capa lineal con `nn.Linear(m, nh)`.
#    - `m`: En este caso es el valor inicial.
#    - `nh`: Es el valor objetivo.
#    En este caso se quiere reducir el tamaño `m` a un tamaño `nh`.
# 3. Se agrega una capa de ReLU con `nn.ReLU`, esta se encarga de transformar los valores negativos de la capa anterior en 0 y deja los valores postivos intactos.
# 4. Y por último se agrega otra capa lineal, en este caso se pasa del valor `nh` a `10`.

# %%
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))

# %%
model

# %% [markdown]
# Este sería el equivalente con una CNN.

# %%
broken_cnn = nn.Sequential(
    nn.Conv2d(1,30, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(30,10, kernel_size=3, padding=1)
)

# %% [markdown]
# Sin embargo, no se puede utilizar directamente dado que no dará 10 posiblidades para cada imagen del _batch_, sino que se tendrán 10 posibilidades, para cada 16 imágenes en el _batch_ para cada 28x28 pixel porque no se dispone de ningún _stride_.

# %%
broken_cnn(xb).shape


# %% [markdown]
# Debido a que con una CNN no se puede realizar una implementación tan simple es necsario definir una función con los siguientes parámetros para facilitar su uso: 
#
# - `ni`: Numbero de inputs.
# - `nf`: Number de _filtros_, es una manera de decir la cantidad de _channels_ que tendrá la CNN, es como decir el numero de outputs.
# - `ks`: Tamaño del kernel
# - `stride`: El stride que se quiera, en ese caso de 2.
# - `act`: Un booleano para decidir si se agrega un ReLU.
#
# Esta función lo que hará entonces es devolver una capa con una convolución sola o un pequeño modelo secuencial con una convolución y una ReLU al final.

# %%
def conv(ni, nf, ks=3, stride=2, act=True):
    res = nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res


# %% [markdown]
# Con la función lista, ya se puede crear la CNN completa:
# - Inicia con un input de 1 y salida de 4, las imágenes son de 28x28 con un _stride_ de 2, por lo que la salida de la primera capa será de 14x14.
# - La siguiente capa tendrá de input la salida de la anterior y la salida a su vez será de 8, de nuevo un _stride_ 2 por lo que ahora internamente la imagen es de 7x7, así sucesivamente hasta que se tenga una "imagen" de 2x2.
# - Luego una capa que pasará de 16 a 10, sin ReLU por lo que `act=False` que dejará la "imagen" en 1x1.
# - Por último una capa de `nn.Flatten` para  quitar las dimensiones o ejes innecesarios.

# %%
simple_cnn = nn.Sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,16),           #2x2
    conv(16,10, act=False), #1x1
    nn.Flatten(),
)

# %% [markdown]
# Ahora cuando se use este modelo se tendrán 10 posibilidades para cada imagen en nuestro _batch_ de 16 en vez del caos que se presentaba antes.

# %%
simple_cnn(xb).shape, broken_cnn(xb).shape

# %% [markdown]
# Además se puede ver que el número de parámetros con respecto al modelo lineal es considerablemente menor, por lo que es más rapido usar la CNN.

# %%
tensor([o.numel() for o in model.parameters()]).sum()

# %%
tensor([o.numel() for o in simple_cnn.parameters()]).sum()

# %% [markdown]
# Y ahora está a disposición una CNN que se puede entrenar.

# %%
x_imgs = x_train.view(-1,1,28,28)
xv_imgs = x_valid.view(-1,1,28,28)
train_ds,valid_ds = Dataset(x_imgs, y_train),Dataset(xv_imgs, y_valid)

# %%
from torch import optim

bs = 256
lr = 0.4
train_dl,valid_dl = get_dls(train_ds, valid_ds, bs, collate_fn=collate_device)
opt = optim.SGD(simple_cnn.parameters(), lr=lr)

# %%
loss,acc = fit(5, simple_cnn.to(def_device), F.cross_entropy, opt, train_dl, valid_dl)

# %% [markdown]
# También se puede refinar el entrenamiento cambiando el _learning rate_

# %%
opt = optim.SGD(simple_cnn.parameters(), lr=lr/4)
loss,acc = fit(5, simple_cnn.to(def_device), F.cross_entropy, opt, train_dl, valid_dl)

# %% [markdown]
# ### Datos Adicionales

# %% [markdown]
# Si se tiene un input de la forma `64x1x28x28` los ejes o dimensiones toman el nombre de `batch, channel, height, width`, esto es comunmente representado como **NCHW** donde N corresponde al tamaño del batch, TensorFlow, otra de las grandes librerías para inteligencia artificial utiliza **NHWC** también conocido como "_Channel Last_", muchos modelos son más rápidos de esta forma y por ese motivo en los últimos años PyTorch paso a soportarlo.
#
# En la CNN recién creada se puede obersvas que la estructura de la primera capa tiene esa forma:
#
# - 1 canal de entrada.
# - 4 canales de salida.
# - Kernel de 3x3.

# %%
simple_cnn

# %%
simple_cnn[0][0]

# %%
conv1 = simple_cnn[0][0]
conv1.weight.shape

# %% [markdown]
# Si es necesario ver de forma más interactiva las multiplicaciones que hay en una CNN puede abrir conv-example.xls del siguiente [repositorio](https://github.com/fastai/course22p2/blob/master/xl/conv-example.xlsx), ir a `Formulas > Ver Precedente` te mostrará de donde salió el resultado.

# %% [markdown]
# ### Imágenes a Color

# %% [markdown]
# Las imágenes a color son solo un tensor de rango 3:

# %%
from torchvision.io import read_image
im = read_image('images/AIEmozione.jpg')
im.shape

# %%
show_image(im.permute(1,2,0))

# %%
_,axis = plt.subplots(1,3)
for car, ax, color in zip(im, axis, ('Reds', 'Greens', 'Blues')): show_image(255-car, ax = ax, cmap=color)

# %% [markdown]
# Todos estos luego se suman para obtener la imagen.

# %% [markdown]
# <img alt="kernel" align="center" src = "images/Colors.svg"/>

# %% [markdown]
# ## Entrenando la CNN

# %% [markdown]
# Ahora se procede a entrenar la CNN en preparación para el Autoencoder.

# %%
learning_rate = 0.4

# %%
cnn = nn.Sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,16),           #2x2
    conv(16,10, act=False),#1x1
    nn.Flatten()).to(def_device)

# %%
opt = optim.SGD(cnn.parameters(), lr=learning_rate)
loss,acc = fit(5, cnn, F.cross_entropy, opt, dt, dv)

# %%
dsd['train'][0]


# %% [markdown]
# ## Autoencoder

# %% [markdown]
# Un Autoencoder es basicamente un algoritmo de compresión, es una CNN que primero comprime los datos y luego los intenta recrear, inicialmente puede parecer que no tiene utilidad porque en principio no se está haciendo nada con la imagen, pero precisamente por su naturaleza de compresor es extremadamente útil para procesos más avanzados, en este caso de generación de imágenes, debido a que es computacionalmente más barato usar imágenes "comprimidas" en medio del proceso de entrenamiento.
#
# Por este motivo el autoencoder es uno de los 3 componentes más importantes de **Stable Diffusion**, siendo los otros dos **CLIP** y la **U-Net**.

# %% [markdown]
# <img alt="kernel" align="center" src = "images/SDArquitectura.png"/>

# %% [markdown]
# En esta parte se creará una `deconvolutional function`, que hará lo contrario de la convolución que se definió antes:
# - La primera capa es una `nn.UpsamplingNearest2d` que lo que hace es copiar los pixeles de la capa anterior y duplicarlos.
# - La segunda capa será una `nn.Conv2d` con un _stride_ de 1, lo que a efectos practicos duplica el tamaño de la cuadricula de la capa anterior.
# - Por último la activación de la ReLU.

# %%
def deconv(ni, nf, ks=3, act=True):
    layers = [nn.UpsamplingNearest2d(scale_factor=2),
              nn.Conv2d(ni, nf, stride=1, kernel_size=ks, padding=ks//2)]
    if act: layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# %% [markdown]
# Aquí se define dos funciones para el `fit` que se tenía antes 

# %%
def eval(model, loss_func, valid_dl, epoch=0):
    model.eval()
    with torch.no_grad():
        tot_loss,count = 0.,0
        for xb,_ in valid_dl:
            pred = model(xb)
            n = len(xb)
            count += n
            tot_loss += loss_func(pred,xb).item()*n
    print(epoch, f'{tot_loss/count:.3f}')


# %% [markdown]
# La diferencia es que ahora en la parte de `Loss` no se está calculando un loss con respecto a un _validation set_, sino que se está comparando consigo mismo (no hay `yb` sino `xb`), porque es necesario entrenarlo para que recree lo que le entro, así funciona un autoencoder.

# %%
def fit_autoencoder(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,_ in train_dl:
            loss = loss_func(model(xb), xb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        eval(model, loss_func, valid_dl, epoch)


# %% [markdown]
# Y Aquí se define el Autoencoder como tal
# - `nn.ZeroPad2d` agrega o quita un _padding_ a la imagen.
# - `nn.Sigmoid` hace que todos los valores estén dentro de 0 y 1

# %%
autoencoder = nn.Sequential(   #28x28
    nn.ZeroPad2d(2),  #32x32
    conv(1,2),        #16x16
    conv(2,4),        #8x8
#     conv(4,8),        #4x4
#     deconv(8,4),      #8x8
    deconv(4,2),      #16x16
    deconv(2,1, act=False), #32x32
    nn.ZeroPad2d(-2), #28x28
    nn.Sigmoid()
).to(def_device)

# %% [markdown]
# Puedes probar a quitar el comentario de las capas más internas, para ver que ocurre.

# %%
eval(autoencoder, F.mse_loss, dv)

# %%
opt = optim.AdamW(autoencoder.parameters(), lr=0.01)
fit_autoencoder(5, autoencoder, F.mse_loss, opt, dt, dv)

# %%
opt = optim.SGD(autoencoder.parameters(), lr=0.1)
fit_autoencoder(5, autoencoder, F.mse_loss, opt, dt, dv)

# %% [markdown]
# Por último se obtienen las imágenes que generó el Autoencoder.

# %%
p = autoencoder(x_batch)
show_images(p[:16].data.cpu(), imsize=1.5)

# %% [markdown]
# Comparadas a las originales no están muy alejadas a pesar de ser un autoencoder extremadamente simple y sin optimizaciones.

# %%
show_images(x_batch[:16].data.cpu(), imsize=1.5)

# %% [markdown]
# ## Bibliografía

# %% [markdown]
# La mayoría de recursos de esta guía de laboratorio fueron tomados y adaptados del curso de [**Practical Deep Learning for Coders**](https://course.fast.ai/) de Jeremy Howard y de su repostirorio en [Github](https://github.com/fastai/course22p2)
#
# - Howard, J. (2022). *Practical Deep Learning for Coders*. FastAi. [https://course.fast.ai/](https://course.fast.ai/).
#
# - Kleinsmith, M. (2017). *CNNs from different viewpoints*. Medium. [https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)
