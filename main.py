import tenserflow as tf
from tenserflow.keras import *
import numpy as np

inputd = 10
outputd = 4

mod = models.Sequential([
    layers.Dense(64,activation = "relu", input_shape=(inputd,)),
    layers.Dense(128,activation = "relu"),
    layers.Dense(64,activation = "relu"),
    layers.Dense(outputd)
])

mod.compile(optimizer = "adam", loss = "mse", metrics=["mae"])

ns = 1000
xt = np.random.rand(ns,inputd)
yt = np.random.rand(ns,outputd)

history = mod.fit(xt, yt, epochs=50, batch_size=32, validation_split=0.2)

xts = np.random.rand(200, inputd)
yts = np.random.rand(200, outputd)

loss, mae = mod.evaluate(xts, yts)
print(f'Test Loss: {loss}, Test MAE: {mae}')